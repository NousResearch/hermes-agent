#!/usr/bin/env bash
# =============================================================================
# Hermes AgentCyber — Live USB Writer
# =============================================================================
#
# Writes the Hermes live ISO to a USB drive and optionally provisions it
# with a pre-configured ~/.hermes directory (API keys, gateway config).
#
# ⚠  This OVERWRITES the target drive. Verify the device path carefully.
#
# Usage:
#   sudo ./write_usb.sh [OPTIONS]
#
# Options:
#   --iso PATH        Path to the ISO file (default: ./hermes-cyber-live.iso)
#   --device PATH     Target USB device e.g. /dev/sdb (prompts if omitted)
#   --provision PATH  Path to a .hermes config dir or .tar.gz to inject
#   --verify          Verify the write with SHA-256 after completion
#   --list            List removable block devices and exit
#   --yes             Skip confirmation prompt (non-interactive mode)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults ---------------------------------------------------------------
ISO="${SCRIPT_DIR}/hermes-cyber-live.iso"
DEVICE=""
PROVISION_PATH=""
VERIFY=false
LIST_ONLY=false
AUTO_YES=false

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iso)       ISO="$2";            shift 2 ;;
    --device)    DEVICE="$2";         shift 2 ;;
    --provision) PROVISION_PATH="$2"; shift 2 ;;
    --verify)    VERIFY=true;         shift   ;;
    --list)      LIST_ONLY=true;      shift   ;;
    --yes)       AUTO_YES=true;       shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---- Privilege check --------------------------------------------------------
if [[ $EUID -ne 0 && "$LIST_ONLY" == "false" ]]; then
  echo "❌  Writing to a block device requires root."
  echo "    Use: sudo $0 $*"
  exit 1
fi

# ---- List removable drives --------------------------------------------------
list_removable() {
  echo "Removable / USB block devices:"
  echo ""
  lsblk -d -o NAME,SIZE,TRAN,MODEL,RM --sort NAME | \
    awk 'NR==1 || $5=="1"' | \
    column -t
  echo ""
  echo "Only drives with RM=1 (removable) are shown."
}

if [[ "$LIST_ONLY" == "true" ]]; then
  list_removable
  exit 0
fi

# ---- Validate ISO -----------------------------------------------------------
if [[ ! -f "$ISO" ]]; then
  echo "❌  ISO not found: $ISO"
  echo "    Build it first: sudo ./build_iso.sh"
  exit 1
fi
ISO_SIZE=$(du -sh "$ISO" | cut -f1)

# ---- Device selection -------------------------------------------------------
if [[ -z "$DEVICE" ]]; then
  echo ""
  echo "Available removable drives:"
  list_removable
  echo -n "Enter target device (e.g. /dev/sdb): "
  read -r DEVICE
fi

# Sanity checks
if [[ ! -b "$DEVICE" ]]; then
  echo "❌  Not a block device: $DEVICE"
  exit 1
fi

# Refuse to write to a mounted system partition
if mount | grep -q "^${DEVICE} "; then
  echo "❌  Device ${DEVICE} appears to be mounted as a system device. Refusing."
  exit 1
fi

# Check removable flag (advisory — not a hard block since some USB docks report 0)
REMOVABLE=$(cat /sys/block/$(basename "$DEVICE")/removable 2>/dev/null || echo "?")
if [[ "$REMOVABLE" == "0" ]]; then
  echo "⚠  WARNING: /sys/block/$(basename "$DEVICE")/removable = 0"
  echo "   This may be a non-removable drive. Double-check the device path."
fi

DRIVE_SIZE=$(lsblk -d -n -o SIZE "$DEVICE" 2>/dev/null || echo "unknown")
DRIVE_MODEL=$(lsblk -d -n -o MODEL "$DEVICE" 2>/dev/null | xargs || echo "unknown")

# ---- Final confirmation -----------------------------------------------------
echo ""
echo "══════════════════════════════════════════════"
echo "  ⚠  ABOUT TO OVERWRITE A DRIVE"
echo "══════════════════════════════════════════════"
echo "  Source ISO : ${ISO} (${ISO_SIZE})"
echo "  Target     : ${DEVICE}  (${DRIVE_SIZE}, ${DRIVE_MODEL})"
if [[ -n "$PROVISION_PATH" ]]; then
  echo "  Provision  : ${PROVISION_PATH}"
fi
echo "══════════════════════════════════════════════"
echo ""

if [[ "$AUTO_YES" == "false" ]]; then
  read -r -p "Type YES to proceed: " CONFIRM
  [[ "$CONFIRM" != "YES" ]] && { echo "Aborted."; exit 1; }
fi

# Unmount any partitions on the target device
echo ""
echo "▶ Unmounting any existing partitions on ${DEVICE}..."
for part in $(lsblk -ln -o NAME "${DEVICE}" | tail -n +2); do
  umount "/dev/${part}" 2>/dev/null && echo "  Unmounted /dev/${part}" || true
done

# ---- Write ISO --------------------------------------------------------------
echo ""
echo "▶ Writing ISO to ${DEVICE}..."
echo "  (This may take several minutes depending on USB speed)"
ISO_BYTES=$(stat -c%s "$ISO")
pv -s "$ISO_BYTES" "$ISO" 2>/dev/null | dd of="$DEVICE" bs=4M conv=fsync,noerror status=none \
  || dd if="$ISO" of="$DEVICE" bs=4M conv=fsync,noerror status=progress
sync
echo "  ✓ ISO written"

# ---- Optional provision step ------------------------------------------------
if [[ -n "$PROVISION_PATH" ]]; then
  echo ""
  echo "▶ Provisioning config..."
  # Re-read partition table
  partprobe "$DEVICE" 2>/dev/null || true
  sleep 2

  # Find the data partition (last partition on the device, or a FAT partition)
  # The ISO leaves space for a casper-rw or config partition after partition 2
  # For simplicity we write config to a file on the ISO's root filesystem
  # by mounting the squashfs overlay — but squashfs is read-only.
  # Instead, we write a config tarball to a small extra FAT32 partition
  # at the end of the USB.
  PROVISION_PART="${DEVICE}3"
  USB_SIZE_SECTORS=$(blockdev --getsz "$DEVICE")
  ISO_SIZE_SECTORS=$(( (ISO_BYTES + 511) / 512 ))
  FREE_SECTORS=$(( USB_SIZE_SECTORS - ISO_SIZE_SECTORS - 2048 ))

  if [[ $FREE_SECTORS -gt 65536 ]]; then
    echo "  Creating config partition on free space..."
    parted -s "$DEVICE" mkpart primary fat32 \
      "${ISO_SIZE_SECTORS}s" "100%" 2>/dev/null || true
    partprobe "$DEVICE" 2>/dev/null || true
    sleep 1
    mkfs.fat -F32 -n HERMESCFG "${PROVISION_PART}" 2>/dev/null || true

    MNT_TMP=$(mktemp -d)
    mount "${PROVISION_PART}" "${MNT_TMP}"
    if [[ -d "$PROVISION_PATH" ]]; then
      tar czf "${MNT_TMP}/hermes-config.tar.gz" -C "$(dirname "$PROVISION_PATH")" "$(basename "$PROVISION_PATH")"
    elif [[ -f "$PROVISION_PATH" ]]; then
      cp "$PROVISION_PATH" "${MNT_TMP}/hermes-config.tar.gz"
    fi
    sync
    umount "${MNT_TMP}"
    rm -rf "${MNT_TMP}"
    echo "  ✓ Config provisioned to ${PROVISION_PART}"
  else
    echo "  ⚠  Not enough free space on USB for config partition (need ≥32 MB free after ISO)"
  fi
fi

# ---- Verify -----------------------------------------------------------------
if [[ "$VERIFY" == "true" ]]; then
  echo ""
  echo "▶ Verifying write (SHA-256)..."
  ISO_SUM=$(sha256sum "$ISO" | cut -d' ' -f1)
  USB_SUM=$(dd if="$DEVICE" bs=4M count=$(( (ISO_BYTES + 4194303) / 4194304 )) 2>/dev/null | sha256sum | cut -d' ' -f1)
  if [[ "$ISO_SUM" == "$USB_SUM" ]]; then
    echo "  ✓ Verify passed — USB matches ISO"
  else
    echo "  ❌  Verify FAILED — sums differ"
    echo "      ISO: $ISO_SUM"
    echo "      USB: $USB_SUM"
    exit 1
  fi
fi

sync
echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅  USB ready. Eject and plug into target PC."
echo "  🔑  Default login: hermes / hermes"
echo "  🤖  Gateway starts automatically after setup."
echo "═══════════════════════════════════════════════"
