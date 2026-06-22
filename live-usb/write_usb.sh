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
#   --iso PATH           Path to the ISO file (default: ./hermes-cyber-live.iso)
#   --device PATH        Target USB device e.g. /dev/sdb (prompts if omitted)
#   --provision PATH     Path to a .hermes config dir or .tar.gz to inject
#   --persistence [SIZE] Create a read-write persistence partition (default: 4G).
#                        Changes made on the live system survive reboots.
#                        Choose "Persistence" from the GRUB boot menu to use it.
#   --encrypt            Encrypt the persistence partition with LUKS2 (AES-256-XTS).
#                        Requires cryptsetup. Prompts for passphrase unless
#                        LUKS_PASSPHRASE env var is set. live-boot will ask for
#                        the passphrase at boot before mounting persistence.
#   --verify             Verify the write with SHA-256 after completion
#   --list               List removable block devices and exit
#   --yes                Skip confirmation prompt (non-interactive mode)
#
# Partition layout after write:
#   p1+p2  ISO hybrid MBR/EFI (written by dd from ISO)
#   p3     FAT32 HERMESCFG  (256 MB, config — created when --provision is used)
#   pN     ext4  HERMESPST  (SIZE,   persistence — created when --persistence is used)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults ---------------------------------------------------------------
ISO="${SCRIPT_DIR}/hermes-cyber-live.iso"
DEVICE=""
PROVISION_PATH=""
PERSISTENCE_SIZE=""   # empty = no persistence partition
ENCRYPT_PERSIST=false
VERIFY=false
LIST_ONLY=false
AUTO_YES=false

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iso)         ISO="$2";              shift 2 ;;
    --device)      DEVICE="$2";           shift 2 ;;
    --provision)   PROVISION_PATH="$2";   shift 2 ;;
    --persistence)
      # Optional size argument: --persistence 8G or just --persistence
      if [[ $# -gt 1 && "$2" =~ ^[0-9]+[GMgm]?$ ]]; then
        PERSISTENCE_SIZE="$2"; shift 2
      else
        PERSISTENCE_SIZE="4G"; shift
      fi
      ;;
    --encrypt)     ENCRYPT_PERSIST=true;   shift ;;
    --verify)      VERIFY=true;           shift ;;
    --list)        LIST_ONLY=true;        shift ;;
    --yes)         AUTO_YES=true;         shift ;;
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

_canonical_removable_device() {
  local input="$1"
  local canonical=""
  canonical="$(readlink -f -- "$input" 2>/dev/null || true)"

  if [[ -z "$canonical" || "$canonical" != /dev/* ]]; then
    echo "❌  Target must resolve to a canonical /dev/... block device: $input" >&2
    return 1
  fi
  if [[ ! -b "$canonical" ]]; then
    echo "❌  Not a block device: $canonical" >&2
    return 1
  fi

  local device_type=""
  device_type="$(lsblk -dn -o TYPE -- "$canonical" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$device_type" != "disk" ]]; then
    echo "❌  Target must be a whole removable disk, not a partition or mapper: $canonical" >&2
    return 1
  fi

  local base="$(basename "$canonical")"
  local removable_path="/sys/class/block/${base}/removable"
  local removable=""
  if [[ -r "$removable_path" ]]; then
    removable="$(tr -d '[:space:]' < "$removable_path")"
  fi
  if [[ "$removable" != "1" ]]; then
    echo "❌  Refusing to write: Linux does not verify ${canonical} as removable media." >&2
    echo "    Expected ${removable_path} to contain 1; got '${removable:-unreadable}'." >&2
    echo "    Root/operator approval is not enough without verifiable removable-media metadata." >&2
    return 1
  fi

  printf '%s\n' "$canonical"
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
ISO_BYTES=$(stat -c%s "$ISO")

# ---- Device selection -------------------------------------------------------
if [[ -z "$DEVICE" ]]; then
  echo ""
  echo "Available removable drives:"
  list_removable
  echo -n "Enter target device (e.g. /dev/sdb): "
  read -r DEVICE
fi

DEVICE="$(_canonical_removable_device "$DEVICE")" || exit 1

if mount | grep -q "^${DEVICE} "; then
  echo "❌  Device ${DEVICE} appears to be mounted as a system device. Refusing."
  exit 1
fi

DRIVE_SIZE=$(lsblk -d -n -o SIZE "$DEVICE" 2>/dev/null || echo "unknown")
DRIVE_MODEL=$(lsblk -d -n -o MODEL "$DEVICE" 2>/dev/null | xargs || echo "unknown")

# ---- Capacity check for persistence -----------------------------------------
if [[ -n "$PERSISTENCE_SIZE" ]]; then
  DRIVE_BYTES=$(blockdev --getsize64 "$DEVICE" 2>/dev/null || echo 0)
  # Parse persistence size to bytes for comparison
  PSZ="${PERSISTENCE_SIZE^^}"
  if [[ "$PSZ" == *G ]]; then
    PERSIST_BYTES=$(( ${PSZ%G} * 1024 * 1024 * 1024 ))
  elif [[ "$PSZ" == *M ]]; then
    PERSIST_BYTES=$(( ${PSZ%M} * 1024 * 1024 ))
  else
    PERSIST_BYTES=$(( PSZ ))
  fi
  CONFIG_BYTES=$(( 256 * 1024 * 1024 ))  # 256 MB for config partition
  NEEDED=$(( ISO_BYTES + CONFIG_BYTES + PERSIST_BYTES + 64 * 1024 * 1024 ))
  if [[ $DRIVE_BYTES -lt $NEEDED ]]; then
    echo "❌  Drive too small for ISO + config + ${PERSISTENCE_SIZE} persistence."
    echo "    Drive: $(( DRIVE_BYTES / 1024 / 1024 / 1024 ))GB  Needed: $(( NEEDED / 1024 / 1024 / 1024 ))GB"
    exit 1
  fi
fi

# ---- Final confirmation -----------------------------------------------------
echo ""
echo "══════════════════════════════════════════════"
echo "  ⚠  ABOUT TO OVERWRITE A DRIVE"
echo "══════════════════════════════════════════════"
echo "  Source ISO    : ${ISO} (${ISO_SIZE})"
echo "  Target        : ${DEVICE}  (${DRIVE_SIZE}, ${DRIVE_MODEL})"
[[ -n "$PROVISION_PATH" ]]  && echo "  Config        : ${PROVISION_PATH} → p3 FAT32 HERMESCFG"
if [[ -n "$PERSISTENCE_SIZE" ]]; then
  _enc_label="ext4 HERMESPST"
  [[ "$ENCRYPT_PERSIST" == "true" ]] && _enc_label="LUKS2+ext4 HERMESPST (encrypted)"
  echo "  Persistence   : ${PERSISTENCE_SIZE} → ${_enc_label} (select in GRUB)"
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
pv -s "$ISO_BYTES" "$ISO" 2>/dev/null | dd of="$DEVICE" bs=4M conv=fsync,noerror status=none \
  || dd if="$ISO" of="$DEVICE" bs=4M conv=fsync,noerror status=progress
sync
echo "  ✓ ISO written"

# Re-read partition table before adding new partitions
partprobe "$DEVICE" 2>/dev/null || true
sleep 2

# ---- Config partition (p3) --------------------------------------------------
_create_config_part() {
  local iso_end_sectors=$(( (ISO_BYTES + 511) / 512 ))
  local start_sector=$(( iso_end_sectors + 2048 ))  # align

  echo "  Creating config partition (256 MB, FAT32, HERMESCFG)..."
  parted -s "$DEVICE" mkpart primary fat32 \
    "${start_sector}s" "$(( start_sector + 524288 ))s" 2>/dev/null || return 1
  partprobe "$DEVICE" 2>/dev/null || true
  sleep 1

  local cfg_part
  cfg_part=$(lsblk -ln -o NAME "$DEVICE" | grep -v "^$(basename "$DEVICE")$" | tail -1)
  cfg_part="/dev/${cfg_part}"

  mkfs.fat -F32 -n HERMESCFG "${cfg_part}" 2>/dev/null
  echo "${cfg_part}"
}

if [[ -n "$PROVISION_PATH" ]]; then
  echo ""
  echo "▶ Creating config partition and provisioning..."
  CFG_PART=$(_create_config_part) || {
    echo "  ⚠  Could not create config partition. Skipping provisioning."
    CFG_PART=""
  }

  if [[ -n "$CFG_PART" ]]; then
    MNT_TMP=$(mktemp -d)
    mount "${CFG_PART}" "${MNT_TMP}"
    if [[ -d "$PROVISION_PATH" ]]; then
      tar czf "${MNT_TMP}/hermes-config.tar.gz" \
        -C "$(dirname "$PROVISION_PATH")" "$(basename "$PROVISION_PATH")"
    elif [[ -f "$PROVISION_PATH" ]]; then
      cp "$PROVISION_PATH" "${MNT_TMP}/hermes-config.tar.gz"
    fi
    sync
    umount "${MNT_TMP}"
    rm -rf "${MNT_TMP}"
    echo "  ✓ Config provisioned to ${CFG_PART} (HERMESCFG)"
  fi
fi

# ---- Persistence partition ---------------------------------------------------
if [[ -n "$PERSISTENCE_SIZE" ]]; then
  echo ""
  echo "▶ Creating persistence partition (${PERSISTENCE_SIZE}, ext4, HERMESPST)..."

  partprobe "$DEVICE" 2>/dev/null || true
  sleep 1

  # Start after whatever's already on the drive, end at 100% minus alignment
  parted -s "$DEVICE" mkpart primary ext4 \
    "-$(( $(numfmt --from=iec "${PERSISTENCE_SIZE}" 2>/dev/null || echo $PERSIST_BYTES) / 512 + 2048 ))s" \
    "100%" 2>/dev/null || \
  parted -s "$DEVICE" mkpart primary ext4 \
    "$(parted -s "$DEVICE" unit s print | awk '/^ [0-9]/{last=$3} END{gsub(/s/,"",last); print last+1}')s" \
    "100%" 2>/dev/null

  partprobe "$DEVICE" 2>/dev/null || true
  sleep 2

  PERSIST_PART=$(lsblk -ln -o NAME "$DEVICE" | grep -v "^$(basename "$DEVICE")$" | tail -1)
  PERSIST_PART="/dev/${PERSIST_PART}"

  if [[ "$ENCRYPT_PERSIST" == "true" ]]; then
    # ---- LUKS2 encrypted persistence ----------------------------------------
    if ! command -v cryptsetup &>/dev/null; then
      echo "❌  cryptsetup not found. Install: apt-get install cryptsetup"
      exit 1
    fi

    # Obtain passphrase — env var for non-interactive CI, interactive prompt otherwise
    if [[ -z "${LUKS_PASSPHRASE:-}" ]]; then
      read -r -s -p "  Enter LUKS passphrase for persistence partition: " LUKS_PASSPHRASE
      echo
      read -r -s -p "  Confirm passphrase: " LUKS_CONFIRM
      echo
      if [[ "$LUKS_PASSPHRASE" != "$LUKS_CONFIRM" ]]; then
        echo "❌  Passphrases do not match."
        exit 1
      fi
    fi

    echo "  Formatting LUKS2 container on ${PERSIST_PART}..."
    echo "$LUKS_PASSPHRASE" | cryptsetup luksFormat \
      --batch-mode \
      --type luks2 \
      --cipher aes-xts-plain64 \
      --key-size 512 \
      --hash sha256 \
      --iter-time 2000 \
      "${PERSIST_PART}"

    echo "  Opening LUKS container..."
    echo "$LUKS_PASSPHRASE" | cryptsetup luksOpen "${PERSIST_PART}" hermespst

    _MAPPED="/dev/mapper/hermespst"
    mkfs.ext4 -L HERMESPST -E lazy_itable_init=0 "${_MAPPED}" 2>&1 | tail -3

    MNT_TMP=$(mktemp -d)
    mount "${_MAPPED}" "${MNT_TMP}"
    echo "/ union" > "${MNT_TMP}/persistence.conf"
    mkdir -p "${MNT_TMP}/home/hermes/.hermes/logs"
    chown -R 1000:1000 "${MNT_TMP}/home" 2>/dev/null || true
    sync
    umount "${MNT_TMP}"
    rm -rf "${MNT_TMP}"

    cryptsetup luksClose hermespst

    echo "  ✓ Persistence partition ready (${PERSIST_PART}, LUKS2-encrypted, label=HERMESPST)"
    echo "  ℹ  live-boot will prompt for the LUKS passphrase at boot."
  else
    # ---- Plain ext4 persistence (default) -----------------------------------
    mkfs.ext4 -L HERMESPST -E lazy_itable_init=0 "${PERSIST_PART}" 2>&1 | tail -3

    MNT_TMP=$(mktemp -d)
    mount "${PERSIST_PART}" "${MNT_TMP}"
    echo "/ union" > "${MNT_TMP}/persistence.conf"
    mkdir -p "${MNT_TMP}/home/hermes/.hermes/logs"
    chown -R 1000:1000 "${MNT_TMP}/home" 2>/dev/null || true
    sync
    umount "${MNT_TMP}"
    rm -rf "${MNT_TMP}"

    echo "  ✓ Persistence partition ready (${PERSIST_PART}, label=HERMESPST)"
    echo "  ℹ  Select 'Persistence' in the GRUB boot menu to use it."
  fi
fi

# ---- Verify -----------------------------------------------------------------
if [[ "$VERIFY" == "true" ]]; then
  echo ""
  echo "▶ Verifying write (SHA-256)..."
  ISO_SUM=$(sha256sum "$ISO" | cut -d' ' -f1)
  USB_SUM=$(dd if="$DEVICE" bs=4M count=$(( (ISO_BYTES + 4194303) / 4194304 )) 2>/dev/null \
    | sha256sum | cut -d' ' -f1)
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
if [[ -n "$PERSISTENCE_SIZE" ]]; then
  [[ "$ENCRYPT_PERSIST" == "true" ]] && \
    echo "  🔒  Encrypted persistence: LUKS passphrase required at boot."
  [[ "$ENCRYPT_PERSIST" != "true" ]] && \
    echo "  💾  Persistence: select it in the GRUB menu."
fi
echo "═══════════════════════════════════════════════"
