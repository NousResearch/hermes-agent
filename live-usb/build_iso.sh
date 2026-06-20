#!/usr/bin/env bash
# =============================================================================
# Hermes AgentCyber — Live USB ISO Builder
# =============================================================================
#
# Builds a bootable Kali Linux (kali-rolling) ISO with Hermes AgentCyber
# pre-installed. Kali gives the full offensive/defensive toolkit out of the
# box: nmap, Metasploit, Burp Suite, sqlmap, Hydra, hashcat, Wireshark,
# aircrack-ng, and hundreds more — all pre-built and maintained by Offensive
# Security.
#
# Supported architectures:
#   amd64 — hybrid BIOS + UEFI boot (write with dd to any USB)
#   arm64 — EFI-only boot (Raspberry Pi 4/5, ARM servers)
#           Requires qemu-user-static + binfmt-support on the build host.
#
# Kali metapackage options (--kali-meta):
#   kali-tools-top10       ~10 essential tools; smallest ISO (~3 GB)
#   kali-linux-headless    Full headless suite; recommended (~5 GB)  ← default
#   kali-linux-default     Everything including GUI tools (~8 GB)
#
# Alternatively build on Debian with --suite bookworm --mirror http://deb.debian.org/debian
# (omit --kali-meta; a base set of tools is installed instead).
#
# Requirements — amd64 Kali/Debian/Ubuntu build host:
#   apt-get install -y debootstrap squashfs-tools xorriso \
#                      grub-efi-amd64-bin grub-pc-bin mtools dosfstools
#
# Additional requirements — ARM64 cross-compilation:
#   apt-get install -y qemu-user-static binfmt-support grub-efi-arm64-bin
#
# Usage:
#   sudo ./build_iso.sh [OPTIONS]
#
# Options:
#   --arch ARCH           Target architecture: amd64 (default) or arm64
#   --suite SUITE         OS suite (default: kali-rolling)
#   --mirror URL          APT mirror URL (default: Kali CDN)
#   --kali-meta PKG       Kali metapackage (default: kali-linux-headless)
#   --output PATH         Output ISO path (default: ./hermes-cyber-live.iso)
#   --source-dir PATH     Path to hermes-agentcyber source (default: parent dir)
#   --no-bundle-source    Don't bundle source; install from pip on first boot
#   --headless-scan       Enable auto-scan on boot (requires config.yaml present)
#   --verbose             Verbose debootstrap output
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."

# ---- Defaults ---------------------------------------------------------------
ARCH="amd64"
SUITE="kali-rolling"
MIRROR="https://http.kali.org/kali"
KALI_META="kali-linux-headless"
OUTPUT="${SCRIPT_DIR}/hermes-cyber-live.iso"
SOURCE_DIR="${REPO_DIR}"
BUNDLE_SOURCE=true
HEADLESS_SCAN=false
VERBOSE=false

WORK_DIR=""   # set after arg parsing; cleaned up on exit

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)            ARCH="$2";         shift 2 ;;
    --suite)           SUITE="$2";        shift 2 ;;
    --mirror)          MIRROR="$2";       shift 2 ;;
    --kali-meta)       KALI_META="$2";    shift 2 ;;
    --output)          OUTPUT="$2";       shift 2 ;;
    --source-dir)      SOURCE_DIR="$2";   shift 2 ;;
    --no-bundle-source) BUNDLE_SOURCE=false; shift ;;
    --headless-scan)   HEADLESS_SCAN=true; shift ;;
    --verbose)         VERBOSE=true;      shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Detect OS family from suite name
if [[ "$SUITE" == kali* ]]; then
  BASE_OS="kali"
else
  BASE_OS="debian"
  KALI_META=""   # no kali meta on Debian builds
fi

# ---- Architecture setup -----------------------------------------------------
case "$ARCH" in
  amd64)
    GRUB_EFI_PKG="grub-efi-amd64-bin"
    EFI_NAME="bootx64.efi"
    EFI_GRUB_FORMAT="x86_64-efi"
    SQUASHFS_BCJ="x86"
    CROSS_COMPILE=false
    ;;
  arm64)
    GRUB_EFI_PKG="grub-efi-arm64-bin"
    EFI_NAME="bootaa64.efi"
    EFI_GRUB_FORMAT="arm64-efi"
    SQUASHFS_BCJ="arm"
    CROSS_COMPILE=true
    ;;
  *)
    echo "❌  Unsupported architecture: ${ARCH}  (supported: amd64, arm64)"
    exit 1
    ;;
esac

# ---- Privilege check --------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "❌  This script must run as root (required for debootstrap + mount)."
  echo "    Use: sudo $0 $*"
  exit 1
fi

# ---- Dependency check -------------------------------------------------------
REQUIRED_CMDS=(debootstrap mksquashfs xorriso mformat mkdosfs)
[[ "$CROSS_COMPILE" == "true" ]] && REQUIRED_CMDS+=(qemu-aarch64-static)
MISSING=()
for cmd in "${REQUIRED_CMDS[@]}"; do
  command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "❌  Missing build dependencies: ${MISSING[*]}"
  echo "    Install with:"
  if [[ "$CROSS_COMPILE" == "true" ]]; then
    echo "    apt-get install -y debootstrap squashfs-tools xorriso ${GRUB_EFI_PKG} mtools dosfstools qemu-user-static binfmt-support"
  else
    echo "    apt-get install -y debootstrap squashfs-tools xorriso grub-efi-amd64-bin grub-pc-bin mtools dosfstools"
  fi
  exit 1
fi

# For ARM64 cross-compilation, ensure binfmt is registered
if [[ "$CROSS_COMPILE" == "true" ]]; then
  if ! update-binfmts --display qemu-aarch64 2>/dev/null | grep -q "enabled"; then
    update-binfmts --enable qemu-aarch64 2>/dev/null || true
  fi
fi

# ---- Working directory ------------------------------------------------------
WORK_DIR="$(mktemp -d /tmp/hermeslive-XXXXXX)"
ROOTFS="${WORK_DIR}/rootfs"
ISO_STAGING="${WORK_DIR}/iso"
SQUASHFS_DIR="${ISO_STAGING}/live"

cleanup() {
  echo "🧹  Cleaning up..."
  # Unmount any leftover mounts
  for mnt in "${ROOTFS}/proc" "${ROOTFS}/sys" "${ROOTFS}/dev/pts" "${ROOTFS}/dev"; do
    mountpoint -q "$mnt" 2>/dev/null && umount -l "$mnt" || true
  done
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

echo "═══════════════════════════════════════════════════════════"
echo "  Hermes AgentCyber — Live USB Builder"
echo "  Base OS: ${BASE_OS^^}  Suite: ${SUITE}  Arch: ${ARCH}"
[[ -n "$KALI_META" ]] && echo "  Kali meta: ${KALI_META}"
echo "  Mirror: ${MIRROR}"
echo "  Output: ${OUTPUT}"
[[ "$CROSS_COMPILE" == "true" ]] && echo "  Mode: ARM64 cross-compilation (via qemu-user-static)"
echo "═══════════════════════════════════════════════════════════"

# ---- Step 1: debootstrap base system ----------------------------------------
echo ""
echo "▶ [1/7] Bootstrapping ${SUITE} (${ARCH})..."
mkdir -p "${ROOTFS}"

# Kali bootstrap: need the Kali keyring and (on non-Kali hosts) the debootstrap script
if [[ "$BASE_OS" == "kali" ]]; then
  # Import Kali archive signing key so debootstrap can verify packages
  if [[ ! -f /usr/share/keyrings/kali-archive-keyring.gpg ]]; then
    echo "  Importing Kali archive key..."
    curl -fsSL https://archive.kali.org/archive-key.asc \
      | gpg --dearmor > /usr/share/keyrings/kali-archive-keyring.gpg
  fi
  # Provide kali-rolling debootstrap script if host debootstrap doesn't have it
  if [[ ! -f /usr/share/debootstrap/scripts/kali-rolling ]]; then
    ln -sf /usr/share/debootstrap/scripts/debian \
           /usr/share/debootstrap/scripts/kali-rolling
  fi
  DEBOOT_KEYRING="--keyring=/usr/share/keyrings/kali-archive-keyring.gpg"
  # Minimal includes: Kali meta installs the tools; live-boot handles squashfs mount
  DEBOOT_INCLUDES="systemd,systemd-sysv,dbus,locales,ca-certificates,curl,wget,sudo,python3,python3-pip,python3-venv,less,vim,tmux,git,live-boot,live-config,live-config-systemd"
else
  DEBOOT_KEYRING=""
  # Debian: install a base set of useful tools (Kali meta not available)
  DEBOOT_INCLUDES="systemd,systemd-sysv,dbus,locales,ca-certificates,curl,wget,iproute2,iputils-ping,net-tools,nmap,tcpdump,openssh-client,git,sudo,python3,python3-pip,python3-venv,less,vim,tmux,parted,usbutils,pciutils,lsof,strace,file,binutils,ncat,live-boot,live-config,live-config-systemd"
fi

DEBOOT_FLAGS="--arch=${ARCH} ${DEBOOT_KEYRING} --include=${DEBOOT_INCLUDES}"

if [[ "$CROSS_COMPILE" == "true" ]]; then
  # Two-stage bootstrap for cross-architecture builds
  if [[ "$VERBOSE" == "true" ]]; then
    debootstrap $DEBOOT_FLAGS --foreign "${SUITE}" "${ROOTFS}" "${MIRROR}"
  else
    debootstrap $DEBOOT_FLAGS --foreign "${SUITE}" "${ROOTFS}" "${MIRROR}" 2>&1 | grep -E "^I:|^E:|^W:" || true
  fi
  # Inject qemu binary so chroot can execute arm64 binaries
  cp /usr/bin/qemu-aarch64-static "${ROOTFS}/usr/bin/"
  echo "  Running debootstrap second stage (in qemu-aarch64 chroot)..."
  chroot "${ROOTFS}" /debootstrap/debootstrap --second-stage 2>&1 | grep -E "^I:|^E:|^W:" || true
else
  if [[ "$VERBOSE" == "true" ]]; then
    debootstrap $DEBOOT_FLAGS "${SUITE}" "${ROOTFS}" "${MIRROR}"
  else
    debootstrap $DEBOOT_FLAGS "${SUITE}" "${ROOTFS}" "${MIRROR}" 2>&1 | grep -E "^I:|^E:|^W:" || true
  fi
fi
echo "  ✓ Base system bootstrapped"

# ---- Step 2: Mount virtual filesystems for chroot ---------------------------
echo ""
echo "▶ [2/7] Preparing chroot environment..."
mount --bind /dev  "${ROOTFS}/dev"
mount --bind /dev/pts "${ROOTFS}/dev/pts"
mount -t proc proc "${ROOTFS}/proc"
mount -t sysfs sysfs "${ROOTFS}/sys"
echo "  ✓ Virtual filesystems mounted"

# ---- Step 3: Bundle hermes source -------------------------------------------
if [[ "$BUNDLE_SOURCE" == "true" ]]; then
  echo ""
  echo "▶ [3/7] Bundling Hermes AgentCyber source..."
  # Create a tarball of the repo, excluding git history and build artifacts
  SOURCE_TARBALL="${WORK_DIR}/hermes-agentcyber.tar.gz"
  tar czf "${SOURCE_TARBALL}" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.mypy_cache' \
    --exclude='.pytest_cache' \
    --exclude='node_modules' \
    --exclude='dist' \
    --exclude='build' \
    -C "${SOURCE_DIR}/.." \
    "$(basename "${SOURCE_DIR}")"
  mkdir -p "${ROOTFS}/opt"
  cp "${SOURCE_TARBALL}" "${ROOTFS}/opt/hermes-agentcyber.tar.gz"
  echo "  ✓ Source bundled ($(du -sh "${SOURCE_TARBALL}" | cut -f1))"
else
  echo ""
  echo "▶ [3/7] Skipping source bundle (will install from PyPI on first boot)"
fi

# ---- Step 4: chroot setup script -------------------------------------------
echo ""
echo "▶ [4/7] Running chroot setup..."
HEADLESS_FLAG=""
[[ "$HEADLESS_SCAN" == "true" ]] && HEADLESS_FLAG="--headless-scan"
BUNDLE_FLAG=""
[[ "$BUNDLE_SOURCE" == "true" ]] && BUNDLE_FLAG="--bundled-source"
KALI_META_FLAG=""
[[ -n "$KALI_META" ]] && KALI_META_FLAG="--kali-meta=${KALI_META}"

# Pass Kali mirror into chroot for sources.list
KALI_MIRROR="${MIRROR}"

cat > "${ROOTFS}/tmp/chroot_setup.sh" << CHROOT_EOF
#!/usr/bin/env bash
set -euo pipefail

BUNDLE_SOURCE=false
HEADLESS_SCAN=false
KALI_META=""
for arg in "\$@"; do
  [[ "\$arg" == "--bundled-source" ]] && BUNDLE_SOURCE=true
  [[ "\$arg" == "--headless-scan" ]]  && HEADLESS_SCAN=true
  [[ "\$arg" == --kali-meta=* ]]      && KALI_META="\${arg#--kali-meta=}"
done

export DEBIAN_FRONTEND=noninteractive
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

# ---- Kali tool installation -------------------------------------------------
if [[ -n "\$KALI_META" ]]; then
  echo "  Setting up Kali repositories and installing \${KALI_META}..."
  # Write full sources.list with all Kali components
  cat > /etc/apt/sources.list << 'SOURCES'
deb ${KALI_MIRROR} kali-rolling main contrib non-free non-free-firmware
SOURCES
  # Kali keyring is already in the rootfs from debootstrap; just update + install
  apt-get update -qq
  # kali-linux-headless can take 15-30 min depending on mirror speed
  apt-get install -y --no-install-recommends "\${KALI_META}" 2>&1 | \
    grep -E "^(Setting up|Unpacking|Get:|Err:|E:)" || true
  echo "  ✓ \${KALI_META} installed"
fi

# ---- hermes user ------------------------------------------------------------
# Kali live creates a 'kali' user; we use 'hermes' (UID 1000) instead.
# If a 'kali' user was created by live-config, remove it.
id kali &>/dev/null && userdel -r kali 2>/dev/null || true
id hermes &>/dev/null || useradd -m -u 1000 -G sudo,adm -s /bin/bash hermes
echo "hermes:hermes" | chpasswd
echo "hermes ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/hermes
chmod 440 /etc/sudoers.d/hermes

# Disable Kali live-config's automatic user creation so it doesn't conflict
mkdir -p /etc/live/config.conf.d
echo 'LIVE_CONFIG_NOROOT=false' > /etc/live/config.conf.d/hermes.conf
echo 'LIVE_CONFIG_USERNAME=hermes' >> /etc/live/config.conf.d/hermes.conf
echo 'LIVE_CONFIG_USER_DEFAULT_GROUPS=sudo,adm,cdrom,floppy,audio,dip,video,plugdev,netdev,bluetooth,wireshark' \
  >> /etc/live/config.conf.d/hermes.conf

# ---- Python + uv + hermes ---------------------------------------------------
# Install uv
curl -fsSL https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
install -m755 /root/.local/bin/uv /usr/local/bin/uv

# Find best available Python (3.11+)
PYTHON_BIN=\$(command -v python3.12 || command -v python3.11 || command -v python3)
if [[ -z "\$PYTHON_BIN" ]]; then
  apt-get install -y python3 python3-venv python3-dev
  PYTHON_BIN=python3
fi

HERMES_VENV=/opt/hermes-venv
uv venv --python "\${PYTHON_BIN}" "\${HERMES_VENV}"
HERMES_PYTHON="\${HERMES_VENV}/bin/python"

if [[ "\$BUNDLE_SOURCE" == "true" && -f /opt/hermes-agentcyber.tar.gz ]]; then
  cd /opt
  tar xzf hermes-agentcyber.tar.gz
  HERMES_SRC_DIR=\$(find /opt -maxdepth 1 -type d -name "hermes-agentcyber*" | head -1)
  uv pip install --python "\${HERMES_PYTHON}" --no-cache-dir -e "\${HERMES_SRC_DIR}"
  echo "\${HERMES_SRC_DIR}" > /opt/hermes-source-path
  rm /opt/hermes-agentcyber.tar.gz
else
  touch /opt/hermes-install-on-firstboot
fi

cat >> /home/hermes/.bashrc << 'BASHRC'
export PATH="/opt/hermes-venv/bin:/usr/local/bin:\$PATH"
export HERMES_HOME="/home/hermes/.hermes"
BASHRC
chown hermes:hermes /home/hermes/.bashrc

mkdir -p /home/hermes/.hermes/logs
chown -R hermes:hermes /home/hermes/.hermes

[[ "\$HEADLESS_SCAN" == "true" ]] && touch /opt/hermes-headless-scan

echo "✓ chroot setup complete"
CHROOT_EOF

chmod +x "${ROOTFS}/tmp/chroot_setup.sh"
chroot "${ROOTFS}" /tmp/chroot_setup.sh ${BUNDLE_FLAG} ${HEADLESS_FLAG} ${KALI_META_FLAG}
rm "${ROOTFS}/tmp/chroot_setup.sh"
echo "  ✓ Hermes installed in chroot"

# ---- Step 5: Install overlay files (services, scripts, config) --------------
echo ""
echo "▶ [5/7] Installing overlay files..."
OVERLAY="${SCRIPT_DIR}/rootfs-overlay"

# Copy the entire overlay tree
cp -a "${OVERLAY}/." "${ROOTFS}/"

# Fix permissions on executables
chmod 755 "${ROOTFS}/usr/local/bin/hermes-live"
chmod 755 "${ROOTFS}/usr/local/bin/hermes-firstboot"
chmod 755 "${ROOTFS}/usr/local/bin/hermes-autoupdate"

# Enable systemd services
chroot "${ROOTFS}" systemctl enable hermes-firstboot.service
chroot "${ROOTFS}" systemctl enable hermes-autoupdate.service
chroot "${ROOTFS}" systemctl enable hermes-gateway.service

# Set hostname
echo "hermes-cyber" > "${ROOTFS}/etc/hostname"
echo "127.0.1.1 hermes-cyber" >> "${ROOTFS}/etc/hosts"

# Auto-login hermes user on tty1 for the first-boot wizard
mkdir -p "${ROOTFS}/etc/systemd/system/getty@tty1.service.d"
cat > "${ROOTFS}/etc/systemd/system/getty@tty1.service.d/autologin.conf" << 'AUTOLOGIN'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin hermes --noclear %I $TERM
AUTOLOGIN

echo "  ✓ Overlay files installed"

# ---- Step 6: Build squashfs + ISO -------------------------------------------
echo ""
echo "▶ [6/7] Building squashfs filesystem..."
mkdir -p "${SQUASHFS_DIR}" "${ISO_STAGING}/boot/grub" "${ISO_STAGING}/EFI/BOOT"
mksquashfs "${ROOTFS}" "${SQUASHFS_DIR}/filesystem.squashfs" \
  -comp xz -b 1M -Xbcj "${SQUASHFS_BCJ}" \
  -e "${ROOTFS}/proc/*" "${ROOTFS}/sys/*" "${ROOTFS}/dev/*" \
  -noappend 2>&1 | tail -3
echo "  ✓ Squashfs built ($(du -sh "${SQUASHFS_DIR}/filesystem.squashfs" | cut -f1))"

# Copy kernel + initrd
KERNEL=$(find "${ROOTFS}/boot" -name "vmlinuz-*" | sort -V | tail -1)
INITRD=$(find "${ROOTFS}/boot" -name "initrd.img-*" | sort -V | tail -1)
cp "${KERNEL}" "${ISO_STAGING}/boot/vmlinuz"
cp "${INITRD}" "${ISO_STAGING}/boot/initrd.img"

# GRUB config
cp "${SCRIPT_DIR}/grub/grub.cfg" "${ISO_STAGING}/boot/grub/grub.cfg"
echo "  ✓ Boot files copied"

# ---- Build EFI bootloader ---------------------------------------------------
echo "  Building GRUB EFI image (${EFI_GRUB_FORMAT})..."
grub-mkstandalone \
  --format="${EFI_GRUB_FORMAT}" \
  --output="${WORK_DIR}/${EFI_NAME}" \
  --locales="" \
  --fonts="" \
  "boot/grub/grub.cfg=${ISO_STAGING}/boot/grub/grub.cfg"
cp "${WORK_DIR}/${EFI_NAME}" "${ISO_STAGING}/EFI/BOOT/"

# FAT ESP image (El Torito EFI entry)
EFIIMG="${WORK_DIR}/efi.img"
dd if=/dev/zero of="${EFIIMG}" bs=1M count=4 2>/dev/null
mkdosfs -F 32 "${EFIIMG}" 2>/dev/null
mmd  -i "${EFIIMG}" ::/EFI ::/EFI/BOOT
mcopy -i "${EFIIMG}" "${WORK_DIR}/${EFI_NAME}" "::/EFI/BOOT/${EFI_NAME^^}"
cp "${EFIIMG}" "${ISO_STAGING}/efi.img"
echo "  ✓ EFI bootloader ready"

echo ""
echo "▶ [7/7] Building bootable ISO with xorriso..."
if [[ "$ARCH" == "amd64" ]]; then
  # Hybrid MBR + EFI (works on legacy BIOS and UEFI)
  # Requires grub-pc-bin for the i386-pc stages
  mkdir -p "${ISO_STAGING}/boot/grub/i386-pc"
  if [[ -d /usr/lib/grub/i386-pc ]]; then
    grub-mkstandalone \
      --format=i386-pc \
      --output="${WORK_DIR}/core.img" \
      --locales="" \
      "boot/grub/grub.cfg=${ISO_STAGING}/boot/grub/grub.cfg" 2>/dev/null || true
    cat /usr/lib/grub/i386-pc/cdboot.img "${WORK_DIR}/core.img" \
      > "${ISO_STAGING}/boot/grub/i386-pc/eltorito.img" 2>/dev/null || true
  fi

  MBR_HYBRID_IMG="/usr/lib/grub/i386-pc/boot_hybrid.img"
  MBR_FLAGS=()
  if [[ -f "${MBR_HYBRID_IMG}" ]]; then
    MBR_FLAGS+=(--grub2-mbr "${MBR_HYBRID_IMG}" -partition_offset 16 --mbr-force-bootable)
  fi

  xorriso -as mkisofs \
    -iso-level 3 \
    -full-iso9660-filenames \
    -volid "HERMES-CYBER" \
    -appid "Hermes AgentCyber Live" \
    "${MBR_FLAGS[@]+"${MBR_FLAGS[@]}"}" \
    -c "/boot/grub/boot.cat" \
    -b "/boot/grub/i386-pc/eltorito.img" \
    -no-emul-boot \
    -boot-load-size 4 \
    -boot-info-table \
    --grub2-boot-info \
    -eltorito-alt-boot \
    -e "efi.img" \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    -o "${OUTPUT}" \
    "${ISO_STAGING}" 2>&1 | tail -5
else
  # ARM64 — EFI only (no MBR/legacy BIOS support)
  xorriso -as mkisofs \
    -iso-level 3 \
    -full-iso9660-filenames \
    -volid "HERMES-CYBER" \
    -appid "Hermes AgentCyber Live" \
    -eltorito-alt-boot \
    -e "efi.img" \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    -o "${OUTPUT}" \
    "${ISO_STAGING}" 2>&1 | tail -5
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅  ISO built successfully!"
echo "  📀  Output: ${OUTPUT}"
echo "  📏  Size:   $(du -sh "${OUTPUT}" | cut -f1)"
echo ""
echo "  Write to USB:  sudo ./write_usb.sh --iso ${OUTPUT}"
echo "  Provision:     ./provision.sh --usb /dev/sdX --config config.yaml"
echo "═══════════════════════════════════════════════════════════"
