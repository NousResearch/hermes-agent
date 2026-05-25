#!/usr/bin/env bash
# =============================================================================
# Hermes AgentCyber — Live USB ISO Builder
# =============================================================================
#
# Builds a bootable Debian 12 (bookworm) ISO with Hermes AgentCyber
# pre-installed and configured to auto-start on boot.
#
# The resulting ISO boots on both UEFI and legacy BIOS systems. When the
# system starts it either:
#   1. Runs the first-boot wizard (interactive, on first boot)
#   2. Starts hermes-gateway as a systemd service (subsequent boots)
#   3. Optionally runs an automated cyber sweep (headless mode)
#
# Requirements (install on the BUILD host):
#   apt-get install -y debootstrap squashfs-tools xorriso grub-efi-amd64-bin \
#                      grub-pc-bin mtools dosfstools isolinux
#
# Usage:
#   sudo ./build_iso.sh [OPTIONS]
#
# Options:
#   --arch ARCH           Target architecture (default: amd64)
#   --suite SUITE         Debian suite (default: bookworm)
#   --mirror URL          Debian mirror (default: http://deb.debian.org/debian)
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
SUITE="bookworm"
MIRROR="http://deb.debian.org/debian"
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
    --output)          OUTPUT="$2";       shift 2 ;;
    --source-dir)      SOURCE_DIR="$2";   shift 2 ;;
    --no-bundle-source) BUNDLE_SOURCE=false; shift ;;
    --headless-scan)   HEADLESS_SCAN=true; shift ;;
    --verbose)         VERBOSE=true;      shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---- Privilege check --------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "❌  This script must run as root (required for debootstrap + mount)."
  echo "    Use: sudo $0 $*"
  exit 1
fi

# ---- Dependency check -------------------------------------------------------
REQUIRED_CMDS=(debootstrap mksquashfs xorriso grub-mkrescue mformat)
MISSING=()
for cmd in "${REQUIRED_CMDS[@]}"; do
  command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "❌  Missing build dependencies: ${MISSING[*]}"
  echo "    Install with:"
  echo "    apt-get install -y debootstrap squashfs-tools xorriso grub-efi-amd64-bin grub-pc-bin mtools dosfstools"
  exit 1
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
echo "  Suite: ${SUITE}  Arch: ${ARCH}  Mirror: ${MIRROR}"
echo "  Output: ${OUTPUT}"
echo "═══════════════════════════════════════════════════════════"

# ---- Step 1: debootstrap base system ----------------------------------------
echo ""
echo "▶ [1/7] Bootstrapping Debian ${SUITE} (${ARCH})..."
mkdir -p "${ROOTFS}"
DEBOOT_FLAGS="--arch=${ARCH} --include=systemd,systemd-sysv,dbus,locales,ca-certificates,curl,wget,iproute2,iputils-ping,net-tools,nmap,tcpdump,openssh-client,git,sudo,python3,python3-pip,python3-venv,less,vim,tmux,parted,usbutils,pciutils,lsof,strace,file,binutils,ncat"
if [[ "$VERBOSE" == "true" ]]; then
  debootstrap $DEBOOT_FLAGS "${SUITE}" "${ROOTFS}" "${MIRROR}"
else
  debootstrap $DEBOOT_FLAGS "${SUITE}" "${ROOTFS}" "${MIRROR}" 2>&1 | grep -E "^I:|^E:|^W:" || true
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

cat > "${ROOTFS}/tmp/chroot_setup.sh" << 'CHROOT_EOF'
#!/usr/bin/env bash
set -euo pipefail

BUNDLE_SOURCE=false
HEADLESS_SCAN=false
for arg in "$@"; do
  [[ "$arg" == "--bundled-source" ]] && BUNDLE_SOURCE=true
  [[ "$arg" == "--headless-scan" ]]  && HEADLESS_SCAN=true
done

# System basics
export DEBIAN_FRONTEND=noninteractive
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

# Create hermes user (UID 1000)
useradd -m -u 1000 -G sudo,adm -s /bin/bash hermes
echo "hermes:hermes" | chpasswd
echo "hermes ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/hermes
chmod 440 /etc/sudoers.d/hermes

# Install uv (hermes package manager)
curl -fsSL https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
install -m755 /root/.local/bin/uv /usr/local/bin/uv

# Install Python 3.11 if not available
python3 --version | grep -q "3.1[1-9]" || {
  apt-get install -y python3.11 python3.11-venv python3.11-dev
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10
}

# Set up hermes virtual environment
HERMES_VENV=/opt/hermes-venv
uv venv --python python3.11 "${HERMES_VENV}"
HERMES_PIP="${HERMES_VENV}/bin/pip"
HERMES_PYTHON="${HERMES_VENV}/bin/python"

# Install hermes-agentcyber
if [[ "$BUNDLE_SOURCE" == "true" && -f /opt/hermes-agentcyber.tar.gz ]]; then
  cd /opt
  tar xzf hermes-agentcyber.tar.gz
  HERMES_SRC_DIR=$(find /opt -maxdepth 1 -type d -name "hermes-agentcyber*" | head -1)
  uv pip install --python "${HERMES_PYTHON}" --no-cache-dir -e "${HERMES_SRC_DIR}"
  echo "${HERMES_SRC_DIR}" > /opt/hermes-source-path
  rm /opt/hermes-agentcyber.tar.gz
else
  # Install from PyPI on first boot (requires internet)
  touch /opt/hermes-install-on-firstboot
fi

# Add uv + venv binaries to PATH for hermes user
cat >> /home/hermes/.bashrc << 'BASHRC'
export PATH="/opt/hermes-venv/bin:/usr/local/bin:$PATH"
export HERMES_HOME="/home/hermes/.hermes"
BASHRC
chown hermes:hermes /home/hermes/.bashrc

# Set up hermes home directory
mkdir -p /home/hermes/.hermes/logs
chown -R hermes:hermes /home/hermes/.hermes

# Headless flag file
[[ "$HEADLESS_SCAN" == "true" ]] && touch /opt/hermes-headless-scan

echo "✓ chroot setup complete"
CHROOT_EOF

chmod +x "${ROOTFS}/tmp/chroot_setup.sh"
chroot "${ROOTFS}" /tmp/chroot_setup.sh ${BUNDLE_FLAG} ${HEADLESS_FLAG}
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

# Enable systemd services
chroot "${ROOTFS}" systemctl enable hermes-firstboot.service
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
mkdir -p "${SQUASHFS_DIR}" "${ISO_STAGING}/boot/grub" "${ISO_STAGING}/EFI/boot"
mksquashfs "${ROOTFS}" "${SQUASHFS_DIR}/filesystem.squashfs" \
  -comp xz -b 1M -Xbcj x86 \
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

echo ""
echo "▶ [7/7] Building bootable ISO with xorriso..."
xorriso -as mkisofs \
  -iso-level 3 \
  -full-iso9660-filenames \
  -volid "HERMES-CYBER" \
  -appid "Hermes AgentCyber Live" \
  --grub2-mbr /usr/lib/grub/i386-pc/boot_hybrid.img \
  -partition_offset 16 \
  --mbr-force-bootable \
  -append_partition 2 28732ac11ff8d211ba4b00a0c93ec93b /usr/lib/grub/i386-pc/cdboot.img \
  -appended_part_as_gpt \
  -iso_mbr_part_type a2a0d0ebe5b9334487c068b6b72699c7 \
  -c "/boot/grub/boot.cat" \
  -b "/boot/grub/i386-pc/eltorito.img" \
  -no-emul-boot \
  -boot-load-size 4 \
  -boot-info-table \
  --grub2-boot-info \
  -eltorito-alt-boot \
  -e "--interval:appended_partition_2_start_2548s_size_16s:all::" \
  -no-emul-boot \
  -boot-load-size 16 \
  -o "${OUTPUT}" \
  "${ISO_STAGING}" 2>&1 | tail -5

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅  ISO built successfully!"
echo "  📀  Output: ${OUTPUT}"
echo "  📏  Size:   $(du -sh "${OUTPUT}" | cut -f1)"
echo ""
echo "  Write to USB:  sudo ./write_usb.sh --iso ${OUTPUT}"
echo "  Provision:     ./provision.sh --usb /dev/sdX --config config.yaml"
echo "═══════════════════════════════════════════════════════════"
