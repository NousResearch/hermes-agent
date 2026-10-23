#!/bin/bash
set -euo pipefail
companion_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$companion_root"
if [[ "${1:-}" == --help ]]; then
  printf 'Usage: ./install-iphone.sh [--no-launch] [DEVICE_UDID]\nUses LocalDevice.txt when the device is omitted.\n'
  exit 0
fi
companion_launch=1
if [[ "${1:-}" == --no-launch ]]; then
  companion_launch=0
  shift
fi
if (( $# > 1 )); then
  printf 'Expected at most one device identifier.\n' >&2
  exit 2
fi
companion_device="${1:-}"
if [[ -z "$companion_device" && -f LocalDevice.txt ]]; then
  IFS= read -r companion_device < LocalDevice.txt
fi
if [[ ! "$companion_device" =~ ^[A-Za-z0-9-]+$ ]]; then
  printf 'Usage: ./install-iphone.sh DEVICE_UDID\nFind the hardware UDID in Xcode > Window > Devices and Simulators.\n' >&2
  exit 2
fi
if [[ ! -f Local.xcconfig ]]; then
  printf 'Configure your DEVELOPMENT_TEAM in Local.xcconfig first. See MAINTENANCE.md.\n' >&2
  exit 2
fi
# Verification never changes the Mac backend or Tailscale configuration.
./rebuild.sh verify
companion_run="BuildEvidence/device-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$companion_run"
git rev-parse HEAD > "$companion_run/app-head.txt"
git --no-optional-locks status --porcelain > "$companion_run/app-status.txt"
companion_actions=(build)
if [[ "$companion_launch" == 0 ]]; then
  # Fresh provisioning must be sealed into a fresh signature.
  companion_actions=(clean build)
fi
xcodebuild -project HermesCompanion.xcodeproj -scheme HermesCompanion \
  -configuration Debug -destination "id=$companion_device" -destination-timeout 30 \
  -derivedDataPath BuildEvidence/DeviceDerivedData \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration "${companion_actions[@]}" \
  > "$companion_run/build.log" 2>&1 || { tail -60 "$companion_run/build.log"; exit 1; }
companion_app="$companion_root/BuildEvidence/DeviceDerivedData/Build/Products/Debug-iphoneos/HermesCompanion.app"
codesign --verify --deep --strict "$companion_app"
python3 scripts/refresh_hermes_phone.py --verify-built
xcrun devicectl device install app --device "$companion_device" "$companion_app" \
  --timeout 60 --json-output "$companion_run/install.json"
if [[ "$companion_launch" == 0 ]]; then
  printf 'Installed Hermes. Launch skipped for unattended signing renewal. Evidence: %s\n' "$companion_run"
  exit 0
fi
printf 'Installed. If iOS requests it, trust your Developer App in Settings > General > VPN & Device Management.\n'
xcrun devicectl device process launch --device "$companion_device" com.local.hermes.companion \
  --timeout 30 --json-output "$companion_run/launch.json" || {
    printf 'Installed, but launch was not confirmed. Unlock the phone and check developer profile trust.\n' >&2
    exit 1
  }
printf 'Device launch confirmed. Evidence: %s\nAuthenticated Mac connection remains a separate check.\n' "$companion_run"
