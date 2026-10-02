#!/usr/bin/env bash
# Replace the volume icon of a finished DMG (the same .VolumeIcon.icns +
# custom-icon flag dmgbuild sets). Tauri's bundler always uses the app's
# .icns as the volume icon and offers no override, so the bootstrap
# installer's DMG gets the shared drive artwork here, before signing.
#
#   scripts/bundles/set_dmg_volume_icon.sh <in.dmg> <icon.icns> <out.dmg>
set -euo pipefail
in_dmg=$1; icns=$2; out_dmg=$3
work=$(mktemp -d "${TMPDIR:-/tmp}/dmg-volicon.XXXXXX")
trap 'rm -rf "$work"' EXIT
rw="$work/rw.dmg"
hdiutil convert "$in_dmg" -format UDRW -o "$rw" >/dev/null
mount=$(hdiutil attach -nobrowse -readwrite -noverify "$rw" | awk -F'\t' '/\/Volumes\//{print $NF; exit}')
test -d "$mount"
cp "$icns" "$mount/.VolumeIcon.icns"
/usr/bin/SetFile -a C "$mount"
hdiutil detach "$mount" >/dev/null
rm -f "$out_dmg"
hdiutil convert "$rw" -format UDZO -o "$out_dmg" >/dev/null
echo "volume icon set: $out_dmg"
