#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
destination=${1:-"$PWD/dist/Hermes.app"}
cargo build --locked --release -p hermes-desktop -p hermes-gateway
target_dir=$(cargo metadata --no-deps --format-version 1 | plutil -extract target_directory raw -o - -)
mkdir -p "$destination/Contents/MacOS" "$destination/Contents/Resources"
install -m 755 "$target_dir/release/hermes-desktop" "$destination/Contents/MacOS/hermes-desktop"
install -m 755 "$target_dir/release/hermes-gateway" "$destination/Contents/MacOS/hermes-gateway"
cp packaging/macos/Info.plist "$destination/Contents/Info.plist"
cp assets/icon.icns "$destination/Contents/Resources/Hermes.icns"
mkdir -p "$destination/Contents/Resources/Notices"
for notice in crates/desktop/assets/fonts/*NOTICE.txt crates/desktop/assets/fonts/*LICENSE*.txt; do
    if [ -f "$notice" ]; then cp "$notice" "$destination/Contents/Resources/Notices/"; fi
done
cp ../../LICENSE "$destination/Contents/Resources/Notices/Hermes-LICENSE"
codesign --force --sign - "$destination/Contents/MacOS/hermes-gateway"
codesign --force --sign - "$destination"
scripts/verify-native-macos.sh "$destination"
printf 'Built %s\n' "$destination"
