#!/bin/sh
# Build a relocatable Linux native distribution. Its bin and share directories
# also follow the XDG layout when installed under the same prefix.
set -eu
cd "$(dirname "$0")/.."
destination=${1:-"$PWD/dist"}
mkdir -p "$destination"
destination=$(cd "$destination" && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
cargo build --locked --release -p hermes-desktop -p hermes-gateway
target=${CARGO_TARGET_DIR:-"$PWD/target"}
stage="$build/Hermes"
install -d "$stage/bin" "$stage/share/applications" "$stage/share/icons/hicolor/512x512/apps" "$stage/share/licenses/hermes"
install -m755 "$target/release/hermes-desktop" "$target/release/hermes-gateway" "$stage/bin/"
install -m644 assets/icon.png "$stage/share/icons/hicolor/512x512/apps/com.nousresearch.hermes.rust-development.png"
install -m644 packaging/linux/com.nousresearch.hermes.rust-development.desktop "$stage/share/applications/"
install -m644 ../../LICENSE "$stage/share/licenses/hermes/LICENSE"
for notice in crates/desktop/assets/fonts/*NOTICE.txt crates/desktop/assets/fonts/*LICENSE*.txt; do
  if [ -f "$notice" ]; then install -m644 "$notice" "$stage/share/licenses/hermes/"; fi
done
tar -C "$build" -czf "$destination/Hermes-linux-$(uname -m).tar.gz" Hermes
