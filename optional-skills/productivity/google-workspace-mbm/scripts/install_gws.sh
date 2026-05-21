#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v gws >/dev/null 2>&1; then
  echo "gws is already installed."
  exit 0
fi

if command -v go >/dev/null 2>&1; then
  echo "Installing gws with go install..."
  go install github.com/googleworkspace/cli@latest
  exit 0
fi

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    target="google-workspace-cli-x86_64-unknown-linux-musl.tar.gz"
    ;;
  Darwin-arm64)
    target="google-workspace-cli-aarch64-apple-darwin.tar.gz"
    ;;
  Darwin-x86_64)
    target="google-workspace-cli-x86_64-apple-darwin.tar.gz"
    ;;
  *)
    echo "Unsupported platform. Install gws manually from https://github.com/googleworkspace/cli/releases" >&2
    exit 1
    ;;
esac

echo "Downloading latest gws release..."
version="$(
  python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://api.github.com/repos/googleworkspace/cli/releases/latest", timeout=15) as response:
    print(json.load(response)["tag_name"])
PY
)"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

archive="$tmp_dir/$target"
url="https://github.com/googleworkspace/cli/releases/download/${version}/${target}"

curl -fsSL "$url" -o "$archive"
tar -xzf "$archive" -C "$tmp_dir"
mv "$tmp_dir/gws" "$SCRIPT_DIR/gws_musl"
chmod +x "$SCRIPT_DIR/gws_musl"
echo "Installed local gws binary at $SCRIPT_DIR/gws_musl"
