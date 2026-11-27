#!/bin/sh
set -eu

version="${LEXMOUNT_BROWSER_CLI_VERSION:-1.1.15}"
download_base_url="${LEXMOUNT_BROWSER_CLI_DOWNLOAD_BASE_URL:-https://cli-bin-1377899528.cos.ap-nanjing.myqcloud.com/releases/browser-cli}"
repo="${download_base_url%/}/v${version}"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) target="aarch64-apple-darwin" ;;
  Linux-x86_64) target="x86_64-unknown-linux-musl" ;;
  *) echo "Unsupported platform: $(uname -s) $(uname -m). This release supports macOS arm64, Linux x86_64, and Windows x86_64." >&2; exit 2 ;;
esac

asset="browser-cli-v${version}-${target}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT INT TERM
curl --proto '=https' --tlsv1.2 -fsSL "$repo/$asset" -o "$tmp_dir/$asset"
curl --proto '=https' --tlsv1.2 -fsSL "$repo/SHA256SUMS" -o "$tmp_dir/SHA256SUMS"
expected="$(awk -v name="$asset" '$2 == name {print $1}' "$tmp_dir/SHA256SUMS")"
[ -n "$expected" ] || { echo "No checksum published for $asset" >&2; exit 3; }
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tmp_dir/$asset" | awk '{print $1}')"
elif command -v openssl >/dev/null 2>&1; then
  actual="$(openssl dgst -sha256 "$tmp_dir/$asset" | awk '{print $NF}')"
else
  echo "Neither sha256sum nor openssl is available for SHA-256 verification" >&2
  exit 5
fi
[ "$expected" = "$actual" ] || { echo "SHA-256 mismatch for $asset" >&2; exit 4; }
skill_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
install_dir="${LEXMOUNT_BROWSER_CLI_INSTALL_DIR:-$skill_dir/bin}"
mkdir -p "$install_dir"
cp "$tmp_dir/$asset" "$install_dir/browser-cli"
chmod 0755 "$install_dir/browser-cli"
"$install_dir/browser-cli" version
echo "Installed browser-cli to $install_dir/browser-cli"
