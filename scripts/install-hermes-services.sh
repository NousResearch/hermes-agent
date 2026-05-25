#!/usr/bin/env bash
set -euo pipefail

# Installs Hermes service-control entry points into /usr/local/bin so they
# are available to sudo from any directory (secure_path includes /usr/local/bin).
#
# Source trust: this installer copies scripts from the invoking user's
# ~/.local/bin by default (or SRC_BIN when explicitly set). Review those source
# scripts before running with sudo; the checks below reject sources owned by an
# unexpected user and sources writable by everyone.

PREFIX="${PREFIX:-/usr/local/bin}"

# When run via sudo, HOME usually points at /root, which is not where the
# Hermes helper scripts live. Prefer the invoking user's home directory so
# `sudo /home/.../install-hermes-services.sh` can locate the source binaries.
if [[ -n "${SRC_BIN:-}" ]]; then
  :
elif [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
  SRC_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
  SRC_BIN="${SRC_HOME:-$HOME}/.local/bin"
else
  SRC_BIN="$HOME/.local/bin"
fi

need_root() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "This installer must be run as root. Re-run with: sudo $0" >&2
    exit 1
  fi
}

expected_source_uid() {
  if [[ -n "${SUDO_UID:-}" ]]; then
    echo "$SUDO_UID"
  else
    id -u
  fi
}

validate_trusted_source() {
  local path="$1"
  local label="$2"
  local owner mode_hex expected_uid

  owner="$(stat -Lc '%u' "$path")"
  mode_hex="$(stat -Lc '%f' "$path")"
  expected_uid="$(expected_source_uid)"

  if [[ "$owner" != "0" && "$owner" != "$expected_uid" ]]; then
    echo "Unsafe $label: $path is owned by uid $owner, expected root or uid $expected_uid" >&2
    exit 1
  fi
  if (( (0x$mode_hex & 0002) != 0 )); then
    echo "Unsafe $label: $path is world-writable" >&2
    exit 1
  fi
}

validate_source_dir() {
  if [[ ! -d "$SRC_BIN" ]]; then
    echo "Missing source directory: $SRC_BIN" >&2
    exit 1
  fi
  SRC_BIN="$(readlink -f "$SRC_BIN")"
  validate_trusted_source "$SRC_BIN" "source directory"
  echo "Using reviewed source directory: $SRC_BIN"
}

install_one() {
  local name="$1"
  local src="$SRC_BIN/$name"
  local dst="$PREFIX/$name"

  if [[ ! -f "$src" ]]; then
    echo "Missing source script: $src" >&2
    exit 1
  fi

  validate_trusted_source "$src" "source script"

  install -D -m 0755 "$src" "$dst"
  echo "Installed $dst"
}

need_root
validate_source_dir

for name in hermes-services hermes-start-all hermes-stop-all hermes-restart-all hermes-status-all; do
  install_one "$name"
done

echo
if command -v hermes-services >/dev/null 2>&1; then
  echo "Verified: $(command -v hermes-services)"
else
  echo "Warning: hermes-services is not on PATH in this environment."
fi
