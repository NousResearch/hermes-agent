#!/usr/bin/env bash
# install.sh — set up the portable vault kit on a fresh machine.
# Checks prerequisites, makes the bin/ scripts executable, and optionally
# symlinks them into ~/.local/bin. Does NOT require root, NAS, or TPM.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$KIT_DIR/bin"

echo "=== vaultkit installer ==="
echo "kit: $KIT_DIR"

# ── prerequisite check ──
missing_required=0
check() {  # name  binary  required(0/1)  hint
  if command -v "$2" >/dev/null 2>&1; then
    printf '  [ ok ] %-16s %s\n' "$1" "$(command -v "$2")"
  else
    if [ "$3" -eq 1 ]; then
      printf '  [MISS] %-16s REQUIRED — %s\n' "$1" "$4"
      missing_required=1
    else
      printf '  [ -- ] %-16s optional — %s\n' "$1" "$4"
    fi
  fi
}

echo "Prerequisites:"
# keepassxc-cli may be named keepassxc-cli or keepassxc.cli (snap)
if command -v keepassxc-cli >/dev/null 2>&1 || command -v keepassxc.cli >/dev/null 2>&1; then
  printf '  [ ok ] %-16s found\n' "keepassxc-cli"
else
  printf '  [MISS] %-16s REQUIRED — apt/dnf/brew install keepassxc\n' "keepassxc-cli"
  missing_required=1
fi
check "python3"  python3 1 "install Python 3.8+"
check "age"      age     0 "apt/brew install age — enables breakglass recovery"
check "age-keygen" age-keygen 0 "ships with age"
# python 'cryptography' is an OPTIONAL speedup for breakglass key derivation;
# the kit falls back to a pure-Python X25519 that yields the identical key.
if python3 -c "import cryptography" 2>/dev/null; then
  printf '  [ ok ] %-16s present (breakglass uses C-backed X25519)\n' "py-cryptography"
else
  printf '  [ -- ] %-16s absent — breakglass falls back to pure-Python X25519 (fine)\n' "py-cryptography"
fi

echo
if [ "$missing_required" -eq 1 ]; then
  echo "Install the REQUIRED tools above, then re-run ./install.sh" >&2
  exit 1
fi

# ── make scripts executable ──
chmod +x "$BIN_DIR"/vault-* 2>/dev/null || true
echo "made bin/ scripts executable"

# ── optional symlink into ~/.local/bin ──
TARGET="${HOME}/.local/bin"
if [ "${1:-}" = "--link" ] || { [ -t 0 ] && read -rp "Symlink commands into $TARGET? (Y/n): " a && [ "${a:-y}" != "n" ]; }; then
  mkdir -p "$TARGET"
  for f in "$BIN_DIR"/vault-*; do
    ln -sf "$f" "$TARGET/$(basename "$f")"
  done
  echo "linked vault-* into $TARGET (ensure it's on your PATH)"
fi

echo
echo "=== done. Next: ==="
echo "  vault-setup                 # bootstrap a fresh vault"
echo "  vault-add <KEY_NAME>        # add a secret (hidden input)"
echo "  vault-unlock                # load secrets into tmpfs env file + manifest"
echo "  vault-breakglass export     # offline recovery bundle (needs age)"
