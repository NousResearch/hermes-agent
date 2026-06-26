#!/usr/bin/env bash
# test_import_roundtrip.sh — vault-import .env onboarding, fully offline/CI-safe.
#
# Creates a throwaway key-file-only vault in /tmp, imports a sample .env, and
# asserts: parsing rules (skips, dedup, quote-strip), idempotency, --update
# overwrite, and value round-trip. No TPM, no root, no real secrets.
#
# Skips (exit 0) if keepassxc-cli is unavailable.
set -uo pipefail

KIT_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../bin" && pwd)"
IMPORT="$KIT_BIN/vault-import"

skip() { echo "SKIP: $*"; exit 0; }
fail() { echo "FAIL: $*" >&2; exit 1; }

CLI=""
for c in keepassxc-cli keepassxc.cli; do
  command -v "$c" >/dev/null 2>&1 && { CLI="$c"; break; }
done
[ -n "$CLI" ] || skip "keepassxc-cli not present"
[ -x "$IMPORT" ] || fail "vault-import not executable at $IMPORT"

SBX="$(mktemp -d "${TMPDIR:-/tmp}/vk-import-test.XXXXXX")"
trap 'rm -rf "$SBX" 2>/dev/null || true' EXIT

export VAULTKIT_DIR="$SBX"
export VAULTKIT_VAULT="$SBX/vault.kdbx"
export VAULTKIT_KEYFILE="$SBX/vault.key"
export VAULTKIT_GROUP="hermes"
export VAULTKIT_CONFIG="$SBX/config"
export VAULTKIT_CLI="$CLI"

# key-file-only vault (no password) so the test is non-interactive
"$CLI" db-create "$VAULTKIT_VAULT" -q --set-key-file "$VAULTKIT_KEYFILE" >/dev/null 2>&1 \
  || fail "db-create failed"
chmod 600 "$VAULTKIT_KEYFILE"
"$CLI" mkdir "$VAULTKIT_VAULT" hermes -k "$VAULTKIT_KEYFILE" --no-password >/dev/null 2>&1 \
  || fail "mkdir group failed"

cat > "$SBX/sample.env" <<'EOF'
# comment line, ignored
ALPHA=plainvalue
export BETA="quoted value"
GAMMA='single quoted'
EMPTY=
bad-name=skipme
DUP=first
DUP=second
EOF

_show() { "$CLI" show -s -a Password "$VAULTKIT_VAULT" "hermes/$1" \
  -k "$VAULTKIT_KEYFILE" --no-password 2>/dev/null; }

echo "== import (fresh) =="
"$IMPORT" "$SBX/sample.env" 2>/dev/null || fail "import returned non-zero"

# expected: ALPHA, BETA, GAMMA, DUP imported; EMPTY + bad-name skipped
for k in ALPHA BETA GAMMA DUP; do
  _show "$k" >/dev/null || fail "$k missing after import"
done
"$CLI" show "$VAULTKIT_VAULT" "hermes/EMPTY" -k "$VAULTKIT_KEYFILE" --no-password \
  >/dev/null 2>&1 && fail "EMPTY should not have been imported"

[ "$(_show BETA)" = "quoted value" ]  || fail "BETA quote-strip wrong: '$(_show BETA)'"
[ "$(_show GAMMA)" = "single quoted" ] || fail "GAMMA quote-strip wrong"
[ "$(_show DUP)" = "second" ]          || fail "DUP should use last occurrence, got '$(_show DUP)'"

echo "== idempotent re-import (no --update) keeps values =="
sed -i 's/^ALPHA=.*/ALPHA=changed/' "$SBX/sample.env"
"$IMPORT" "$SBX/sample.env" 2>/dev/null || fail "re-import non-zero"
[ "$(_show ALPHA)" = "plainvalue" ] || fail "ALPHA changed without --update"

echo "== --update overwrites =="
"$IMPORT" "$SBX/sample.env" --update 2>/dev/null || fail "--update non-zero"
[ "$(_show ALPHA)" = "changed" ] || fail "ALPHA not updated with --update"

echo
echo "PASS: vault-import parsing, idempotency, --update, and round-trip all OK."
