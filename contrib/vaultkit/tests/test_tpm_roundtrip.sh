#!/usr/bin/env bash
# test_tpm_roundtrip.sh — end-to-end TPM seal/unseal round-trip for vault-seal-tpm.
#
# Verified working on a real TPM2 machine (the canonical "ROUND-TRIP OK" path).
# Designed to be CI-safe: if there is no TPM, no systemd-creds, or no usable
# root, it SKIPS (exit 0 with a clear notice) rather than failing — so it can
# live in a normal test matrix on runners that lack a TPM.
#
# What it proves:
#   1. seal      : a throwaway random key is encrypted to the TPM (blob created)
#   2. unseal    : the blob decrypts back to a tmpfs target
#   3. round-trip: the unsealed bytes are byte-identical to the source key
#   4. teardown  : the tmpfs key is wiped
#
# Nothing here touches a real vault — the key is 128 random bytes in /tmp.
set -uo pipefail

KIT_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../bin" && pwd)"
SEAL="$KIT_BIN/vault-seal-tpm"

skip() { echo "SKIP: $*"; exit 0; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# ── preconditions (skip, don't fail, when the environment can't support it) ──
command -v systemd-creds >/dev/null 2>&1 || skip "systemd-creds not present"
[ -x "$SEAL" ] || fail "vault-seal-tpm not executable at $SEAL"

# TPM2 present? Try the modern and legacy probes; also accept a device node.
_tpm_ok=""
if command -v systemd-analyze >/dev/null 2>&1 && systemd-analyze has-tpm2 >/dev/null 2>&1; then
  _tpm_ok=1
elif systemd-creds has-tpm2 >/dev/null 2>&1; then
  _tpm_ok=1
elif [ -e /dev/tpmrm0 ] || [ -e /dev/tpm0 ]; then
  _tpm_ok=1
fi
[ -n "$_tpm_ok" ] || skip "no TPM2 available"

# We need a way to reach root non-interactively for the systemd-creds tpm2 ops.
# If we're already root, great. Otherwise require passwordless sudo; if neither,
# skip (an interactive password prompt has no place in CI).
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if sudo -n true >/dev/null 2>&1; then
    SUDO="sudo -n"
  else
    skip "not root and no passwordless sudo (TPM ops need root)"
  fi
fi

# ── sandbox ──
SBX="$(mktemp -d "${TMPDIR:-/tmp}/vk-tpm-test.XXXXXX")"
cleanup() {
  # best-effort teardown + remove sandbox
  VAULTKIT_TPM_KEYFILE="$SBX/run/test.key" VAULTKIT_TPM_BLOB="$SBX/test.key.cred" \
    VAULTKIT_TPM_CRED_NAME="vaultkit-citest" \
    "$SEAL" teardown >/dev/null 2>&1 || true
  rm -rf "$SBX" 2>/dev/null || true
}
trap cleanup EXIT

head -c 128 /dev/urandom > "$SBX/test.key"
chmod 600 "$SBX/test.key"

export VAULTKIT_KEYFILE="$SBX/test.key"
export VAULTKIT_TPM_BLOB="$SBX/test.key.cred"
export VAULTKIT_TPM_KEYFILE="$SBX/run/test.key"
export VAULTKIT_TPM_CRED_NAME="vaultkit-citest"

echo "== [1/4] status =="
"$SEAL" status || fail "status command errored"

echo "== [2/4] seal =="
# vault-seal-tpm self-elevates via sudo; in CI we may already be root, in which
# case the internal _reexec_as_root() is a no-op. Either way this must succeed.
"$SEAL" seal || fail "seal failed"
[ -f "$SBX/test.key.cred" ] || fail "sealed blob was not created"

echo "== [3/4] unseal + compare =="
"$SEAL" unseal || fail "unseal failed"
[ -f "$SBX/run/test.key" ] || fail "unsealed key not written"
if $SUDO cmp -s "$SBX/test.key" "$SBX/run/test.key"; then
  echo "   round-trip: identical"
else
  fail "unsealed key does NOT match source"
fi

echo "== [4/4] teardown =="
"$SEAL" teardown || fail "teardown failed"
# the tmpfs key should be gone (a root-owned file may need sudo to stat)
if $SUDO test -f "$SBX/run/test.key"; then
  fail "tmpfs key still present after teardown"
fi

echo
echo "PASS: TPM seal -> unseal -> round-trip -> teardown all succeeded."
