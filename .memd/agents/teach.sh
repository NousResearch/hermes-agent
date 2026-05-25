#!/usr/bin/env bash
set -euo pipefail

_memd_default_root="/home/aparcedodev/.hermes/hermes-agent/.memd"
_memd_resolve_root() {
if [[ -n "${MEMD_BUNDLE_ROOT:-}" && -d "$MEMD_BUNDLE_ROOT" ]]; then
printf '%s\n' "$MEMD_BUNDLE_ROOT"; return 0
fi
local d="$PWD"
while [[ "$d" != "/" && -n "$d" ]]; do
if [[ -d "$d/.memd" ]]; then printf '%s\n' "$d/.memd"; return 0; fi
d="$(dirname "$d")"
done
local gcd
if gcd="$(git rev-parse --git-common-dir 2>/dev/null)"; then
[[ "$gcd" != /* ]] && gcd="$PWD/$gcd"
local main_root
if main_root="$(cd "$(dirname "$gcd")" 2>/dev/null && pwd)" && [[ -n "$main_root" && -d "$main_root/.memd" ]]; then
printf '%s\n' "$main_root/.memd"; return 0
fi
fi
printf '%s\n' "$_memd_default_root"
}
export MEMD_BUNDLE_ROOT="$(_memd_resolve_root)"
source "$MEMD_BUNDLE_ROOT/backend.env" 2>/dev/null || true
source "$MEMD_BUNDLE_ROOT/env"

memd_cmd="${MEMD_BIN:-memd}"
if [[ -z "${MEMD_BIN:-}" ]]; then
candidate="$MEMD_BUNDLE_ROOT/../target/debug/memd"
[[ -x "$candidate" ]] && memd_cmd="$candidate"
fi
args=(teach --output "$MEMD_BUNDLE_ROOT")
exec "$memd_cmd" "${args[@]}" "$@"
