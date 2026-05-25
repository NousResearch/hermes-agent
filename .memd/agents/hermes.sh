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
set -a
source "$MEMD_BUNDLE_ROOT/backend.env" 2>/dev/null || true
source "$MEMD_BUNDLE_ROOT/env"
set +a
if [[ -z "${MEMD_TAB_ID:-}" ]]; then
  if [[ -n "${WT_SESSION:-}" ]]; then
    export MEMD_TAB_ID="tab-${WT_SESSION:0:8}"
  elif [[ -n "${TERM_SESSION_ID:-}" ]]; then
    export MEMD_TAB_ID="tab-${TERM_SESSION_ID:0:8}"
  else
    tty_id="$(tty 2>/dev/null || true)"
    if [[ -n "$tty_id" && "$tty_id" != "not a tty" ]]; then
      export MEMD_TAB_ID="tab-${tty_id//\//-}"
    else
      export MEMD_TAB_ID="tab-$$"
    fi
  fi
fi
export MEMD_AGENT="hermes"
export MEMD_WORKER_NAME="Hermes"
memd wake --output "$MEMD_BUNDLE_ROOT" --route auto --intent current_task --write >/dev/null 2>&1 || true
nohup memd heartbeat --output "$MEMD_BUNDLE_ROOT" --watch --interval-secs 30 --probe-base-url >/tmp/memd-heartbeat.log 2>&1 &
memd hive --output "$MEMD_BUNDLE_ROOT" --publish-heartbeat --summary >/dev/null 2>&1 || true
exec memd wake --output "$MEMD_BUNDLE_ROOT" --route auto --intent current_task --write "$@"
