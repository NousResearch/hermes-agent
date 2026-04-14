#!/usr/bin/env bash
set -euo pipefail

# OpenClaw verification runner (read-only)
# Exits non-zero if any invariant fails.

fail=0
say(){ printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }

say "verify: gateway reachability"
if ! openclaw status >/dev/null 2>&1; then
  say "FAIL: openclaw status failed"
  fail=1
fi

say "verify: systemd unit effective settings"
EFF=$(systemctl --user show openclaw-gateway.service -p KillMode -p TimeoutStopUSec -p RestartUSec -p ActiveState -p SubState --no-pager || true)
if ! grep -q "KillMode=control-group" <<<"$EFF"; then
  say "FAIL: KillMode not control-group"
  say "$EFF"
  fail=1
fi
if ! grep -q "ActiveState=active" <<<"$EFF"; then
  say "FAIL: systemd service not active"
  say "$EFF"
  fail=1
fi

say "verify: telegram probe (walletdb account)"
# openclaw health is structured; we just ensure walletdb probe ok.
HJSON=$(openclaw health --json 2>/dev/null || true)
if ! grep -q '"walletdb"' <<<"$HJSON"; then
  say "WARN: could not find walletdb in health JSON"
else
  if ! grep -q '"accountId": "walletdb"' <<<"$HJSON"; then
    say "WARN: walletdb accountId not present (schema mismatch?)"
  fi
  if ! grep -q '"username": "WalletDBbot"' <<<"$HJSON"; then
    say "WARN: WalletDBbot username not observed in probe"
  fi
  if ! grep -q '"probe": {[^}]*"ok": true' <<<"(echo "$HJSON" | tr '\n' ' ')"; then
    say "WARN: telegram probe did not show ok=true (check openclaw health --json)"
  fi
fi

say "verify: check recent telegram getUpdates 409 conflicts (last 6h)"
LOG=/tmp/openclaw/openclaw-$(date +%F).log
if [ -f "$LOG" ]; then
  if rg -n "getUpdates conflict" "$LOG" | tail -n 1 >/dev/null 2>&1; then
    say "WARN: found getUpdates conflict in logs (possible duplicate bot instance)"
    rg -n "getUpdates conflict" "$LOG" | tail -n 5 || true
  else
    say "ok: no recent getUpdates conflict lines found"
  fi
else
  say "WARN: log file not found: $LOG"
fi

say "verify: openclaw security audit (fast)"
if ! openclaw security audit >/dev/null 2>&1; then
  say "FAIL: openclaw security audit failed"
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  say "PASS"
else
  say "FAIL"
fi
exit "$fail"
