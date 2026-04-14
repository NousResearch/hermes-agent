#!/usr/bin/env bash
set -euo pipefail

# Detect Telegram getUpdates 409 conflicts (usually means >1 bot poller instance)
# Robust against historical log noise by time-window filtering.
# Exit codes:
# 0 = no recent conflict
# 2 = recent conflict detected

LOG=/tmp/openclaw/openclaw-$(date +%F).log
if [ ! -f "$LOG" ]; then
  # After midnight the new daily log may not exist yet. Fall back to the newest
  # available log file so the watchdog continues working across date boundaries.
  LOG=$(ls -1t /tmp/openclaw/openclaw-*.log 2>/dev/null | head -n 1 || true)
  if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "missing log: /tmp/openclaw/openclaw-$(date +%F).log (and no fallback log files found)" >&2
    exit 0
  fi
fi

TAIL_LINES=${TAIL_LINES:-8000}
WINDOW_MIN=${WINDOW_MIN:-15}

python3 - <<PY
import json, re, sys, time
from datetime import datetime, timezone

log_path = "$LOG"
tail_lines = int("$TAIL_LINES")
window_min = int("$WINDOW_MIN")
cutoff = time.time() - window_min*60

# Read tail efficiently
with open(log_path, 'rb') as f:
    try:
        f.seek(0, 2)
        end = f.tell()
        # read last ~2MB to cover tail_lines in most cases
        f.seek(max(0, end - 2_000_000))
    except Exception:
        f.seek(0)
    data = f.read().decode('utf-8', errors='ignore').splitlines()[-tail_lines:]

hits=[]
for i,line in enumerate(data, start=1):
    # Avoid false positives from summary/info lines that may quote the phrase.
    # We only want the actual gateway Telegram channel ERROR lines.
    if 'Telegram getUpdates conflict:' not in line:
        continue
    if '409: Conflict' not in line:
        continue
    if 'gateway/channels/telegram' not in line:
        continue
    if '"logLevelName":"ERROR"' not in line and '"logLevelName": "ERROR"' not in line:
        continue
    # logs are JSON-ish per-line with "time":"...Z"
    m=re.search(r'"time"\s*:\s*"([^"]+)"', line)
    ts=None
    if m:
        t=m.group(1)
        try:
            # handle Z or offset
            if t.endswith('Z'):
                dt=datetime.fromisoformat(t.replace('Z','+00:00'))
            else:
                dt=datetime.fromisoformat(t)
            ts=dt.timestamp()
        except Exception:
            ts=None
    if ts is None:
        # fallback: treat as recent if we can't parse (conservative)
        ts=time.time()
    if ts >= cutoff:
        hits.append(line)

if hits:
    print(f"recent getUpdates conflict detected in last {window_min}m ({len(hits)} hits).")
    for l in hits[-10:]:
        print(l)
    sys.exit(2)

print(f"ok: no getUpdates conflict in last {window_min}m")
sys.exit(0)
PY
