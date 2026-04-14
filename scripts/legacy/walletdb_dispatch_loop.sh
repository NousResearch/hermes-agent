#!/usr/bin/env bash
set -euo pipefail

# Legacy debug loop (DEPRECATED). Delivery is owned by walletdb-alerts-deliver.service.
# Disabled by default to avoid DB contention.
if [[ "${ALLOW_LEGACY_DISPATCH:-}" != "1" ]]; then
  echo "walletdb_dispatch_loop.sh is deprecated and disabled by default." >&2
  echo "Use: walletdb alerts status|simulate|analytics or systemctl --user status walletdb-alerts-deliver.service" >&2
  echo "Set ALLOW_LEGACY_DISPATCH=1 to run anyway." >&2
  exit 2
fi

LOG=${LOG:-/tmp/walletdb_dispatch_$$.log}
touch "$LOG" || true
ln -sf "$LOG" /tmp/walletdb_dispatch_latest.log || true

echo "walletdb_dispatch_loop pid=$$ log=$LOG" >>"$LOG"

cd /home/yeqiuqiu/projects/walletdb
source .venv/bin/activate

end=$((SECONDS+55))
while [ $SECONDS -lt $end ]; do
  out=$(timeout 15s python -m walletdb.cli alerts dispatch --db-path data/walletdb.sqlite --limit 10 --no-mark --json 2>>"$LOG") || {
    err=$(timeout 15s python -m walletdb.cli alerts dispatch --db-path data/walletdb.sqlite --limit 10 --no-mark --json 2>&1 || true)
    echo "__ERROR__:${err}" | tee -a "$LOG"
    exit 0
  }

  # Only parse when we got a non-empty, non-empty-list JSON payload.
  out_trim=$(echo "$out" | tr -d '\r' | sed 's/^ *//;s/ *$//')
  if [ -n "$out_trim" ] && [ "$out_trim" != "[]" ]; then
    python -c "import json,sys
s = (sys.argv[1] or '').strip()
for i,ch in enumerate(s):
    if ch in '[{':
        s=s[i:]
        break
try:
    data=json.loads(s)
except Exception as e:
    print('PARSE_ERROR', str(e))
    raise SystemExit(0)
if isinstance(data,dict): data=[data]
for a in data:
    if isinstance(a,str): txt=a
    elif isinstance(a,dict): txt=a.get('text') or a.get('message') or a.get('body') or json.dumps(a,ensure_ascii=False)
    else: txt=str(a)
    print('__ALERT__:' + (txt or '').replace('\\n',' ').strip())
" "$out_trim" 2>>"$LOG" | tee -a "$LOG"
  fi

  sleep 5
done
