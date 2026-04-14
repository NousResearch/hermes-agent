#!/usr/bin/env bash
set -euo pipefail

# Weekly session hygiene: archive + delete session transcripts older than 14 days.

ARCHDIR=/home/yeqiuqiu/.openclaw/_archives/sessions
mkdir -p "$ARCHDIR"
TS=$(date +%F_%H%M%S)
ARCH="$ARCHDIR/sessions_older_than_14d_${TS}.tgz"
LIST=/tmp/openclaw_sessions_prune_list_${TS}.txt

find /home/yeqiuqiu/.openclaw/agents -maxdepth 3 -type f -name '*.jsonl' -mtime +14 -print > "$LIST"
COUNT=$(wc -l < "$LIST" | tr -d ' ')
if [ "$COUNT" = "0" ]; then
  echo "OK: no session .jsonl older than 14 days"
  exit 0
fi

BYTES=$(python - <<PY
import os
p="$LIST"
size=0
with open(p) as f:
    for line in f:
        line=line.strip()
        if line and os.path.exists(line):
            size+=os.path.getsize(line)
print(size)
PY
)

echo "Found $COUNT files older than 14d (bytes=$BYTES). Archiving to $ARCH"

tar -czf "$ARCH" -T "$LIST" --transform='s#^/##'

xargs -d '\n' -a "$LIST" rm -f

echo "Done. Archive size=$(stat -c %s "$ARCH") bytes"
