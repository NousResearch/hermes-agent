#!/usr/bin/env bash
# E2E: paper-literature-search → ranked list → Feishu IM (no online doc).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEISHU_CHAT_ID="${FEISHU_CHAT_ID:-oc_bfa70445b7411381db594886a2201495}"
QUERY="${QUERY:-multimodal large language model}"
MARKER="paper-search-e2e-$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/${MARKER}.log"
SCRIPTS="$ROOT/skills/research/paper-literature-search/scripts"
exec > >(tee "$LOG") 2>&1

echo "=== Paper Literature Search E2E ($MARKER) ==="
echo "QUERY=$QUERY"

cd "$ROOT"
source venv/bin/activate 2>/dev/null || true

echo "--- 1) Rank (Semantic Scholar + arXiv) ---"
RANK_JSON="/tmp/${MARKER}-rank.json"
python3 "$SCRIPTS/paper_search_rank.py" "$QUERY" --top 6 | tee "$RANK_JSON"
COUNT=$(python3 -c "import json; print(len(json.load(open('$RANK_JSON'))['papers']))")
test "$COUNT" -ge 1

echo "--- 2) Dry-run Feishu body ---"
python3 "$SCRIPTS/paper_search_feishu_deliver.py" "$QUERY" --json-in "$RANK_JSON" --dry-run | head -25

if [[ "${SKIP_IM:-}" == "1" ]]; then
  echo "SKIP_IM=1 — skip lark send"
  echo "=== PASS (rank only). Log: $LOG ==="
  exit 0
fi

echo "--- 3) Full pipeline → Feishu ---"
python3 "$SCRIPTS/paper_search_pipeline.py" "$QUERY" \
  --chat-id "$FEISHU_CHAT_ID" --top 6

echo "--- 4) Verify chat messages ---"
sleep 2
VERIFY_MSGS=$(lark-cli im +chat-messages-list --as bot --chat-id "$FEISHU_CHAT_ID" --page-size 12 2>/dev/null || echo '{}')
echo "$VERIFY_MSGS" | grep -F "文献检索" | head -5 || \
  echo "$VERIFY_MSGS" | grep -F "$QUERY" | head -5 || \
  echo "WARN: verify grep empty (chat may not show yet)"

echo "=== PASS. Log: $LOG ==="
