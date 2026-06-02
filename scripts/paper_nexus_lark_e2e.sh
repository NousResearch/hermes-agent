#!/usr/bin/env bash
# E2E: kanban-paper-nexus full lark-cli loop (arXiv + S2), fetch IM + check logs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES="${HERMES:-$ROOT/venv/bin/hermes}"
FEISHU_CHAT_ID="${FEISHU_CHAT_ID:-oc_bfa70445b7411381db594886a2201495}"
MARKER="paper-nexus-e2e-$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/${MARKER}.log"
REPORT="/tmp/${MARKER}-report.md"
SKILL_SCRIPTS="$ROOT/skills/research/kanban-paper-nexus/scripts"
LIVE_SCRIPTS="$ROOT/skills/devops/kanban-feishu-live/scripts"
SCORE="$ROOT/scripts/paper_flow_score.py"
AGENT_LOG="${HERMES_HOME:-$HOME/.hermes}/logs/agent.log"
GW_LOG="${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
ERR_LOG="${HERMES_HOME:-$HOME/.hermes}/logs/errors.log"

exec > >(tee "$LOG") 2>&1

cd "$ROOT"
source venv/bin/activate 2>/dev/null || true

fetch_msgs() {
  local tag="$1"
  local out="/tmp/${MARKER}-${tag}-msgs.json"
  sleep 2
  lark-cli im +chat-messages-list --as bot --chat-id "$FEISHU_CHAT_ID" \
    --page-size 20 --format json > "$out" 2>/dev/null || echo '{"data":{"messages":[]}}' > "$out"
  echo "$out"
}

extract_text() {
  python3 - <<'PY' "$1"
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for it in data.get("data", {}).get("messages") or []:
    print(it.get("content") or "")
PY
}

run_nexus_case() {
  local case_name="$1"
  local paper_input="$2"
  local title_zh="$3"
  local score_mode="$4"
  local expect_id="$5"
  local meta="/tmp/${MARKER}-${case_name}-meta.json"

  echo ""
  echo "========== CASE: $case_name =========="
  echo "input=$paper_input"

  echo "--- metadata ---"
  if ! python3 "$SKILL_SCRIPTS/paper_nexus_metadata.py" "$paper_input" > "$meta" 2>/dev/null; then
    echo "WARN: metadata failed (S2 429?) — using resolve_canonical_id only"
    python3 -c "
import sys
sys.path.insert(0, '$SKILL_SCRIPTS')
from paper_nexus_metadata import resolve_canonical_id
cid = resolve_canonical_id('$paper_input')
import json
json.dump({'canonical_id': cid, 'paper_id': cid, 'title': '$title_zh'}, open('$meta','w'))
"
  fi
  cat "$meta"

  local cid
  cid=$(python3 -c "import json; print(json.load(open('$meta'))['canonical_id'])")

  echo "--- memory query ---"
  python3 "$SKILL_SCRIPTS/paper_memory_search_query.py" "$paper_input" --meta-json "$meta" || true

  echo "--- registry ---"
  python3 "$SKILL_SCRIPTS/paper_doc_registry.py" resolve "$cid"

  echo "--- feishu doc sync ---"
  local sync_json doc_url doc_action
  if sync_json=$(python3 "$SKILL_SCRIPTS/paper_feishu_doc_sync.py" "$paper_input" "$MARKER" --title-zh "$title_zh" 2>&1); then
    echo "$sync_json"
    doc_url=$(echo "$sync_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('doc_url',''))" 2>/dev/null || echo "")
    doc_action=$(echo "$sync_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('action',''))" 2>/dev/null || echo "")
  else
    echo "WARN: doc sync failed: $sync_json"
    doc_url=""
    doc_action="skipped"
  fi

  echo "--- kanban smoke ---"
  "$HERMES" kanban boards switch paper-nexus 2>/dev/null || true
  local create_out task_id
  create_out=$("$HERMES" kanban create "[paper] ${cid} E2E ${MARKER}" --assignee kanban-researcher --json 2>/dev/null || echo '{}')
  echo "$create_out"
  task_id=$(echo "$create_out" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id') or d.get('task_id') or '')" 2>/dev/null || echo "")
  if [[ -z "$task_id" ]]; then
    echo "WARN: kanban create returned no id; subscribe will skip"
    task_id="t_e2e_skip"
  fi

  local tasks_json
  tasks_json=$(python3 -c "import json; print(json.dumps({'T0':'$task_id','T5':'$task_id','T6':'$task_id'}))")

  echo "--- feishu live IM ---"
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus init "$cid" \
    --chat-id "$FEISHU_CHAT_ID" --title-zh "$title_zh" --doc-url "$doc_url" --tasks-inline "$tasks_json"
  python3 "$LIVE_SCRIPTS/kanban_feishu_subscribe.py" --board paper-nexus "$cid" || true
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
    --entity-id "$cid" --event pipeline_started --summary "E2E $case_name 启动"
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
    --entity-id "$cid" --event stage_done --stage T0 --summary "E2E T0 完成" --update-doc-url "$doc_url"
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
    --entity-id "$cid" --event pipeline_done --summary "E2E $case_name 收尾" --update-doc-url "$doc_url"

  local msgf textf scoref
  msgf=$(fetch_msgs "$case_name")
  textf="/tmp/${MARKER}-${case_name}-text.txt"
  extract_text "$msgf" > "$textf"
  scoref="/tmp/${MARKER}-${case_name}-score.json"
  python3 "$SCORE" --mode "$score_mode" --text-file "$textf" --expect-id "$expect_id" | tee "$scoref"

  echo "CASE $case_name: doc_action=$doc_action doc_url=${doc_url:-none}"
  rg -F "$MARKER" "$textf" >/dev/null && echo "IM marker OK" || echo "WARN: marker missing in captured text"
}

echo "=== Paper Nexus lark E2E ($MARKER) ==="
echo "CHAT=$FEISHU_CHAT_ID"
LOG_SNIP_START=$(wc -l < "$AGENT_LOG" 2>/dev/null || echo 0)

run_nexus_case "gcn" "1609.02907" "图卷积网络 GCN 半监督节点分类" "gcn_kanban" "1609.02907"
run_nexus_case "doi-dexnet" "10.1126/scirobotics.aau4984" "学习双手抓取机器人策略" "s2_kanban" "10.1126/scirobotics.aau4984"
run_nexus_case "s2-ids" "https://www.semanticscholar.org/paper/ceced53f349f7e425352ecf4813b307667cd8aa6" \
  "图神经网络入侵检测系统综述" "s2_kanban" "s2:ceced53f349f7e425352ecf4813b307667cd8aa6"

echo ""
echo "========== LOG CHECK (since test start line $LOG_SNIP_START) =========="
for f in "$AGENT_LOG" "$GW_LOG" "$ERR_LOG"; do
  echo "--- $f ---"
  if [[ -f "$f" ]]; then
    tail -n +"$LOG_SNIP_START" "$f" 2>/dev/null | rg -i "paper.nexus|paper_nexus|1609\.02907|ceced53f|kanban_feishu|error|WARN|429|doc sync|lark-cli" | tail -40 || echo "(no matches)"
  else
    echo "missing"
  fi
done

{
  echo "# Paper Nexus lark E2E · $MARKER"
  echo ""
  echo "- Log: $LOG"
  echo "- Chat: $FEISHU_CHAT_ID"
  echo ""
  for c in gcn doi-dexnet s2-ids; do
    echo "## $c"
    echo '```json'
    cat "/tmp/${MARKER}-${c}-score.json" 2>/dev/null || echo '{}'
    echo '```'
  done
} > "$REPORT"

echo ""
echo "=== DONE ==="
echo "Report: $REPORT"
cat "$REPORT"
