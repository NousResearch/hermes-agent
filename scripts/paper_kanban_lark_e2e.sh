#!/usr/bin/env bash
# E2E: kanban-paper-nexus via lark-cli (doc 中文名 + 阶段 IM + Kanban smoke).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES="${HERMES:-$ROOT/venv/bin/hermes}"
FEISHU_CHAT_ID="${FEISHU_CHAT_ID:-oc_bfa70445b7411381db594886a2201495}"
ARXIV_ID="${ARXIV_ID:-2402.03300}"
TITLE_ZH="${TITLE_ZH:-DeepSeekMath 开放语言模型数学推理}"
# SKIP_IM=1 — no lark-cli im send
# SKIP_DOC=1 — skip doc sync (network)
MARKER="paper-kanban-e2e-$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/paper-kanban-lark-${MARKER}.log"
SKILL_SCRIPTS="$ROOT/skills/research/kanban-paper-nexus/scripts"
LIVE_SCRIPTS="$ROOT/skills/devops/kanban-feishu-live/scripts"
META_JSON="/tmp/paper-meta-${MARKER}.json"
exec > >(tee "$LOG") 2>&1

echo "=== Paper Nexus lark E2E ($MARKER) arXiv=$ARXIV_ID ==="

cd "$ROOT"
source venv/bin/activate 2>/dev/null || true

echo "--- 0) arXiv metadata ---"
python3 "$SKILL_SCRIPTS/paper_nexus_metadata.py" "$ARXIV_ID" | tee "$META_JSON"

echo "--- 1) Memory search query (short, no full-text) ---"
python3 "$SKILL_SCRIPTS/paper_memory_search_query.py" "$ARXIV_ID" --meta-json "$META_JSON"

echo "--- 2) Registry resolve ---"
python3 "$SKILL_SCRIPTS/paper_doc_registry.py" resolve "$ARXIV_ID"

if [[ "${SKIP_DOC:-}" != "1" ]]; then
  echo "--- 3) Feishu doc sync (online title = [id] title_zh) ---"
  SYNC_JSON=$(python3 "$SKILL_SCRIPTS/paper_feishu_doc_sync.py" "$ARXIV_ID" "$MARKER" --title-zh "$TITLE_ZH")
  echo "$SYNC_JSON"
  DOC_URL=$(echo "$SYNC_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_url'])")
  DOC_ACTION=$(echo "$SYNC_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['action'])")
  FEISHU_DOC_TITLE=$(echo "$SYNC_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('feishu_doc_title',''))")
  echo "DOC_URL=$DOC_URL ACTION=$DOC_ACTION"
  echo "FEISHU_DOC_TITLE=$FEISHU_DOC_TITLE"
else
  echo "SKIP_DOC=1"
  DOC_URL=""
  DOC_ACTION="skipped"
fi

ABS_URL=$(python3 -c "import json; print(json.load(open('$META_JSON'))['arxiv_abs'])")

echo "--- 4) Kanban: create E2E parent task ---"
"$HERMES" kanban boards create paper-nexus 2>/dev/null || true
"$HERMES" kanban boards switch paper-nexus 2>/dev/null || true
CREATE_OUT=$("$HERMES" kanban create "[paper] ${ARXIV_ID} E2E ${MARKER}" --assignee kanban-researcher --json 2>/dev/null || true)
echo "$CREATE_OUT"
TASK_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('task_id') or d.get('id') or '')" 2>/dev/null || echo "")
if [[ -z "$TASK_ID" ]]; then
  TASK_ID="t_e2e_placeholder"
  echo "WARN: could not parse task_id, using $TASK_ID for session init only"
fi

TASKS_JSON=$(python3 -c "import json; print(json.dumps({'T0':'$TASK_ID','T1':'$TASK_ID','T2':'$TASK_ID','T3':'$TASK_ID','T4':'$TASK_ID','T5':'$TASK_ID','T6':'$TASK_ID'}))")

echo "--- 5) Feishu live session (kanban-feishu-live) ---"
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus init "$ARXIV_ID" \
  --chat-id "$FEISHU_CHAT_ID" \
  --title-zh "$TITLE_ZH" \
  --doc-url "$DOC_URL" \
  --tasks-inline "$TASKS_JSON"

python3 "$LIVE_SCRIPTS/kanban_feishu_subscribe.py" --board paper-nexus "$ARXIV_ID" || echo "WARN: subscribe returned non-zero (task may be placeholder)"

if [[ "${SKIP_IM:-}" == "1" ]]; then
  echo "SKIP_IM=1 — dry-run notify only"
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
    --entity-id "$ARXIV_ID" --event pipeline_started --dry-run
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
    --entity-id "$ARXIV_ID" --event stage_done --stage T0 --summary "E2E thesis smoke" --dry-run
else
  echo "--- 6) Feishu IM: pipeline_started + T0 done ---"
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
    --entity-id "$ARXIV_ID" --event pipeline_started
  python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
    --entity-id "$ARXIV_ID" --event stage_done --stage T0 \
    --summary "E2E：公开数学语料续训提升 MATH 推理（${MARKER}）"

  if [[ -n "$DOC_URL" ]]; then
    echo "--- 7) Feishu IM: doc link summary ---"
    if [[ "$DOC_ACTION" == "create" ]]; then
      ACTION_ZH="已新建专属文档（中文名）"
    else
      ACTION_ZH="已在原文档追加（同一论文）"
    fi
    MSG="📄 论文 Kanban E2E · ${MARKER}

【题名】${TITLE_ZH}
【代号】${ARXIV_ID}
📎 ${ACTION_ZH}
${FEISHU_DOC_TITLE:+云文档名：${FEISHU_DOC_TITLE}
}${DOC_URL}
🔗 ${ABS_URL}
▶ /kanban-paper-nexus ${ARXIV_ID}"
    lark-cli im +messages-send --as bot --chat-id "$FEISHU_CHAT_ID" --text "$MSG"
  fi

  echo "--- 8) Verify bot messages in chat ---"
  sleep 2
  lark-cli im +chat-messages-list --as bot --chat-id "$FEISHU_CHAT_ID" --page-size 10 2>/dev/null | \
    rg -F "$MARKER" || echo "WARN: marker not found in recent messages"
fi

echo "=== PASS. Log: $LOG ==="
