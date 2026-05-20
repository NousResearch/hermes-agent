#!/usr/bin/env bash
# E2E: one Feishu doc per paper (registry); same paper → update only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES="${HERMES:-$ROOT/venv/bin/hermes}"
FEISHU_CHAT_ID="${FEISHU_CHAT_ID:-oc_bfa70445b7411381db594886a2201495}"
ARXIV_ID="${ARXIV_ID:-2402.03300}"
# SKIP_IM=1 — doc sync only, no chat message
MARKER="paper-kanban-e2e-$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/paper-kanban-lark-${MARKER}.log"
SKILL_SCRIPTS="$ROOT/skills/research/kanban-paper-nexus/scripts"
exec > >(tee "$LOG") 2>&1

echo "=== Paper Kanban lark E2E ($MARKER) arXiv=$ARXIV_ID ==="
echo "Doc policy: new paper → +create; same canonical arXiv id → +append only."

cd "$ROOT"
source venv/bin/activate 2>/dev/null || true

echo "--- 1) Registry resolve ---"
python3 "$SKILL_SCRIPTS/paper_doc_registry.py" resolve "$ARXIV_ID"

echo "--- 2) Sync bilingual doc (create or update) ---"
SYNC_JSON=$(python3 "$SKILL_SCRIPTS/paper_feishu_doc_sync.py" "$ARXIV_ID" "$MARKER")
echo "$SYNC_JSON"
DOC_URL=$(echo "$SYNC_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_url'])")
DOC_ACTION=$(echo "$SYNC_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['action'])")
echo "DOC_URL=$DOC_URL ACTION=$DOC_ACTION"

ABS_URL=$(python3 "$SKILL_SCRIPTS/paper_nexus_metadata.py" "$ARXIV_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['arxiv_abs'])")

if [[ "${SKIP_IM:-}" == "1" ]]; then
  echo "SKIP_IM=1 — no Feishu message."
else
  echo "--- 3) Single IM ---"
  if [[ "$DOC_ACTION" == "create" ]]; then
    ACTION_ZH="已新建专属文档"
  else
    ACTION_ZH="已在原文档追加更新（同一论文不新建）"
  fi
  MSG="📄 论文 Kanban · ${MARKER}

【核心总结】
① 7B 模型在 MATH 竞赛题达 51.7%，接近 GPT-4/Gemini-Ultra 档（无工具）。
② 关键：120B 数学 token 数据筛选 + GRPO 强化学习（比 PPO 省显存）。
③ 参考方向：复现→读数据管线+GRPO；产品→加步骤校验；工具→PaperQA/GROBID。

📎 ${ACTION_ZH}：${DOC_URL}
🔗 arXiv：${ABS_URL}
▶ /kanban-paper-nexus ${ARXIV_ID}"
  lark-cli im +messages-send --as bot --chat-id "$FEISHU_CHAT_ID" --text "$MSG"
fi

echo "--- 4) Kanban smoke ---"
"$HERMES" kanban boards create paper-nexus 2>/dev/null || true
"$HERMES" kanban boards switch paper-nexus 2>/dev/null || true
"$HERMES" kanban create "[paper] ${ARXIV_ID} E2E验收" --assignee kanban-researcher 2>&1 | head -4 || true

echo "=== DONE. Log: $LOG ==="
