#!/usr/bin/env bash
# Closed-loop: 3 flows × 3 themes → lark-cli send → fetch messages → score vs A/B tiers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEISHU_CHAT_ID="${FEISHU_CHAT_ID:-oc_bfa70445b7411381db594886a2201495}"
MARKER="paper-3flow-$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/${MARKER}.log"
REPORT="/tmp/${MARKER}-report.md"
SEARCH_SCRIPTS="$ROOT/skills/research/paper-literature-search/scripts"
KANBAN_SCRIPTS="$ROOT/skills/research/kanban-paper-nexus/scripts"
LIVE_SCRIPTS="$ROOT/skills/devops/kanban-feishu-live/scripts"
SCORE="$ROOT/scripts/paper_flow_score.py"

exec > >(tee "$LOG") 2>&1

cd "$ROOT"
source venv/bin/activate 2>/dev/null || true

fetch_messages() {
  local tag="$1"
  local out="/tmp/${MARKER}-${tag}-msgs.json"
  sleep 3
  lark-cli im +chat-messages-list --as bot --chat-id "$FEISHU_CHAT_ID" \
    --page-size 25 --format json > "$out" 2>/dev/null || echo '{"items":[]}' > "$out"
  echo "$out"
}

extract_messages_text() {
  local json_file="$1"
  python3 - <<'PY' "$json_file"
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
msgs = data.get("data", {}).get("messages") or data.get("items") or []
texts = []
for it in msgs:
    c = it.get("content") or ""
    if not c and isinstance(it.get("body"), dict):
        c = it["body"].get("content") or ""
    texts.append(str(c))
print("\n".join(texts))
PY
}

echo "=== Three-flow lark E2E ($MARKER) ==="
echo "CHAT=$FEISHU_CHAT_ID"

# --- Flow 1: paper-search (主题: 图神经网络 经典) ---
echo ""
echo "========== FLOW 1: /paper-search · 图神经网络 经典 =========="
QUERY1="图神经网络 经典 graph neural network survey"
RANK1="/tmp/${MARKER}-search-rank.json"
python3 "$SEARCH_SCRIPTS/paper_search_rank.py" "$QUERY1" \
  --top 8 --profile survey --min-citations 30 | tee "$RANK1"
python3 "$SEARCH_SCRIPTS/paper_search_feishu_deliver.py" "$QUERY1" \
  --json-in "$RANK1" --chat-id "$FEISHU_CHAT_ID"
MSG1=$(fetch_messages "search")
TEXT1=$(extract_messages_text "$MSG1")
echo "$TEXT1" > "/tmp/${MARKER}-search-text.txt"
python3 "$SCORE" --mode gnn_search --text-file "/tmp/${MARKER}-search-text.txt" \
  | tee "/tmp/${MARKER}-search-score.json"

# --- Flow 2: kanban-paper-nexus arXiv (主题: GCN 经典 A档) ---
echo ""
echo "========== FLOW 2: /kanban-paper-nexus · 1609.02907 GCN =========="
ARXIV2="1609.02907"
TITLE_ZH2="图卷积网络 GCN 半监督节点分类"
META2="/tmp/${MARKER}-meta2.json"
python3 "$KANBAN_SCRIPTS/paper_nexus_metadata.py" "$ARXIV2" | tee "$META2"
python3 "$KANBAN_SCRIPTS/paper_memory_search_query.py" "$ARXIV2" --meta-json "$META2"
SYNC2=$(python3 "$KANBAN_SCRIPTS/paper_feishu_doc_sync.py" "$ARXIV2" "$MARKER" --title-zh "$TITLE_ZH2" 2>&1) || SYNC2='{"action":"skipped"}'
echo "$SYNC2"
DOC_URL2=$(echo "$SYNC2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('doc_url',''))" 2>/dev/null || echo "")
TASKS2='{"T0":"t_e2e","T5":"t_e2e"}'
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus init "$ARXIV2" \
  --chat-id "$FEISHU_CHAT_ID" --title-zh "$TITLE_ZH2" --doc-url "$DOC_URL2" --tasks-inline "$TASKS2"
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
  --entity-id "$ARXIV2" --event pipeline_started --summary "E2E GCN 精读流水线启动"
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
  --entity-id "$ARXIV2" --event stage_done --stage T0 --summary "E2E: GCN 论点种子" --update-doc-url "$DOC_URL2"
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
  --entity-id "$ARXIV2" --event pipeline_done --summary "E2E GCN 流水线收尾" --update-doc-url "$DOC_URL2"
MSG2=$(fetch_messages "kanban-arxiv")
TEXT2=$(extract_messages_text "$MSG2")
echo "$TEXT2" > "/tmp/${MARKER}-kanban-arxiv-text.txt"
python3 "$SCORE" --mode gcn_kanban --text-file "/tmp/${MARKER}-kanban-arxiv-text.txt" \
  --expect-id "$ARXIV2" | tee "/tmp/${MARKER}-kanban-arxiv-score.json"

# --- Flow 3: kanban-paper-nexus S2 (主题: GNN 入侵检测期刊综述) ---
echo ""
echo "========== FLOW 3: /kanban-paper-nexus · s2:ceced53f… IDS survey =========="
S2URL="https://www.semanticscholar.org/paper/ceced53f349f7e425352ecf4813b307667cd8aa6"
TITLE_ZH3="图神经网络入侵检测系统综述"
META3="/tmp/${MARKER}-meta3.json"
if ! python3 "$KANBAN_SCRIPTS/paper_nexus_metadata.py" "$S2URL" > "$META3" 2>/dev/null; then
  echo "WARN: S2 metadata 429 — using cached canonical for notify-only"
  cat > "$META3" <<'EOF'
{
  "canonical_id": "s2:ceced53f349f7e425352ecf4813b307667cd8aa6",
  "paper_id": "s2:ceced53f349f7e425352ecf4813b307667cd8aa6",
  "title": "A survey on graph neural networks for intrusion detection systems: Methods, trends and challenges",
  "s2_url": "https://www.semanticscholar.org/paper/ceced53f349f7e425352ecf4813b307667cd8aa6"
}
EOF
fi
cat "$META3"
CID3=$(python3 -c "import json; print(json.load(open('$META3'))['canonical_id'])")
python3 "$KANBAN_SCRIPTS/paper_memory_search_query.py" "$S2URL" --meta-json "$META3"
SYNC3=$(python3 "$KANBAN_SCRIPTS/paper_feishu_doc_sync.py" "$S2URL" "$MARKER-s2" --title-zh "$TITLE_ZH3" 2>&1) || SYNC3='{"action":"skipped"}'
echo "$SYNC3"
DOC_URL3=$(echo "$SYNC3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('doc_url',''))" 2>/dev/null || echo "")
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus init "$CID3" \
  --chat-id "$FEISHU_CHAT_ID" --title-zh "$TITLE_ZH3" --doc-url "$DOC_URL3" --tasks-inline "$TASKS2"
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
  --entity-id "$CID3" --event pipeline_started --summary "E2E GNN-IDS 期刊综述启动"
python3 "$LIVE_SCRIPTS/kanban_feishu_stage_notify.py" --board paper-nexus notify \
  --entity-id "$CID3" --event pipeline_done --summary "E2E S2 IDS 综述收尾" --update-doc-url "$DOC_URL3"
MSG3=$(fetch_messages "kanban-s2")
TEXT3=$(extract_messages_text "$MSG3")
echo "$TEXT3" > "/tmp/${MARKER}-kanban-s2-text.txt"
python3 "$SCORE" --mode s2_kanban --text-file "/tmp/${MARKER}-kanban-s2-text.txt" \
  --expect-id "$CID3" | tee "/tmp/${MARKER}-kanban-s2-score.json"

# --- Report ---
python3 - <<'PY' "$MARKER" "$REPORT" "$RANK1"
import json, sys
marker, report, rank_path = sys.argv[1], sys.argv[2], sys.argv[3]
rank = json.load(open(rank_path, encoding="utf-8"))
lines = [
    f"# Paper 三流 lark 闭环报告 · `{marker}`",
    "",
    "## 主题分工",
    "| 流 | 命令类型 | 测试主题 |",
    "|---|---------|---------|",
    "| 1 | `/paper-search` | 图神经网络 经典（检索 Top 列表） |",
    "| 2 | `/kanban-paper-nexus` | **1609.02907** GCN 经典（A档基石） |",
    "| 3 | `/kanban-paper-nexus` | **s2:ceced53f…** GNN-IDS 期刊综述 |",
    "",
    "## Flow 1 检索 Top（脚本输出，供与 A/B 档对照）",
]
for i, p in enumerate(rank.get("papers", [])[:8], 1):
    aid = p.get("arxiv_id") or p.get("paper_id", "")
    sc = p.get("scores", {}).get("display", "?")
    lines.append(f"{i}. **[{sc}]** `{aid}` · {p.get('year','?')} · cite={p.get('citation_count',0)}")
    lines.append(f"   - EN: {p.get('title','')[:90]}")
lines.append("")
for tag, name in [
    ("search", "paper-search 飞书抓取"),
    ("kanban-arxiv", "kanban Transformer 飞书抓取"),
    ("kanban-s2", "kanban S2 飞书抓取"),
]:
    sp = f"/tmp/{marker}-{tag}-score.json"
    try:
        s = json.load(open(sp, encoding="utf-8"))
    except FileNotFoundError:
        continue
    lines.append(f"## {name}")
    lines.append(f"- label: {s.get('label')}")
    lines.append(f"- tier A hits: {s.get('tier_a_hits')}/{s.get('tier_a_expected')}")
    lines.append(f"- tier B hits: {s.get('tier_b_hits')}/{s.get('tier_b_expected')}")
    if "expect_hit" in s:
        lines.append(f"- expect_id hit: {s.get('expect_hit')} (`{s.get('expect_id')}`)")
    for h in s.get("hits", []):
        lines.append(f"  - [{h['tier']}] `{h['id']}` {h['title_zh']}")
    lines.append("")
open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(report)
PY

echo ""
echo "=== DONE ==="
echo "Log: $LOG"
echo "Report: $REPORT"
cat "$REPORT"
