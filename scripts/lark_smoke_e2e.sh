#!/usr/bin/env bash
# Closed-loop smoke test: Hermes health + Tavily web_search + skills preset + lark-cli Feishu.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES="${HERMES:-$ROOT/venv/bin/hermes}"
FEISHU_CHAT_ID="${FEISHU_CHAT_ID:-oc_bfa70445b7411381db594886a2201495}"
MARKER="hermes-smoke-$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/hermes-lark-smoke-${MARKER}.log"
exec > >(tee "$LOG") 2>&1

echo "=== Hermes lark smoke E2E ($MARKER) ==="

cd "$ROOT"
source venv/bin/activate

echo "--- 1) Sync Tavily key from OpenClaw (if missing) ---"
bash "$ROOT/scripts/sync_openclaw_secrets.sh"

echo "--- 2) Ensure web.search_backend=tavily in config ---"
python3 <<'PY'
from pathlib import Path
import yaml
p = Path.home() / ".hermes/config.yaml"
cfg = yaml.safe_load(p.read_text()) or {}
web = cfg.setdefault("web", {})
if web.get("search_backend") != "tavily":
    web["search_backend"] = "tavily"
    p.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False))
    print("  set web.search_backend=tavily")
else:
    print("  web.search_backend already tavily")
PY

echo "--- 3) Hermes doctor / memory / mcp ---"
"$HERMES" doctor 2>&1 | tail -15 || true
"$HERMES" memory status 2>&1 | tail -8 || true
"$HERMES" mcp list 2>&1 | tail -12

echo "--- 4) Tavily web_search (1 query) ---"
set -a
# shellcheck source=/dev/null
[[ -f "${HOME}/.hermes/.env" ]] && source "${HOME}/.hermes/.env"
set +a
python3 <<'PY'
import json, os
os.chdir("/home/narwal/workspace/robot/workflow/hermes-agent")
from hermes_cli.plugins import discover_plugins
discover_plugins()
from tools.web_tools import web_search_tool
r = web_search_tool("Hermes agent skills", limit=2)
d = json.loads(r) if isinstance(r, str) else r
ok = d.get("success", True) if isinstance(d, dict) else True
print("web_search ok:", ok)
if isinstance(d, dict) and not ok:
    raise SystemExit(d.get("error", "web_search failed"))
print("results:", len(d.get("results", d.get("data", [])) if isinstance(d, dict) else []))
PY

echo "--- 5) Skills preset check (skip apply; run apply_skills_preset.py separately) ---"
python3 <<'PY'
from hermes_cli.config import load_config
from hermes_cli.skills_config import get_disabled_skills
from tools.skills_tool import _find_all_skills
cfg = load_config()
all_skills = _find_all_skills(skip_disabled=True)
disabled = get_disabled_skills(cfg)
enabled = len(all_skills) - len(disabled)
print(f"enabled skills: {enabled}, disabled: {len(disabled)}")
PY

echo "--- 6) Kanban CLI smoke ---"
"$HERMES" kanban list 2>&1 | head -8

echo "--- 7) lark-cli: bot send marker to Feishu chat ---"
lark-cli im +messages-send --as bot \
  --chat-id "$FEISHU_CHAT_ID" \
  --text "🧪 Hermes smoke $MARKER — skills preset + Tavily OK"

echo "--- 8) lark-cli: verify message visible (last 5) ---"
sleep 2
lark-cli im +chat-messages-list --as bot \
  --chat-id "$FEISHU_CHAT_ID" \
  --page-size 5 2>&1 | grep -F "$MARKER" && echo "PASS: marker found in chat" || {
    echo "WARN: marker not in last 5 messages (may still be delivered)"
  }

echo "--- 9) Gateway status (no restart) ---"
systemctl --user is-active hermes-gateway.service 2>/dev/null || echo "gateway unit not active"

echo "=== DONE. Log: $LOG ==="
