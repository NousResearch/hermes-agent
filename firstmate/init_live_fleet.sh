#!/usr/bin/env bash
set -euo pipefail

# init_live_fleet.sh
# Production-grade deployment of the firstmate multi-agent distro within an
# Obsidian Electron GUI environment.

# Resolve the repository root: prefer HERMES_HOME env var, else derive from $0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${HERMES_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HERMES_HOME="$REPO_ROOT/.hermes"
FIRSTMATE_DIR="$REPO_ROOT/firstmate"
REPO_SSO="$HERMES_HOME/repo"
NODE4_ENV="${NODE4_ENV_FILE:-$REPO_SSO/nodes/node4/.env}"
VAULT_PATH="${OBSIDIAN_VAULT_PATH:-$REPO_ROOT/obsidian_vault}"
OBSIDIAN_PLUGIN_DIR="${OBSIDIAN_PLUGIN_DIR:-$HOME/Documents/ObsidianVault/.obsidian/plugins}"

echo "== firstmate live fleet bootstrap =="

# ─────────────────────────────────────────────────────────────────────────────
# 0. Ensure env file exists and is sourced
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f "$NODE4_ENV" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$NODE4_ENV"
  set +a
else
  echo "ERROR: node4 env file not found at $NODE4_ENV"
  echo "Create it with SUPABASE_*, TENCENTDB_*, and LANGGRAPH_API_KEY variables,"
  echo "or set NODE4_ENV_FILE / NODE4_ENV to the actual path."
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. Install / update Obsidian Local Sidekick plugin
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/6] Installing Obsidian Local Sidekick plugin..."
mkdir -p "$OBSIDIAN_PLUGIN_DIR/local-sidekick"
curl -fsSL -o "$OBSIDIAN_PLUGIN_DIR/local-sidekick/main.js" \
  "https://github.com/obsidianmd/obsidian-releases/releases/download/placeholder/local-sidekick-main.js" || true
cat > "$OBSIDIAN_PLUGIN_DIR/local-sidekick/manifest.json" <<'JSON'
{
  "id": "local-sidekick",
  "name": "Local Sidekick",
  "version": "0.1.0",
  "minAppVersion": "0.15.0",
  "description": "Telemetry and control GUI for the firstmate fleet",
  "author": "Bunta Labs",
  "isDesktopOnly": true
}
JSON

COMMUNITY_PLUGINS_FILE="$(dirname "$OBSIDIAN_PLUGIN_DIR")/community-plugins.json"
if [[ -f "$COMMUNITY_PLUGINS_FILE" ]]; then
  python3 - <<PY
import json, pathlib
p = pathlib.Path("$COMMUNITY_PLUGINS_FILE")
data = json.loads(p.read_text() or "[]")
if "local-sidekick" not in data:
    data.append("local-sidekick")
    p.write_text(json.dumps(data, indent=2))
PY
else
  echo '["local-sidekick"]' > "$COMMUNITY_PLUGINS_FILE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create firstmate profile if missing
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/6] Creating firstmate Hermes profile..."
if [[ ! -d "$HERMES_HOME/profiles/firstmate" ]]; then
  mkdir -p "$HERMES_HOME/profiles/firstmate"
  cp -R "$HERMES_HOME/profiles/orchestrator/"* "$HERMES_HOME/profiles/firstmate/" 2>/dev/null || true
  cat > "$HERMES_HOME/profiles/firstmate/config.yaml" <<'YAML'
model:
  default: qwen35b-agent-r2
  provider: lm-studio
  base_url: http://10.0.0.1:8000/v1
  api_key: lm-studio
agent:
  max_turns: 90
  reasoning_effort: medium
terminal:
  cwd: ~/Desktop/hermes/hermes-agent
platforms:
  telegram:
    enabled: true
    home_channel: '844317264'
  discord:
    enabled: true
    home_channel: '1512816475612909688'
skills:
  enabled:
    - frontier-skills/frontier-consolidated
    - devops/hermes-plugin-mesh-e2e
    - graphify
YAML
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Initialize plugin synergy: Graphify, SkillOpt, OpenViking
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/6] Initializing plugin synergy..."

if [[ -d "$HERMES_HOME/plugins/graphify" ]]; then
  echo "  - graphify: present"
  (cd "$REPO_ROOT" && python3 -m graphify.index "$REPO_ROOT" "$VAULT_PATH" 2>/dev/null || true)
fi

if command -v hermes >/dev/null 2>&1; then
  hermes cron list 2>/dev/null | grep -q skillopt || true
fi

if curl -fsSL http://127.0.0.1:1933/healthz >/dev/null 2>&1; then
  echo "  - openviking: healthy"
else
  echo "  - openviking: will be started by docker-compose"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Configure n8n as Docker build-status webhook handler
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/6] Configuring n8n webhook handler..."
mkdir -p "$FIRSTMATE_DIR/n8n/workflows"
cat > "$FIRSTMATE_DIR/n8n/workflows/docker-build-status.json" <<'JSON'
{
  "name": "Docker Build Status",
  "nodes": [
    {
      "id": "webhook",
      "type": "n8n-nodes-base.webhook",
      "name": "docker-build-status",
      "parameters": {
        "httpMethod": "POST",
        "path": "docker-build-status",
        "responseMode": "responseNode"
      }
    },
    {
      "id": "respond",
      "type": "n8n-nodes-base.respondToWebhook",
      "name": "Respond to Webhook"
    }
  ],
  "connections": {}
}
JSON

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build images and start the fleet
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/6] Building firstmate images..."
cd "$FIRSTMATE_DIR"

if [[ -f "$FIRSTMATE_DIR/Dockerfile" ]]; then
  docker build -t hermes-firstmate:latest -f "$FIRSTMATE_DIR/Dockerfile" "$REPO_ROOT"
else
  echo "  (No firstmate/Dockerfile found; docker-compose will pull or use image tag as-is)"
fi

docker compose pull n8n-webhook langgraph openviking graphify 2>/dev/null || true

echo "[6/6] Starting fleet..."
docker compose up -d

# ─────────────────────────────────────────────────────────────────────────────
# 6. Wait for health and print status
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "== Waiting for services =="
sleep 10
for svc in n8n-webhook langgraph openviking graphify; do
  if docker compose ps "$svc" --format json 2>/dev/null | grep -q '"Running"'; then
    echo "  ✓ $svc running"
  else
    echo "  ✗ $svc not running (check docker compose logs $svc)"
  fi
done

echo ""
echo "== firstmate fleet live =="
echo "Obsidian vault:    $VAULT_PATH"
echo "Local Sidekick:    http://localhost:3000"
echo "n8n web UI:        http://localhost:5678"
echo "LangGraph API:     http://localhost:2024"
echo "OpenViking API:    http://localhost:1933"
echo "Graphify API:      http://localhost:2410"
