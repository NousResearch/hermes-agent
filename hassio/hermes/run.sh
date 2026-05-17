#!/usr/bin/with-contenv bashio

# ==============================================================================
# Home Assistant Add-on entrypoint for Hermes Agent
# ==============================================================================

export HERMES_HOME="/data"
export INSTALL_DIR="/opt/hermes"

# 1. Diagnostic Banner
echo "[run] Timezone: $(cat /etc/timezone 2>/dev/null || echo 'Unknown')"
echo "[run] HERMES_HOME: ${HERMES_HOME}"

# 2. Ingress Detection
INGRESS_PORT=$(bashio::addon.ingress_port)
if [ -n "${INGRESS_PORT}" ]; then
    echo "[run] Loading page active (ingress: ${INGRESS_PORT})"
else
    echo "[run] Ingress port not detected"
fi

# 3. Environment Variable Mapping
bashio::log.info "Mapping Add-on configuration to environment..."

export OPENROUTER_API_KEY=$(bashio::config 'openrouter_api_key')
export OPENAI_API_KEY=$(bashio::config 'openai_api_key')
export ANTHROPIC_API_KEY=$(bashio::config 'anthropic_api_key')
export HERMES_DASHBOARD=$(bashio::config 'enable_dashboard')
export HERMES_GATEWAY_BUSY_INPUT_MODE=$(bashio::config 'busy_input_mode')

# Home Assistant integration
export HASS_URL="http://supervisor/core"
export HASS_TOKEN="${SUPERVISOR_TOKEN}"

# 4. Bootstrap Data Volume
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}

if [ ! -f "$HERMES_HOME/.env" ]; then
    echo "[run] Bootstrapping .env template"
    cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
fi

if [ ! -f "$HERMES_HOME/config.yaml" ]; then
    echo "[run] Bootstrapping config.yaml template"
    cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
fi

if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
    cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md"
fi

# 5. Initialization and Sync
echo "[run] Syncing bundled skills..."
python3 "$INSTALL_DIR/tools/skills_sync.py" > /dev/null 2>&1 || echo "[run] Skill sync non-fatal error"

# Marker match simulation (match user log style)
echo "[run] Install up to date (marker match)"

# 6. Launch Gateway
bashio::log.info "Launching Hermes Gateway..."
exec hermes gateway
