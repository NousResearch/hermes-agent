#!/usr/bin/env bash
# hermes-agent-updater.sh — automatic update, dependency sync, service restart, Telegram notify
# Runs via cron every 2 hours. Checks for upstream updates, applies them, restarts services, sends summary.
#
# Deployment example — paths and REMOTE_NAME/REMOTE_BRANCH below are tuned for
# a single Hetzner host running the gateway as a systemd unit. Adjust before
# reusing on another box. Companion: scripts/hermes-agent-warmup.py.

set -uo pipefail

REPO_DIR="/home/leos/hermes-agent"
VENV="$REPO_DIR/venv/bin"
ENV_FILE="/home/leos/.hermes/.env"
LOG_FILE="/tmp/hermes-updater.log"
LOCK_FILE="/tmp/hermes-updater.lock"

# uv binary — venv is uv-managed (no pip/ensurepip inside on purpose).
# Use full path because cron's PATH does not include ~/.local/bin.
UV_BIN="/home/leos/.local/bin/uv"

# Which remote/branch to track. We follow the masserfx fork's OAuth
# content-filter workaround branch, not upstream NousResearch/main,
# because the running gateway depends on patches that only exist there
# (HERMES_OAUTH_NO_MCP_PREFIX, COMPACT_GUIDANCE, FORCE_DROP_1M_BETA gates).
REMOTE_NAME="fork"
REMOTE_BRANCH="fix/oauth-content-filter-workarounds"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S'): $*" >> "$LOG_FILE"; }

# Load Telegram credentials from .env
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
while IFS='=' read -r key value; do
    key="${key%%#*}"        # strip comments
    key="${key// /}"        # strip spaces
    value="${value## }"     # trim leading space
    value="${value%% }"     # trim trailing space
    case "$key" in
        TELEGRAM_BOT_TOKEN) TELEGRAM_BOT_TOKEN="$value" ;;
        TELEGRAM_ALLOWED_USERS) TELEGRAM_CHAT_ID="$value" ;;
    esac
done < "$ENV_FILE"

if [[ -z "$TELEGRAM_BOT_TOKEN" || -z "$TELEGRAM_CHAT_ID" ]]; then
    log "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USERS in $ENV_FILE"
    exit 1
fi

send_telegram() {
    local message="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="$message" \
        -d parse_mode="Markdown" \
        --max-time 30 > /dev/null 2>&1 || true
}

# Prevent concurrent runs
if [[ -f "$LOCK_FILE" ]]; then
    pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Already running (PID $pid), skipping"
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

cd "$REPO_DIR"

# Fetch from tracked remote
git fetch "$REMOTE_NAME" --quiet 2>> "$LOG_FILE"

LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse "$REMOTE_NAME/$REMOTE_BRANCH")

if [[ "$LOCAL_HEAD" == "$REMOTE_HEAD" ]]; then
    log "Up to date ($(echo "$LOCAL_HEAD" | cut -c1-8))"
    exit 0
fi

# Count commits behind
BEHIND=$(git rev-list "HEAD..$REMOTE_NAME/$REMOTE_BRANCH" --count)
COMMIT_LOG=$(git log --oneline "HEAD..$REMOTE_NAME/$REMOTE_BRANCH" 2>/dev/null | head -15 || true)
FIRST_COMMIT=$(echo "$COMMIT_LOG" | tail -1 | cut -c1-8)
LAST_COMMIT=$(echo "$COMMIT_LOG" | head -1 | cut -c1-8)

log "Update available — $BEHIND commits behind. Updating..."

# Stash any local changes
STASHED=false
if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then
    git stash push -m "auto-updater $(date +%Y%m%d_%H%M%S)" --quiet 2>> "$LOG_FILE"
    STASHED=true
fi

# Pull updates (fast-forward only for safety)
if ! git pull --ff-only "$REMOTE_NAME" "$REMOTE_BRANCH" >> "$LOG_FILE" 2>&1; then
    log "Fast-forward failed, resetting to $REMOTE_NAME/$REMOTE_BRANCH"
    git reset --hard "$REMOTE_NAME/$REMOTE_BRANCH" >> "$LOG_FILE" 2>&1
fi

NEW_HEAD=$(git rev-parse --short HEAD)

# Sync dependencies via uv (project is uv-managed, uv.lock is authoritative).
log "Syncing dependencies via uv..."
"$UV_BIN" sync --quiet >> "$LOG_FILE" 2>&1 || log "WARN: uv sync failed"

# Quick smoke test
log "Running smoke tests..."
TEST_OUTPUT=$("$VENV/python" -m pytest tests/test_imports.py -q --tb=short 2>&1 | tail -3 || true)

# Collect services to restart
SERVICES=(hermes-agent paperclip-memory-api paperclip-agent-daemon)
RESTART_RESULTS=""

for svc in "${SERVICES[@]}"; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        sudo systemctl restart "$svc" 2>> "$LOG_FILE"
        sleep 3
        if systemctl is-active --quiet "$svc" 2>/dev/null; then
            RESTART_RESULTS="${RESTART_RESULTS}  ✅ ${svc}
"
        else
            RESTART_RESULTS="${RESTART_RESULTS}  ❌ ${svc} (failed)
"
        fi
    else
        RESTART_RESULTS="${RESTART_RESULTS}  ⏭ ${svc} (not running)
"
    fi
done

# Pop stash if we stashed
if [[ "$STASHED" == "true" ]]; then
    git stash pop --quiet 2>> "$LOG_FILE" || true
fi

# Warm-up: verify hermes-agent's OAuth path actually works after the restart.
# Without this, observed regression: auto-update + restart leaves the service
# in a state where the first real Telegram request fails with HTTP 400
# "out of extra usage" until a second restart clears it. The warm-up makes
# a tiny direct API call exercising the same env-flag patches the gateway
# uses; on failure we restart hermes-agent once more.
WARMUP_RESULT="skipped"
if [[ -x /home/leos/hermes-agent-warmup.py ]] && systemctl is-active --quiet hermes-agent 2>/dev/null; then
    sleep 8  # give hermes-agent a moment to fully bind its sockets / load env
    log "Running warm-up check..."
    if "$VENV/python" /home/leos/hermes-agent-warmup.py >> "$LOG_FILE" 2>&1; then
        WARMUP_RESULT="OK"
        log "Warm-up OK"
    else
        log "Warm-up FAILED — restarting hermes-agent once more"
        sudo systemctl restart hermes-agent 2>> "$LOG_FILE"
        sleep 5
        if "$VENV/python" /home/leos/hermes-agent-warmup.py >> "$LOG_FILE" 2>&1; then
            WARMUP_RESULT="recovered"
            log "Warm-up recovered after second restart"
        else
            WARMUP_RESULT="STILL FAILING"
            log "Warm-up still failing after retry — manual intervention needed"
        fi
    fi
fi

# Build Telegram message
COMMIT_PREVIEW=$(echo "$COMMIT_LOG" | head -8)

MSG="🔄 *Hermes Agent Auto-Update*

📦 *${BEHIND} nových commitů* (${FIRST_COMMIT} → ${LAST_COMMIT})
🏷 HEAD: \`${NEW_HEAD}\`

*Poslední změny:*
\`\`\`
${COMMIT_PREVIEW}
\`\`\`

*Testy:* ${TEST_OUTPUT}

*Služby:*
${RESTART_RESULTS}
*Warm-up:* ${WARMUP_RESULT}
🕐 $(date '+%Y-%m-%d %H:%M:%S')"

send_telegram "$MSG"

log "Update complete — $BEHIND commits, services restarted, notification sent"
