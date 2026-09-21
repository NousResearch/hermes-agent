#!/usr/bin/env bash
# Post-remediation audit — one-time check 24h after Phase 0–5 work on 2026-05-19/20.
# Reports MCP surface, jiter health, schema migration, tool-result persistence,
# cron stagger, and bloat indicators. Output formatted for Discord delivery.
set -uo pipefail

LOG_DIR="/home/kensei/.hermes/logs"
NOW="$(date '+%d/%m/%y %H:%M:%S')"
SINCE="$(date -d '24 hours ago' '+%Y-%m-%d %H:%M')"

# Helper: count regex matches in last 24h of a log file (uses log timestamp prefix).
count_recent() {
    local pattern="$1" file="$2"
    [ -f "$file" ] || { echo 0; return; }
    awk -v since="$SINCE" -v pat="$pattern" '
        /^2026-[0-9-]+ [0-9:]+/ {
            ts=substr($0,1,16)
            keep = (ts >= since)
        }
        keep && $0 ~ pat { c++ }
        END { print c+0 }
    ' "$file"
}

# 1. MCP tool count from most recent registration log line
MCP_LINE=$(grep "MCP: registered" "$LOG_DIR/agent.log" 2>/dev/null | tail -1)
MCP_TOOLS=$(echo "$MCP_LINE" | grep -oE 'registered [0-9]+ tool' | grep -oE '[0-9]+' || echo "?")
MCP_SERVERS=$(echo "$MCP_LINE" | grep -oE 'from [0-9]+ server' | grep -oE '[0-9]+' || echo "?")

# 2. jiter ImportError count today
JITER_ERRORS=$(count_recent "No module named 'jiter" "$LOG_DIR/errors.log")
JITER_ERRORS=$((JITER_ERRORS + $(count_recent "No module named 'jiter" "$LOG_DIR/errors.log.1")))

# 3. Kanban schema errors since migration
SCHEMA_ERRORS_24H=$(count_recent "no such column" "$LOG_DIR/gateway.log")

# 4. Tool result persistence — how many times did Hermes spill to /tmp/hermes-results?
PERSISTED_24H=$(count_recent "Persisted large tool result" "$LOG_DIR/agent.log")
PERSISTED_FILES=$(ls /tmp/hermes-results/ 2>/dev/null | wc -l)

# 5. Cron stagger — confirm no :00 collisions in last 24h cron logs (heartbeat-audit should now be :15)
HEARTBEAT_RUNS=$(ls /home/kensei/.hermes/cron/output/084352cdeafd/ 2>/dev/null | grep "2026-05-2[01]_" | wc -l)
HEARTBEAT_LAST=$(ls -t /home/kensei/.hermes/cron/output/084352cdeafd/ 2>/dev/null | head -1)

# 6. Top error categories in last 24h (excluding known noise)
RATE_LIMITS=$(count_recent "RateLimitError|HTTP 429" "$LOG_DIR/errors.log")
MNEMOSYNE_BEAM=$(count_recent "mnemosyne.beam" "$LOG_DIR/errors.log")

# 8. Largest session JSON in last 24h (catches runaway-session regressions)
LARGEST_SESSION=$(find /home/kensei/.hermes/sessions -name "session_*.json" -mtime -1 -type f -printf "%s %p\n" 2>/dev/null | sort -rn | head -1)
LARGEST_SIZE_KB=$(echo "$LARGEST_SESSION" | awk '{printf "%d", $1/1024}')
LARGEST_NAME=$(echo "$LARGEST_SESSION" | awk '{print $2}' | xargs -I{} basename {} 2>/dev/null)

# 9. Verify Layer B fork commit still present
LAYER_B_OK="no"
grep -q "DEFAULT_RESULT_SIZE_CHARS: int = 25_000" /home/kensei/repos/KenseiAgent/tools/budget_config.py 2>/dev/null && LAYER_B_OK="yes"

# 10. Verify shim repoint still in place (could be reverted by pipx upgrade)
SHIM_TARGET=$(readlink /home/kensei/.local/bin/hermes 2>/dev/null)
SHIM_OK="no"
[[ "$SHIM_TARGET" == "/home/kensei/repos/KenseiAgent/.venv/bin/hermes" ]] && SHIM_OK="yes"

# Status emoji
STATUS="✅"
[ "$JITER_ERRORS" -gt 0 ] && STATUS="⚠️"
[ "$SCHEMA_ERRORS_24H" -gt 0 ] && STATUS="⚠️"
[ "$MCP_TOOLS" = "?" ] && STATUS="⚠️"
[ "$LAYER_B_OK" = "no" ] && STATUS="❌"
[ "$SHIM_OK" = "no" ] && STATUS="❌"

# Verb · count · signal
SIGNAL="clean"
[ "$STATUS" = "⚠️" ] && SIGNAL="degraded"
[ "$STATUS" = "❌" ] && SIGNAL="regression"

cat <<EOF
$STATUS Post-remediation audit · $NOW
audit · 24h · $SIGNAL

• MCP: \`$MCP_TOOLS tools / $MCP_SERVERS servers\` (target: 73/2)
• jiter errors (24h): \`$JITER_ERRORS\` (target: 0)
• kanban schema errors (24h): \`$SCHEMA_ERRORS_24H\` (target: 0)
• tool-result persistence: \`$PERSISTED_24H\` spills logged, \`$PERSISTED_FILES\` files in \`/tmp/hermes-results/\`
• heartbeat-audit runs: \`$HEARTBEAT_RUNS\` (last: \`$HEARTBEAT_LAST\`, expected \`:15\`)
• rate-limit hits: \`$RATE_LIMITS\` · mnemosyne.beam fails: \`$MNEMOSYNE_BEAM\` (pre-existing latent bug)
• largest 24h session: \`${LARGEST_SIZE_KB}KB\` (\`$LARGEST_NAME\`) — regression if >500KB
• fork commit c2e5e4fd7 (Layer B): \`$LAYER_B_OK\` · CLI shim points at 3.11 venv: \`$SHIM_OK\`
EOF
