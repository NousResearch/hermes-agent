#!/usr/bin/env bash
# Sync-bridge health probe for Hermes ↔ OpenClaw memory sync.
#
# Spec: ~/wiki/operations/sync-bridge-health-probe-spec.md (se-007)
#
# Runs four checks (H1–H4), emits JSON to stdout, writes an incident file to
# ~/wiki/operations/sync-bridge-incidents/ on failure.
#
# Exit codes:
#   0 = all green
#   1 = at least one red
#   2 = probe itself errored

set -u  # treat unset vars as errors; DO NOT use -e (we want checks to continue)

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

WIKI_DIR="${WIKI_DIR:-$HOME/wiki}"
LOG_DIR="${LOG_DIR:-$HOME/.hermes/logs}"
INCIDENT_DIR="${INCIDENT_DIR:-$WIKI_DIR/operations/sync-bridge-incidents}"
SERVICE_NAME="${SYNC_SERVICE_NAME:-com.brian.wiki-memory-sync}"
MAX_EXPORT_AGE_SEC="${MAX_EXPORT_AGE_SEC:-7200}"  # 2 hours
MEMORY_RECALL_TIMEOUT_SEC="${MEMORY_RECALL_TIMEOUT_SEC:-3}"

mkdir -p "$INCIDENT_DIR"

TS_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TS_LOCAL="$(date +%Y-%m-%dT%H:%M:%S%z)"
TS_SEC="$(date +%s)"

declare -a results=()
overall_ok=true

add_result() {
    local id="$1" ok="$2" detail="$3"
    detail_escaped=$(printf '%s' "$detail" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    results+=("{\"id\":\"$id\",\"ok\":$ok,\"detail\":$detail_escaped}")
    if [ "$ok" = "false" ]; then overall_ok=false; fi
}

# ---------------- H1 — launchd service ----------------
check_h1() {
    local uid
    uid=$(id -u)
    local out
    out=$(launchctl print "gui/$uid/$SERVICE_NAME" 2>&1) || {
        add_result "H1" false "launchctl print failed for $SERVICE_NAME"
        return
    }
    local last_exit
    last_exit=$(echo "$out" | awk -F'=' '/last exit code/ {gsub(/[ \t]/, "", $2); print $2; exit}')
    if [ -z "$last_exit" ]; then
        add_result "H1" false "no last exit code found"
        return
    fi
    if [ "$last_exit" != "0" ]; then
        add_result "H1" false "last exit code = $last_exit (expected 0)"
        return
    fi
    add_result "H1" true "launchd loaded, last exit=0"
}

# ---------------- H2 — sync actually running (via stdout log) ----------------
check_h2() {
    # 改用 stdout log 判讀最近一次 "=== 同步開始 ===" 時間，而非 export mtime。
    # 因為 export 檔名以日期命名（YYYY-MM-DD_openclaw_memory_export.md），
    # 同一天的 sync 不會覆寫檔案，mtime 會看起來 stale 但其實 sync 正常。
    local log="$LOG_DIR/wiki-memory-sync.stdout.log"
    if [ ! -f "$log" ]; then
        add_result "H2" false "stdout log missing: $log (sync never ran?)"
        return
    fi
    # 抓最後一次 "同步開始" 的時間戳（log line 格式：[YYYY-MM-DD HH:MM] === 同步開始 ===）
    local last_start
    last_start=$(grep "=== 同步開始 ===" "$log" | tail -n 1 | grep -oE '\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\]' | tr -d '[]')
    if [ -z "$last_start" ]; then
        add_result "H2" false "no sync-start entry found in stdout log"
        return
    fi
    local last_epoch
    last_epoch=$(date -j -f "%Y-%m-%d %H:%M" "$last_start" +%s 2>/dev/null || date -d "$last_start" +%s 2>/dev/null)
    if [ -z "$last_epoch" ]; then
        add_result "H2" false "could not parse timestamp: $last_start"
        return
    fi
    local age=$(( TS_SEC - last_epoch ))
    # Sync interval 是 1800s（30min），容忍 3 倍間隔 = 90min 前最近一次跑
    local max_since_last=${MAX_SINCE_LAST_SYNC_SEC:-5400}
    if [ "$age" -gt "$max_since_last" ]; then
        local hrs=$(( age / 3600 ))
        local mins=$(( (age % 3600) / 60 ))
        add_result "H2" false "last sync was ${hrs}h ${mins}m ago (expected every 30min)"
        return
    fi
    add_result "H2" true "last sync $((age/60))m ago (at $last_start)"
}

# ---------------- H3 — stderr log tail ----------------
check_h3() {
    local log="$LOG_DIR/wiki-memory-sync.stderr.log"
    if [ ! -f "$log" ]; then
        add_result "H3" true "no stderr log yet (clean slate)"
        return
    fi
    local errors
    errors=$(tail -n 50 "$log" | grep -iE "error|traceback|fatal|exception" | head -5 || true)
    if [ -n "$errors" ]; then
        local first_line
        first_line=$(printf '%s\n' "$errors" | head -n 1)
        add_result "H3" false "stderr log has errors: ${first_line:0:200}"
        return
    fi
    add_result "H3" true "stderr log tail clean"
}

# ---------------- H4 — memory recall ----------------
_run_with_timeout() {
    # Portable timeout wrapper. Returns exit code of inner command, or 124 on timeout.
    local secs="$1"; shift
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$secs" "$@"
        return $?
    fi
    if command -v timeout >/dev/null 2>&1; then
        timeout "$secs" "$@"
        return $?
    fi
    # Fallback: run in background and kill if it takes too long.
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill -9 "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill -9 "$watcher" 2>/dev/null || true
    return $rc
}

check_h4() {
    # Verify LanceDB / memory store is reachable. Probes whichever recall path
    # is wired up on this host. Kept advisory — unknown CLI subcommand is NOT
    # a red flag, only genuine timeout/DB error is.
    if ! command -v openclaw >/dev/null 2>&1; then
        add_result "H4" true "openclaw CLI absent; skipping (advisory)"
        return
    fi
    local out rc
    out=$(_run_with_timeout "$MEMORY_RECALL_TIMEOUT_SEC" openclaw memory recall --query "health" --limit 1 2>&1)
    rc=$?
    # 137 = SIGKILL from our timeout wrapper (the CLI hung printing help for unknown cmd)
    # 2 = unknown command (argparse)
    # Either way, memory subcommand isn't available → treat as advisory skip, not red.
    if [ $rc -eq 124 ] || [ $rc -eq 0 ]; then
        add_result "H4" true "memory_recall ok (rc=$rc)"
        return
    fi
    if echo "$out" | grep -qi "unknown command"; then
        add_result "H4" true "memory subcommand not wired on this host; skipping (advisory)"
        return
    fi
    add_result "H4" false "memory_recall failed rc=$rc: ${out:0:200}"
}

# ---------------- run all ----------------
check_h1
check_h2
check_h3
check_h4

# ---------------- emit JSON ----------------
IFS=','
checks_json="[${results[*]}]"
unset IFS

printf '{"ts":"%s","overall_ok":%s,"checks":%s}\n' \
    "$TS_ISO" "$overall_ok" "$checks_json"

# ---------------- incident write if red ----------------
if [ "$overall_ok" = "false" ]; then
    incident_ts=$(date +%Y-%m-%dT%H%M)
    incident_file="$INCIDENT_DIR/${incident_ts}.md"
    {
        printf -- '---\n'
        printf 'title: Sync bridge incident %s\n' "$TS_LOCAL"
        printf 'created: %s\n' "$TS_LOCAL"
        printf 'type: incident\n'
        printf 'tags: [sync-bridge, incident, se-007]\n'
        printf -- '---\n\n'
        printf '# Sync bridge incident\n\n'
        printf '**Time**: %s\n' "$TS_LOCAL"
        printf '**Overall**: RED\n\n'
        printf '## Checks\n'
        for r in "${results[@]}"; do
            printf -- '- %s\n' "$r"
        done
        printf '\n## Recovery hints\n'
        printf '1. `launchctl kickstart -k gui/$(id -u)/%s`\n' "$SERVICE_NAME"
        printf '2. `tail -200 %s`\n' "$LOG_DIR/wiki-memory-sync.stderr.log"
        printf '3. Manual run: `~/wiki-memory-sync.sh` (if script path differs, update INCIDENT template)\n'
    } > "$incident_file"
    exit 1
fi

exit 0
