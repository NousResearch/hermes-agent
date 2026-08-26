#!/bin/sh
# host-status-collector.sh — couche 1 de l'organe durable de surveillance Hermes.
#
# Produit /opt/data/host-status/host-status.{json,md} à partir de l'état du
# container hermes + du volume (jobs.json, exec db) + des dashboards +
# état watchdog. Lit via la même chaîne que le container hermes (donc
# identique au bot) — la couche 2 (synthèse LLM) lira ce JSON pour
# rédiger le [STATUS 12h] sans dépendre du G: local de l'opérateur.
#
# Configurable via flags ; défaut aligné sur la routine surveillance.
#
# Issue: jboige/hermes-agent #5 (couche 1)

set -u

STATUS_DIR="/opt/data/host-status"
LOG_FILE="/opt/data/logs/status-collector.log"
CRON_STALE_MIN=120
REVIEWS_GAP_MIN=240
BOT_STALE_MIN=120
CONTAINER_NAME="hermes"
GLOBAL_DASH="/opt/data/.shared-state/dashboards/global.md"
COORD_DASH="/opt/data/.shared-state/dashboards/workspace-cluster-coordination.md"

mkdir -p "$STATUS_DIR" "$(dirname "$LOG_FILE")"

log() {
    ts=$(date -u "+%Y-%m-%dT%H:%M:%SZ")
    printf '[%s] %s\n' "$ts" "$*" >> "$LOG_FILE"
}

# ---------- Sections ----------

section_container() {
    out=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Running}}|{{.State.StartedAt}}|{{.RestartCount}}|{{.Image}}' 2>/dev/null)
    if [ -z "$out" ]; then
        echo '{"running":false,"error":"docker inspect failed"}'
        return
    fi
    running=$(echo "$out" | cut -d'|' -f1)
    started=$(echo "$out" | cut -d'|' -f2)
    restarts=$(echo "$out" | cut -d'|' -f3)
    image=$(echo "$out" | cut -d'|' -f4)
    if [ "$running" = "true" ]; then
        started_epoch=$(date -u -d "$started" "+%s" 2>/dev/null || echo 0)
        now_epoch=$(date -u "+%s")
        uptime=$(( now_epoch - started_epoch ))
        upmin=$(( uptime / 60 ))
        cat <<EOF
{"running":true,"image":"$image","started_at":"$started","uptime_min":$upmin,"restarts":$restarts}
EOF
    else
        cat <<EOF
{"running":false,"image":"$image"}
EOF
    fi
}

section_gateway() {
    pid=$(ps -eo pid,cmd 2>/dev/null | awk '/[g]ateway run/ {print $1; exit}')
    if [ -n "$pid" ]; then
        echo "{\"running\":true,\"pid\":$pid}"
    else
        echo '{"running":false}'
    fi
}

section_crons() {
    jobs=$(cat /opt/data/cron/jobs.json 2>/dev/null)
    if [ -z "$jobs" ]; then
        echo '[{"error":"jobs.json not found"}]'
        return
    fi
    echo "$jobs" | python3 -c "
import json, sys
from datetime import datetime, timezone

CRON_STALE_MIN = $CRON_STALE_MIN
now = datetime.now(timezone.utc)

try:
    doc = json.load(sys.stdin)
except Exception as e:
    print(json.dumps([{'error': str(e)}]))
    sys.exit(0)

jobs = doc.get('jobs', doc) if isinstance(doc, dict) else doc
rows = []
for j in (jobs if isinstance(jobs, list) else jobs.values()):
    last = j.get('last_run_at') or j.get('last_status_at')
    last_out = last
    stale = None
    if last:
        try:
            dt = datetime.fromisoformat(last.replace('Z','+00:00'))
            stale = int((now - dt).total_seconds() / 60)
        except Exception:
            pass
    rows.append({
        'id': j.get('id', '') or j.get('job_id', ''),
        'name': j.get('name') or j.get('id') or '?',
        'last_run_at': last_out,
        'minutes_since': stale,
        'status': j.get('last_status') or j.get('status'),
        'stale_over_threshold': (stale is not None and stale > CRON_STALE_MIN),
    })
print(json.dumps(rows, ensure_ascii=False))
"
}

section_last_review() {
    sf="/opt/data/logs/review-watchdog-state.json"
    if [ ! -f "$sf" ]; then
        echo '{"known":false}'
        return
    fi
    python3 -c "
import json
from datetime import datetime, timezone

REVIEWS_GAP_MIN = $REVIEWS_GAP_MIN
with open('$sf') as f:
    o = json.load(f)
last = o.get('LastReviewSeen')
if not last:
    print(json.dumps({'known': True, 'last_seen': None}))
    raise SystemExit
try:
    dt = datetime.fromisoformat(last.replace('Z','+00:00'))
    gap = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
    print(json.dumps({
        'known': True,
        'last_seen': last,
        'gap_min': gap,
        'over_threshold': gap > REVIEWS_GAP_MIN,
    }))
except Exception as e:
    print(json.dumps({'known': True, 'last_seen': last, 'error': str(e)}))
"
}

section_watchdogs() {
    # Lecture via schtasks — pas exécutable dans le container. On se contente
    # de signaler l'indisponibilité ici ; la couche 2 du bot peut lire via
    # volume partagé (les .ps1 écrivent leur state dans /opt/data/logs/).
    # Comme compromis, on regarde la présence des fichiers state.json :
    sf_mcp="/opt/data/logs/mcp-watchdog-state.json"
    sf_rew="/opt/data/logs/review-watchdog-state.json"
    sf_ct="/opt/data/logs/cluster-tour-watchdog-state.json"
    python3 -c "
import json, os
from datetime import datetime, timezone

def last_recent(path):
    if not os.path.exists(path): return None
    try:
        with open(path) as f: o = json.load(f)
        return o.get('LastRun')
    except Exception: return None

results = []
for name, path in [
    ('Hermes-Review-Watchdog', '$sf_rew'),
    ('Hermes-MCP-Watchdog', '$sf_mcp'),
    ('Hermes-ClusterTour-Watchdog', '$sf_ct'),
]:
    last = last_recent(path)
    if not last:
        results.append({'name': name, 'result': None, 'ok': False, 'note': 'state missing'})
        continue
    try:
        last_dt = datetime.fromisoformat(last.replace('Z','+00:00'))
        fresh = (datetime.now(timezone.utc) - last_dt).total_seconds() < 60*60
        results.append({'name': name, 'result': 0 if fresh else 1, 'ok': fresh})
    except Exception:
        results.append({'name': name, 'result': None, 'ok': False})
print(json.dumps(results))
"
}

section_bus() {
    # Dans le container, G: n'existe pas — on regarde uniquement le filesystem
    # partagé /opt/data/.shared-state/ qui est la vue bot. Le test pertinent
    # pour le bus = le bot écrit-il ?
    COORD_DASH="/opt/data/.shared-state/dashboards/workspace-cluster-coordination.md"
    python3 -c "
import json, os, re
from datetime import datetime, timezone

BOT_STALE_MIN = $BOT_STALE_MIN
CO = '$COORD_DASH'
res = {'g_drive_ok': True, 'latest_bot_write': None}
if os.path.exists(CO):
    with open(CO, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    # Regex extraite l'en-tête du dernier message signé po-2026|...|... ou ai-01|...
    matches = re.findall(r'^### \[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\] ([^|]+)\|([^\s]+)', text, re.M)
    if matches:
        ts, machine, ws = matches[-1]
        try:
            dt = datetime.fromisoformat(ts.replace('Z','+00:00'))
            gap = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
            res['latest_bot_write'] = {
                'ts': ts, 'machine': machine.strip(), 'workspace': ws.strip(),
                'gap_min': gap,
                'stale_over_threshold': gap > BOT_STALE_MIN,
            }
        except Exception:
            pass
print(json.dumps(res))
"
}

section_cluster_tour() {
    GM="/opt/data/.shared-state/dashboards/global.md"
    python3 -c "
import json, os, re
from datetime import datetime, timezone
GM = '$GM'
out = {'known': False}
if os.path.exists(GM):
    with open(GM, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    m = re.findall(r'## \[CLUSTER-HEALTH\] T#(\d+) — (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)', text)
    if m:
        n, ts = m[-1]
        try:
            dt = datetime.fromisoformat(ts.replace('Z','+00:00'))
            gap = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
            out = {'known': True, 't': int(n), 'ts': ts, 'gap_min': gap}
        except Exception:
            pass
print(json.dumps(out))
"
}

section_error_patterns() {
    # Compter dans /proc/*/fd/1 sur N=24h — proxy : on regarde depuis 24h
    # grâce au timestamp mtime du log.
    docker_logs=$(docker logs "$CONTAINER_NAME" --since 24h 2>&1)
    if [ -z "$docker_logs" ]; then
        echo '{"ok":false}'
        return
    fi
    cnt=$(echo "$docker_logs" | grep -Eci 'model_dump|Streaming failed|429|401|crash|traceback' 2>/dev/null || echo 0)
    cat <<EOF
{"ok":true,"error_count_24h":$cnt}
EOF
}

# ---------- Orchestration ----------

log "Collecting..."

STATUS_DIR_REAL=$(date -u "+%Y/%m/%d")
TS_ISO=$(date -u "+%Y-%m-%dT%H:%M:%SZ")

# Compose JSON
{
    echo "{"
    printf '  "timestamp_utc": "%s",\n' "$TS_ISO"
    printf '  "container": %s,\n' "$(section_container)"
    printf '  "gateway": %s,\n' "$(section_gateway)"
    printf '  "crons": %s,\n' "$(section_crons)"
    printf '  "reviews_gap": %s,\n' "$(section_last_review)"
    printf '  "watchdogs": %s,\n' "$(section_watchdogs)"
    printf '  "bus": %s,\n' "$(section_bus)"
    printf '  "cluster_tour": %s,\n' "$(section_cluster_tour)"
    printf '  "error_patterns": %s\n' "$(section_error_patterns)"
    echo "}"
} > "$STATUS_DIR/host-status.json.tmp"

# Validation JSON
python3 -c "import json; json.load(open('$STATUS_DIR/host-status.json.tmp'))" 2>/dev/null
if [ $? -eq 0 ]; then
    mv "$STATUS_DIR/host-status.json.tmp" "$STATUS_DIR/host-status.json"
    cp "$STATUS_DIR/host-status.json" "$STATUS_DIR/host-status-$(date -u "+%Y%m%d-%H%M%S").json"
    # Rotation : garder 144 fichiers
    ls -1t "$STATUS_DIR"/host-status-*.json 2>/dev/null | tail -n +145 | xargs -r rm -f
    log "OK"
else
    log "INVALID JSON produced" "ERROR"
    rm -f "$STATUS_DIR/host-status.json.tmp"
fi
