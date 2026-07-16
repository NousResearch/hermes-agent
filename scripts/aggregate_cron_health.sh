#!/bin/bash
# aggregate_cron_health.sh — Daily cron health aggregation with SLO monitoring
# Designed for Hermes Agent cron system. Zero dependencies, no_agent compatible.
# Schedule: 0 6 * * * (daily at 06:00), no_agent=true
#
# Output: JSON to $HERMES_HOME/logs/cron_health_YYYY-MM-DD.json
# Alert: when overall success rate < 95% or LLM cron output anomalies detected
# Silent: when all healthy (empty stdout = no delivery)

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
OUTDIR="$HERMES_HOME/logs"
mkdir -p "$OUTDIR"

TODAY=$(date +%Y-%m-%d)
OUTFILE="$OUTDIR/cron_health_${TODAY}.json"
JOBS_FILE="$HERMES_HOME/cron/jobs.json"
export HERMES_HOME JOBS_FILE OUTFILE TODAY

python3 << 'PYEOF'
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

HERMES_HOME = os.environ.get('HERMES_HOME', str(Path.home() / '.hermes'))
jobs_file = os.environ.get('JOBS_FILE', f'{HERMES_HOME}/cron/jobs.json')
outfile = os.environ['OUTFILE']

with open(jobs_file) as f:
    data = json.load(f)
jobs = data.get('jobs', [])

total = 0
success = 0
failures = []
llm_anomalies = []

for j in jobs:
    if j.get('paused_at'):
        continue
    total += 1
    status = j.get('last_status', 'unknown')

    if status == 'ok':
        success += 1
    if status in ('error', 'timeout'):
        failures.append({
            'id': j['id'][:12],
            'name': j.get('name', '?')[:60],
            'reason': status,
        })

    # LLM cron output quality check
    if not j.get('no_agent'):
        output_dir = Path(HERMES_HOME) / 'cron' / 'output' / j['id']
        if output_dir.exists():
            files = sorted(output_dir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                try:
                    lines = len(files[0].read_text().split('\n'))
                    if lines < 10:
                        llm_anomalies.append({
                            'id': j['id'][:12],
                            'output_lines': lines,
                            'expected_min': 10,
                        })
                except Exception as e:
                    print('[warn] cannot read output:', j['id'][:12], str(e), file=sys.stderr)

success_rate = round(success / total, 2) if total > 0 else 1.0

result = {
    'date': os.environ['TODAY'],
    'total_crons': total,
    'overall_success_rate': success_rate,
    'failures': failures,
    'llm_output_anomalies': llm_anomalies,
}

with open(outfile, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# Summary line for health check consumers (stdout → delivery / machine-readable)

# Exit code strategy (compatible with hermes health #51028):
# 0 = healthy, 1 = warning, 2 = critical
exit_code = 0
alerts = []

if success_rate < 0.95:
    alerts.append(f'CRITICAL: success rate {success_rate:.0%} below 95% SLO')
    exit_code = 2
elif failures:
    alerts.append(f'WARNING: {len(failures)} cron job(s) failing')
    exit_code = 1
if llm_anomalies:
    alerts.append(f'WARNING: {len(llm_anomalies)} LLM cron(s) with thin output (<10 lines)')
    exit_code = max(exit_code, 1)

if alerts:
    for a in alerts:
        print(a)

sys.exit(exit_code)
# Healthy: no output → no_agent mode stays silent
PYEOF
