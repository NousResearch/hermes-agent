#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/yeqiuqiu/clawd-architect/ops/openclaw')
MAP_PATH = ROOT / 'codex_lane_account_map.json'
RECONCILE = ROOT / 'codex_lane_reconcile.py'
USAGE = ROOT / 'codex_usage_report.py'


def run_json(cmd):
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    ap = argparse.ArgumentParser(description='Detect Codex lane drift and optionally auto-heal it.')
    ap.add_argument('--heal', action='store_true', help='Run lane reconciliation when drift is detected')
    ap.add_argument('--strict', action='store_true', help='Exit non-zero if any desired lane has <= 5%% week remaining')
    args = ap.parse_args()

    desired = {x['agent']: x for x in json.loads(MAP_PATH.read_text()).get('lanes', [])}

    rc, out, err = run_json([sys.executable, str(USAGE), '--json'])
    if rc != 0:
        print(json.dumps({'status': 'error', 'stderr': err}, indent=2))
        raise SystemExit(rc)
    payload = json.loads(out)
    issues = []

    rc_reconcile, out_reconcile, _ = run_json([sys.executable, str(RECONCILE), '--dry-run'])
    if rc_reconcile == 0:
        reconcile = json.loads(out_reconcile)
        for result in reconcile.get('results', []):
            if result.get('status') == 'global-only-source':
                issues.append({
                    'kind': 'global-only-source',
                    'agent': result.get('agent'),
                    'email': result.get('email'),
                    'globalProfileId': result.get('globalProfileId'),
                })

    for lane in payload.get('lanes', []):
        agent = lane.get('agent')
        if agent not in desired:
            continue
        want = desired[agent]
        if lane.get('email') != want.get('email'):
            issues.append({'kind': 'email-mismatch', 'agent': agent, 'expected': want.get('email'), 'actual': lane.get('email')})
        if lane.get('boundProfile') != want.get('desiredProfileId'):
            issues.append({'kind': 'profile-mismatch', 'agent': agent, 'expected': want.get('desiredProfileId'), 'actual': lane.get('boundProfile')})
        usage = lane.get('usage') or {}
        for w in usage.get('windows', []):
            if w.get('label') == 'Week' and args.strict and (w.get('remainingPercent') or 0) <= 5:
                issues.append({'kind': 'low-week-headroom', 'agent': agent, 'email': lane.get('email'), 'remainingPercent': w.get('remainingPercent')})

    healed = False
    if issues and args.heal:
        healed = subprocess.run([sys.executable, str(RECONCILE)], check=False).returncode == 0
        rc2, out2, _ = run_json([sys.executable, str(USAGE), '--json'])
        if rc2 == 0:
            payload = json.loads(out2)

    print(json.dumps({'status': 'ok' if not issues else 'warn', 'issues': issues, 'healed': healed}, indent=2))
    raise SystemExit(0 if not issues else 2)


if __name__ == '__main__':
    main()
