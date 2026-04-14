#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/yeqiuqiu/clawd-architect/ops/openclaw')
LANE_MAP = ROOT / 'codex_lane_account_map.json'
RECONCILE = ROOT / 'codex_lane_reconcile.py'
USAGE = ROOT / 'codex_usage_report.py'
AUDIT = ROOT / 'codex_worker_pool_audit.py'
DISPATCH_HEALTH = ROOT / 'codex_dispatch_health.py'

EXIT_CODE_ISSUES_PRESENT = 2
EXIT_CODE_DISPATCH_HEALTH_MISSING_AGENT = 4
ISSUE_KIND_DISPATCH_HEALTH_MISSING_AGENT = 'dispatch-health-missing-agent'
FAILURE_CODE_DISPATCH_HEALTH_MISSING_AGENT = 'dispatch_health_missing_agent'


def run_json(cmd):
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    import argparse

    ap = argparse.ArgumentParser(description='Codex lane auth + dispatch preflight audit.')
    ap.add_argument('--agent', action='append', help='Optionally require one or more agents to be dispatch-healthy')
    ap.add_argument(
        '--all-lanes',
        action='store_true',
        help='When used with --agent, also enforce global lane auth readiness checks (fleet mode).',
    )
    ap.add_argument('--lookback-hours', type=float, default=24.0)
    ap.add_argument('--dispatch-health-json', help='Optional output path for machine-checkable dispatch-health JSON')
    args = ap.parse_args()

    issues = []
    lane_map = json.loads(LANE_MAP.read_text())
    desired_agents = {lane['agent'] for lane in lane_map.get('lanes', [])}
    targeted_agents = {agent for agent in (args.agent or []) if agent}
    scope_agents = desired_agents if (not targeted_agents or args.all_lanes) else (desired_agents & targeted_agents)

    unknown_targeted = sorted(targeted_agents - desired_agents)
    for unknown in unknown_targeted:
        issues.append({'kind': 'target-agent-not-in-lane-map', 'agent': unknown})

    rc, out, err = run_json([sys.executable, str(RECONCILE), '--dry-run'])
    if rc != 0:
        print(json.dumps({'status': 'error', 'step': 'reconcile-dry-run', 'stderr': err}, indent=2))
        raise SystemExit(rc)
    reconcile = json.loads(out)
    for result in reconcile.get('results', []):
        agent = result.get('agent')
        if targeted_agents and not args.all_lanes and agent not in scope_agents:
            continue
        if result.get('status') != 'ok':
            issues.append({'kind': result.get('status') or 'missing-source', **result})

    rc, out, err = run_json([sys.executable, str(AUDIT), '--lane-map', str(LANE_MAP)])
    if rc != 0:
        print(json.dumps({'status': 'error', 'step': 'worker-pool-audit', 'stderr': err}, indent=2))
        raise SystemExit(rc)
    audit = json.loads(out)

    for lane in audit.get('lanes', []):
        if targeted_agents and not args.all_lanes and lane.get('agent') not in scope_agents:
            continue
        if lane.get('agent') in desired_agents and not lane.get('ready'):
            issues.append({'kind': 'lane-not-ready', 'agent': lane.get('agent'), 'detail': lane})

    rc, out, err = run_json([sys.executable, str(USAGE), '--json'])
    if rc != 0:
        print(json.dumps({'status': 'error', 'step': 'usage-report', 'stderr': err}, indent=2))
        raise SystemExit(rc)
    usage = json.loads(out)

    dispatch_cmd = [
        sys.executable,
        str(DISPATCH_HEALTH),
        '--lookback-hours',
        str(args.lookback_hours),
        '--include-orchestrator',
    ]
    if args.dispatch_health_json:
        dispatch_cmd.extend(['--json-output', args.dispatch_health_json])
    rc, out, err = run_json(dispatch_cmd)
    if rc != 0:
        print(json.dumps({'status': 'error', 'step': 'dispatch-health', 'stderr': err}, indent=2))
        raise SystemExit(rc)
    dispatch_health = json.loads(out)

    dispatch_lane_by_agent = {
        row.get('agent'): row
        for row in dispatch_health.get('lanes', [])
        if row.get('agent')
    }
    dispatch_check_agents = scope_agents if (not targeted_agents or args.all_lanes) else targeted_agents
    for agent in sorted(dispatch_check_agents):
        row = dispatch_lane_by_agent.get(agent)
        if not row:
            issues.append({'kind': ISSUE_KIND_DISPATCH_HEALTH_MISSING_AGENT, 'agent': agent})
            continue
        if row.get('status') != 'healthy':
            issues.append(
                {
                    'kind': 'dispatch-health-blocked-agent',
                    'agent': agent,
                    'dispatchStatus': row.get('status'),
                    'quarantineReasons': row.get('quarantineReasons') or [],
                    'probationReasons': row.get('probationReasons') or [],
                    'recommendedActions': row.get('recommendedActions') or [],
                }
            )

    hard_fail_kinds = {ISSUE_KIND_DISPATCH_HEALTH_MISSING_AGENT}
    hard_fail_issues = [
        issue
        for issue in issues
        if isinstance(issue, dict) and str(issue.get('kind') or '') in hard_fail_kinds
    ]
    has_hard_fail_issues = bool(hard_fail_issues)
    failure_code = None
    if has_hard_fail_issues:
        failure_code = FAILURE_CODE_DISPATCH_HEALTH_MISSING_AGENT
    elif issues:
        failure_code = 'preflight_issues_present'

    summary = {
        'status': 'fail' if has_hard_fail_issues else ('ok' if not issues else 'warn'),
        'hardFail': has_hard_fail_issues,
        'hardFailKinds': sorted({str(issue.get('kind') or '') for issue in hard_fail_issues if str(issue.get('kind') or '')}),
        'failureCode': failure_code,
        'issues': issues,
        'desiredLanes': sorted(desired_agents),
        'scope': {
            'targetedAgents': sorted(targeted_agents),
            'globalFleetChecks': bool(not targeted_agents or args.all_lanes),
            'authCheckAgents': sorted(scope_agents),
            'dispatchCheckAgents': sorted(dispatch_check_agents),
        },
        'dispatchHealth': {
            'statusCounts': dispatch_health.get('statusCounts'),
            'sharedAccountBindings': dispatch_health.get('sharedAccountBindings'),
            'sharedAccountBindingsAllowlisted': dispatch_health.get('sharedAccountBindingsAllowlisted'),
            'sharedAccountBindingsBlocked': dispatch_health.get('sharedAccountBindingsBlocked'),
            'lanes': dispatch_health.get('lanes'),
            'lookbackHours': dispatch_health.get('lookbackHours'),
            'generatedAt': dispatch_health.get('generatedAt'),
            'jsonOutputPath': dispatch_health.get('jsonOutputPath'),
        },
        'accounts': [
            {
                'email': acct.get('email'),
                'planType': acct.get('planType'),
                'agents': acct.get('agents'),
                'windows': acct.get('usage', {}).get('windows', []) if isinstance(acct.get('usage'), dict) else None,
            }
            for acct in usage.get('accounts', [])
        ],
    }
    print(json.dumps(summary, indent=2))
    if has_hard_fail_issues:
        raise SystemExit(EXIT_CODE_DISPATCH_HEALTH_MISSING_AGENT)
    raise SystemExit(0 if not issues else EXIT_CODE_ISSUES_PRESENT)


if __name__ == '__main__':
    main()
