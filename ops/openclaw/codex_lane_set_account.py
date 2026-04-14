#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

MAP_PATH = Path('/home/yeqiuqiu/clawd-architect/ops/openclaw/codex_lane_account_map.json')
RECONCILE = Path('/home/yeqiuqiu/clawd-architect/ops/openclaw/codex_lane_reconcile.py')
PREFLIGHT = Path('/home/yeqiuqiu/clawd-architect/ops/openclaw/codex_preflight_audit.py')


def main():
    ap = argparse.ArgumentParser(description='Safely update authoritative Codex lane->account mapping.')
    ap.add_argument('--agent', required=True)
    ap.add_argument('--email', required=True)
    ap.add_argument('--profile-id', help='Override desired profile id (default: openai-codex:<agent>)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-runtime-reload', action='store_true', help='Do not run `openclaw secrets reload` after reconcile.')
    args = ap.parse_args()

    obj = json.loads(MAP_PATH.read_text())
    lanes = obj.setdefault('lanes', [])
    desired = args.profile_id or f'openai-codex:{args.agent}'

    found = False
    for lane in lanes:
        if lane.get('agent') == args.agent:
            lane['email'] = args.email
            lane['desiredProfileId'] = desired
            found = True
            break
    if not found:
        lanes.append({'agent': args.agent, 'email': args.email, 'desiredProfileId': desired})

    if args.dry_run:
        print(json.dumps(obj, indent=2))
        return

    MAP_PATH.write_text(json.dumps(obj, indent=2) + '\n')
    print(f'updated {MAP_PATH}: {args.agent} -> {args.email} ({desired})')

    rc = subprocess.run([sys.executable, str(RECONCILE)], check=False).returncode
    if rc != 0:
        raise SystemExit(rc)

    if not args.skip_runtime_reload:
        reload_cmd = ['openclaw', 'secrets', 'reload', '--json']
        reload_proc = subprocess.run(reload_cmd, check=False, capture_output=True, text=True)
        if reload_proc.returncode != 0:
            print('runtime reload failed:', ' '.join(reload_cmd), file=sys.stderr)
            if reload_proc.stdout.strip():
                print(reload_proc.stdout.strip(), file=sys.stderr)
            if reload_proc.stderr.strip():
                print(reload_proc.stderr.strip(), file=sys.stderr)
            raise SystemExit(reload_proc.returncode)
        payload = reload_proc.stdout.strip()
        if payload:
            print(f'runtime reload: {payload}')

    rc = subprocess.run([sys.executable, str(PREFLIGHT)], check=False).returncode
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
