#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/yeqiuqiu/clawd-architect')
INSPECT = ROOT / 'ops' / 'openclaw' / 'codex_auth_inspect.py'
NORMALIZE = ROOT / 'ops' / 'openclaw' / 'codex_auth_normalize.py'
RECONCILE = ROOT / 'ops' / 'openclaw' / 'codex_lane_reconcile.py'
USAGE = ROOT / 'ops' / 'openclaw' / 'codex_usage_report.py'
LANE_MAP = ROOT / 'ops' / 'openclaw' / 'codex_lane_account_map.json'
GLOBAL_CONFIG = Path('/home/yeqiuqiu/.openclaw/openclaw.json')

SAFE_ALIAS_RE = re.compile(r'^[A-Za-z0-9._-]+$')


def load_store(agent: str):
    path = Path(f'/home/yeqiuqiu/.openclaw/agents/{agent}/agent/auth-profiles.json')
    if not path.exists():
        return {}, path
    try:
        return json.loads(path.read_text()), path
    except Exception:
        return {}, path


def provider_profiles(store: dict, provider: str = 'openai-codex'):
    profiles = store.get('profiles', {}) or {}
    return {k: v for k, v in profiles.items() if isinstance(v, dict) and v.get('provider') == provider}


def run(cmd: list[str]):
    return subprocess.run(cmd, check=False)


def load_global_provider_profiles(provider: str = 'openai-codex'):
    try:
        cfg = json.loads(GLOBAL_CONFIG.read_text())
    except Exception:
        return {}
    profiles = ((cfg.get('auth') or {}).get('profiles') or {})
    return {
        profile_id: meta for profile_id, meta in profiles.items()
        if isinstance(meta, dict) and meta.get('provider') == provider
    }


def expected_email_for_agent(agent: str):
    try:
        obj = json.loads(LANE_MAP.read_text())
    except Exception:
        return None
    for lane in obj.get('lanes', []):
        if lane.get('agent') == agent:
            return lane.get('email')
    return None


def main():
    ap = argparse.ArgumentParser(
        description='Alias-aware OpenAI Codex login wrapper for worker-lane isolation.'
    )
    ap.add_argument('--agent', required=True, help='Target agent id, e.g. codex-worker-plus-2')
    ap.add_argument(
        '--alias',
        help='Stable alias to pin after login (default: agent id). Stored as openai-codex:<alias>.',
    )
    ap.add_argument('--provider', default='openai-codex', help='Provider id (default: openai-codex)')
    ap.add_argument('--method', help='Optional auth method id')
    ap.add_argument('--set-default', action='store_true', help='Pass through --set-default to openclaw')
    ap.add_argument(
        '--from-profile-id',
        help='Explicit source profile id to normalize when multiple profiles exist after login',
    )
    ap.add_argument(
        '--from-email',
        help='Explicit source email to normalize when multiple profiles exist after login',
    )
    ap.add_argument(
        '--set-order-only',
        action='store_true',
        help='Repin order only; do not rename profile id (use when desired alias already exists).',
    )
    args = ap.parse_args()

    if args.provider != 'openai-codex':
        raise SystemExit('This wrapper currently supports only --provider openai-codex.')

    alias = (args.alias or args.agent).strip()
    expected_email = expected_email_for_agent(args.agent)
    if not SAFE_ALIAS_RE.match(alias):
        raise SystemExit(
            'Alias must match [A-Za-z0-9._-]+. Example: codex-worker-plus-2'
        )

    desired_profile_id = f'{args.provider}:{alias}'

    before_store, store_path = load_store(args.agent)
    before_profiles = set(provider_profiles(before_store, args.provider).keys())
    before_global = set(load_global_provider_profiles(args.provider).keys())

    login_cmd = [
        'openclaw',
        'models',
        'auth',
        '--agent',
        args.agent,
        'login',
        '--provider',
        args.provider,
    ]
    if args.method:
        login_cmd.extend(['--method', args.method])
    if args.set_default:
        login_cmd.append('--set-default')

    print('== Running interactive login ==')
    print(' '.join(login_cmd))
    rc = run(login_cmd)
    if rc.returncode != 0:
        raise SystemExit(rc.returncode)

    after_store, _ = load_store(args.agent)
    after_provider_profiles = provider_profiles(after_store, args.provider)
    after_profiles = set(after_provider_profiles.keys())
    new_profiles = sorted(after_profiles - before_profiles)
    after_global = set(load_global_provider_profiles(args.provider).keys())
    new_global_profiles = sorted(after_global - before_global)

    normalize_cmd = [
        sys.executable,
        str(NORMALIZE),
        args.agent,
        '--profile-id',
        desired_profile_id,
    ]
    if args.set_order_only:
        normalize_cmd.append('--set-order-only')

    if args.from_profile_id:
        normalize_cmd.extend(['--from-profile-id', args.from_profile_id])
    elif args.from_email:
        normalize_cmd.extend(['--from-email', args.from_email])
    elif len(new_profiles) == 1:
        normalize_cmd.extend(['--from-profile-id', new_profiles[0]])

    print('\n== Normalizing to stable lane alias ==')
    print(' '.join(normalize_cmd))
    rc = run(normalize_cmd)
    if rc.returncode != 0:
        if not after_provider_profiles and new_global_profiles:
            print('\nGlobal-only OAuth landing detected.', file=sys.stderr)
            print(
                'New global auth profiles: ' + ', '.join(new_global_profiles),
                file=sys.stderr,
            )
            print(
                'OpenClaw stored profile metadata in ~/.openclaw/openclaw.json but no usable lane-local credential was written.',
                file=sys.stderr,
            )
            print(
                'Do not keep retrying blindly; inspect/capture the landed account and fix the lane explicitly.',
                file=sys.stderr,
            )
            raise SystemExit(3)
        print('\nNormalization failed. Current auth store summary:', file=sys.stderr)
        run([sys.executable, str(INSPECT), args.agent])
        print(
            '\nIf multiple profiles are present, rerun with --from-profile-id or --from-email.',
            file=sys.stderr,
        )
        raise SystemExit(rc.returncode)

    print('\n== Reconcile authoritative lane mapping ==')
    reconcile_cmd = [sys.executable, str(RECONCILE)]
    print(' '.join(reconcile_cmd))
    reconcile_rc = run(reconcile_cmd)
    if reconcile_rc.returncode != 0:
        raise SystemExit(reconcile_rc.returncode)

    print('\n== Verify pinned auth order ==')
    verify_cmd = [
        'openclaw',
        'models',
        'auth',
        'order',
        'get',
        '--agent',
        args.agent,
        '--provider',
        args.provider,
        '--json',
    ]
    print(' '.join(verify_cmd))
    verify_rc = run(verify_cmd)
    if verify_rc.returncode != 0:
        raise SystemExit(verify_rc.returncode)

    print('\n== Final lane summary ==')
    run([sys.executable, str(INSPECT), args.agent])
    if expected_email:
        final_store, _ = load_store(args.agent)
        final_profiles = provider_profiles(final_store, args.provider)
        landed = final_profiles.get(desired_profile_id, {})
        landed_email = (landed.get('email') or '').strip()
        if landed_email and landed_email.lower() != expected_email.lower():
            raise SystemExit(
                f'Post-reconcile mismatch for {args.agent}: expected {expected_email}, got {landed_email}'
            )
    print('\n== Current Codex usage report ==')
    run([sys.executable, str(USAGE)])
    print(f'\nDone. Stable profile target: {desired_profile_id}')


if __name__ == '__main__':
    main()
