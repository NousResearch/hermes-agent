#!/usr/bin/env python3
import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/yeqiuqiu/.openclaw/agents')
BACKUP_DIR = Path('/home/yeqiuqiu/.openclaw/_archives/auth-profiles')
CONFIG_PATH = Path('/home/yeqiuqiu/.openclaw/openclaw.json')
PROVIDER = 'openai-codex'


def decode_jwt_payload(token: str):
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return {}
        payload = parts[1] + ('=' * (-len(parts[1]) % 4))
        return json.loads(base64.urlsafe_b64decode(payload).decode('utf-8'))
    except Exception:
        return {}


def profile_email(cred: dict):
    payload = decode_jwt_payload(cred.get('access') or '')
    profile = payload.get('https://api.openai.com/profile') or {}
    return (profile.get('email') or cred.get('email') or '').strip()


def profile_account_id(cred: dict):
    payload = decode_jwt_payload(cred.get('access') or '')
    auth = payload.get('https://api.openai.com/auth') or {}
    return (auth.get('chatgpt_account_id') or cred.get('accountId') or '').strip()


def load_store(path: Path):
    if not path.exists():
        return {'version': 1, 'profiles': {}, 'order': {}, 'lastGood': {}, 'usageStats': {}}
    return json.loads(path.read_text())


def backup(path: Path):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = BACKUP_DIR / f'{path.parent.parent.name}_{ts}.auth-profiles.json'
    out.write_text(path.read_text() if path.exists() else '')
    return out


def all_auth_paths():
    return sorted(ROOT.glob('*/agent/auth-profiles.json'))


def build_index():
    index = {}
    for path in all_auth_paths():
        store = load_store(path)
        for pid, cred in (store.get('profiles') or {}).items():
            if not isinstance(cred, dict) or cred.get('provider') != PROVIDER:
                continue
            email = profile_email(cred)
            if not email:
                continue
            # prefer non-default aliases over default if both exist
            score = 1 if pid.endswith(':default') else 2
            existing = index.get(email)
            if not existing or score > existing['score']:
                index[email] = {'path': path, 'profileId': pid, 'cred': cred, 'score': score}
    return index


def build_global_index():
    if not CONFIG_PATH.exists():
        return {}
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}
    index = {}
    profiles = ((cfg.get('auth') or {}).get('profiles') or {})
    for profile_id, meta in profiles.items():
        if not isinstance(meta, dict) or meta.get('provider') != PROVIDER:
            continue
        if ':' not in profile_id:
            continue
        email = profile_id.split(':', 1)[1].strip()
        if '@' not in email:
            continue
        index[email.lower()] = {
            'path': str(CONFIG_PATH),
            'profileId': profile_id,
            'email': email,
        }
    return index


def reconcile_lane(spec: dict, index: dict, global_index: dict, dry_run: bool):
    agent = spec['agent']
    desired = spec['desiredProfileId']
    email = spec['email']
    target_path = ROOT / agent / 'agent' / 'auth-profiles.json'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    store = load_store(target_path)
    found = index.get(email)
    if not found:
        global_found = global_index.get(email.lower())
        if global_found:
            return {
                'agent': agent,
                'status': 'global-only-source',
                'email': email,
                'globalProfileId': global_found['profileId'],
                'globalPath': global_found['path'],
            }
        return {'agent': agent, 'status': 'missing-source', 'email': email}
    cred = json.loads(json.dumps(found['cred']))
    if not cred.get('email'):
        cred['email'] = email
    next_store = {
        'version': store.get('version', 1),
        'profiles': {desired: cred},
        'order': {PROVIDER: [desired]},
        'lastGood': {PROVIDER: desired},
        'usageStats': {desired: (store.get('usageStats') or {}).get(desired, {'errorCount': 0})}
    }
    result = {
        'agent': agent,
        'email': email,
        'sourcePath': str(found['path']),
        'sourceProfileId': found['profileId'],
        'targetPath': str(target_path),
        'targetProfileId': desired,
        'accountId': profile_account_id(cred),
        'status': 'ok'
    }
    if dry_run:
        return result
    backup_path = backup(target_path)
    target_path.write_text(json.dumps(next_store, indent=2) + '\n')
    result['backupPath'] = str(backup_path)
    return result


def main():
    ap = argparse.ArgumentParser(description='Authoritatively reconcile Codex lane stores from a desired email->lane map.')
    ap.add_argument('--map', default='/home/yeqiuqiu/clawd-architect/ops/openclaw/codex_lane_account_map.json')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    spec = json.loads(Path(args.map).read_text())
    index = build_index()
    global_index = build_global_index()
    out = {'provider': spec.get('provider', PROVIDER), 'results': []}
    for lane in spec.get('lanes', []):
        out['results'].append(reconcile_lane(lane, index, global_index, args.dry_run))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
