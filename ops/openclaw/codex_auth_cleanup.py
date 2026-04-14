#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_store(agent: str):
    path = Path(f'/home/yeqiuqiu/.openclaw/agents/{agent}/agent/auth-profiles.json')
    if not path.exists():
        raise SystemExit(f'Auth store not found: {path}')
    return path, json.loads(path.read_text())


def backup_path_for(path: Path):
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup_dir = Path('/home/yeqiuqiu/.openclaw/_archives/auth-profiles')
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir / f'{path.parent.parent.name}_{ts}.auth-profiles.json'


def main():
    ap = argparse.ArgumentParser(description='Prune/pin OpenAI Codex auth store profiles for one agent.')
    ap.add_argument('agent')
    ap.add_argument('--provider', default='openai-codex')
    ap.add_argument('--keep-profile-id', action='append', default=[], help='Profile id to keep (repeatable)')
    ap.add_argument('--pin-profile-id', help='Set auth order + lastGood to this profile id')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    path, obj = load_store(args.agent)
    profiles = obj.get('profiles', {}) or {}
    provider = args.provider
    keep = set(args.keep_profile_id)

    removed = []
    kept = []
    next_profiles = {}
    for pid, prof in profiles.items():
        if not isinstance(prof, dict):
            next_profiles[pid] = prof
            kept.append(pid)
            continue
        if prof.get('provider') != provider:
            next_profiles[pid] = prof
            kept.append(pid)
            continue
        if pid in keep:
            next_profiles[pid] = prof
            kept.append(pid)
        else:
            removed.append(pid)

    obj['profiles'] = next_profiles
    order = obj.setdefault('order', {})
    if args.pin_profile_id:
        order[provider] = [args.pin_profile_id]
        last_good = obj.setdefault('lastGood', {})
        last_good[provider] = args.pin_profile_id
    elif provider in order:
        order[provider] = [pid for pid in order.get(provider, []) if pid in next_profiles]

    usage = obj.get('usageStats') or {}
    for pid in list(usage.keys()):
        if pid not in next_profiles:
            usage.pop(pid, None)
    if usage:
        obj['usageStats'] = usage
    elif 'usageStats' in obj:
        obj['usageStats'] = {}

    result = {
        'agent': args.agent,
        'path': str(path),
        'pinProfileId': args.pin_profile_id,
        'keptProfiles': kept,
        'removedProfiles': removed,
        'order': obj.get('order', {}).get(provider, []),
        'lastGood': obj.get('lastGood', {}).get(provider),
    }

    if args.dry_run:
        print(json.dumps(result, indent=2))
        return

    backup = backup_path_for(path)
    backup.write_text(path.read_text())
    path.write_text(json.dumps(obj, indent=2) + '\n')
    result['backupPath'] = str(backup)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
