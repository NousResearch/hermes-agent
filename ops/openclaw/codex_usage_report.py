#!/usr/bin/env python3
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

AGENTS = [
    'codex-orchestrator-pro',
    'codex-orchestrator-plus',
    'codex-worker-pro',
    'codex-worker-plus-1',
    'codex-worker-plus-2',
    'codex-worker-plus-3',
    'codex-worker-plus-4',
    'codex-worker-plus-5',
    'codex-worker-plus-6',
    'codex-worker-plus-7',
    'codex-worker-plus-8',
    'codex-worker-plus-9',
    'codex-worker-plus-10',
    'codex-worker-plus-11',
]

WEEKLY_RESET_GAP_SECONDS = 4320 * 60


def decode_jwt_payload(token: str):
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return {}
        payload = parts[1] + ('=' * (-len(parts[1]) % 4))
        return json.loads(base64.urlsafe_b64decode(payload).decode('utf-8'))
    except Exception:
        return {}


def read_store(agent: str):
    path = Path(f'/home/yeqiuqiu/.openclaw/agents/{agent}/agent/auth-profiles.json')
    if not path.exists():
        return path, {}
    return path, json.loads(path.read_text())


def resolve_ordered_profiles(store: dict):
    profiles = store.get('profiles', {}) or {}
    provider_profiles = {k: v for k, v in profiles.items() if isinstance(v, dict) and v.get('provider') == 'openai-codex'}
    order = list((store.get('order', {}) or {}).get('openai-codex', []) or [])
    out = []
    seen = set()
    for pid in order:
        if pid in provider_profiles:
            out.append((pid, provider_profiles[pid]))
            seen.add(pid)
    for pid, prof in provider_profiles.items():
        if pid not in seen:
            out.append((pid, prof))
    return out


def profile_summary(profile_id: str, prof: dict):
    payload = decode_jwt_payload(prof.get('access') or '')
    auth = (payload.get('https://api.openai.com/auth') or {}) if isinstance(payload, dict) else {}
    profile = (payload.get('https://api.openai.com/profile') or {}) if isinstance(payload, dict) else {}
    email = profile.get('email') or prof.get('email')
    return {
        'profileId': profile_id,
        'email': email,
        'accountId': auth.get('chatgpt_account_id') or prof.get('accountId'),
        'planType': auth.get('chatgpt_plan_type'),
        'access': prof.get('access'),
    }


def resolve_secondary_label(primary_reset_at, secondary_reset_at, limit_window_seconds):
    hours = round((limit_window_seconds or 86400) / 3600)
    if hours >= 168:
        return 'Week'
    if hours < 24:
        return f'{hours}h'
    if isinstance(primary_reset_at, (int, float)) and isinstance(secondary_reset_at, (int, float)):
        if secondary_reset_at - primary_reset_at >= WEEKLY_RESET_GAP_SECONDS:
            return 'Week'
    return 'Day'


def fetch_usage(access_token: str, account_id: str | None, timeout: int = 8):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'CodexBar',
        'Accept': 'application/json',
    }
    if account_id:
        headers['ChatGPT-Account-Id'] = account_id
    req = urllib.request.Request('https://chatgpt.com/backend-api/wham/usage', headers=headers, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8', errors='ignore')
        except Exception:
            body = ''
        return {'error': f'HTTP {e.code}', 'body': body[:500]}
    except Exception as e:
        return {'error': str(e)}

    windows = []
    rate = data.get('rate_limit') or {}
    pw = rate.get('primary_window') or {}
    if pw:
        hours = round((pw.get('limit_window_seconds') or 10800) / 3600)
        used = float(pw.get('used_percent') or 0)
        windows.append({
            'label': f'{hours}h',
            'usedPercent': max(0, min(100, used)),
            'remainingPercent': max(0, min(100, 100 - used)),
            'resetAt': pw.get('reset_at'),
        })
    sw = rate.get('secondary_window') or {}
    if sw:
        label = resolve_secondary_label(
            pw.get('reset_at'),
            sw.get('reset_at'),
            sw.get('limit_window_seconds') or 86400,
        )
        used = float(sw.get('used_percent') or 0)
        windows.append({
            'label': label,
            'usedPercent': max(0, min(100, used)),
            'remainingPercent': max(0, min(100, 100 - used)),
            'resetAt': sw.get('reset_at'),
        })
    return {'windows': windows, 'raw': data}


def fmt_reset(ts_seconds):
    if not ts_seconds:
        return None
    try:
        return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).astimezone().isoformat(timespec='minutes')
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description='Report Codex OAuth account usage/headroom and lane bindings.')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--agent', action='append', help='Limit to one or more agents')
    args = ap.parse_args()

    agents = args.agent or AGENTS
    account_map = OrderedDict()
    lane_rows = []

    for agent in agents:
        path, store = read_store(agent)
        ordered = resolve_ordered_profiles(store)
        if not ordered:
            lane_rows.append({'agent': agent, 'boundProfile': None, 'email': None, 'planType': None, 'accountId': None, 'usage': None})
            continue
        active_id, active_prof = ordered[0]
        summary = profile_summary(active_id, active_prof)
        lane_rows.append({'agent': agent, 'boundProfile': active_id, 'email': summary['email'], 'planType': summary['planType'], 'accountId': summary['accountId'], 'usage': None})
        key = summary['accountId'] or summary['email'] or active_id
        if key not in account_map:
            account_map[key] = {
                'email': summary['email'],
                'planType': summary['planType'],
                'accountId': summary['accountId'],
                'profileId': active_id,
                'agents': [agent],
                'access': summary['access'],
            }
        else:
            account_map[key]['agents'].append(agent)

    reports = []
    for key, account in account_map.items():
        usage = fetch_usage(account['access'], account['accountId']) if account.get('access') else {'error': 'No access token'}
        reports.append({
            'email': account['email'],
            'planType': account['planType'],
            'accountId': account['accountId'],
            'profileId': account['profileId'],
            'agents': account['agents'],
            'usage': usage,
        })
        for row in lane_rows:
            if row['accountId'] == account['accountId'] and row['email'] == account['email']:
                row['usage'] = usage

    payload = {'accounts': reports, 'lanes': lane_rows, 'generatedAt': datetime.now(timezone.utc).isoformat()}
    if args.json:
        safe = json.loads(json.dumps(payload))
        for acct in safe['accounts']:
            pass
        print(json.dumps(payload, indent=2))
        return

    print('Codex usage by account')
    print('======================')
    for acct in reports:
        print(f"- {acct['email'] or '(unknown email)'} | plan={acct['planType'] or '?'} | agents={', '.join(acct['agents'])}")
        usage = acct['usage'] or {}
        if usage.get('error'):
            print(f"  usage: ERROR {usage['error']}")
            continue
        for window in usage.get('windows', []):
            reset = fmt_reset(window.get('resetAt'))
            suffix = f" | resets {reset}" if reset else ''
            print(f"  {window['label']}: {window['remainingPercent']:.1f}% left ({window['usedPercent']:.1f}% used){suffix}")
    print('\nLane bindings')
    print('=============')
    for row in lane_rows:
        if not row['boundProfile']:
            print(f"- {row['agent']}: unbound")
            continue
        usage = row.get('usage') or {}
        week_left = None
        for w in usage.get('windows', []):
            if w.get('label') == 'Week':
                week_left = w.get('remainingPercent')
                break
        suffix = f" | week-left={week_left:.1f}%" if isinstance(week_left, (int, float)) else ''
        print(f"- {row['agent']}: {row['email']} ({row['planType']}) via {row['boundProfile']}{suffix}")


if __name__ == '__main__':
    main()
