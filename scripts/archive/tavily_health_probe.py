#!/usr/bin/env python3
"""Lightweight Tavily API health probe for KENSEI heartbeat audits.

Runs a minimal Tavily search, logs each result, and tracks consecutive failures.
The script never prints or logs the API key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERMES = Path('/home/kensei/.hermes')
DEFAULT_ENV = HERMES / '.env'
DEFAULT_STATE = HERMES / 'state' / 'tavily_health.json'
DEFAULT_LOG = HERMES / 'logs' / 'tavily_health.jsonl'
API_URL = 'https://api.tavily.com/search'
LONDON = ZoneInfo('Europe/London')


def now_iso() -> str:
    return datetime.now(LONDON).isoformat(timespec='seconds')


def load_env_key(env_file: Path = DEFAULT_ENV) -> str | None:
    key = os.environ.get('TAVILY_API_KEY')
    if key:
        return key.strip()
    if not env_file.exists():
        return None
    for raw in env_file.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        if name.strip() == 'TAVILY_API_KEY':
            return value.strip().strip('"').strip("'") or None
    return None


def load_state(path: Path) -> dict:
    if not path.exists():
        return {'consecutive_failures': 0}
    try:
        data = json.loads(path.read_text(encoding='utf-8', errors='replace'))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {'consecutive_failures': 0, 'state_read_error': True}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def append_log(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(event, sort_keys=True) + '\n')


def classify_failure(status_code: int | None, message: str) -> tuple[str, str]:
    text = (message or '').lower()
    if status_code == 432 or 'quota' in text or 'usage limit' in text:
        return 'quota_exhausted', 'Tavily quota exhausted or usage limit reached'
    if status_code in (401, 403):
        return 'auth_failed', 'Tavily authentication failed'
    if status_code == 429:
        return 'rate_limited', 'Tavily rate limited the probe'
    if status_code and status_code >= 500:
        return 'server_error', 'Tavily server-side error'
    if 'timed out' in text or 'timeout' in text:
        return 'timeout', 'Tavily probe timed out'
    if status_code:
        return 'http_error', f'Tavily returned HTTP {status_code}'
    return 'network_error', 'Tavily probe could not reach the API'


def call_tavily(api_key: str, timeout: int) -> dict:
    payload = {
        'api_key': api_key,
        'query': 'Hermes Agent health check',
        'search_depth': 'basic',
        'max_results': 1,
        'include_answer': False,
        'include_raw_content': False,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'User-Agent': 'kensei-heartbeat/1.0'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(5120).decode('utf-8', errors='replace')
            data = json.loads(body) if body else {}
            results = data.get('results') if isinstance(data, dict) else None
            return {
                'ok': True,
                'status_code': response.status,
                'result_count': len(results) if isinstance(results, list) else 0,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(2048).decode('utf-8', errors='replace')
        kind, summary = classify_failure(exc.code, body)
        return {'ok': False, 'status_code': exc.code, 'failure_kind': kind, 'summary': summary}
    except Exception as exc:
        kind, summary = classify_failure(None, str(exc))
        return {'ok': False, 'status_code': None, 'failure_kind': kind, 'summary': summary}


def run_probe(args: argparse.Namespace) -> dict:
    state_path = Path(args.state)
    log_path = Path(args.log)
    threshold = int(args.threshold)
    checked_at = now_iso()
    key = load_env_key(Path(args.env_file))
    state = load_state(state_path)

    if not key:
        result = {
            'ok': False,
            'status': 'failure',
            'checked_at': checked_at,
            'failure_kind': 'missing_key',
            'summary': 'TAVILY_API_KEY is missing from environment and Hermes .env',
        }
    else:
        result = call_tavily(key, int(args.timeout))
        result['checked_at'] = checked_at
        result['status'] = 'ok' if result.get('ok') else 'failure'

    previous_failures = int(state.get('consecutive_failures') or 0)
    if result.get('ok'):
        consecutive = 0
        alert = False
    else:
        consecutive = previous_failures + 1
        alert = consecutive >= threshold

    new_state = {
        'consecutive_failures': consecutive,
        'threshold': threshold,
        'last_checked_at': checked_at,
        'last_status': result['status'],
        'last_failure_kind': result.get('failure_kind'),
        'last_summary': result.get('summary'),
        'last_status_code': result.get('status_code'),
        'alert_active': alert,
    }
    if alert:
        new_state['last_alert_at'] = checked_at
    elif state.get('last_alert_at') and not result.get('ok'):
        new_state['last_alert_at'] = state.get('last_alert_at')

    save_state(state_path, new_state)

    event = dict(new_state)
    event['checked_at'] = checked_at
    event['probe'] = 'tavily_api'
    event['result_count'] = result.get('result_count')
    append_log(log_path, event)

    output = dict(event)
    output['ok'] = bool(result.get('ok'))
    output['alert'] = alert
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description='Probe Tavily API health and track consecutive failures.')
    parser.add_argument('--threshold', type=int, default=3, help='Consecutive failures required before alerting.')
    parser.add_argument('--timeout', type=int, default=10, help='HTTP timeout in seconds.')
    parser.add_argument('--state', default=str(DEFAULT_STATE), help='State JSON path.')
    parser.add_argument('--log', default=str(DEFAULT_LOG), help='JSONL log path.')
    parser.add_argument('--env-file', default=str(DEFAULT_ENV), help='Hermes .env path.')
    parser.add_argument('--json', action='store_true', help='Print JSON result.')
    args = parser.parse_args()

    result = run_probe(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        if result.get('ok'):
            print(f"OK Tavily API healthy · results={result.get('result_count', 0)}")
        elif result.get('alert'):
            print(f"ALERT Tavily API {result.get('failure_kind')} · failures={result.get('consecutive_failures')}/{result.get('threshold')} · {result.get('last_summary')}")
        else:
            print(f"WARN Tavily API {result.get('failure_kind')} · failures={result.get('consecutive_failures')}/{result.get('threshold')} · {result.get('last_summary')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
