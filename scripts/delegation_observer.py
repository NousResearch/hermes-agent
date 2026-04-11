#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path.home() / '.hermes' / 'sessions'


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _iter_recent_session_files(limit: int, since_hours: int | None = None) -> list[Path]:
    files = sorted(SESSIONS_DIR.glob('session_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if since_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        files = [p for p in files if datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) >= cutoff]
    return files[:limit]


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_delegate_events(session: dict[str, Any]) -> list[dict[str, Any]]:
    messages = session.get('messages') or []
    events: list[dict[str, Any]] = []
    tool_calls_by_id: dict[str, dict[str, Any]] = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get('role') == 'assistant':
            for tc in msg.get('tool_calls') or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get('function') or {}
                if fn.get('name') == 'delegate_task':
                    tool_calls_by_id[tc.get('id', '')] = {
                        'id': tc.get('id', ''),
                        'args': _safe_json_loads(fn.get('arguments', '{}')) or {},
                    }
        elif msg.get('role') == 'tool':
            tcid = msg.get('tool_call_id', '')
            if tcid in tool_calls_by_id:
                payload = _safe_json_loads(msg.get('content', '')) or {}
                events.append({
                    'tool_call_id': tcid,
                    'args': tool_calls_by_id[tcid]['args'],
                    'payload': payload,
                })
    return events


def summarize(limit: int, since_hours: int | None = None) -> dict[str, Any]:
    files = _iter_recent_session_files(limit, since_hours=since_hours)
    summary: dict[str, Any] = {
        'sessions_scanned': 0,
        'sessions_with_delegation': 0,
        'delegate_calls': 0,
        'delegated_results': 0,
        'declined_results': 0,
        'proof_rich_results': 0,
        'legacy_results': 0,
        'worker_classes': Counter(),
        'models': Counter(),
        'decisions': Counter(),
        'per_session': [],
    }

    for path in files:
        session = _load_json(path)
        if not session:
            continue
        summary['sessions_scanned'] += 1
        events = _extract_delegate_events(session)
        if not events:
            continue
        summary['sessions_with_delegation'] += 1

        per = {
            'session_file': str(path),
            'session_id': session.get('session_id'),
            'delegate_calls': len(events),
            'events': [],
        }

        for event in events:
            summary['delegate_calls'] += 1
            results = event['payload'].get('results') or []
            if not isinstance(results, list):
                results = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                decision = result.get('delegation_decision') or 'delegated'
                worker_class = result.get('worker_class') or 'unknown'
                lane = result.get('resolved_lane') or {}
                model = lane.get('model') or result.get('model') or 'unknown'
                proof_rich = any(k in result for k in ('worker_class', 'resolved_lane', 'delegation_score', 'delegation_reasons', 'delegation_decision'))
                if proof_rich:
                    summary['proof_rich_results'] += 1
                else:
                    summary['legacy_results'] += 1

                summary['decisions'][decision] += 1
                summary['worker_classes'][worker_class] += 1
                summary['models'][model] += 1

                if decision == 'declined_low_leverage':
                    summary['declined_results'] += 1
                else:
                    summary['delegated_results'] += 1

                args = event.get('args') or {}
                event_goal = args.get('goal') or ''
                task_index = result.get('task_index')
                if not event_goal and args.get('tasks') and isinstance(task_index, int):
                    try:
                        event_goal = (args['tasks'][task_index] or {}).get('goal') or ''
                    except (IndexError, TypeError):
                        event_goal = ''

                per['events'].append({
                    'decision': decision,
                    'worker_class': worker_class,
                    'model': model,
                    'score': result.get('delegation_score'),
                    'reasons': result.get('delegation_reasons') or [],
                    'goal': event_goal,
                })
        summary['per_session'].append(per)

    summary['worker_classes'] = dict(summary['worker_classes'])
    summary['models'] = dict(summary['models'])
    summary['decisions'] = dict(summary['decisions'])
    return summary


def render_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('Hermes Delegation Observer')
    lines.append(f"sessions_scanned: {summary['sessions_scanned']}")
    lines.append(f"sessions_with_delegation: {summary['sessions_with_delegation']}")
    lines.append(f"delegate_calls: {summary['delegate_calls']}")
    lines.append(f"delegated_results: {summary['delegated_results']}")
    lines.append(f"declined_results: {summary['declined_results']}")
    lines.append(f"proof_rich_results: {summary['proof_rich_results']}")
    lines.append(f"legacy_results: {summary['legacy_results']}")
    lines.append('')
    lines.append('decisions:')
    for k, v in sorted(summary['decisions'].items()):
        lines.append(f'  {k}: {v}')
    lines.append('worker_classes:')
    for k, v in sorted(summary['worker_classes'].items()):
        lines.append(f'  {k}: {v}')
    lines.append('models:')
    for k, v in sorted(summary['models'].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f'  {k}: {v}')
    lines.append('')
    lines.append('recent delegation events:')
    for session in summary['per_session'][:10]:
        lines.append(f"- {session['session_id']} ({session['delegate_calls']} calls)")
        for ev in session['events'][:5]:
            goal = (ev.get('goal') or '')[:80]
            lines.append(
                f"    - {ev['decision']} | {ev['worker_class']} | {ev['model']} | score={ev.get('score')} | goal={goal}"
            )
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description='Observe Hermes delegation usage from recent sessions.')
    parser.add_argument('--limit', type=int, default=20, help='Number of recent session files to scan')
    parser.add_argument('--since-hours', type=int, default=None, help='Only scan sessions modified within the last N hours')
    parser.add_argument('--json', action='store_true', help='Emit JSON instead of text')
    args = parser.parse_args()

    summary = summarize(args.limit, since_hours=args.since_hours)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(render_text(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
