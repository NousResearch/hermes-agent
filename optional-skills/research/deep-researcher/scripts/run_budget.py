"""Persist the effective query allowance and receipts before native retrieval."""
from __future__ import annotations

from pathlib import Path

PLANS = {'direct': {'quick': 4, 'standard': 8, 'deep': 12},
         'batched': {'quick': 6, 'standard': 16, 'deep': 40}}
GAP_ALLOWANCE = 2


def usage(data: dict) -> dict:
    searches = data.get('searches', [])
    regular = sum(not row['gap'] for row in searches)
    gaps = len(searches) - regular
    return {'regular': regular, 'gap': gaps,
            'remaining': max(0, data['budgets']['search_calls'] - regular),
            'gap_remaining': max(0, data['budgets'].get('gap_search_calls', GAP_ALLOWANCE) - gaps),
            'pending': [row['id'] for row in searches if row['outcome'] == 'reserved']}


def _active(root: Path) -> dict:
    from research_run import load
    data = load(root)
    if data['status'] != 'active':
        raise ValueError('Search accounting requires an active run')
    return data


def reserve(root: Path, request_id: str, query: str, question: str, gap: bool = False) -> dict:
    from research_run import now, save
    if not all(isinstance(x, str) and x.strip() for x in (request_id, query, question)):
        raise ValueError('Search id, query and question must be nonempty strings')
    if not isinstance(gap, bool):
        raise ValueError('gap must be a boolean')
    data = _active(root)
    searches = data.setdefault('searches', [])
    existing = next((row for row in searches if row['id'] == request_id), None)
    identity = {'id': request_id, 'query': query, 'question': question, 'gap': gap}
    if existing:
        if any(existing[key] != value for key, value in identity.items()):
            raise ValueError('Search id already belongs to a different request')
        return existing
    remaining = usage(data)['gap_remaining' if gap else 'remaining']
    if not remaining:
        raise ValueError('Search budget exhausted; record an explicit budget change before continuing')
    record = {**identity, 'outcome': 'reserved', 'reserved_at': now()}
    searches.append(record)
    save(root, data)
    return record


def finish(root: Path, request_id: str, outcome: str, urls: list[str], error: str = '',
           *, candidates: list | None = None, new_count: int | None = None) -> dict:
    from evidence_ledger import canonical_url
    from research_run import now, save
    if outcome not in {'success', 'empty', 'failed'}:
        raise ValueError('Search outcome must be success, empty or failed')
    data = _active(root)
    record = next((row for row in data.get('searches', []) if row['id'] == request_id), None)
    if record is None:
        raise ValueError('Reserve the search before recording its result')
    result = {'outcome': outcome, 'urls': list(dict.fromkeys(canonical_url(url) for url in urls)), 'error': error}
    if candidates is not None:
        result['candidates'] = candidates
    if new_count is not None:
        result['new_count'] = new_count
    if outcome == 'failed' and not error.strip():
        raise ValueError('A failed search needs an error description')
    if outcome != 'success' and urls:
        raise ValueError('Only a successful search can record candidate URLs')
    if record['outcome'] != 'reserved':
        if any(record.get(key) != value for key, value in result.items()):
            raise ValueError('Search result already recorded with different contents')
        return record
    record.update(result, finished_at=now())
    save(root, data)
    return record


def change(root: Path, allowance: int, reason: str) -> dict:
    from research_run import now, save
    if not isinstance(allowance, int) or isinstance(allowance, bool) or allowance <= 0 or not reason.strip():
        raise ValueError('A budget change needs a positive allowance and a reason')
    data = _active(root)
    if allowance < usage(data)['regular']:
        raise ValueError('Cannot lower the budget below queries already reserved')
    previous = data['budgets']['search_calls']
    data['budgets']['search_calls'] = allowance
    data.setdefault('budget_changes', []).append({'from': previous, 'to': allowance, 'reason': reason, 'at': now(),
                                                'search_index': len(data.get('searches', []))})
    save(root, data)
    return usage(data)


def command(args) -> None:
    import json
    root = Path(args.run).expanduser()
    if args.budget_action == 'reserve':
        result = reserve(root, args.id, args.query, args.question, args.gap)
    elif args.budget_action == 'record':
        result = finish(root, args.id, args.outcome, args.new_url, args.error)
    else:
        result = change(root, args.search_budget, args.reason)
    print(json.dumps(result, ensure_ascii=False))


def add_commands(commands) -> None:
    command_parser = commands.add_parser('budget', help='Record an explicit change to the effective query allowance')
    command_parser.add_argument('run')
    command_parser.add_argument('--search-budget', type=int, required=True)
    command_parser.add_argument('--reason', required=True)
    command_parser.set_defaults(func=command, budget_action='change')
    command_parser = commands.add_parser('reserve-search', help='Count a query before calling the native search tool')
    command_parser.add_argument('run')
    command_parser.add_argument('--id', required=True)
    command_parser.add_argument('--query', required=True)
    command_parser.add_argument('--question', required=True)
    command_parser.add_argument('--gap', action='store_true')
    command_parser.set_defaults(func=command, budget_action='reserve')
    command_parser = commands.add_parser('record-search', help='Record a reserved query result without refunding its allowance')
    command_parser.add_argument('run')
    command_parser.add_argument('--id', required=True)
    command_parser.add_argument('--outcome', required=True, choices=('success', 'empty', 'failed'))
    command_parser.add_argument('--new-url', action='append', default=[])
    command_parser.add_argument('--error', default='')
    command_parser.set_defaults(func=command, budget_action='record')
