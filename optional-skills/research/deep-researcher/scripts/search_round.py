"""Bounded discovery through a caller-supplied native Hermes web_search function."""
from __future__ import annotations

from pathlib import Path


def _candidates(response: dict) -> list[dict]:
    from evidence_ledger import canonical_url
    if not isinstance(response, dict) or response.get('success') is False:
        raise ValueError(str(response.get('error', 'Native search failed')) if isinstance(response, dict) else 'Invalid native search response')
    data = response.get('data')
    rows = data.get('web') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError('Native search response is missing data.web; not an empty search')
    found = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('url'), str):
            continue
        try:
            canonical = canonical_url(row['url'])
        except ValueError:
            continue
        found.setdefault(canonical, {'url': row['url'], 'canonical_url': canonical,
            'title': str(row.get('title') or '')[:120], 'description': str(row.get('description') or '')[:200]})
    return list(found.values())


def shortlist(data: dict) -> list[dict]:
    found = {}
    for record in data.get('searches', []):
        for candidate in record.get('candidates', []):
            url = candidate['canonical_url']
            value = found.setdefault(url, {**candidate, 'questions': []})
            if record['question'] not in value['questions']:
                value['questions'].append(record['question'])
    return sorted(found.values(), key=lambda value: -len(value['questions']))


def stop_reason(data: dict) -> str:
    changes = data.get('budget_changes', [])
    start = changes[-1].get('search_index', 0) if changes else 0
    records = [row for row in data.get('searches', [])[start:] if row['outcome'] != 'reserved']
    if sum(row['outcome'] == 'empty' for row in records) >= 4:
        return 'Four empty responses; synthesize or report the gap'
    if len(records) >= 2 and all(row['outcome'] == 'failed' for row in records[-2:]):
        return 'Two failed searches; resolve the backend failure before more queries'
    if len(records) >= 2 and all(row['outcome'] != 'failed' and row.get('new_count') == 0 for row in records[-2:]):
        return 'Two searches added no new canonical URLs; reassess evidence gaps'
    return ''


def _one(root: Path, query: dict, search, limit: int) -> dict:
    from research_run import load
    from run_budget import finish, reserve
    allowed = {'id', 'query', 'question', 'gap'}
    if not isinstance(query, dict) or set(query) - allowed or not {'id', 'query', 'question'} <= query.keys():
        raise ValueError('Each query needs id, query, question and optional gap')
    if not isinstance(query.get('gap', False), bool):
        raise ValueError('gap must be a boolean')
    before = load(root)
    prior = next((row for row in before.get('searches', []) if row['id'] == query['id']), None)
    record = reserve(root, query['id'], query['query'], query['question'], query.get('gap', False))
    if prior:
        if record['outcome'] == 'reserved':
            raise ValueError('Pending search has an uncertain outcome; record it before retrying with a new id')
        return record
    seen = {row['canonical_url'] for row in shortlist(before)}
    try:
        candidates = _candidates(search(query['query'], limit=limit))
    except (ValueError, OSError, RuntimeError) as exc:
        return finish(root, query['id'], 'failed', [], str(exc), candidates=[])
    outcome = 'success' if candidates else 'empty'
    new_count = sum(row['canonical_url'] not in seen for row in candidates)
    return finish(root, query['id'], outcome, [row['url'] for row in candidates],
                  candidates=candidates, new_count=new_count)


def discover(run: str | Path, queries: list[dict], search, limit: int = 8, max_rows: int = 25) -> dict:
    """Invoke only the supplied native tool, counting each query before calling it.

    Backend errors remain failures, not empty results. Completed request IDs replay
    their saved candidates without another retrieval. Never retry a pending request.
    """
    from research_run import load
    from run_budget import usage
    root = Path(run)
    if not isinstance(queries, list) or limit <= 0 or max_rows <= 0:
        raise ValueError('Queries must be an array and output limits must be positive')
    completed = []
    reason = stop_reason(load(root))
    for query in queries:
        if reason:
            break
        record = _one(root, query, search, limit)
        completed.append({'id': record['id'], 'outcome': record['outcome'],
                          'new_count': record.get('new_count', 0), 'error': record.get('error', '')})
        reason = stop_reason(load(root))
    data = load(root)
    rows = shortlist(data)
    return {'queries': completed, 'stop_reason': reason, 'usage': usage(data),
            'candidate_count': len(rows), 'shortlist': rows[:max_rows],
            'omitted_rows': max(0, len(rows) - max_rows),
            'note': 'All candidates are retained in the run manifest. URL novelty is not a credibility judgment.'}
