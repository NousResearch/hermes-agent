"""Compact review views and explicit, atomic updates to a hash-bound review packet."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path: Path, data: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix='.review-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _selected(value: str) -> set[str]:
    return {item.strip() for item in value.split(',') if item.strip()}


def _brief(item: dict) -> dict:
    result = {key: item[key] for key in ('id', 'kind', 'status', 'line', 'evidence_ids') if key in item}
    for key in ('text', 'context'):
        if key in item:
            result[key] = item[key][:600]
            if len(item[key]) > 600:
                result[key + '_clipped'] = True
    if 'claim_status' in item:
        result['claim_status'] = item['claim_status']
        if any(c.isdigit() for c in item['text']):
            result['numeric_review'] = 'Check period, population/geography, units, denominator and comparison; separate incompatible series.'
    return result


def compact(root: Path, ids: set[str], evidence_ids: set[str], max_chars: int) -> dict:
    from review_validation import inspect_review
    if max_chars < 2000:
        raise ValueError('max-chars must be at least 2000 so hashes, omissions and guidance remain visible')
    expected, review, errors = inspect_review(root)
    items = review['claims'] + review['assertions']
    by_id = {item['id']: item for item in items}
    if ids - by_id.keys() or evidence_ids - expected['evidence'].keys():
        raise ValueError('Unknown item or evidence ID; show RUN lists the current IDs')
    chosen = [item for item in items if item['id'] in ids or (not ids and item.get('claim_status') is not None)]
    passages = set(evidence_ids)
    if ids:
        passages.update(key for item in chosen for key in item.get('evidence_ids', []))
    result = {'report_sha256': expected['report_sha256'], 'ledger_sha256': expected['ledger_sha256'],
              'checks': review['checks'], 'items': [_brief(item) for item in chosen],
              'passages': [], 'warnings': [{k: item[k] for k in ('id', 'kind', 'text', 'line')} for item in review['assertions']],
              'errors': errors, 'omitted': {'items': [], 'passages': [], 'warnings': [], 'errors': 0},
              'next': 'Run template RUN; edit its unapproved review-decisions.json; apply RUN RUN/review-decisions.json. Correct report/ledger with native patch, then prepare RUN --replace. Never approve automatically.'}
    for key in sorted(passages):
        evidence = expected['evidence'][key]
        cap = max(800, (max_chars - 1500) // max(1, len(passages)))
        result['passages'].append({**evidence, 'excerpt': evidence['excerpt'][:cap],
                                   'excerpt_clipped': len(evidence['excerpt']) > cap})
    return _fit(result, max_chars)


def _fit(result: dict, max_chars: int) -> dict:
    while len(json.dumps(result, ensure_ascii=False)) > max_chars:
        if result['errors']:
            result['errors'].pop()
            result['omitted']['errors'] += 1
        elif result['warnings']:
            result['omitted']['warnings'].insert(0, result['warnings'].pop()['id'])
        elif result['items']:
            result['omitted']['items'].insert(0, result['items'].pop()['id'])
        elif result['passages']:
            result['omitted']['passages'].insert(0, result['passages'].pop()['id'])
        else:
            raise ValueError('Output metadata exceeds max-chars; request a larger view')
    return result


def show(args) -> dict:
    return compact(args.run, _selected(args.ids), _selected(args.evidence), args.max_chars)


def prepare(args) -> dict:
    from research_run import safe_path
    from semantic_review import packet
    path = safe_path(args.run, 'semantic-review.json')
    if path.exists() and not args.replace:
        raise ValueError('Review exists; use --replace after edits to prepare a fresh pending packet')
    content, _ = packet(args.run)
    atomic_json(path, content)
    return compact(args.run, set(), set(), 12000)



def template(args) -> dict:
    from research_run import load, safe_path
    from review_validation import inspect_review
    expected, review, _ = inspect_review(args.run)
    protected = set(load(args.run)['artifacts']) | {'report.md', 'evidence.json', 'evidence-brief.md', 'semantic-review.json'}
    if args.output in protected:
        raise ValueError('Decision templates cannot overwrite research artifacts')
    path = safe_path(args.run, args.output)
    if path.exists() and not args.replace:
        raise ValueError('Decision template exists; edit it or use --replace explicitly')
    result = {key: expected[key] for key in ('report_sha256', 'ledger_sha256')}
    result.update(reviewer=review['reviewer'], checks=copy.deepcopy(review['checks']))
    atomic_json(path, result)
    return {'path': str(path), 'automatically_approved': 0,
            'next': 'Write the four critique notes with native patch; apply RUN FILE, check RUN, then checkpoint complete.'}

def apply_decisions(args) -> dict:
    from research_run import safe_path
    from review_validation import inspect_review, review_errors
    expected, current, _ = inspect_review(args.run)
    payload = json.loads(args.payload.read_text(encoding='utf-8'))
    allowed = {'report_sha256', 'ledger_sha256', 'reviewer', 'checks'}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError('Decision payload allows only hashes, reviewer and checks. Use template RUN to generate the exact format')
    if any(payload.get(key) != expected[key] for key in ('report_sha256', 'ledger_sha256')):
        raise ValueError('Decision hashes are stale or missing; show RUN for current hashes, then review the current material')
    staged = copy.deepcopy(current)
    if 'reviewer' in payload:
        staged['reviewer'] = payload['reviewer']
    checks = payload.get('checks', {})
    if not isinstance(checks, dict) or set(checks) - staged['checks'].keys():
        raise ValueError('Unknown review checklist entries')
    for key, decision in checks.items():
        if isinstance(decision, dict) and decision.get('status') == 'pending' and not decision.get('note'):
            continue
        staged['checks'][key] = decision
    errors = review_errors(expected, staged, allow_pending=True)
    if errors:
        raise ValueError('No decisions saved:\n' + '\n'.join(errors))
    atomic_json(safe_path(args.run, 'semantic-review.json'), staged)
    remaining = review_errors(expected, staged)
    return {'saved': True, 'ready': not remaining, 'remaining': remaining,
            'next': 'Resolve every remaining issue, run check RUN, then checkpoint RUN --status complete.'}
