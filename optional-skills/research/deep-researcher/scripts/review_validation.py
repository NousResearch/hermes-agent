"""A fresh, explicit four-part critique; no machine claim of semantic truth."""
from __future__ import annotations

import json
from pathlib import Path


def review_errors(expected: dict, review, allow_pending: bool = False) -> list[str]:
    from semantic_review import CHECKS
    if not isinstance(review, dict):
        raise ValueError('Semantic review must be a JSON object')
    for key in ('kind', 'report_sha256', 'ledger_sha256', 'report', 'evidence', 'claims', 'assertions'):
        if review.get(key) != expected[key]:
            raise ValueError('Semantic review is stale or source material changed; prepare RUN --replace, then review again')
    errors = []
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        errors.append('reviewer: name the person or model performing the critique')
    checks = review.get('checks')
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        return errors + ['checks: review finality, independence, entities and numbers']
    for key, value in checks.items():
        if not isinstance(value, dict) or set(value) != {'status', 'note'}:
            errors.append(f'{key}: use only status and note from the template')
            continue
        status, note = value['status'], value['note']
        if status not in {'pending', 'needs-correction', 'checked'} or not isinstance(note, str):
            errors.append(f'{key}: status must be pending, needs-correction or checked; note must be text')
        elif status != 'pending' and not note.strip():
            errors.append(f'{key}: explain the review finding')
        elif not allow_pending and status != 'checked':
            errors.append(f'{key}: {status}: {note or "inspect the material claims and record the critique"}')
    return errors


def inspect_review(root: Path) -> tuple[dict, dict, list[str]]:
    from research_run import safe_path
    from semantic_review import packet
    expected, _ = packet(root)
    review = json.loads(safe_path(root, 'semantic-review.json').read_text(encoding='utf-8'))
    return expected, review, review_errors(expected, review)
