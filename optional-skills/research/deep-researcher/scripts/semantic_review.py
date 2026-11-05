#!/usr/bin/env python3
"""Prepare and validate an explicit claim/evidence review; never infer semantic truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from review_signals import assertions

KIND = 'deep-research-semantic-review-v5'
CHECKS = ('finality', 'independence', 'entities', 'numbers')


def evidence_index(data: dict) -> dict:
    result = {}
    for source in data['sources']:
        for item in source.get('evidence', []):
            result[item['id']] = {**item, 'source_id': source['id'], 'url': source['url'],
                'title': source['title'], 'independence_group': source.get('independence_group'),
                'independence_assessed': source.get('independence_assessed', False),
                'independence_note': source.get('independence_note', '')}
    return result


def packet(root: Path) -> tuple[dict, dict]:
    from evidence_ledger import read_ledger, validate
    from research_run import digest, safe_path
    ledger_path = safe_path(root, 'evidence.json')
    report_path = safe_path(root, 'report.md')
    data = read_ledger(ledger_path)
    errors, _ = validate(data)
    if errors:
        raise ValueError('Fix the ledger before semantic review: ' + '; '.join(errors))
    report = report_path.read_text(encoding='utf-8')
    claims = [{'id': c['id'], 'text': c['text'], 'evidence_ids': c.get('evidence_ids', []),
               'claim_status': c['status'], 'importance': c.get('importance', 'major'),
               'caveat': c.get('caveat', ''), 'reasoning': c.get('reasoning', '')} for c in data['claims']]
    return {'kind': KIND, 'report_sha256': digest(report_path), 'ledger_sha256': digest(ledger_path),
            'reviewer': '', 'report': report, 'evidence': evidence_index(data), 'claims': claims,
            'assertions': [{**{k: v for k, v in item.items() if k not in {'status', 'note', 'qualification', 'evidence_ids'}},
                            'status': 'warning'} for item in assertions(report, data)],
            'checks': {key: {'status': 'pending', 'note': ''} for key in CHECKS}}, data


def check(root: Path) -> dict:
    from review_validation import inspect_review
    _, review, errors = inspect_review(root)
    if errors:
        raise ValueError('Review is incomplete:\n' + '\n'.join(errors))
    return {'reviewer': review['reviewer'], 'claims_in_dossier': len(review['claims']),
            'heuristic_warnings': len(review['assertions']),
            'note': 'Review records are attestations; structural validation cannot establish semantic truth.'}


def main() -> None:
    from research_run import load
    from review_commands import apply_decisions, prepare, show, template
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    command = commands.add_parser('prepare', help='Bind current artifacts and show compact pending review items')
    command.add_argument('run', type=Path)
    command.add_argument('--replace', action='store_true')
    command = commands.add_parser('show', help='Show bounded pending items, or selected items/passages by ID')
    command.add_argument('run', type=Path)
    command.add_argument('--ids', default='', help='Comma-separated claim or assertion IDs')
    command.add_argument('--evidence', default='', help='Comma-separated passage IDs')
    command.add_argument('--max-chars', type=int, default=12000)
    command = commands.add_parser('template', help='Write exact decision IDs and fields without approving anything')
    command.add_argument('run', type=Path)
    command.add_argument('--output', default='review-decisions.json', help='Run-relative output path')
    command.add_argument('--replace', action='store_true')
    command = commands.add_parser('apply', help='Atomically apply explicit JSON decisions; never approve automatically')
    command.add_argument('run', type=Path)
    command.add_argument('payload', type=Path)
    command = commands.add_parser('check', help='Report all unresolved review issues together')
    command.add_argument('run', type=Path)
    args = parser.parse_args()
    try:
        if load(args.run)['status'] == 'archived':
            raise ValueError('Restore the run before reviewing it')
        handlers = {'prepare': prepare, 'show': show, 'template': template, 'apply': apply_decisions,
                    'check': lambda args: check(args.run)}
        print(json.dumps(handlers[args.action](args), ensure_ascii=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
