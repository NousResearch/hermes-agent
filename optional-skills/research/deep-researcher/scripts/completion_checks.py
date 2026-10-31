"""Minimum completion gates for a research dossier; no live network or model calls."""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path


def _provenance(data: dict, cited: set[int]) -> list[str]:
    errors = []
    evidence = {e['id']: s['id'] for s in data['sources'] for e in s.get('evidence', [])}
    located = set()
    for source in data['sources']:
        if source['id'] in cited and not source.get('summary', '').strip():
            errors.append(f"Cited source [{source['id']}] needs a compact summary")
    for claim in data['claims']:
        references = set(claim.get('source_ids', []) + claim.get('counter_source_ids', []))
        passages = {evidence[key] for key in claim.get('evidence_ids', []) if key in evidence}
        located.update(passages)
        if references - passages:
            errors.append(f"Claim {claim['id']} needs precise passages for sources {sorted(references - passages)}")
    if cited - located:
        errors.append(f'Cited sources have no linked claims/passages: {sorted(cited - located)}')
    return errors


def _budget(data: dict) -> None:
    from run_budget import GAP_ALLOWANCE, usage
    consumed = usage(data)
    if consumed['pending']:
        raise ValueError('Resolve reserved search outcomes before completion: ' + ', '.join(consumed['pending']))
    if consumed['regular'] > data['budgets']['search_calls'] or consumed['gap'] > data['budgets'].get('gap_search_calls', GAP_ALLOWANCE):
        raise ValueError('Recorded searches exceed the effective allowance; record an explicit budget change')


def require_completion(root: Path, manifest: dict) -> dict:
    from audit_report import analytical_body, bibliography_entries, citations_in, run, split_document
    from evidence_ledger import canonical_url, read_ledger, validate
    from research_run import digest, now, safe_path
    report = safe_path(root, manifest['report'])
    ledger_path = safe_path(root, 'evidence.json')
    brief = safe_path(root, 'evidence-brief.md')
    for path in (report, ledger_path, brief):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Completion requires nonempty {path.name}')
    data = read_ledger(ledger_path)
    errors, warnings = validate(data)
    if not data.get('sources') or not data.get('claims'):
        errors.append('Completion requires sources and atomic claims')
    if errors:
        raise ValueError('Ledger is not ready: ' + '; '.join(errors))
    body, lines = split_document(report.read_text(encoding='utf-8'))
    entries, parse_errors = bibliography_entries(lines)
    by_url = {canonical_url(source['url']): source['id'] for source in data['sources']}
    cited = set()
    for number in citations_in(analytical_body(body), set(entries)):
        if number not in entries or canonical_url(entries[number]) not in by_url:
            raise ValueError(f'Citation [{number}] must identify a ledger source')
        cited.add(by_url[canonical_url(entries[number])])
    errors = parse_errors + _provenance(data, cited)
    if errors:
        raise ValueError('Provenance is incomplete: ' + '; '.join(errors))
    audit = io.StringIO()
    with contextlib.redirect_stdout(audit):
        result = run(argparse.Namespace(report=str(report), ledger=str(ledger_path), no_live=True,
                                        workers=1, timeout=1, max_heuristic_warnings=12))
    if result:
        raise ValueError('Report audit failed: ' + audit.getvalue())
    _budget(manifest)
    return {'checked_at': now(), 'report_sha256': digest(report), 'ledger_sha256': digest(ledger_path),
            'ledger_warnings': warnings,
            'report_warnings': [line for line in audit.getvalue().splitlines() if line.startswith('WARN:')],
            'note': 'Structural gates passed; this is not proof of source truth or research quality.'}
