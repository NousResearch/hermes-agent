"""CLI contracts for bounded research, atomic evidence writes and completion review."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'optional-skills/research/deep-researcher/scripts'


def cli(script, *args, ok=True):
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                            check=False, capture_output=True, text=True, encoding='utf-8')
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def init(tmp_path, *args):
    root = tmp_path / 'research with spaces'
    cli('research_run.py', 'init', root, '--question', 'Growth?', '--scope', 'Synthetic', *args)
    cli('evidence_ledger.py', 'init', root / 'evidence.json', '--topic', 'Fixture', '--question', 'Q1: Growth?')
    return root


def batch():
    return {'batch_id': 'round-1', 'sources': [{
        'key': 'dataset', 'url': 'https://example.org/data', 'title': 'Dataset',
        'source_type': 'primary', 'questions': ['Q1'], 'summary': 'Reports growth of 10%.',
        'evidence': [{'excerpt': 'The dataset reports growth of 10%.', 'locator': 'Results, paragraph 1'}],
    }], 'claims': [{
        'key': 'growth', 'text': 'The dataset reports growth of 10%.', 'question': 'Q1',
        'status': 'single-source', 'sources': ['dataset'], 'evidence': ['dataset:1'],
    }]}


def add_batch(root, payload=None, ok=True):
    path = root / 'batch.json'
    path.write_text(json.dumps(payload or batch()), encoding='utf-8')
    return cli('evidence_ledger.py', 'add-batch', root / 'evidence.json', path, ok=ok)


def draft(root, text='The dataset reports growth of 10%. [1]'):
    report = text + '\n\nSingle-origin evidence.\n\n## References\n1. [Dataset](https://example.org/data)\n'
    (root / 'report.md').write_text(report, encoding='utf-8')
    brief = cli('evidence_ledger.py', 'brief', root / 'evidence.json').stdout
    (root / 'evidence-brief.md').write_text(brief, encoding='utf-8')


def approve(root):
    cli('semantic_review.py', 'prepare', root, '--replace')
    path = root / 'semantic-review.json'
    data = json.loads(path.read_text())
    data['reviewer'] = 'Synthetic fixture reviewer'
    for check in data['checks'].values():
        check.update(status='checked', note='Reviewed the synthetic report and retained evidence.')
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_effective_budget_is_persisted_and_exhaustion_does_not_mutate(tmp_path):
    root = init(tmp_path, '--execution', 'batched', '--search-budget', '1')
    path = root / 'run_manifest.json'
    assert json.loads(path.read_text())['budgets']['search_calls'] == 1
    cli('research_run.py', 'reserve-search', root, '--id', 'q1', '--query', 'growth', '--question', 'Q1')
    cli('research_run.py', 'record-search', root, '--id', 'q1', '--outcome', 'empty')
    before = path.read_bytes()
    cli('research_run.py', 'reserve-search', root, '--id', 'q2', '--query', 'other', '--question', 'Q1', ok=False)
    assert path.read_bytes() == before
    cli('research_run.py', 'budget', root, '--search-budget', '16', '--reason', 'Explicit scope expansion')
    assert json.loads(path.read_text())['budgets']['search_calls'] == 16
    cli('research_run.py', 'reserve-search', root, '--id', 'q2', '--query', 'other', '--question', 'Q1')
    assert len(json.loads(path.read_text())['searches']) == 2


def test_batch_retry_is_idempotent_and_conflicting_retry_is_rejected(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    ledger = root / 'evidence.json'
    before = ledger.read_bytes()
    add_batch(root)
    assert ledger.read_bytes() == before
    payload = batch()
    payload['claims'][0]['text'] = 'A changed conclusion.'
    add_batch(root, payload, ok=False)
    assert ledger.read_bytes() == before
    data = json.loads(before)
    assert data['claims'][0]['source_ids'] == [data['sources'][0]['id']]
    assert data['claims'][0]['evidence_ids'] == [data['sources'][0]['evidence'][0]['id']]


def test_invalid_late_claim_rolls_back_whole_batch(tmp_path):
    root = init(tmp_path)
    ledger = root / 'evidence.json'
    before = ledger.read_bytes()
    payload = batch()
    payload['claims'][0]['evidence'] = ['missing:1']
    add_batch(root, payload, ok=False)
    assert ledger.read_bytes() == before


def test_report_alone_cannot_complete_or_change_manifest(tmp_path):
    root = init(tmp_path)
    (root / 'report.md').write_text('# Plausible report\nThe result is final.\n')
    path = root / 'run_manifest.json'
    before = path.read_bytes()
    cli('research_run.py', 'checkpoint', root, '--status', 'complete', ok=False)
    assert path.read_bytes() == before


def test_passage_keeps_late_match_in_a_long_line_and_bounds_output(tmp_path):
    path = tmp_path / 'flattened PDF.txt'
    text = 'start ' + 'irrelevant ' * 3000 + 'decisive 10% finding' + ' trailing' * 3000
    path.write_text(text, encoding='utf-8')
    result = cli('passages.py', path, '--term', 'decisive', '--max-chars', '180')
    data = json.loads(result.stdout)
    assert data['status'] == 'matched'
    assert sum(len(p['text']) for p in data['passages']) <= 180
    passage = data['passages'][0]
    assert 'decisive 10%' in passage['text']
    assert passage['text'] == text[passage['start']:passage['end']]
    assert passage['line_start'] == passage['line_end'] == 1


def test_completion_needs_no_optional_review_packet(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    draft(root)
    cli('research_run.py', 'checkpoint', root, '--status', 'complete')
    data = json.loads((root / 'run_manifest.json').read_text())
    assert data['status'] == 'complete'
    assert 'semantic-review.json' not in data['artifacts']
    assert 'review_sha256' not in data['completion_checks']
    cli('research_run.py', 'archive', root)


def test_optional_review_does_not_replace_current_provenance_checks(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    draft(root)
    approve(root)
    draft(root, 'Unsupported source citation. [2]')
    cli('semantic_review.py', 'check', root, ok=False)
    before = (root / 'run_manifest.json').read_bytes()
    cli('research_run.py', 'checkpoint', root, '--status', 'complete', ok=False)
    assert (root / 'run_manifest.json').read_bytes() == before


def test_discovery_counts_native_queries_and_retains_all_question_associations(tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(SCRIPTS))
    discovery = importlib.import_module('search_round')
    root = init(tmp_path, '--execution', 'batched', '--search-budget', '3')
    calls = []

    def search(query, limit):
        calls.append(query)
        return {'success': True, 'data': {'web': [
            {'url': 'https://example.org/data', 'title': None, 'description': None},
            {'url': 'https://example.org/' + query, 'title': query},
        ]}}

    queries = [{'id': 'q1', 'question': 'Q1', 'query': 'first'},
               {'id': 'q2', 'question': 'Q2', 'query': 'second'}]
    result = discovery.discover(root, queries, search, max_rows=1)
    assert calls == ['first', 'second']
    assert result['shortlist'][0]['questions'] == ['Q1', 'Q2']
    assert result['omitted_rows'] == 2
    replay = discovery.discover(root, queries, search)
    assert calls == ['first', 'second']
    assert replay['usage']['regular'] == 2
    assert len(replay['shortlist']) == 3


def test_discovery_stops_on_failures_without_reporting_them_as_empty(tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(SCRIPTS))
    discovery = importlib.import_module('search_round')
    root = init(tmp_path)
    calls = []

    def search(query, limit):
        calls.append(query)
        return {'success': False, 'error': 'Backend is unavailable'}

    queries = [{'id': str(i), 'question': 'Q1', 'query': 'query ' + str(i)} for i in range(4)]
    result = discovery.discover(root, queries, search)
    assert len(calls) == 2
    assert [q['outcome'] for q in result['queries']] == ['failed', 'failed']
    assert result['usage']['regular'] == 2
    assert 'failed' in result['stop_reason']


@pytest.mark.parametrize('missing', ['summary', 'evidence_ids'])
def test_completion_rejects_missing_provenance_without_saving_status(tmp_path, missing):
    root = init(tmp_path)
    add_batch(root)
    draft(root)
    path = root / 'evidence.json'
    data = json.loads(path.read_text())
    if missing == 'summary':
        data['sources'][0]['summary'] = ''
    else:
        data['claims'][0]['evidence_ids'] = []
    path.write_text(json.dumps(data), encoding='utf-8')
    manifest = root / 'run_manifest.json'
    before = manifest.read_bytes()
    cli('research_run.py', 'checkpoint', root, '--status', 'complete', ok=False)
    assert manifest.read_bytes() == before


def test_no_match_and_overlapping_windows_preserve_literal_text(tmp_path):
    path = tmp_path / 'cache.txt'
    text = 'α\r\n' + 'x' * 40 + 'firstword secondword' + 'z' * 80
    path.write_text(text, encoding='utf-8', newline='')
    data = json.loads(cli('passages.py', path, '--term', 'firstword', '--term', 'secondword',
                          '--context', '3', '--max-chars', '80').stdout)
    for term in ['firstword', 'secondword']:
        assert any(term in row['text'] for row in data['passages'])
    for row in data['passages']:
        assert row['text'] == text[row['start']:row['end']]
        assert row['line_start'] == row['line_end'] == 2
    empty = json.loads(cli('passages.py', path, '--term', 'absent').stdout)
    assert empty['status'] == 'no-match' and not empty['passages']
    assert 'not proof' in empty['note']


def test_review_cannot_drop_assertions_or_edit_retained_evidence(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    draft(root, 'The final status is unverified. [1]')
    approve(root)
    path = root / 'semantic-review.json'
    original = json.loads(path.read_text())
    changed = json.loads(json.dumps(original))
    changed['assertions'] = []
    path.write_text(json.dumps(changed), encoding='utf-8')
    cli('semantic_review.py', 'check', root, ok=False)
    changed = json.loads(json.dumps(original))
    changed['evidence']['S1E1']['excerpt'] = 'This is definitely final.'
    path.write_text(json.dumps(changed), encoding='utf-8')
    cli('semantic_review.py', 'check', root, ok=False)


def test_budget_change_resumes_fixed_backend_without_refunding_queries(tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(SCRIPTS))
    discovery = importlib.import_module('search_round')
    root = init(tmp_path)
    queries = [{'id': str(i), 'question': 'Q1', 'query': 'growth'} for i in range(3)]
    result = discovery.discover(root, queries, lambda *a, **kw: {'success': True, 'data': None})
    assert result['usage']['regular'] == 2
    assert all(row['outcome'] == 'failed' for row in result['queries'])
    cli('research_run.py', 'budget', root, '--search-budget', '8', '--reason', 'Backend repaired; resume discovery')
    result = discovery.discover(root, queries[2:], lambda *a, **kw: {
        'success': True, 'data': {'web': [{'url': 'https://example.org/data'}]}})
    assert result['usage']['regular'] == 3
    assert result['queries'][0]['outcome'] == 'success'


def test_later_batch_links_existing_keys_without_duplicating_sources(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    payload = batch()
    payload['batch_id'] = 'round-2'
    payload['sources'] = []
    payload['claims'][0]['key'] = 'growth-second-question'
    add_batch(root, payload)
    data = json.loads((root / 'evidence.json').read_text())
    assert len(data['sources']) == 1 and len(data['claims']) == 2
    assert data['claims'][1]['evidence_ids'] == ['S1E1']


def test_stale_completed_report_cannot_be_archived(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    draft(root)
    cli('research_run.py', 'checkpoint', root, '--status', 'complete')
    with (root / 'report.md').open('a') as out:
        out.write('\nA changed conclusion.\n')
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    cli('research_run.py', 'archive', root, '--prune', ok=False)
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()} == before


def test_readonly_claims_cannot_be_rewritten_inside_the_review(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    draft(root)
    path = approve(root)
    data = json.loads(path.read_text())
    data['claims'][0]['text'] = 'A different claim.'
    path.write_text(json.dumps(data))
    cli('semantic_review.py', 'check', root, ok=False)
