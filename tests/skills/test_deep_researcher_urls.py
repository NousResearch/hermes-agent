"""Citation and evidence identity regressions from the contributed skill review."""
import json

import pytest

from tests.skills.test_deep_researcher_workflow import cli, init, add_batch


def audit(tmp_path, urls, body='A finding. [1]', bibliography=None, ledger_urls=None, ok=True):
    report = tmp_path / 'report.md'
    entries = bibliography or '\n'.join(f'{i}. [Source]({url})' for i, url in enumerate(urls, 1))
    report.write_text(body + '\n\n## References\n' + entries, encoding='utf-8')
    ledger = tmp_path / 'audit-ledger.json'
    ledger.write_text(json.dumps({'sources': [{'url': url, 'canonical_url': url.split('?')[0]}
                                              for url in (ledger_urls or urls)]}))
    return cli('audit_report.py', report, '--ledger', ledger, '--no-live', ok=ok)


@pytest.mark.parametrize('first,second', [
    ('https://example.org/item?id=111', 'https://example.org/item?id=222'),
    ('https://youtube.com/watch?v=first', 'https://youtube.com/watch?v=second'),
    ('https://example.org/pdf?document=111', 'https://example.org/pdf?document=222'),
])
def test_audit_preserves_query_identifiers(tmp_path, first, second):
    audit(tmp_path, [first, second], body='Two findings. [1,2]')


def test_audit_does_not_match_different_query_through_legacy_canonical_field(tmp_path):
    result = audit(tmp_path, ['https://example.org/item?id=222'],
                   ledger_urls=['https://example.org/item?id=111'], ok=False)
    assert 'absent from the evidence ledger' in result.stdout


@pytest.mark.parametrize('url,markup', [
    ('https://en.wikipedia.org/wiki/Python_(programming_language)', '[Python](https://en.wikipedia.org/wiki/Python_(programming_language))'),
    ('https://example.org/a_(b_(c))', '[Nested](https://example.org/a_(b_(c)))'),
    ('https://example.org/a_(b)', '[Angle](<https://example.org/a_(b)> "Title")'),
    ('https://example.org/a_(b)', r'[Escaped](https://example.org/a_\(b\))'),
])
def test_markdown_destinations_preserve_balanced_parentheses(tmp_path, url, markup):
    audit(tmp_path, [url], bibliography='1. ' + markup)


def test_bracketed_year_is_not_a_missing_citation(tmp_path):
    audit(tmp_path, ['https://example.org/a'], body='In [2024] it grew. [1]')


def test_actual_citation_errors_still_fail(tmp_path):
    result = audit(tmp_path, ['https://example.org/a'], body='In [2024] it grew. [1,2]', ok=False)
    assert 'Citation [2]' in result.stdout and 'Citation [2024]' not in result.stdout


def test_following_section_is_not_parsed_as_bibliography(tmp_path):
    audit(tmp_path, ['https://example.org/a'],
          bibliography='1. [Source](https://example.org/a)\n\n## Methodology\n1. Search the source.\n2. Review evidence.')


@pytest.mark.parametrize('parameter', ['ref', 'source'])
def test_source_identifying_query_parameters_remain_distinct(tmp_path, parameter):
    root = init(tmp_path)
    ledger = root / 'evidence.json'
    for version in ['v1', 'v2']:
        cli('evidence_ledger.py', 'add-source', ledger, '--url', f'https://github.com/example/repo?{parameter}={version}',
            '--title', version, '--source-type', 'primary', '--question', 'Q1', '--summary', 'Fixture version.')
    assert len(json.loads(ledger.read_text())['sources']) == 2


@pytest.mark.parametrize('status', ['failed', 'pending'])
def test_question_coverage_counts_only_extracted_sources(tmp_path, status):
    root = init(tmp_path)
    ledger = root / 'evidence.json'
    for i in range(2):
        cli('evidence_ledger.py', 'add-source', ledger, '--url', f'https://example.org/{i}',
            '--title', 'Fixture', '--source-type', 'primary', '--question', 'Q1', '--summary', 'Fixture summary.')
    data = json.loads(ledger.read_text())
    data['sources'][1]['extraction_status'] = status
    ledger.write_text(json.dumps(data))
    result = cli('evidence_ledger.py', 'check', ledger)
    assert 'fewer than two extracted sources' in result.stdout


@pytest.mark.parametrize('sources,evidence', [('999', 'S1E1'), ('1', 'S999E1'), ('1', 'S1E999')])
def test_invalid_individual_claim_never_writes(tmp_path, sources, evidence):
    root = init(tmp_path)
    add_batch(root)
    ledger = root / 'evidence.json'
    before = ledger.read_bytes()
    cli('evidence_ledger.py', 'add-claim', ledger, '--text', 'A claim.', '--question', 'Q1',
        '--status', 'single-source', '--source', sources, '--evidence', evidence, ok=False)
    assert ledger.read_bytes() == before


def test_native_truncated_display_is_not_accepted_as_full_cache(tmp_path):
    path = tmp_path / 'display.txt'
    path.write_text('head\n\n[... middle omitted — see footer ...]\n\ntail\n──────── [TRUNCATED] ────────\nFull text saved to: native-cache.md\n')
    result = cli('passages.py', path, '--term', 'decisive', ok=False)
    assert 'full' in result.stderr.lower() and 'truncated' in result.stderr.lower()


def test_later_batch_uses_assigned_ids_after_failed_source_addition(tmp_path):
    root = init(tmp_path)
    add_batch(root)
    ledger = root / 'evidence.json'
    cli('evidence_ledger.py', 'add-source', ledger, '--url', 'https://example.org/data',
        '--title', 'Duplicate', '--source-type', 'primary', '--question', 'Q1', ok=False)
    from tests.skills.test_deep_researcher_workflow import batch
    payload = batch()
    payload['batch_id'] = 'round-2'
    payload['sources'][0].update(key='second', url='https://example.org/second')
    payload['claims'][0].update(key='second-claim', sources=['second'], evidence=['second:1'])
    add_batch(root, payload)
    data = json.loads(ledger.read_text())
    assert data['claims'][1]['source_ids'] == [data['sources'][1]['id']]
    assert data['claims'][1]['evidence_ids'] == [data['sources'][1]['evidence'][0]['id']]
    assert data['claims'][1]['source_ids'] != data['claims'][0]['source_ids']


def test_query_order_and_www_identity_are_preserved(tmp_path):
    root = init(tmp_path)
    ledger = root / 'evidence.json'
    urls = ['https://www.example.org/item?step=A&step=B', 'https://www.example.org/item?step=B&step=A']
    for url in urls:
        cli('evidence_ledger.py', 'add-source', ledger, '--url', url, '--title', 'Fixture',
            '--source-type', 'primary', '--question', 'Q1', '--summary', 'Fixture summary.')
    sources = json.loads(ledger.read_text())['sources']
    assert [s['url'] for s in sources] == urls
    assert [s['canonical_url'] for s in sources] == urls


def test_capped_native_cache_discloses_incomplete_input(tmp_path):
    path = tmp_path / 'native-cache.txt'
    path.write_text('A decisive finding.\n\n[... stored copy truncated at 100 chars of 1000; re-extract for the rest ...]')
    result = json.loads(cli('passages.py', path, '--term', 'decisive').stdout)
    assert result['input_truncated']
    assert 'unreviewed' in result['note']


def test_live_audit_receives_the_complete_markdown_destination(tmp_path, monkeypatch):
    import argparse
    import importlib
    from tests.skills.test_deep_researcher_workflow import SCRIPTS
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = importlib.import_module('audit_report')
    url = 'https://www.example.org/Python_(programming_language)?ref=v2'
    report = tmp_path / 'report.md'
    report.write_text(f'A finding. [1]\n\n## References\n1. [Source]({url})\n')
    requested = []

    def live_check(item, timeout):
        requested.append(item[1])
        return module.LinkResult(item[0], item[1], 'ok', 'Fixture HTTP 200')

    monkeypatch.setattr(module, 'check_url', live_check)
    code = module.run(argparse.Namespace(report=str(report), ledger=None, no_live=False,
                                        workers=1, timeout=1, max_heuristic_warnings=12))
    assert code == 0 and requested == [url]
