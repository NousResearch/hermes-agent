"""Regression contracts from the live research review, exercised through the CLI."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'optional-skills/research/deep-researcher/scripts'


def cli(*args, ok=True):
    result = subprocess.run([sys.executable, str(SCRIPTS / 'semantic_review.py'), *map(str, args)],
                            capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


@pytest.fixture
def dossier(tmp_path):
    from tests.skills.test_deep_researcher_workflow import init, add_batch, draft
    root = init(tmp_path)
    add_batch(root)
    draft(root)
    return root


def prepare(root, text=None):
    if text:
        (root / 'report.md').write_text(text + '\n\n## References\n1. [Dataset](https://example.org/data)\n')
    cli('prepare', root, '--replace')
    return json.loads((root / 'semantic-review.json').read_text())


def ready(packet):
    packet['reviewer'] = 'Fixture reviewer'
    for check in packet['checks'].values():
        check.update(status='checked', note='Compared the synthetic report with retained passages.')
    return packet


def save(root, packet):
    (root / 'semantic-review.json').write_text(json.dumps(packet), encoding='utf-8')


def test_repeated_same_line_entity_has_one_stable_review_id(dossier):
    packet = prepare(dossier, 'The dataset mentions Cyprus twice: Cyprus is unverified; Cyprus remains unverified. [1]')
    ids = [i['id'] for i in packet['assertions']]
    assert len(ids) == len(set(ids))
    entries = [i for i in packet['assertions'] if i['text'] == 'Cyprus']
    assert len(entries) == 1
    again = prepare(dossier)
    assert [i['id'] for i in again['assertions']] == ids
    ready(packet)
    save(dossier, packet)
    cli('check', dossier)


def test_dates_and_demonyms_do_not_hide_real_organizations(dossier):
    packet = prepare(dossier, 'In June and March, H1 German and French figures mention MCEI and Alpine Research Agency. [1]')
    names = {i['text'] for i in packet['assertions'] if i['kind'] == 'entity'}
    assert not names.intersection({'June', 'March', 'H1', 'German', 'French'})
    assert {'MCEI', 'Alpine Research Agency'} <= names



def test_compact_view_and_atomic_decision_application(dossier):
    packet = prepare(dossier)
    view = json.loads(cli('show', dossier, '--max-chars', '3000').stdout)
    assert len(json.dumps(view, ensure_ascii=False)) <= 3000
    assert 'report' not in view and 'evidence' not in view
    assert any(item['id'] == 'C001' for item in view['items'])
    ready(packet)
    payload = {key: packet[key] for key in ['report_sha256', 'ledger_sha256', 'reviewer', 'checks']}
    path = dossier / 'decisions.json'
    path.write_text(json.dumps(payload))
    cli('apply', dossier, path)
    initial = (dossier / 'semantic-review.json').read_bytes()
    cli('apply', dossier, path)
    assert (dossier / 'semantic-review.json').read_bytes() == initial
    cli('check', dossier)
    before = (dossier / 'semantic-review.json').read_bytes()
    payload['checks']['invented'] = {'status': 'checked', 'note': 'invalid'}
    path.write_text(json.dumps(payload))
    cli('apply', dossier, path, ok=False)
    assert (dossier / 'semantic-review.json').read_bytes() == before


def test_apply_rejects_stale_material_without_reusing_approval(dossier):
    packet = prepare(dossier)
    payload = {key: packet[key] for key in ['report_sha256', 'ledger_sha256']}
    payload.update(reviewer='Fixture reviewer', checks={})
    path = dossier / 'decisions.json'
    path.write_text(json.dumps(payload))
    before = (dossier / 'semantic-review.json').read_bytes()
    with (dossier / 'report.md').open('a') as stream:
        stream.write('\nNew interpretation.\n')
    cli('apply', dossier, path, ok=False)
    assert (dossier / 'semantic-review.json').read_bytes() == before
    fresh = prepare(dossier)
    assert all(i['status'] == 'pending' for i in fresh['checks'].values())


def test_check_reports_every_unresolved_category(dossier):
    packet = ready(prepare(dossier))
    for key in ['finality', 'entities', 'numbers']:
        packet['checks'][key]['status'] = 'pending'
    save(dossier, packet)
    result = cli('check', dossier, ok=False)
    assert all(word in result.stderr for word in ['finality', 'entities', 'numbers'])


def test_same_entity_on_distinct_lines_keeps_distinct_contexts(dossier):
    packet = prepare(dossier, 'The dataset names Cyprus. [1]\nAnother text names Cyprus. [1]')
    entries = [i for i in packet['assertions'] if i['text'] == 'Cyprus']
    assert len(entries) == 2 and entries[0]['id'] != entries[1]['id']


def test_selected_view_shows_passages_and_reports_omissions(dossier):
    prepare(dossier)
    view = json.loads(cli('show', dossier, '--ids', 'C001', '--max-chars', '5000').stdout)
    assert [i['id'] for i in view['items']] == ['C001']
    assert view['passages'][0]['id'] == 'S1E1'
    assert '10%' in view['passages'][0]['excerpt']
    assert 'numeric_review' in view['items'][0]


def test_apply_valid_partial_decision_leaves_other_requirements_pending(dossier):
    packet = prepare(dossier)
    payload = {key: packet[key] for key in ['report_sha256', 'ledger_sha256']}
    payload.update(reviewer='Fixture reviewer', checks={'numbers': {'status': 'checked', 'note': 'Reviewed sample percentage scope.'}})
    path = dossier / 'partial.json'
    path.write_text(json.dumps(payload))
    result = json.loads(cli('apply', dossier, path).stdout)
    assert result['saved'] and not result['ready']
    cli('check', dossier, ok=False)


def test_article_prefix_keeps_known_names_known_and_flags_unknown_agency(dossier):
    packet = prepare(dossier, 'The Dataset reports growth. The Alpine Research Agency disagrees. [1]')
    names = {i['text'] for i in packet['assertions'] if i['kind'] == 'entity'}
    assert 'Dataset' not in names and 'The Dataset' not in names
    assert 'Alpine Research Agency' in names


def test_template_writes_valid_unapproved_decisions_without_overwriting(dossier):
    packet = prepare(dossier)
    result = json.loads(cli('template', dossier).stdout)
    path = Path(result['path'])
    draft = json.loads(path.read_text())
    assert draft['report_sha256'] == packet['report_sha256']
    assert set(draft) == {'report_sha256', 'ledger_sha256', 'reviewer', 'checks'}
    assert all(i['status'] == 'pending' for i in draft['checks'].values())
    cli('template', dossier, ok=False)
    draft['reviewer'] = 'Fixture reviewer'
    path.write_text(json.dumps(draft))
    assert not json.loads(cli('apply', dossier, path).stdout)['ready']
    ready(draft)
    path.write_text(json.dumps(draft))
    cli('apply', dossier, path)
    cli('check', dossier)


def test_template_cannot_replace_core_artifacts(dossier):
    prepare(dossier)
    path = dossier / 'evidence.json'
    before = path.read_bytes()
    cli('template', dossier, '--output', 'evidence.json', '--replace', ok=False)
    assert path.read_bytes() == before


def test_entity_review_ignores_report_labels_but_keeps_substantive_names(dossier):
    text = ('> **Mode:** standard | **Sources:** 1 | **Scope:** fixture\n'
            'Its March update reports growth. **Confidence: Medium–High**. [1]\n'
            '1. **Affordability.** MCEI comments on the result. [1]\n'
            '2. **Installer capacity/cost.** The Alpine Research Agency disagrees. [1]\n'
            '*Prepared with the Hermes Deep Researcher workflow. Citations establish provenance; confidence reflects independence.*')
    packet = prepare(dossier, text)
    names = {i['text'] for i in packet['assertions'] if i['kind'] == 'entity'}
    assert not names.intersection({'Mode', 'Sources', 'Scope', 'Its March', 'Medium', 'High', 'Affordability', 'Installer', 'Hermes Deep Researcher'})
    assert {'MCEI', 'Alpine Research Agency'} <= names


def test_heuristic_warnings_do_not_require_word_level_decisions(dossier):
    packet = ready(prepare(dossier, 'MCEI identity is unverified. The dataset reports growth of 10%. [1]'))
    assert any(i['text'] == 'MCEI' for i in packet['assertions'])
    save(dossier, packet)
    result = json.loads(cli('check', dossier).stdout)
    assert result['claims_in_dossier'] == 1 and result['heuristic_warnings'] >= 1
    template_result = json.loads(cli('template', dossier).stdout)
    payload = json.loads(Path(template_result['path']).read_text())
    assert 'assertions' not in payload


def test_identified_corrections_are_saved_and_block_completion(dossier):
    packet = prepare(dossier)
    payload = {key: packet[key] for key in ['report_sha256', 'ledger_sha256']}
    payload.update(reviewer='Fixture critic', checks={'numbers': {'status': 'needs-correction', 'note': 'Correct the population comparison.'}})
    path = dossier / 'critique.json'
    path.write_text(json.dumps(payload))
    result = json.loads(cli('apply', dossier, path).stdout)
    assert result['saved'] and not result['ready']
    result = cli('check', dossier, ok=False)
    assert 'Correct the population comparison' in result.stderr
