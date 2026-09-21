"""Synthetic, isolated witnesses for the independent QA bypasses."""
import json
import sqlite3
import pytest
from test_x_stage_safety import artifact
import x_manager as xm
import x_voice_gate as voice
from x_delivery import prepare_delivery

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(xm, 'DB_PATH', tmp_path / 'state.db')
    monkeypatch.setattr(voice, 'load_voice_corpus', lambda: [{'approved': True, 'url': 'https://x.com/Sahil_Saghir/status/123', 'provenance': {'kind': 'synthetic'}, 'text': 'A factual test exemplar.'}])


def test_missing_soft_references_do_not_block_valid_review(isolated, monkeypatch):
    item = artifact()
    xm.stage_for_approval(item)
    monkeypatch.setattr(voice, 'load_voice_corpus', lambda: [])
    for call in (lambda: xm.stage_for_approval(artifact(identity='next')), lambda: xm.format_approval_card(item), lambda: prepare_delivery(item.id)):
        call()  # Style references are optional; freshness and factuality still gate.

@pytest.mark.parametrize('body', ['I migrated our database last week.', 'We doubled revenue with this workflow.', 'My revenue tripled.', "I've completely rewritten the parser.", 'Yesterday, we quietly replaced everything.'])
def test_all_first_person_assertions_need_exact_evidence(isolated, body):
    item = artifact(); item.body = body
    with pytest.raises(xm.XManagerError, match='experience'):
        xm.stage_for_approval(item)


def test_real_approved_export_allows_only_exact_first_person_sentence(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(xm, 'DB_PATH', tmp_path/'state.db')
    item = artifact()
    own = dict(item.pack.context['sources'][0])
    own['url'] = 'https://x.com/Sahil_Saghir/status/' + own['id']
    own['text'] = 'I migrated our database last week.'
    own.update(author='Sahil_Saghir', observed_at=item.pack.context['sources'][0]['created_at'], provenance={'kind':'browser','source':'https://x.com/Sahil_Saghir'})
    path = tmp_path/'data/x-analytics/observations.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'account':'Sahil_Saghir','posts':[own]}))
    item.body = own['text']
    xm.stage_for_approval(item)
    assert prepare_delivery(item.id).body == item.body
    for body in ('I migrated our database last week. We doubled revenue.',
                 'I migrated our database last week; we doubled revenue.',
                 'I migrated our database last week without downtime.'):
        item.body = body
        with pytest.raises(xm.XManagerError, match='experience'):
            xm.format_approval_card(item)
    path.unlink()
    with pytest.raises(xm.XManagerError, match='experience'):
        prepare_delivery(item.id)


@pytest.mark.parametrize('legacy_context', [False, True])
def test_legacy_source_index_is_backfilled(isolated, legacy_context):
    item = artifact(); xm.stage_for_approval(item)
    with sqlite3.connect(xm.DB_PATH) as conn:
        conn.execute('DROP TABLE x_manager_sources')
        if legacy_context:
            source = item.pack.context['sources'][0]
            conn.execute('UPDATE x_manager_artifacts SET context=?',
                         (json.dumps({'tweet_id': source['id'], 'source_url': source['url']}),))
    item.id = 'different-id'
    with pytest.raises(xm.XManagerError, match='source already staged'):
        xm.stage_for_approval(item)
    with sqlite3.connect(xm.DB_PATH) as conn:
        assert conn.execute('SELECT artifact_id FROM x_manager_sources').fetchone()[0] == 'random'


def test_unrecoverable_legacy_history_blocks_new_staging(isolated):
    item = artifact(); xm.stage_for_approval(item)
    with sqlite3.connect(xm.DB_PATH) as conn:
        conn.execute('DROP TABLE x_manager_sources')
        conn.execute("UPDATE x_manager_artifacts SET context='{}'")
    with pytest.raises(xm.XManagerError, match='legacy.*review'):
        xm.stage_for_approval(artifact(identity='new'))


def test_quote_builder_keeps_pack_free_factual_pointers(isolated):
    candidates = []
    for index in range(xm.QUOTE_SCAN_MIN):
        item = artifact(identity=str(index))
        source = item.pack.context['sources'][0]
        item.pack.claim = item.pack.evidence = item.pack.mechanism = item.pack.position = ''
        candidates.append(dict(tweet_id=source['id'], author='test', text='Documented source.',
                               quote_draft='Source: ' + source['url'], pack=item.pack))
    assert len(xm.scan_quote_tweet_candidates(candidates)) == xm.QUOTE_SCAN_MIN


def test_conversational_opinion_stages_without_essay_pack(isolated):
    item = artifact(); item.pack.claim = item.pack.evidence = item.pack.mechanism = item.pack.position = ''
    item.body = 'That conclusion is wrong. The slow runs matter too.'
    # Conversational opinions are approval drafts, not independent essays.
    xm.stage_for_approval(item)
