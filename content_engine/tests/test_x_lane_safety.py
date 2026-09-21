"""Offline lane contracts: real builders and file state; no live services."""
import importlib.util
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'content_engine'))
sys.path.insert(0, str(ROOT / 'scripts' / 'content_engine'))


@pytest.fixture
def lanes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    def load(name, folder):
        spec = importlib.util.spec_from_file_location(name, ROOT / folder / (name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod
    monkeypatch.setitem(sys.modules, 'x_manager_report', Mock(render_report=Mock(side_effect=AssertionError('unexpected rendering'))))
    for name in ('x_ingest', 'x_voice_gate', 'x_manager', 'llm_generate'):
        load(name, 'content_engine')
    scout = load('x_quote_scout', 'scripts/content_engine')
    morning = load('x_morning_article', 'scripts/content_engine')
    thesis = load('x_thesis_incubator', 'scripts/content_engine')
    for mod in (scout, morning, thesis):
        monkeypatch.setattr(mod, 'CE', tmp_path)
        monkeypatch.setattr(mod, '_load_env', lambda: None)
    for key in ('VERDICT_FILE', 'STANDALONE_SEEDS', 'STATE_FILE2', 'CANDIDATES_JSON'):
        monkeypatch.setattr(scout, key, tmp_path / (key + '.json'))
    monkeypatch.setattr(morning, 'STATE_FILE', tmp_path / 'morning.json')
    monkeypatch.setattr(thesis, 'THESIS_STATE', tmp_path / 'thesis.json')
    monkeypatch.setattr(thesis, 'STANDALONE_SEEDS', scout.STANDALONE_SEEDS, raising=False)
    monkeypatch.setattr(scout, 'blog_cross_reference', lambda text: [])
    return scout, morning, thesis


def source(hours=1, **kwargs):
    created = datetime.now(timezone.utc) - timedelta(hours=hours)
    tid = str((int(created.timestamp() * 1000) - 1288834974657) << 22)
    row = dict(id=tid, url=f'https://x.com/builder/status/{tid}',
               created_at=created.isoformat(),
               origin='for_you', source='for_you', author='builder',
               text='A specific agent benchmark with measured latency and a documented failure.',
               summary='A specific agent benchmark with measured latency and a documented failure.',
               signal_id='research-1', signal_type='research_signal')
    row.update(kwargs)
    return row


def draft():
    return dict(verdict='quote', reason='A specific tradeoff worth examining',
                post='That latency claim needs the slow runs too.', stance='Tail latency matters',
                claim='Tail latency matters', evidence='Measured benchmark',
                mechanism='Slow runs determine usability', position='Report slow runs',
                body='That latency claim needs the slow runs too.')


@pytest.mark.parametrize('hours,created', [(7, None), (-1, None), (1, ''), (1, 'bad'), (1, '2026-09-12T08:00:00')])
def test_every_lane_rejects_unverifiable_sources_before_llm(lanes, monkeypatch, hours, created):
    scout, morning, thesis = lanes
    row = source(hours)
    if created is not None:
        row['created_at'] = created
    calls = Mock(return_value=draft())
    monkeypatch.setattr(scout, '_draft', calls)
    monkeypatch.setattr(morning, '_draft_article', calls)
    monkeypatch.setattr(thesis, '_draft_thesis', calls)
    assert scout._candidate_artifacts([row])[0] == []
    with pytest.raises((ValueError, RuntimeError)):
        morning._build_artifact(row)
    monkeypatch.setattr(thesis, '_load_inbox', lambda: [row, dict(row, id='another')])
    monkeypatch.setattr(thesis, '_load_seeds', lambda: [])
    thesis.main()
    calls.assert_not_called()


def test_scout_rejects_one_voice_issue_and_persists_source_dedupe(lanes, monkeypatch):
    scout, _, _ = lanes
    row = source()
    calls = Mock(return_value={**draft(), 'post': 'Great point, latency needs the slow runs too.'})
    monkeypatch.setattr(scout, '_draft', calls)
    assert scout._candidate_artifacts([row])[0] == []
    scout._candidate_artifacts([row])
    assert calls.call_count == 1


def test_collection_preserves_honest_coverage_and_context(lanes, monkeypatch):
    scout, _, _ = lanes
    row = source(thread_context=[{'text': 'A parent post'}], context_status='partial')
    coverage = {'following': {'complete': False, 'partial_reason': 'page bound'}}
    def ingest(**kwargs):
        kwargs['diagnostics'].update(coverage)
        return [row]
    monkeypatch.setattr(scout.x_ingest, 'ingest', ingest)
    rows = scout._collect()
    monkeypatch.setattr(scout, '_draft', lambda tweet: draft())
    arts = scout._candidate_artifacts(rows)[0]
    assert arts[0].pack.context['feed_coverage']['following'] == coverage['following']
    assert arts[0].pack.context['feed_coverage']['own_reference_refresh']['status'] == 'disabled'
    assert arts[0].pack.context['thread_context'] == row['thread_context']
    assert arts[0].pack.context['context_status'] == 'partial'


def test_morning_collector_failure_is_empty_not_invented(lanes, monkeypatch):
    _, morning, _ = lanes
    monkeypatch.setitem(sys.modules, 'activity_collector', Mock(collect_all=Mock(side_effect=RuntimeError('offline'))))
    assert morning._collect_signals() == []


def test_provenance_and_article_source_dedupe_survive_new_artifact_ids(lanes, monkeypatch):
    scout, morning, _ = lanes
    row = source()
    monkeypatch.setattr(scout, '_draft', lambda row: draft())
    monkeypatch.setattr(morning, '_draft_article', lambda row: draft())
    arts = scout._candidate_artifacts([row])[0]
    article = morning._build_artifact(row)
    for art in [*arts, article]:
        assert art.pack.context['sources'] == [{k: row[k] for k in ('id', 'url', 'created_at', 'origin')}]
    morning._record_reported([article])
    with pytest.raises((ValueError, RuntimeError), match='duplicate'):
        morning._build_artifact(row)


def test_seeds_expire_and_consumption_is_persistent(lanes):
    scout, _, thesis = lanes
    fresh = source()
    old = source(7)
    def seed(row):
        return dict(source_id=row['id'], source_url=row['url'], created_at=row['created_at'],
                    origin=row['origin'], claim='Tail latency', evidence='benchmark',
                    sources=[{k: row[k] for k in ('id', 'url', 'created_at', 'origin')}])
    scout._merge_standalone_seeds([seed(old), seed(fresh)])
    seeds = thesis._load_seeds()
    assert len(seeds) == 1
    thesis._mark_seeds_used(seeds)
    assert thesis._load_seeds() == []
    scout._merge_standalone_seeds(seeds)
    assert thesis._load_seeds() == []  # A producer replay must not resurrect a consumed seed.


def test_voice_runtime_requires_real_approved_own_history(lanes, tmp_path, monkeypatch):
    scout, morning, thesis = lanes
    for mod in (scout, morning, thesis):
        monkeypatch.setattr(mod, 'VOICE_SKILL', tmp_path / 'missing-skill')
    for mod in (scout, morning, thesis):
        with pytest.raises(FileNotFoundError):
            mod._runtime_voice()
    data = tmp_path / 'data/x-analytics'
    data.mkdir(parents=True)
    row = source(72, author='Sahil_Saghir')
    row['url'] = f"https://x.com/Sahil_Saghir/status/{row['id']}"
    corpus = data / 'observations.json'
    row.update(observed_at=datetime.now(timezone.utc).isoformat(),provenance={'kind':'browser','source':'https://x.com/Sahil_Saghir'})
    raw = json.dumps({'account':'Sahil_Saghir','posts':[row]}).encode()
    corpus.write_bytes(raw)
    Path(str(corpus) + '.approval.json').write_text(json.dumps({
        'approved': True, 'sha256': hashlib.sha256(raw).hexdigest()}))
    # Runtime now also requires the actual voice instructions and calibration.
    skill = tmp_path / 'voice/SKILL.md'
    skill.parent.mkdir()
    skill.write_text('Synthetic voice guidance')
    (skill.parent / 'references').mkdir()
    (skill.parent / 'references/runtime-voice.md').write_text('Synthetic voice guidance')
    (skill.parent / 'references/approved-conversational-calibration.md').write_text('Synthetic calibration')
    monkeypatch.setattr(scout, 'VOICE_SKILL', skill)
    for mod in (scout, morning, thesis):
        assert row['text'] in mod._runtime_voice()
    corpus.unlink()
    for mod in (scout, morning, thesis):
        assert 'unavailable' in mod._runtime_voice()
        assert 'Synthetic voice guidance' in mod._runtime_voice()


def test_article_and_thesis_reject_invented_experience_and_preserve_sources(lanes, monkeypatch, tmp_path):
    scout, morning, thesis = lanes
    row = source()
    monkeypatch.setattr(morning, '_draft_article', lambda row: {**draft(), 'body': 'I deployed this model yesterday.'})
    with pytest.raises(RuntimeError, match='voice'):
        morning._build_artifact(row)
    a, b = source(), source(2)
    monkeypatch.setattr(thesis, '_load_inbox', lambda: [a, b])
    monkeypatch.setattr(thesis, '_load_seeds', lambda: [])
    monkeypatch.setattr(thesis, '_draft_thesis', lambda material: {**draft(), 'post': 'I deployed this model yesterday.'})
    staged = []
    monkeypatch.setattr(thesis.xm, 'stage_for_approval', lambda art: staged.append(art))
    monkeypatch.setattr(thesis, '_mark_inbox_used', lambda rows: None)
    monkeypatch.setattr(thesis, 'render_report', lambda *args, **kwargs: tmp_path / 'review.html')
    thesis.main()
    assert staged == []
    monkeypatch.setattr(thesis, '_draft_thesis', lambda material: draft())
    thesis.main()
    assert len(staged) == 1
    assert staged[0].pack.context['sources'] == scout._sources(a) + scout._sources(b)
    thesis.main()
    assert len(staged) == 1  # New random draft IDs do not permit reuse.


def test_source_identity_conflict_and_stale_direct_draft_fail_before_llm(lanes, monkeypatch):
    scout, morning, _ = lanes
    conflict = source(url='https://x.com/builder/status/99')
    llm = Mock(side_effect=AssertionError('LLM must not run'))
    monkeypatch.setattr(scout, '_call_llm_chain', llm)
    monkeypatch.setattr(morning, '_call_llm_chain', llm)
    for row in (conflict, source(7)):
        with pytest.raises(ValueError):
            scout._draft(row)
        with pytest.raises(ValueError):
            morning._draft_article(row)
    llm.assert_not_called()
