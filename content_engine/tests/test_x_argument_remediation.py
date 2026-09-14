"""Offline argument-policy regression witnesses; no publishing or staging."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'content_engine'), str(ROOT / 'scripts/content_engine')]


@pytest.mark.parametrize('verdict', ['reply', 'quote'])
@pytest.mark.parametrize('form', ['pointer', 'attribution', 'opinion', 'packed_opinion'])
def test_scout_keeps_pack_free_source_pointer(tmp_path, monkeypatch, verdict, form):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import x_quote_scout as scout
    import x_voice_gate
    # Explicit synthetic readiness for the real common builder, not a loader test.
    corpus = [{"text": "Small queues keep failures visible.", "approved": True,
               "url": "https://example.test/synthetic-corpus",
               "provenance": {"kind": "synthetic_test_fixture"}}]
    monkeypatch.setattr(x_voice_gate, "load_voice_corpus", lambda: corpus)
    created = datetime.now(timezone.utc) - timedelta(hours=1)
    tid = str((int(created.timestamp() * 1000) - 1288834974657) << 22)
    row = dict(id=tid, url=f'https://x.com/builder/status/{tid}',
               created_at=created.isoformat(), origin='for_you', author='builder',
               text='The benchmark reports latency across all measured runs.')
    monkeypatch.setattr(scout, 'VERDICT_FILE', tmp_path / 'verdicts.json')
    monkeypatch.setattr(scout, 'blog_cross_reference', lambda text: [])
    posts = {
        'pointer': 'Source: ' + row['url'],
        'attribution': f'The source says: "{row["text"]}"',
        'opinion': 'That conclusion is wrong. The slow runs matter too.',
        'packed_opinion': 'That conclusion is wrong. The slow runs matter too.',
    }
    post = posts[form]
    data = dict(verdict=verdict, post=post, intent='factual', argument_required=False)
    if form == 'packed_opinion':
        data.update(claim='The conclusion omits tail latency', evidence=row['text'],
                    mechanism='Aggregate latency can hide slow runs', position='Include tail latency')
    monkeypatch.setattr(scout, '_draft', lambda row: data)
    artifacts, seeds, discards = scout._candidate_artifacts([row])
    # The approved contract permits both conversational reactions and pointers.
    assert len(artifacts) == 1
    assert artifacts[0].body == post
    assert artifacts[0].pack.is_complete() is (form == 'packed_opinion')
    assert not seeds and not discards
    assert json.loads((tmp_path / 'verdicts.json').read_text())['source_keys']


@pytest.mark.parametrize('lane', ['reply_draft', 'quote_tweet_scan'])
def test_conversation_pack_policy_ignores_generated_intent_labels(lane):
    import x_manager as xm
    from x_argument_policy import requires_argument_pack
    source_text = 'The benchmark reports latency across all measured runs.'
    pack = xm.ArgumentPack('', '', '', '', {
        'sources': [{'url': 'https://example.com/benchmark'}],
        'source_text': source_text,
        'argument_required': False, 'intent': 'factual', 'stance': 'ordinary',
    })
    art = xm.XArtifact('policy-test', lane, 'sahil_twitter',
                       f'The source says: "{source_text}"', pack)
    assert requires_argument_pack(art) is False
    for body in (
        'That conclusion is wrong. The slow runs matter too.',
        'The benchmark also needs the slow runs.',
        'One wonders how that inference survived scrutiny.',
        'Source: https://example.com/other',
        'Source: https://example.com/benchmark That conclusion is wrong.',
        f'The source says: "{source_text}" This proves my point.',
        'The source says: "Invented supporting data."',
    ):
        art.body = body
        # Pack policy is not a factuality classifier; source/voice validation
        # and human review remain mandatory for these conversational lanes.
        assert requires_argument_pack(art) is False, body
    art.body = f'The source says: "{source_text}"'
    art.lane = xm.LANE_ARTICLE
    assert requires_argument_pack(art) is True
