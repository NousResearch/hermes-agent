import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x_manager as xm


@pytest.fixture
def synthetic_voice_corpus(monkeypatch):
    """Keep freshness/voice/dedupe tests past the unrelated corpus-readiness gate."""
    import x_voice_gate
    corpus = [{"text": "Small queues keep failures visible.", "approved": True,
               "url": "https://example.test/synthetic-corpus",
               "provenance": {"kind": "synthetic_test_fixture"}}]
    monkeypatch.setattr(x_voice_gate, "load_voice_corpus", lambda: corpus)
    return corpus


def artifact(hours=1, identity='random'):
    created = datetime.now(timezone.utc) - timedelta(hours=hours)
    tid = str((int(created.timestamp()*1000)-1288834974657)<<22)
    source = dict(id=tid,url=f'https://x.com/test/status/{tid}',created_at=created.isoformat(),origin='for_you')
    return xm.XArtifact(identity,xm.LANE_QUOTE_SCAN,'sahil_twitter','A smaller queue makes failures easier to inspect.',xm.ArgumentPack('claim','evidence','mechanism','position',{'sources':[source]}))


def test_stale_and_unknown_fail_at_stage_and_delivery(tmp_path,monkeypatch,synthetic_voice_corpus):
    monkeypatch.setattr(xm,'DB_PATH',tmp_path/'state.db')
    for item in (artifact(7),artifact(-1)):
        with pytest.raises(xm.XManagerError, match="source provenance/freshness invalid"): xm.stage_for_approval(item)
        with pytest.raises(xm.XManagerError, match="source provenance/freshness invalid"): xm.format_approval_card(item)
    item=artifact();item.pack.context={}
    with pytest.raises(xm.XManagerError, match="source provenance missing"): xm.stage_for_approval(item)


def test_source_dedupe_and_staged_timestamp(tmp_path,monkeypatch,synthetic_voice_corpus):
    monkeypatch.setattr(xm,'DB_PATH',tmp_path/'state.db')
    first=artifact();xm.stage_for_approval(first)
    first.id='another-random-id'
    with pytest.raises(xm.XManagerError, match="source already staged"): xm.stage_for_approval(first)
    rows=xm.list_artifacts()
    assert len(rows)==1
    import json
    assert json.loads(rows[0]['context'])['staged_at']
    from x_delivery import prepare_delivery
    prepare_delivery(rows[0]['id'])
    assert json.loads(xm.list_artifacts()[0]['context'])['delivery_checked_at']
    import sqlite3
    stale=artifact(7)
    with sqlite3.connect(str(xm.DB_PATH)) as conn:
        conn.execute('UPDATE x_manager_artifacts SET context=?',(json.dumps(stale.pack.context),))
    with pytest.raises(xm.XManagerError, match="source provenance/freshness invalid"): prepare_delivery(rows[0]['id'])


def test_stage_rejects_voice_violation(tmp_path,monkeypatch,synthetic_voice_corpus):
    monkeypatch.setattr(xm,'DB_PATH',tmp_path/'state.db')
    item=artifact();item.body='Great point, a smaller queue makes failures easier to inspect.'
    with pytest.raises(xm.XManagerError, match='generic praise'): xm.stage_for_approval(item)
