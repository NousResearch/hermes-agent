"""Adversarial tests for living source refresh, entirely fixture-backed."""
import json
from datetime import datetime, timezone, timedelta
import pytest
from x_voice_gate import load_voice_corpus


def observation():
    now=datetime.now(timezone.utc)
    created=now-timedelta(hours=1)
    sid=str((int(created.timestamp()*1000)-1288834974657)<<22)
    return {'id':sid,'url':f'https://x.com/Sahil_Saghir/status/{sid}',
            'author':'Sahil_Saghir','text':'Fixture only; never a production voice reference.',
            'created_at':created.isoformat(),'observed_at':now.isoformat(),
            'metrics':{'likes':1,'replies':0,'reposts':0,'views':10},
            'provenance':{'kind':'browser','source':'https://x.com/Sahil_Saghir','sha256':'a'*64}}


def test_latest_incomplete_cannot_revive_old_text(monkeypatch,tmp_path):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    row=observation();row['author']='sahil_saghir'
    path=tmp_path/'data/x-analytics/observations.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'posts':[row]}))
    assert len(load_voice_corpus())==1
    latest={**row,'observed_at':datetime.now(timezone.utc).isoformat(),'truncated':True}
    path.write_text(json.dumps({'posts':[row,latest]}))
    assert load_voice_corpus()==[]
    latest={**row,'observed_at':(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}
    path.write_text(json.dumps({'posts':[latest]}))
    assert load_voice_corpus()==[]


def test_refresh_uses_replies_and_preserves_history(monkeypatch,tmp_path):
    import x_reference_refresh as refresh
    path=tmp_path/'observations.json'
    monkeypatch.setenv('X_OBSERVATIONS_PATH',str(path))
    row=observation();calls=[]
    def collect(account,**kwargs):
        assert account=='Sahil_Saghir'
        calls.append(kwargs)
        return {'schema_version':1,'account':'Sahil_Saghir','posts':[row]}
    result=refresh.refresh_references(collector=collect)
    assert result['posts']==1 and calls[0]['include_replies'] is True
    assert len(json.loads(path.read_text())['posts'])==1
    saved=path.read_bytes()
    def failed(*args,**kwargs): raise RuntimeError('fixture browser unavailable')
    with pytest.raises(RuntimeError):refresh.refresh_references(collector=failed)
    assert path.read_bytes()==saved
