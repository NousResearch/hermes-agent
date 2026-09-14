"""Living-reference policy replaces fixed-corpus authorisation."""
import json
from pathlib import Path
from datetime import datetime, timezone
import x_voice_gate as voice


def test_default_does_not_read_deleted_legacy_archive(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    legacy = tmp_path/'research/x-voice/sahil-public-corpus-2026.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('[]')
    assert voice.load_voice_corpus() == []


def test_valid_published_observation_needs_no_approval_sidecar(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = tmp_path/'data/x-analytics/observations.json'
    path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    sid = str((int(now.timestamp()*1000-3600000)-1288834974657)<<22)
    row = {'id':sid,'url':f'https://x.com/Sahil_Saghir/status/{sid}',
           'author':'Sahil_Saghir','text':'Synthetic published-observation fixture.',
           'created_at':datetime.fromtimestamp(((int(sid)>>22)+1288834974657)/1000,timezone.utc).isoformat(),
           'observed_at':now.isoformat(),'provenance':{'kind':'browser','source':'https://x.com/Sahil_Saghir'}}
    path.write_text(json.dumps({'account':'Sahil_Saghir','posts':[row]}))
    result = voice.load_voice_corpus()
    assert len(result) == 1
    assert result[0]['text'] == row['text']
    assert result[0]['provenance']['kind'] == 'observed_published_post'


def test_generated_drafts_are_not_published_references(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    path = tmp_path/'data/x-analytics/observations.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'account':'Sahil_Saghir','posts':[{'text':'made up','provenance':{'kind':'generated'}}]}))
    assert voice.load_voice_corpus() == []
