"""A profile's native context watermark applies equally to new and resumed threads."""
from types import SimpleNamespace

import pytest

from agent import codex_runtime
from agent.transports import codex_app_server_session as wire


class RecordingClient:
    def __init__(self, **kwargs):
        self.requests = []

    def initialize(self, **kwargs):
        return {}

    def request(self, method, params=None, timeout=None):
        self.requests.append((method, params))
        return {'thread': {'id': params.get('threadId', 'fresh')}}

    def close(self):
        pass


def start_profile(tmp_path, monkeypatch, *, limit=None, mode='native', resume=False):
    (tmp_path/'config.yaml').write_text(
        'compression:\n  codex_app_server_auto: '+mode+'\n'+
        (f'  codex_auto_compact_token_limit: {limit}\n' if limit is not None else ''))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    client = RecordingClient()
    monkeypatch.setattr(wire, 'CodexAppServerClient', lambda **kw: client)
    db = SimpleNamespace(get_session_model_config_value=lambda *a: 'original' if resume else None)
    agent = SimpleNamespace(_codex_session=None, session_cwd=str(tmp_path),
                            _cached_system_prompt='Stable instructions', ephemeral_system_prompt=None,
                            _session_db=db, session_id='watermark')
    codex_runtime._ensure_codex_session(agent, [])
    assert agent._codex_session.ensure_started() == ('original' if resume else 'fresh')
    return client


@pytest.mark.parametrize('resume', [False, True])
def test_native_watermark_is_thread_scoped_not_global(tmp_path, monkeypatch, resume):
    client = start_profile(tmp_path, monkeypatch, limit=120000, resume=resume)
    method, params = client.requests[-1]
    assert method == ('thread/resume' if resume else 'thread/start')
    assert params['config']['model_auto_compact_token_limit'] == 120000
    assert not any(m.startswith('config/') or m.startswith('thread/goal/') for m,p in client.requests)
    assert params['developerInstructions'] == 'Stable instructions'


@pytest.mark.parametrize('mode', ['native', 'hermes', 'off'])
def test_unset_watermark_preserves_codex_defaults(tmp_path, monkeypatch, mode):
    assert 'config' not in start_profile(tmp_path, monkeypatch, mode=mode).requests[-1][1]


@pytest.mark.parametrize('mode', ['hermes', 'off'])
def test_native_watermark_does_not_enable_other_modes(tmp_path, monkeypatch, mode):
    assert 'config' not in start_profile(tmp_path, monkeypatch, limit=120000, mode=mode).requests[-1][1]


@pytest.mark.parametrize('limit', ['false', 'true', '0', '-1', '3.5', '"100000"'])
def test_invalid_watermark_fails_before_spawning(tmp_path, monkeypatch, limit):
    with pytest.raises(ValueError, match='codex_auto_compact_token_limit'):
        start_profile(tmp_path, monkeypatch, limit=limit)


def test_next_profile_does_not_inherit_previous_watermark(tmp_path, monkeypatch):
    a=tmp_path/'a'; b=tmp_path/'b'; a.mkdir(); b.mkdir()
    start_profile(a, monkeypatch, limit=120000)
    assert 'config' not in start_profile(b, monkeypatch).requests[-1][1]
