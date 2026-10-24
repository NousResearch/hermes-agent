"""An OAuth token refresh must not silently bypass a session's proxy."""
from types import SimpleNamespace

from agent.client_lifecycle import ClientLifecycleMixin
from hermes_cli import auth
from hermes_cli.auth_constants import DEFAULT_CODEX_BASE_URL


def test_codex_refresh_keeps_session_endpoint(monkeypatch):
    calls = []
    def credentials(**kwargs):
        return {'api_key': 'fresh' if kwargs.get('force_refresh') else 'old',
                'base_url': DEFAULT_CODEX_BASE_URL}
    monkeypatch.setattr(auth, 'resolve_codex_runtime_credentials', credentials)
    agent = SimpleNamespace(
        api_mode='codex_responses', provider='openai-codex', api_key='old',
        base_url='http://127.0.0.1:8787/v1',
        _adopt_openai_credentials=lambda key, url, **kw: calls.append((key, url)) or True,
    )
    assert ClientLifecycleMixin._try_refresh_codex_client_credentials(agent, force=True)
    assert calls == [('fresh', agent.base_url)]
