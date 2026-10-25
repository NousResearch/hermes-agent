"""The first handoff turn must retain the selected session route (#135150)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.run import GatewayRunner


@pytest.mark.asyncio
@pytest.mark.parametrize('origin_provider,unavailable,existing', [
    ('anthropic', False, False), ('anthropic', False, True),
    ('anthropic', True, False), ('anthropic', True, True), (None, False, True),
])
@pytest.mark.parametrize('route_shape', ['top_level', 'nested', 'billing'])
async def test_handoff_first_turn_uses_origin_model(
    tmp_path, monkeypatch, origin_provider, unavailable, existing, route_shape,
):
    store = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    db = store._db
    route = {'provider': origin_provider, 'base_url': 'https://api.anthropic.com'}
    config = route if route_shape == 'top_level' else (
        {'gateway_runtime': route} if route_shape == 'nested' else {}
    )
    db.create_session('origin-chat', source='desktop', model='origin-model', model_config=config)
    if route_shape == 'billing':
        db._write_sql('UPDATE sessions SET billing_provider = ? WHERE id = ?',
                      (origin_provider, 'origin-chat'))
    row = db.get_session('origin-chat')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='42', user_id='42', chat_type='dm')
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._sessions = {}
    dest = SimpleNamespace(source=source, platform_name='telegram', home=SimpleNamespace(chat_id='42'),
                           effective_thread_id=None)
    runner._handoff_resolve_destination = AsyncMock(return_value=dest)
    key = store.get_or_create_session(source).session_key
    runner._handoff_session_key = lambda *_: key
    runner._evict_cached_agent = lambda *_: None
    runner._release_running_agent_state = lambda *_: None
    monkeypatch.setattr('gateway.run._resolve_gateway_model', lambda *_: 'default-model')
    monkeypatch.setattr('gateway.run._resolve_runtime_agent_kwargs', lambda: {'provider': 'default', 'api_key': 'default-token'})
    if existing:
        old = {'model': 'old-model', 'provider': 'openai', 'api_key': 'old-token'}
        store.set_model_override(key, old)
        runner._session_state(key).conversation.model_override = old
    resolutions = []

    def resolve(provider, *, target_model=None):
        resolutions.append((provider, target_model))
        if unavailable:
            raise RuntimeError('credentials unavailable')
        return {'provider': provider, 'api_key': 'origin-token', 'api_mode': 'chat_completions'}
    monkeypatch.setattr('gateway.run._resolve_runtime_agent_kwargs_for_provider', resolve)
    monkeypatch.setattr('gateway.run._credential_pool_for_provider', lambda *_: None)
    observed = []

    async def handle(event):
        observed.append(runner._resolve_session_agent_runtime(session_key=key))
        return None

    runner._handle_message = handle
    try:
        await runner._process_handoff(row)
        model, runtime = observed[0]
        expected = 'default-model' if unavailable else ('origin-model' if origin_provider else 'old-model')
        assert model == expected, 'handoff silently selected the wrong route'
        expected_provider = 'default' if unavailable else (origin_provider or 'openai')
        assert runtime['provider'] == expected_provider
        if origin_provider:
            assert resolutions
            assert set(resolutions) == {(origin_provider, 'origin-model')}
        else:
            assert resolutions == []  # Keep the destination's already-resolved override.
        assert store.peek_session_id(key) == 'origin-chat'
        persisted = store.get_model_override(key)
        assert persisted['model'] == ('origin-model' if origin_provider else 'old-model')
        assert persisted['provider'] == (origin_provider or 'openai')
        assert 'api_key' not in persisted
        if unavailable:
            assert runner._pre_agent_fallback_notice
        runner._session_state(key).conversation.model_override = None
        resolutions.clear()
        again, restored = runner._resolve_session_agent_runtime(session_key=key)
        assert again == expected
        assert restored['provider'] == expected_provider
        assert resolutions
        assert set(resolutions) == {(origin_provider or 'openai', persisted['model'])}
    finally:
        db.close()
