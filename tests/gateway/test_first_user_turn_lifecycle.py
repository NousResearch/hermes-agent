from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.session_transcript import TranscriptReadError

@pytest.mark.asyncio
@pytest.mark.parametrize('abort', ['history', 'text', 'internal'])
async def test_aborted_or_internal_preparation_preserves_first_user_turn(tmp_path, monkeypatch, abort):
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {})
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='first', user_id='u')
    entry = store.get_or_create_session(source)
    entry.updated_at += timedelta(seconds=1)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *a, **kw: ''
    runner._rehydrate_prompt_pins = AsyncMock()
    runner._hmwa_auto_load_skills = Mock()
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(_store=store, load_transcript=AsyncMock(return_value=[]), update_session=AsyncMock())
    if abort == 'history':
        runner._async_session_store.load_transcript.side_effect = TranscriptReadError('unavailable')
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *a: None
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=None if abort == 'text' else '')
    runner._hmwa_apply_message_timestamp = lambda event, text: (text, text, None)
    runner._delivery_adapter_for = lambda source: None
    runner._bind_adapter_run_generation = lambda *a: None
    event = MessageEvent(source=source, text='', internal=abort == 'internal', auto_skill='alpha')
    await runner._hmwa_prepare_turn(event, source, entry, entry.session_key, entry.session_key, 1)
    assert entry.metadata['first_agent_turn_pending'] is True
    if abort == 'internal':
        runner._hmwa_auto_load_skills.assert_not_called()
        runner.hooks.emit.assert_not_awaited()
    runner.hooks.emit.reset_mock()
    assert await runner._hmwa_open_session(entry, entry.session_key, source) == (False, True)
    runner.hooks.emit.assert_awaited_once()

@pytest.mark.asyncio
async def test_success_only_consumes_user_marker_and_persists(tmp_path):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='first', user_id='u')
    entry = store.get_or_create_session(source)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = store
    async def save(*args, **kwargs):
        store.update_session(*args, **kwargs)
    runner._async_session_store = SimpleNamespace(_store=store, update_session=save)
    user = MessageEvent(source=source, text='hello')
    internal = MessageEvent(source=source, text='', internal=True)
    await runner._hmwa_complete_first_user_turn(internal, entry, {'messages': []})
    await runner._hmwa_complete_first_user_turn(user, entry, {'failed': True})
    assert entry.metadata['first_agent_turn_pending'] is True
    await runner._hmwa_complete_first_user_turn(user, entry, {'messages': []})
    assert 'first_agent_turn_pending' not in entry.metadata
    restored = SessionStore(sessions_dir=tmp_path, config=GatewayConfig()).get_or_create_session(source)
    assert 'first_agent_turn_pending' not in restored.metadata
