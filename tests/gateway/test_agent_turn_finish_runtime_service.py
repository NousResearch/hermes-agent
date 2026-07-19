"""Production-path unit tests for promoted agent_turn_finish service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.agent_turn_finish_runtime_service import (
    GatewayFinishedAgentTurn,
    finish_gateway_agent_turn,
)
from gateway.config import Platform
from gateway.session import SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _base_runner(**overrides):
    adapter = SimpleNamespace(
        stop_typing=AsyncMock(),
        send=AsyncMock(),
    )
    store = SimpleNamespace(
        clear_resume_pending=AsyncMock(),
        _save=AsyncMock(),
        _record_gateway_session_peer=AsyncMock(),
        append_to_transcript=AsyncMock(),
        update_session=AsyncMock(),
        has_platform_message_id=AsyncMock(return_value=False),
        reset_session=AsyncMock(return_value=None),
    )
    base = dict(
        _adapter_for_source=MagicMock(return_value=adapter),
        _thread_metadata_for_source=MagicMock(return_value={}),
        _reply_anchor_for_event=MagicMock(return_value=None),
        _is_session_run_current=MagicMock(return_value=True),
        _is_admin_user=MagicMock(return_value=False),
        _clear_restart_failure_count=MagicMock(),
        async_session_store=store,
        _rebind_turn_lease=MagicMock(),
        _sync_telegram_topic_binding=MagicMock(),
        _evict_cached_agent=MagicMock(),
        _clear_conversation_scope=MagicMock(),
        _refresh_agent_cache_message_count=AsyncMock(),
        _should_send_voice_reply=MagicMock(return_value=False),
        _send_voice_reply=AsyncMock(),
        _deliver_media_from_response=AsyncMock(),
        hooks=SimpleNamespace(emit=AsyncMock()),
        _session_db=None,
        _show_reasoning=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _event():
    return SimpleNamespace(text="hi", message_id="m1", raw_message=None)


async def _call(runner, agent_result, **kw):
    defaults = dict(
        runner=runner,
        event=_event(),
        source=_source(),
        session_entry=SimpleNamespace(
            session_id="sid",
            session_key="sk",
        ),
        session_key="sk",
        history=[{"role": "user", "content": "old"}],
        agent_result=agent_result,
        run_start_session_id="sid",
        message_text="hi",
        persist_user_message="hi",
        persist_user_timestamp=None,
        quick_key="qk",
        run_generation=1,
        msg_start_time=0.0,
        platform_name="telegram",
        hook_ctx={"platform": "telegram", "user_id": "u1", "chat_id": "c1",
                  "thread_id": "", "chat_type": "dm", "session_id": "sid",
                  "message": "hi"},
        logger=MagicMock(),
    )
    defaults.update(kw)
    return await finish_gateway_agent_turn(**defaults)


@pytest.mark.asyncio
async def test_finish_returns_final_response_on_happy_path(monkeypatch):
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._sanitize_gateway_final_response",
        lambda _p, text: text,
    )
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._normalize_empty_agent_response",
        lambda _ar, response, history_len=0: response,
    )
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._should_clear_resume_pending_after_turn",
        lambda _ar: False,
    )
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._load_gateway_config",
        lambda: {},
    )
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._resolve_gateway_model",
        lambda: "test-model",
    )
    # no process registry side effects
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        SimpleNamespace(pending_watchers=[], completion_queue=[]),
        raising=False,
    )

    runner = _base_runner()
    finished = await _call(
        runner,
        {
            "final_response": "hello there",
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ],
            "api_calls": 1,
            "history_offset": 1,
            "last_prompt_tokens": 10,
            "agent_persisted": False,
        },
    )

    assert isinstance(finished, GatewayFinishedAgentTurn)
    assert finished.response == "hello there"
    runner.hooks.emit.assert_awaited()
    runner.async_session_store.update_session.assert_awaited()
    runner._refresh_agent_cache_message_count.assert_awaited_once()


@pytest.mark.asyncio
async def test_finish_discards_stale_generation():
    # Production checks the method on the adapter *type*, not only the instance.
    class _StaleAdapter:
        def __init__(self):
            self.pop_calls = []

        def pop_post_delivery_callback(self, key, generation=None):
            self.pop_calls.append((key, generation))

        async def stop_typing(self, _chat_id):
            return None

    adapter = _StaleAdapter()
    runner = _base_runner(
        _is_session_run_current=MagicMock(return_value=False),
        _adapter_for_source=MagicMock(return_value=adapter),
    )

    finished = await _call(
        runner,
        {"final_response": "should not deliver", "messages": []},
    )

    assert finished.response is None
    assert adapter.pop_calls == [("qk", 1)]


@pytest.mark.asyncio
async def test_finish_intentional_silence_returns_empty_string(monkeypatch):
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._is_gateway_hidden_reasoning_incomplete_turn",
        lambda _ar: False,
    )
    monkeypatch.setattr(
        "gateway.response_filters.is_intentional_silence_agent_result",
        lambda _ar, _response: True,
    )
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._should_clear_resume_pending_after_turn",
        lambda _ar: False,
    )
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._load_gateway_config",
        lambda: {},
    )
    monkeypatch.setattr(
        "gateway.agent_turn_finish_runtime_service._resolve_gateway_model",
        lambda: "test-model",
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        SimpleNamespace(pending_watchers=[], completion_queue=[]),
        raising=False,
    )

    runner = _base_runner()
    finished = await _call(
        runner,
        {
            "final_response": "NO_REPLY",
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "NO_REPLY"},
            ],
            "history_offset": 1,
            "last_prompt_tokens": 0,
            "agent_persisted": False,
        },
    )

    # Intentional silence is empty string (not None) for adapters.
    assert finished.response == ""
