"""Regression test for streamed Telegram trailing-footer delivery metadata.

When the agent already streamed the main body (``already_sent=True``),
``GatewayRunner._handle_message_with_agent`` sends the runtime footer as a small
trailing message. That footer is still part of the terminal reply, so it must
carry ``suppress_post_send_typing=True`` to avoid re-triggering Telegram's
typing bubble after the answer already landed.
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="208214988",
        user_name="tester",
        chat_type="dm",
        thread_id="17585",
    )


def _make_event(text: str = "hi") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    r = cast(Any, runner)
    r.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.stop_typing = AsyncMock()
    r.adapters = {Platform.TELEGRAM: adapter}
    r.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    source = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    r.session_store = MagicMock()
    r.session_store.get_or_create_session.return_value = session_entry
    r.session_store.load_transcript.return_value = []
    r.session_store.append_to_transcript = MagicMock()
    r.session_store.update_session = MagicMock()
    r.session_store._save = MagicMock()

    r._session_db = None
    r._voice_mode = {}
    r._reasoning_config = None
    r._provider_routing = {}
    r._fallback_model = None
    r._show_reasoning = False
    r._pending_model_notes = {}
    r._session_model_overrides = {}
    r._pending_messages = {}
    r._pending_approvals = {}
    r._recover_telegram_topic_thread_id = lambda source: None
    r._cache_session_source = lambda *_a, **_kw: None
    r._is_telegram_topic_lane = lambda source: False
    r._set_session_env = lambda context: []
    r._bind_adapter_run_generation = lambda *_a, **_kw: None
    r._is_session_run_current = lambda *_a, **_kw: True
    r._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    r._emit_gateway_run_progress = AsyncMock()
    r._run_process_watcher = AsyncMock()
    r._deliver_media_from_response = AsyncMock()
    r._should_send_voice_reply = lambda *_a, **_kw: False
    r._clear_restart_failure_count = lambda *_a, **_kw: None
    r._evict_cached_agent = lambda *_a, **_kw: None
    r._set_session_reasoning_override = lambda *_a, **_kw: None

    from gateway.run import GatewayRunner as _GR

    r._thread_metadata_for_source = _GR._thread_metadata_for_source.__get__(runner, _GR)
    r._reply_anchor_for_event = _GR._reply_anchor_for_event
    return runner


@pytest.mark.asyncio
async def test_streamed_trailing_footer_suppresses_post_send_typing(monkeypatch):
    import gateway.run as gateway_run
    import gateway.runtime_footer as runtime_footer

    runner = _make_runner()
    event = _make_event()
    session_key = build_session_key(event.source)

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "main body",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "main body"},
            ],
            "history_offset": 0,
            "already_sent": True,
            "failed": False,
            "model": "openai/gpt-5.4",
            "last_prompt_tokens": 25,
            "context_length": 100,
            "api_calls": 1,
            "session_id": "sess-1",
            "tools": [],
        }
    )

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(runtime_footer, "build_footer_line", lambda **_kw: "gpt-5.4 · 25%")

    result = await runner._handle_message_with_agent(
        event,
        event.source,
        session_key,
        run_generation=1,
    )

    r = cast(Any, runner)
    assert result is None
    r._deliver_media_from_response.assert_awaited_once_with(
        "main body", event, r.adapters[Platform.TELEGRAM]
    )
    assert r.adapters[Platform.TELEGRAM].send.await_count >= 1

    footer_call = r.adapters[Platform.TELEGRAM].send.await_args_list[-1]
    send_chat_id = footer_call.kwargs.get("chat_id", footer_call.args[0] if footer_call.args else None)
    send_content = footer_call.kwargs.get("content", footer_call.args[1] if len(footer_call.args) > 1 else None)
    send_metadata = footer_call.kwargs.get("metadata", footer_call.args[2] if len(footer_call.args) > 2 else None)
    assert send_chat_id == event.source.chat_id
    assert send_content == "gpt-5.4 · 25%"
    assert send_metadata["thread_id"] == "17585"
    assert send_metadata["suppress_post_send_typing"] is True

    fresh_meta = runner._thread_metadata_for_source(
        event.source, runner._reply_anchor_for_event(event)
    )
    assert fresh_meta is not None
    assert "suppress_post_send_typing" not in fresh_meta
