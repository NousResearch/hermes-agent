from copy import deepcopy
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.run import _AGENT_PENDING_SENTINEL, GatewayRunner


class _PendingAdapter:
    def __init__(self):
        self._pending_messages = {}


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: _PendingAdapter()}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
async def test_handle_message_does_not_priority_interrupt_photo_followup():
    runner = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")
    session_key = build_session_key(source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    event = MessageEvent(
        text="caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo-a.jpg"],
        media_types=["image/jpeg"],
    )

    result = await runner._handle_message(event)

    assert result is None
    running_agent.interrupt.assert_not_called()
    assert runner.adapters[Platform.TELEGRAM]._pending_messages[session_key] is event


@pytest.mark.asyncio
@pytest.mark.parametrize("path,boundary,kind", [
    *[pytest.param(path, boundary, MessageType.PHOTO if path == "photo" else MessageType.TEXT,
                   id=f"{boundary}-{path}")
      for boundary in ("same", "sender", "plugin", "control", "reply", "queued", "orphaned", "voice")
      for path in ("photo", "startup", "grace")],
    pytest.param("fifo", "sender", MessageType.TEXT, id="text"),
    pytest.param("fifo", "sender", MessageType.PHOTO, id="photo"),
    pytest.param("debounce-slot", "sender", MessageType.TEXT, id="other-sender-text"),
    pytest.param("debounce-slot", "empty", MessageType.TEXT, id="empty-slot"),
])
async def test_busy_coalescing_preserves_independent_pending_events(path, boundary, kind, monkeypatch):
    import time

    if path == "fifo":
        from tests.gateway.test_queue_consumption import TestBusyInputModeQueueFifo

        fixture = TestBusyInputModeQueueFifo()
        runner, adapter = fixture._make_runner_and_adapter()
        session_key = "telegram:group:shared"
        runner._queue_or_replace_pending_event(
            session_key, fixture._media_event("/tmp/a.jpg", "image/jpeg", MessageType.PHOTO, text="look at this"))
        followup = (fixture._text_event("unrelated question", user_id="u2") if kind == MessageType.TEXT
                    else fixture._media_event("/tmp/b.jpg", "image/jpeg", MessageType.PHOTO, text="mine", user_id="u2"))
        runner._queue_or_replace_pending_event(session_key, followup)
        head = adapter._pending_messages[session_key]
        assert ((head.text, head.media_urls, head.source.user_id), runner._queued_events.get(session_key, [])) == (
            ("look at this", ["/tmp/a.jpg"], "u1"), [followup],
        )
        return

    if path == "debounce-slot":
        import asyncio
        import types
        from tests.gateway.test_active_session_text_merge import _make_adapter, _make_event

        pending_sender = "u1" if boundary == "sender" else None
        adapter = _make_adapter()
        monkeypatch.setattr(adapter, "_event_session_key", lambda event: "shared")
        queued_text_calls: list[tuple[str, str]] = []
        adapter.gateway_runner = types.SimpleNamespace(
            _queue_or_replace_pending_event=lambda session_key, event: queued_text_calls.append((session_key, event.text)),
        )
        adapter._active_sessions["shared"] = asyncio.Event()
        if pending_sender is not None:
            adapter._pending_messages["shared"] = _make_event("one", chat_type="group", user_id=pending_sender)
        for text, sender in [("two", "u2"), ("three", "u3")]:
            await adapter.handle_message(_make_event(text, chat_type="group", user_id=sender))
        await adapter._flush_text_debounce_now("shared")
        pending = adapter._pending_messages.get("shared")
        assert (pending.text if pending else None, queued_text_calls, adapter._text_debounce) == (
            ("one", [("shared", "two"), ("shared", "three")], {}) if pending_sender is not None
            else (None, [("shared", "two"), ("shared", "three")], {})
        )
        return

    runner = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100", chat_type="group",
        user_id="user-a", thread_id="thread",
    )
    key = build_session_key(source)
    runner._running_agents[key] = _AGENT_PENDING_SENTINEL if path == "startup" else MagicMock()
    if path == "grace":
        monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "10000")
        runner._session_state(key).turn.started_ts = time.time()
    pending = MessageEvent(
        text="first", source=source, message_id="first",
        message_type=MessageType.PHOTO if path == "photo" else MessageType.TEXT,
        media_urls=["/tmp/first.jpg"] if path == "photo" else [],
        media_types=["image/jpeg"] if path == "photo" else [],
        media_text_inlined=[False] if path == "photo" else [],
    )
    incoming = MessageEvent(
        text="second", source=source, message_id="second",
        message_type=MessageType.PHOTO if path == "photo" else MessageType.TEXT,
        media_urls=["/tmp/second.jpg"] if path == "photo" else [],
        media_types=["image/jpeg"] if path == "photo" else [],
        media_text_inlined=[False] if path == "photo" else [],
    )
    if boundary == "sender":
        pending.source = replace(source, user_id="user-b")
    if boundary == "plugin":
        pending.internal = True
        pending.metadata = {"hermes_plugin_id": "demo", "hermes_plugin_injection": True}
    if boundary == "control":
        pending.allow_gateway_control = False
    if boundary == "voice":
        pending.message_type = MessageType.VOICE
        pending.media_urls, pending.media_types = ["/tmp/voice.ogg"], ["audio/ogg"]
        pending.media_text_inlined = [False]
    if boundary == "reply":
        pending.reply_to_message_id = "quote-a"
        incoming.reply_to_message_id = "quote-b"
    adapter = runner.adapters[Platform.TELEGRAM]
    if boundary == "orphaned":
        runner._session_state(key).conversation.queued_events.append(pending)
    else:
        adapter._pending_messages[key] = pending
    queued = MessageEvent(text="between", source=source, message_id="between")
    if boundary == "queued":
        runner._enqueue_fifo(key, queued, adapter)
    expected = deepcopy([pending] + ([queued] if boundary == "queued" else []) + [incoming])
    if boundary == "same":
        expected = [
            replace(
                deepcopy(pending),
                text="first\n\nsecond" if path == "photo" else "first\nsecond",
                media_urls=pending.media_urls + incoming.media_urls,
                media_types=pending.media_types + incoming.media_types,
                media_text_inlined=pending.media_text_inlined
                + incoming.media_text_inlined,
                merged_message_ids=[*pending.merged_message_ids, incoming.message_id],
            )
        ]

    await runner._handle_message(incoming)

    actual = [adapter._pending_messages[key], *list(runner._overflow_queue(key) or [])]
    assert actual == expected
