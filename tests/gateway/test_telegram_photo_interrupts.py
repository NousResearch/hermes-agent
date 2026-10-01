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
@pytest.mark.parametrize("path", ["photo", "startup", "grace"])
@pytest.mark.parametrize("boundary", ["same", "sender", "plugin", "control", "reply", "queued", "orphaned", "voice"])
async def test_busy_coalescing_preserves_independent_pending_events(path, boundary, monkeypatch):
    import time

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
        expected = [replace(
            deepcopy(pending), text="first\n\nsecond" if path == "photo" else "first\nsecond",
            media_urls=pending.media_urls + incoming.media_urls,
            media_types=pending.media_types + incoming.media_types,
            media_text_inlined=pending.media_text_inlined + incoming.media_text_inlined,
            merged_message_ids=[*pending.merged_message_ids, incoming.message_id],
        )]

    await runner._handle_message(incoming)

    actual = [adapter._pending_messages[key], *list(runner._overflow_queue(key) or [])]
    assert actual == expected
