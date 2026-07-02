"""Regression tests for image/photo messages arriving while a gateway session is busy."""

from types import SimpleNamespace
import time

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakeAgent:
    def __init__(self):
        self.steered = []

    def steer(self, text):
        self.steered.append(text)
        return True

    def get_activity_summary(self):
        return {
            "api_call_count": 2,
            "max_iterations": 10,
            "current_tool": "terminal",
            "seconds_since_activity": 0,
            "last_activity_desc": "test-active",
        }


class _FakeAdapter:
    def __init__(self):
        self.sent = []
        self._pending_messages = {}

    async def _send_with_retry(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SimpleNamespace(success=True)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return await self._send_with_retry(
            chat_id,
            content,
            reply_to=reply_to,
            metadata=metadata,
        )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5767139348",
        chat_type="dm",
        user_id="5767139348",
    )


def _photo_event(source=None):
    return MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source or _source(),
        media_urls=["/tmp/busy-photo.png"],
        media_types=["image/png"],
        message_id="photo-1",
    )


def _make_busy_runner(adapter, agent, session_key):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.session_store = None
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._update_prompt_pending = {}
    runner._running_agents = {session_key: agent}
    runner._running_agents_ts = {session_key: time.time()}
    runner._busy_input_mode = "steer"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._startup_restore_in_progress = False
    runner._agent_has_active_subagents = lambda _agent: False
    runner._session_has_compression_in_flight = lambda _session_key: False
    runner._is_user_authorized = lambda _source: True
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._scale_to_zero_note_real_inbound = lambda: None
    return runner


@pytest.mark.asyncio
async def test_busy_steer_photo_is_vision_enriched_and_steered(monkeypatch):
    """A captionless photo during a busy steer run should not fall back opaque.

    The busy path used to look only at event.text.  Telegram photos often have
    empty text, so the image was queued/captured without a vision pre-analysis
    even though the live agent could accept steer text.
    """
    source = _source()
    event = _photo_event(source)
    session_key = "agent:main:telegram:dm:5767139348"
    adapter = _FakeAdapter()
    agent = _FakeAgent()
    runner = _make_busy_runner(adapter, agent, session_key)

    queued = []
    background_calls = []
    vision_calls = []

    runner._queue_or_replace_pending_event = lambda _session_key, _event: queued.append(_event)
    runner._schedule_frontdesk_background_task = lambda **kwargs: background_calls.append(kwargs) or "fd_photo"

    async def fake_vision(user_text, image_paths):
        vision_calls.append((user_text, list(image_paths)))
        return "[vision] the image shows a handwritten deployment checklist"

    runner._enrich_message_with_vision = fake_vision

    import agent.onboarding as onboarding
    monkeypatch.setattr(onboarding, "is_seen", lambda *_args, **_kwargs: True)

    handled = await runner._handle_active_session_busy_message(event, session_key)

    assert handled is True
    assert vision_calls == [("", ["/tmp/busy-photo.png"])]
    assert agent.steered == ["[vision] the image shows a handwritten deployment checklist"]
    assert queued == []
    assert background_calls == []
    assert adapter.sent
    assert "Steered into current run" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_busy_steer_photo_is_not_masked_by_pending_clarify(monkeypatch):
    """A pending clarify must not swallow a media-only photo follow-up.

    Telegram can route the next message to the runner to give an active clarify
    a chance to resolve.  If that next message is a captionless photo, it has no
    clarify text; it must keep flowing to busy steer with vision context instead
    of being left as an opaque pending photo behind the run.
    """
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()

    source = _source()
    event = _photo_event(source)
    adapter = _FakeAdapter()
    agent = _FakeAgent()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.session_store = None
    session_key = runner._session_key_for_source(source)
    runner = _make_busy_runner(adapter, agent, session_key)

    queued = []
    background_calls = []
    vision_calls = []

    runner._queue_or_replace_pending_event = lambda _session_key, _event: queued.append(_event)
    runner._schedule_frontdesk_background_task = lambda **kwargs: background_calls.append(kwargs) or "fd_photo"

    async def fake_vision(user_text, image_paths):
        vision_calls.append((user_text, list(image_paths)))
        return "[vision] the image shows a Telegram screenshot with an error banner"

    runner._enrich_message_with_vision = fake_vision

    cm.register("clarify-image", session_key, "Choose", ["A", "B"])
    cm.mark_awaiting_text("clarify-image")

    response = await runner._handle_message(event)

    assert response is None
    assert vision_calls == [("", ["/tmp/busy-photo.png"])]
    assert agent.steered == ["[vision] the image shows a Telegram screenshot with an error banner"]
    assert queued == []
    assert adapter._pending_messages == {}
    assert background_calls == []
    with cm._lock:
        entry = cm._entries["clarify-image"]
    assert entry.response is None
    assert not entry.event.is_set()
