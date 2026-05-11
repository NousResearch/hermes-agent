from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner


def _entry(*, minutes_idle: int, session_key: str = "agent:main:telegram:dm:123"):
    now = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        session_key=session_key,
        session_id="sid-123",
        updated_at=now - timedelta(minutes=minutes_idle),
        origin=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="123", thread_id=None),
        suspended=False,
        expiry_finalized=False,
    ), now


def test_next_afk_followup_selects_first_unfired_threshold():
    from gateway.afk_followup import DEFAULT_AFK_THRESHOLDS_MINUTES, next_afk_followup

    entry, now = _entry(minutes_idle=16)

    decision = next_afk_followup(
        entry,
        now=now,
        thresholds_minutes=DEFAULT_AFK_THRESHOLDS_MINUTES,
        fired_thresholds={5},
        running_session_keys=set(),
        queued_session_keys=set(),
    )

    assert decision is not None
    assert decision.threshold_minutes == 15
    assert decision.idle_label == "15m"


def test_next_afk_followup_skips_running_or_queued_sessions():
    from gateway.afk_followup import DEFAULT_AFK_THRESHOLDS_MINUTES, next_afk_followup

    entry, now = _entry(minutes_idle=61)

    assert next_afk_followup(
        entry,
        now=now,
        thresholds_minutes=DEFAULT_AFK_THRESHOLDS_MINUTES,
        fired_thresholds=set(),
        running_session_keys={entry.session_key},
        queued_session_keys=set(),
    ) is None

    assert next_afk_followup(
        entry,
        now=now,
        thresholds_minutes=DEFAULT_AFK_THRESHOLDS_MINUTES,
        fired_thresholds=set(),
        running_session_keys=set(),
        queued_session_keys={entry.session_key},
    ) is None


def test_build_afk_followup_prompt_is_safety_scoped_and_not_silent():
    from gateway.afk_followup import build_afk_followup_prompt

    prompt = build_afk_followup_prompt("15m", cycle_index=1)

    assert "[Automated AFK follow-up]" in prompt
    assert "The user has been AFK for 15m." in prompt
    assert "top 3 loose ends" in prompt
    assert "first" in prompt.lower()
    assert "destructive" in prompt
    assert "credential" in prompt
    assert "deployment" in prompt
    assert "reply" in prompt.lower()
    assert "Stay silent" not in prompt
    assert "SILENT" not in prompt


@pytest.mark.asyncio
async def test_afk_followup_tick_injects_internal_virtual_turn_once(monkeypatch):
    from gateway.afk_followup import AfkFollowupConfig

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._afk_followup_config = AfkFollowupConfig(enabled=True, thresholds_minutes=(5, 15), interval_seconds=1)
    runner._afk_fired_thresholds = {}
    runner._running_agents = {}
    runner._queued_events = {}

    entry, now = _entry(minutes_idle=6)
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={entry.session_key: entry},
    )

    sent_events = []

    class Adapter:
        def __init__(self):
            self._pending_messages = {}

        async def handle_message(self, event):
            sent_events.append(event)

    runner.adapters = {Platform.TELEGRAM: Adapter()}
    monkeypatch.setattr("gateway.run.datetime", SimpleNamespace(now=lambda tz=None: now))

    await runner._afk_followup_tick()
    await runner._afk_followup_tick()

    assert len(sent_events) == 1
    event = sent_events[0]
    assert event.internal is True
    assert event.message_type == MessageType.TEXT
    assert "[Automated AFK follow-up]" in event.text
    assert "AFK for 5m" in event.text
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "123"
    assert runner._afk_fired_thresholds[entry.session_key] == {5}


@pytest.mark.asyncio
async def test_afk_followup_tick_skips_adapter_pending_message(monkeypatch):
    from gateway.afk_followup import AfkFollowupConfig

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._afk_followup_config = AfkFollowupConfig(enabled=True, thresholds_minutes=(5,), interval_seconds=1)
    runner._afk_fired_thresholds = {}
    runner._running_agents = {}
    runner._queued_events = {}

    entry, now = _entry(minutes_idle=10)
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={entry.session_key: entry},
    )
    adapter = SimpleNamespace(
        _pending_messages={entry.session_key: MessageEvent(text="queued", source=entry.origin)},
        handle_message=AsyncMock(),
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    monkeypatch.setattr("gateway.run.datetime", SimpleNamespace(now=lambda tz=None: now))

    await runner._afk_followup_tick()

    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_afk_followup_tick_does_not_inject_when_disabled():
    from gateway.afk_followup import AfkFollowupConfig

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._afk_followup_config = AfkFollowupConfig(enabled=False, thresholds_minutes=(5,), interval_seconds=1)
    runner._afk_fired_thresholds = {}
    runner._running_agents = {}
    runner._queued_events = {}

    entry, _now = _entry(minutes_idle=10)
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={entry.session_key: entry},
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: adapter}

    await runner._afk_followup_tick()

    adapter.handle_message.assert_not_called()


class _TestAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="token", extra={}),
            Platform.TELEGRAM,
        )
        self.sent = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content, kwargs))
        return SendResult(success=True, message_id="sent-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_internal_afk_response_delivers_even_if_user_message_arrived_mid_run():
    adapter = _TestAdapter()
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="u1",
        thread_id=None,
    )
    event = MessageEvent(
        text="[Automated AFK follow-up]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
    )
    session_key = "agent:main:telegram:dm:123"
    interrupt = __import__("asyncio").Event()
    interrupt.set()
    adapter._active_sessions[session_key] = interrupt
    adapter._pending_messages[session_key] = MessageEvent(
        text="user returned",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter.set_message_handler(AsyncMock(return_value="AFK task completed: checked logs."))

    await adapter._process_message_background(event, session_key)

    assert adapter.sent[0][0] == "123"
    assert adapter.sent[0][1] == "AFK task completed: checked logs."
    assert adapter.sent[0][2].get("reply_to") is None
    assert len(adapter.sent) >= 1
