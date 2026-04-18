from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.busy_followup_runtime_service import handle_gateway_busy_followup
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_source(*, chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        user_name="發發發",
        chat_id="179033731" if chat_type == "dm" else "726109087",
        chat_type=chat_type,
    )


def _make_event(text: str, *, chat_type: str = "dm") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(chat_type=chat_type),
        message_id="m1",
    )


def _make_runner(*, running_agent, busy_input_mode: str = "smart", adapter=None):
    source = _make_source()
    session_entry = SimpleNamespace(session_id="sess-1")
    return SimpleNamespace(
        _running_agents={"qq_napcat:dm:179033731": running_agent},
        _running_agents_ts={},
        _pending_messages={},
        adapters={Platform.QQ_NAPCAT: adapter} if adapter is not None else {},
        session_store=SimpleNamespace(
            get_or_create_session=MagicMock(return_value=session_entry),
            load_transcript=MagicMock(return_value=[]),
        ),
        _get_busy_input_mode=lambda _platform: busy_input_mode,
        _busy_followup_force_queue_reason=lambda _session_key, _running_agent: "",
        _admin_only_message=lambda _source, _action: None,
        _handle_status_command=AsyncMock(return_value="status"),
        _handle_reset_command=AsyncMock(return_value="reset"),
        _handle_approve_command=AsyncMock(return_value="approve"),
        _handle_deny_command=AsyncMock(return_value="deny"),
    )


@pytest.mark.asyncio
async def test_busy_followup_smart_queue_uses_fallback_ack_without_adapter_specific_ack(
    monkeypatch,
):
    event = _make_event("继续")
    queued = {}

    class _Adapter:
        def queue_message(self, session_key, queued_event):
            queued["session_key"] = session_key
            queued["event"] = queued_event

    adapter = _Adapter()
    runner = _make_runner(running_agent=MagicMock(), adapter=adapter)
    logger = MagicMock()

    monkeypatch.setattr(
        "gateway.busy_followup_runtime_service.try_handle_direct_gateway_shortcuts",
        lambda *args, **kwargs: None,
    )

    result = await handle_gateway_busy_followup(
        runner=runner,
        event=event,
        source=event.source,
        session_key="qq_napcat:dm:179033731",
        logger=logger,
        pending_agent_sentinel=object(),
        truncate_status_preview=lambda value: str(value),
        fallback_busy_ack=lambda source, text: f"fallback:{text}",
    )

    assert result.handled is True
    assert result.response == "fallback:继续"
    assert queued["session_key"] == "qq_napcat:dm:179033731"
    assert queued["event"] is event


@pytest.mark.asyncio
async def test_busy_followup_force_queue_uses_fallback_ack_without_adapter_specific_ack(
    monkeypatch,
):
    event = _make_event("继续")
    queued = {}

    class _Adapter:
        def queue_message(self, session_key, queued_event):
            queued["session_key"] = session_key
            queued["event"] = queued_event

    adapter = _Adapter()
    runner = _make_runner(running_agent=MagicMock(), adapter=adapter)
    runner._busy_followup_force_queue_reason = lambda _session_key, _running_agent: "background_job"
    logger = MagicMock()

    monkeypatch.setattr(
        "gateway.busy_followup_runtime_service.try_handle_direct_gateway_shortcuts",
        lambda *args, **kwargs: None,
    )

    result = await handle_gateway_busy_followup(
        runner=runner,
        event=event,
        source=event.source,
        session_key="qq_napcat:dm:179033731",
        logger=logger,
        pending_agent_sentinel=object(),
        truncate_status_preview=lambda value: str(value),
        fallback_busy_ack=lambda source, text: f"fallback:{text}",
    )

    assert result.handled is True
    assert result.response == "fallback:继续"
    assert queued["session_key"] == "qq_napcat:dm:179033731"
    assert queued["event"] is event


@pytest.mark.asyncio
async def test_busy_followup_pending_sentinel_uses_fallback_ack_without_adapter_specific_ack(
    monkeypatch,
):
    pending_agent = object()
    event = _make_event("继续")
    queued = {}

    class _Adapter:
        def queue_message(self, session_key, queued_event):
            queued["session_key"] = session_key
            queued["event"] = queued_event

    adapter = _Adapter()
    runner = _make_runner(running_agent=pending_agent, adapter=adapter)
    logger = MagicMock()

    monkeypatch.setattr(
        "gateway.busy_followup_runtime_service.try_handle_direct_gateway_shortcuts",
        lambda *args, **kwargs: None,
    )

    result = await handle_gateway_busy_followup(
        runner=runner,
        event=event,
        source=event.source,
        session_key="qq_napcat:dm:179033731",
        logger=logger,
        pending_agent_sentinel=pending_agent,
        truncate_status_preview=lambda value: str(value),
        fallback_busy_ack=lambda source, text: f"fallback:{text}",
    )

    assert result.handled is True
    assert result.response == "fallback:继续"
    assert queued["session_key"] == "qq_napcat:dm:179033731"
    assert queued["event"] is event


@pytest.mark.asyncio
async def test_queue_command_preserves_followup_metadata_when_rebuilding_text_event(
    monkeypatch,
):
    event = _make_event("/queue @马嘎 后面继续", chat_type="group")
    event.raw_message = {"source": "qq"}
    event.metadata = {
        "group_trigger_reason": "require_mention_disabled",
        "explicit_group_trigger": True,
        "explicit_group_trigger_reason": "bot_mention",
        "explicit_addressed": True,
        "address_reason": "bot_mention",
        "requires_reply": True,
    }
    event.reply_to_message_id = "bot-msg-9"
    event.reply_to_text = "上一条"
    queued = {}

    class _Adapter:
        def queue_message(self, session_key, queued_event):
            queued["session_key"] = session_key
            queued["event"] = queued_event

    adapter = _Adapter()
    runner = _make_runner(running_agent=MagicMock(), busy_input_mode="queue", adapter=adapter)
    runner._running_agents = {"qq_napcat:group:726109087:179033731": MagicMock()}

    monkeypatch.setattr(
        "gateway.busy_followup_runtime_service.try_handle_direct_gateway_shortcuts",
        lambda *args, **kwargs: None,
    )

    result = await handle_gateway_busy_followup(
        runner=runner,
        event=event,
        source=event.source,
        session_key="qq_napcat:group:726109087:179033731",
        logger=MagicMock(),
        pending_agent_sentinel=object(),
        truncate_status_preview=lambda value: str(value),
        fallback_busy_ack=lambda source, text: f"fallback:{text}",
    )

    assert result.handled is True
    assert result.response == "Queued for the next turn."
    assert queued["session_key"] == "qq_napcat:group:726109087:179033731"
    assert queued["event"].text == "@马嘎 后面继续"
    assert queued["event"].raw_message == {"source": "qq"}
    assert queued["event"].metadata["explicit_addressed"] is True
    assert queued["event"].reply_to_message_id == "bot-msg-9"
    assert queued["event"].reply_to_text == "上一条"
