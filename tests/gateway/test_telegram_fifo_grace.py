"""Telegram grace applies before durable-FIFO and relay busy preemption (#152516).

The old fast path queued a quick follow-up; authority and relay admissions bypassed
that guard and interrupted the running agent instead.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_ingress_live_turn import _route


def _busy_telegram_turn():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="grace-chat", chat_type="dm", user_id="u1")
    event = MessageEvent(text="one more detail", message_type=MessageType.TEXT, source=source)
    key = "telegram:grace-chat"
    runner = object.__new__(GatewayRunner)
    state = SimpleNamespace(turn=SimpleNamespace(started_ts=100.0, agent=object()))
    runner._draining = False
    runner._peek_session_state = lambda _key: state
    runner._effective_busy_input_mode = lambda _source: "interrupt"
    runner._effective_busy_text_mode = lambda _source: "interrupt"
    runner._delivery_adapter_for = lambda _source: object()
    runner._session_key_for_source = lambda _source: key
    runner._is_session_running = lambda _key: True
    runner._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    runner._hm_post_admission_consume = AsyncMock(return_value=(False, None))
    runner._resolve_busy_steer_or_redirect = AsyncMock(
        side_effect=AssertionError("grace follow-up preempted the running turn")
    )
    runner._busy_interrupt_and_ack = AsyncMock()
    return runner, event, source, key


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["durable", "relay"])
async def test_telegram_followup_within_grace_uses_durable_fifo(monkeypatch, ingress):
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3")
    runner, event, source, key = _busy_telegram_turn()

    with (
        patch("gateway.run_inbound.time.time", return_value=102.0),
        patch("gateway.session_ingress.admit_message", new_callable=AsyncMock, return_value="queued") as admit,
    ):
        if ingress == "durable":
            assert await runner._hm_admit_to_fifo(object(), event, source, key, True) == (True, "queued")
            admit.assert_awaited_once()
        else:
            assert await _route(runner, event) is False  # relay caller admits its own FIFO row
            admit.assert_not_awaited()

    runner._resolve_busy_steer_or_redirect.assert_not_awaited()
    runner._busy_interrupt_and_ack.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_followup_after_grace_can_still_preempt(monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3")
    runner, event, source, key = _busy_telegram_turn()
    runner._resolve_busy_steer_or_redirect = AsyncMock(
        return_value=SimpleNamespace(steered=False, redirected=False)
    )

    with patch("gateway.run_inbound.time.time", return_value=105.0):
        assert await runner._hm_busy_preempt(event, source, key) is False

    runner._resolve_busy_steer_or_redirect.assert_awaited_once()
    runner._busy_interrupt_and_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_long_followup_uses_send_time_before_default_batch_flush(monkeypatch):
    """A +2s Telegram text must not interrupt when its default 1.5s batch flushes at +3.5s."""
    from datetime import UTC, datetime

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", raising=False)
    runner, event, source, key = _busy_telegram_turn()
    event.text = "One more detail " * 80  # >1,024 chars: default long-text delay, not the fast tier
    event.timestamp = datetime.fromtimestamp(102.0, UTC)
    event.raw_message = SimpleNamespace(date=event.timestamp)  # Telegram's provider send time
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    delay = adapter._text_batch_delay_for(event)
    assert delay == pytest.approx(1.5)

    with patch("gateway.run_inbound.time.time", return_value=102.0 + delay):
        assert await runner._hm_busy_preempt(event, source, key) is False

    runner._resolve_busy_steer_or_redirect.assert_not_awaited()
    runner._busy_interrupt_and_ack.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("sent_at, edited_at, waits", [(95.0, 102.0, True), (101.0, 110.0, False)])
async def test_telegram_edited_followup_is_judged_by_its_edit_time(monkeypatch, sent_at, edited_at, waits):
    """An edit keeps the message's original ``date``; the grace reads the edit's own time instead."""
    from datetime import UTC, datetime

    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)
    runner, event, source, key = _busy_telegram_turn()
    runner._resolve_busy_steer_or_redirect.side_effect = None  # the out-of-grace edit is expected to preempt
    event.timestamp = datetime.fromtimestamp(sent_at, UTC)  # the turn started at 100
    event.raw_message = SimpleNamespace(date=event.timestamp, edit_date=datetime.fromtimestamp(edited_at, UTC))
    with patch("gateway.run_inbound.time.time", return_value=edited_at + 0.2):
        await runner._hm_busy_preempt(event, source, key)
    preempted = runner._busy_interrupt_and_ack.await_count + runner._resolve_busy_steer_or_redirect.await_count
    assert (preempted == 0) is waits
