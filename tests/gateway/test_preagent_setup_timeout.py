"""Pre-agent setup bound and PENDING busy-ack (#138358 complementary).

A turn wedged in session-resolve / turn-prepare / agent:start hooks never reaches
the inactivity watchdog. Follow-ups against a PENDING sentinel were queued with
no user-visible notice.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource


KEY = "agent:main:telegram:dm:138358-preagent"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="138358",
        chat_type="dm",
        user_id="user-138358",
        user_name="reporter",
    )


def _event(text: str = "hello after cron") -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source(), message_id="m1")


def _bare_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    return runner


@pytest.mark.asyncio
async def test_hung_preagent_resolve_is_cancelled_and_notified(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0.05")
    runner = _bare_runner()

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(10)
        raise AssertionError("hung resolve should have been cancelled")

    runner._hmwa_resolve_session = _hang

    with caplog.at_level("ERROR", logger="gateway.run"):
        started = time.monotonic()
        result = await GatewayRunner._handle_message_with_agent(
            runner, _event(), _source(), KEY, 1,
        )
        elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert "Turn setup stalled" in result
    assert "Please resend" in result
    assert "Pre-agent setup timed out" in caplog.text
    assert KEY in caplog.text
    runner.hooks.emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_preagent_timeout_zero_does_not_bound_setup(monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    runner = _bare_runner()
    seen = asyncio.Event()

    async def _quick(*_args, **_kwargs):
        seen.set()
        return None

    runner._hmwa_resolve_session = _quick
    result = await GatewayRunner._handle_message_with_agent(
        runner, _event(), _source(), KEY, 1,
    )
    assert seen.is_set()
    assert result is None


@pytest.mark.asyncio
async def test_pending_followup_acks_after_stall_timeout(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_STALL_TIMEOUT", "60")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    runner = _bare_runner()
    runner._draining = False
    runner._hm_busy_slash_or_photo = AsyncMock(return_value=(False, None))
    runner._effective_busy_input_mode = lambda _source: "interrupt"
    runner._hm_busy_telegram_grace_queue = lambda *_args, **_kwargs: False
    runner._hm_merge_pending_for_source = lambda *_args, **_kwargs: None
    runner._peek_session_state = lambda _key: SimpleNamespace(
        turn=SimpleNamespace(
            agent=_AGENT_PENDING_SENTINEL,
            started_ts=time.time() - 400,
            busy_ack_ts=0.0,
        ),
    )
    runner._session_state = runner._peek_session_state

    result = await GatewayRunner._hm_handle_running_session_message(
        runner, _event("later"), _source(), KEY,
    )
    assert "Still starting this turn" in result
    assert "6 min" in result
    assert "queued" in result.lower()


@pytest.mark.asyncio
async def test_fresh_pending_followup_stays_silent(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_STALL_TIMEOUT", "300")
    runner = _bare_runner()
    runner._draining = False
    runner._hm_busy_slash_or_photo = AsyncMock(return_value=(False, None))
    runner._effective_busy_input_mode = lambda _source: "interrupt"
    runner._hm_busy_telegram_grace_queue = lambda *_args, **_kwargs: False
    merged = []
    runner._hm_merge_pending_for_source = lambda *_args, **_kwargs: merged.append(True)
    runner._peek_session_state = lambda _key: SimpleNamespace(
        turn=SimpleNamespace(
            agent=_AGENT_PENDING_SENTINEL,
            started_ts=time.time() - 5,
            busy_ack_ts=0.0,
        ),
    )
    runner._session_state = runner._peek_session_state

    result = await GatewayRunner._hm_handle_running_session_message(
        runner, _event("later"), _source(), KEY,
    )
    assert result is None
    assert merged == [True]
