"""Regression: a rapid follow-up on a just-started Slack turn must not interrupt.

Root cause of the "Red always starts with ⚡ Interrupting current task" report.

``_handle_message`` claims the session with ``_AGENT_PENDING_SENTINEL`` *before*
the many awaits that precede real agent construction (hooks, vision enrichment,
STT, permalink/thread hydration, session-hygiene compression).  A second
physical Slack event for the same session that lands inside that window sees a
busy session and, with ``busy_input_mode: interrupt``, emits the interrupt ack.

The gateway already has a grace window that queues such a follow-up *without*
interrupting — but it is gated on ``source.platform == Platform.TELEGRAM``, so
Slack never gets it.  These tests pin the intended behavior:

1. Telegram (existing behavior) — follow-up inside the grace window queues.
2. Slack (the bug) — the identical shape must also queue, not interrupt.
"""

from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, build_session_key


class _PendingAdapter:
    def __init__(self):
        self._pending_messages = {}


def _make_runner(platform: Platform) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {platform: _PendingAdapter()}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    return runner


def _source(platform: Platform) -> SessionSource:
    if platform is Platform.SLACK:
        return SessionSource(
            platform=platform,
            chat_id="C0BF1EYUA9H",
            chat_type="group",
            user_id="U0374GH838U",
            scope_id="T025KND0E",
            thread_id="1787104575.961829",
        )
    return SessionSource(
        platform=platform, chat_id="12345", chat_type="dm", user_id="u1"
    )


async def _claim_and_followup(platform: Platform):
    """Claim a session with the pending sentinel, then deliver a follow-up."""
    runner = _make_runner(platform)
    src = _source(platform)
    session_key = build_session_key(src)

    # Reproduce the real pre-agent claim from _handle_message: the sentinel is
    # installed and started_ts is set to "just now", exactly the state a second
    # near-simultaneous event observes.
    state = runner._session_state(session_key)
    state.turn.agent = _AGENT_PENDING_SENTINEL
    import time

    state.turn.started_ts = time.time()

    event = MessageEvent(
        text="같은 메시지가 1초 안에 두 번 도착",
        message_type=MessageType.TEXT,
        source=src,
    )
    result = await runner._handle_message(event)
    return runner, session_key, event, result


@pytest.mark.asyncio
async def test_telegram_followup_inside_grace_queues_without_interrupt():
    """Baseline: the existing Telegram grace window queues instead of acking."""
    runner, session_key, event, result = await _claim_and_followup(Platform.TELEGRAM)

    assert result is None, "grace-window follow-up must not produce a busy ack"
    pending = runner.adapters[Platform.TELEGRAM]._pending_messages
    assert session_key in pending, "follow-up should be queued for the pending agent"


@pytest.mark.asyncio
async def test_slack_followup_inside_grace_queues_without_interrupt():
    """The bug: Slack must get the same grace treatment as Telegram.

    Before the fix this returns the "⚡ Interrupting current task" ack (or
    otherwise fails to queue), which is exactly what users saw as Red
    "starting with an interrupt".
    """
    runner, session_key, event, result = await _claim_and_followup(Platform.SLACK)

    assert result is None, (
        "Slack follow-up inside the post-claim grace window must not produce "
        "an interrupt ack"
    )
    pending = runner.adapters[Platform.SLACK]._pending_messages
    assert session_key in pending, "follow-up should be queued for the pending agent"
