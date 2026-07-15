"""Regression test for hardened HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS parsing.

Salvage of #13873 by @Junass1 (remaining gap): three of the four timeout knobs
that PR hardened already route through ``gateway.run._float_env`` on current
main, but the Telegram follow-up grace window was still parsed with a raw
``float(os.getenv(...))``.  A config/env typo (e.g.
``HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS=abc``) therefore raised ``ValueError``
inside the inbound Telegram message handler instead of falling back to the
documented 3.0s default.  This test pins the helper's behaviour so the knob
can never crash message handling again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from gateway.run import _float_env


def test_followup_grace_valid_value(monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "5.5")
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 5.5


def test_followup_grace_invalid_value_falls_back(monkeypatch):
    # The exact typo from #13873 must NOT raise — it falls back to the default.
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "abc")
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 3.0


def test_followup_grace_empty_falls_back(monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "")
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 3.0


def test_followup_grace_unset_falls_back(monkeypatch):
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 3.0


@pytest.mark.asyncio
async def test_handle_message_queues_followup_when_grace_env_is_malformed(monkeypatch):
    """Handler-level regression: abc grace env must not raise in _handle_message.

    Pattern mirrors tests/gateway/test_busy_session_ack.py Telegram fixture
    (queue mode + active agent). Before _float_env, float("abc") crashed the
    inbound path; after, the follow-up is still queued (#51324 review).
    """
    import time
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    # Defer imports used only by this handler test so module import stays light.
    from gateway.platforms.base import (
        MessageEvent,
        MessageType,
        Platform,
        SessionSource,
        build_session_key,
    )
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "abc")

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_text_mode = "interrupt"
    runner._busy_input_mode = "queue"
    runner._queued_events = {}
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._enqueue_fifo = lambda key, event, adapter: runner._queued_events.setdefault(
        key, []
    ).append(event)

    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.TELEGRAM
    adapter._text_debounce = {}
    adapter._busy_text_debounce_seconds = 0.6

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="user1",
    )
    sk = build_session_key(source)
    runner.adapters[source.platform] = adapter

    agent = MagicMock()
    agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time()

    event = MessageEvent(
        text="follow-up with bad grace env",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m-abc",
    )

    result = await GatewayRunner._handle_message(runner, event)
    assert result is None
    # Queued without raising on HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS=abc
    queued = runner._queued_events.get(sk) or []
    pending = adapter._pending_messages.get(sk)
    assert queued or pending is event
    agent.interrupt.assert_not_called()
