"""Tests for the gateway /queue command when NO agent is running (idle path).

/queue queues a prompt for the next turn without interrupting the current
run. When an agent IS running, the gateway stores the payload in the FIFO
(adapter._pending_messages + the runner overflow) and it is drained at the
tail of the completed run — see ``test_queue_consumption.py``.

When the session is IDLE there is no in-flight turn to queue behind, and the
FIFO drain only runs after a completed agent run. Enqueuing in that case
would strand the message forever (nothing consumes it). So the idle path must
instead strip the ``/queue`` prefix and run the payload now as a normal user
turn — mirroring the idle ``/steer`` behavior.

Regression: previously the idle path had no ``queue`` branch. Because ``queue``
is in ``GATEWAY_KNOWN_COMMANDS`` it also skipped the unknown-command notice, so
the literal ``"/queue <prompt>"`` text was forwarded verbatim to the agent.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _make_runner():
    """Construct a GatewayRunner wired for the cold/idle dispatch path.

    No agent is placed in ``_running_agents`` so ``_handle_message`` flows
    through the canonical command chain and (for a non-returning command)
    falls through to ``_handle_message_with_agent``, which we capture.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = _session_entry()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._draining = False
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    # Keep the dispatch path out of Telegram topic-lobby handling.
    runner._is_telegram_topic_root_lobby = lambda _source: False
    return runner, adapter


@pytest.mark.asyncio
async def test_queue_idle_strips_prefix_and_runs_now():
    """With no active agent, /queue must strip the prefix and send the
    payload as a normal user turn — not enqueue it (which would strand it),
    and not forward the literal '/queue ...' text to the agent."""
    runner, adapter = _make_runner()
    sk = build_session_key(_make_source())

    seen = {}

    async def _capture(event, source, _quick_key, _run_generation):
        seen["text"] = event.text
        return ""

    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("/queue investigate the logs"))

    # Prefix stripped — the agent saw the bare payload, not "/queue ...".
    assert seen.get("text") == "investigate the logs"
    # Nothing was stranded in the FIFO (slot or overflow).
    assert adapter._pending_messages == {}
    assert not getattr(runner, "_queued_events", {}).get(sk)


@pytest.mark.asyncio
async def test_queue_alias_q_idle_strips_prefix():
    """The /q alias resolves to canonical 'queue' and gets the same idle
    treatment."""
    runner, adapter = _make_runner()

    seen = {}

    async def _capture(event, source, _quick_key, _run_generation):
        seen["text"] = event.text
        return ""

    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("/q ship it"))

    assert seen.get("text") == "ship it"
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_queue_idle_empty_payload_returns_usage():
    """/queue with no payload and no active agent surfaces a usage hint and
    does NOT dispatch an agent turn."""
    runner, adapter = _make_runner()

    called = {"count": 0}

    async def _capture(event, source, _quick_key, _run_generation):
        called["count"] += 1
        return ""

    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_event("/queue"))

    assert result is not None
    assert "Usage" in result or "usage" in result
    assert called["count"] == 0
    assert adapter._pending_messages == {}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
