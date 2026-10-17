"""The ``pre_gateway_dispatch`` hook must also run on the busy path.

``gateway/platforms/base.py`` routes a message that arrives while a session is busy to
``_handle_active_session_busy_message`` *instead of* ``_handle_message``, so the ingress gate
(and with it ``pre_gateway_dispatch``) never ran for it: a plugin silencing a chat (``skip``)
still got the busy ack — and a steer / interrupt — for every follow-up that landed while the
previous turn was winding down (#140442).

The hook is once per inbound message: the busy path offers it, and a follow-up that then falls
through to the adapter's own queueing (``busy_text_mode: queue``) must not offer it again when
its turn is finally dispatched.
"""
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import Platform, SessionSource, build_session_key
from gateway.platforms.event import MessageEvent, MessageType


def _make_event(text="hello", platform=Platform.TELEGRAM):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=platform,
            chat_id="123",
            chat_type="dm",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _make_runner():
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._queued_events = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_text_mode = "interrupt"
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
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter():
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.TELEGRAM
    adapter._text_debounce = {}
    adapter._busy_text_debounce_seconds = 0.6
    return adapter


def _busy_runner(monkeypatch, hook, mode="interrupt", text_mode=None):
    """Runner with a live agent for one session and the plugin hook stubbed."""
    calls = {"count": 0}

    async def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            calls["count"] += 1
            return hook(kwargs.get("event"))
        return []

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _fake_hook)

    runner, _sentinel = _make_runner()
    runner._busy_input_mode = mode
    if text_mode is not None:
        runner._busy_text_mode = text_mode
    adapter = _make_adapter()
    event = _make_event()
    sk = build_session_key(event.source)
    runner.adapters[event.source.platform] = adapter

    agent = MagicMock()
    agent.get_activity_summary.return_value = {
        "api_call_count": 3,
        "max_iterations": 60,
        "current_tool": "terminal",
        "last_activity_ts": time.time(),
        "last_activity_desc": "terminal",
        "seconds_since_activity": 1.0,
    }
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time()
    return runner, adapter, event, sk, agent, calls


@pytest.mark.asyncio
async def test_skip_on_the_busy_path_sends_no_ack(monkeypatch):
    """A chat a plugin silences must not receive the busy ack (nor be interrupted)."""
    runner, adapter, event, sk, agent, calls = _busy_runner(
        monkeypatch, lambda _ev: [{"action": "skip", "reason": "archive-only"}]
    )

    result = await runner._handle_active_session_busy_message(event, sk)

    assert calls["count"] == 1, "the busy path must offer pre_gateway_dispatch"
    assert result is True, "a skipped event is handled (dropped), not dispatched"
    adapter._send_with_retry.assert_not_called()
    agent.interrupt.assert_not_called()
    assert not runner._queued_events.get(sk)


@pytest.mark.asyncio
async def test_skip_on_the_busy_path_prevents_steering(monkeypatch):
    """skip must also stop a steer: the text never enters the running turn."""
    runner, _adapter, event, sk, agent, _calls = _busy_runner(
        monkeypatch, lambda _ev: [{"action": "skip", "reason": "archive-only"}], mode="steer"
    )
    agent.steer = MagicMock(return_value=True)

    result = await runner._handle_active_session_busy_message(event, sk)

    assert result is True
    agent.steer.assert_not_called()


@pytest.mark.asyncio
async def test_hook_runs_once_when_the_busy_path_falls_through_to_queue(monkeypatch):
    """busy_text_mode=queue hands the event back to the adapter; the hook must not run twice."""
    runner, _adapter, event, sk, _agent, calls = _busy_runner(
        monkeypatch, lambda _ev: [], mode="queue", text_mode="queue"
    )

    handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is False, "queue-mode text falls through to the adapter's own queueing"
    assert calls["count"] == 1
    # The event was offered already, so the dispatch that later runs this turn is a no-op.
    assert await runner._hm_pre_gateway_dispatch_hook(event, event.source) is event
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_rewrite_on_the_busy_path_reaches_the_steered_text(monkeypatch):
    """A plugin rewrite applies to what the busy path injects, not just to cold-path turns."""
    runner, _adapter, event, sk, agent, _calls = _busy_runner(
        monkeypatch,
        lambda _ev: [{"action": "rewrite", "text": "rewritten by plugin"}],
        mode="steer",
    )
    agent.steer = MagicMock(return_value=True)

    await runner._handle_active_session_busy_message(event, sk)

    injected = agent.steer.call_args.args[0]
    assert "rewritten by plugin" in injected
