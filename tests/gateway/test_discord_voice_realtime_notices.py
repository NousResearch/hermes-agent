from types import SimpleNamespace
from unittest.mock import MagicMock

import tools.approval as approval
import tools.approval_context as approval_context
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner


def _make_turn_runner(controller):
    """A real TurnRunner bound to a real GatewayRunner carrying the mixin."""
    runner = object.__new__(GatewayRunner)
    runner._voice_realtime_controllers = {(None, 5): controller}
    ctx = SimpleNamespace(
        _voice_ack_guild=(5, 77),
        _status_adapter=MagicMock(),
        _status_chat_id="chat-1",
        _status_thread_metadata=None,
        _loop_for_step=None,
        session_key="sess-voice",
        stream_consumer_holder=[None],
    )
    return TurnRunner(runner, ctx)


def test_active_consult_speaks_blocking_notice():
    runner = object.__new__(GatewayRunner)
    controller = MagicMock(consult_active=True)
    runner._voice_realtime_controllers = {(None, 5): controller}

    runner._notify_voice_realtime_blocked(
        SimpleNamespace(_voice_ack_guild=(5, 77)),
        "Hermes needs your approval in Discord.",
    )

    controller.notify.assert_called_once_with(
        "Hermes needs your approval in Discord."
    )


def test_notice_is_silent_without_active_consult():
    runner = object.__new__(GatewayRunner)
    controller = MagicMock(consult_active=False)
    runner._voice_realtime_controllers = {(None, 5): controller}

    runner._notify_voice_realtime_blocked(
        SimpleNamespace(_voice_ack_guild=(5, 77)),
        "Hermes needs your approval in Discord.",
    )

    controller.notify.assert_not_called()


def test_approval_notify_sync_reaches_voice_controller():
    """The production notify callback (TurnRunner._approval_notify_sync)
    resolves the mixin method through the owning GatewayRunner — a plain
    ``self._notify_voice_realtime_blocked`` would raise AttributeError and
    auto-deny every gateway dangerous-command approval."""
    controller = MagicMock(consult_active=True)
    turn_runner = _make_turn_runner(controller)

    turn_runner._approval_notify_sync(
        {"command": "rm -rf /tmp/x", "description": "dangerous command"},
    )

    controller.notify.assert_called_once_with(
        "Hermes needs your approval in Discord."
    )
    turn_runner._ctx._status_adapter.pause_typing_for_chat.assert_called_once_with(
        "chat-1"
    )


def test_await_gateway_decision_drives_notify_without_notify_failed(monkeypatch):
    """End-to-end over tools.approval: the registered callback must survive
    _await_gateway_decision's notify step (notify_failed would silently
    BLOCK the tool call before the user ever sees a prompt)."""
    controller = MagicMock(consult_active=True)
    turn_runner = _make_turn_runner(controller)
    notify_cb = turn_runner._approval_notify_sync
    # Expire the wait immediately: this test covers the notify leg, not the
    # human-response poll loop.
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 0)

    decision = approval._await_gateway_decision(
        "sess-voice",
        notify_cb,
        {
            "command": "rm -rf /tmp/x",
            "description": "dangerous command",
            "pattern_key": "rm",
            "pattern_keys": ["rm"],
        },
    )

    assert not decision.get("notify_failed")
    controller.notify.assert_called_once_with(
        "Hermes needs your approval in Discord."
    )
