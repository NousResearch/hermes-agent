"""Every boundary of the gateway approval delivery path must leave a bounded log record (#138133).

A Telegram DM reported an approval card the user never saw and a five-minute tool timeout; the
delivery chain (terminal gate → notifier handoff → platform send/ack → wait settle) had exactly
one outcome with no log at all ("sent"), so the failing boundary could not be identified from
logs. These tests pin the correlation records — request_id / session / message_id / settle
state only; never the command text.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

# Imported at MODULE level on purpose: gateway.run's import side effects (hermes_bootstrap's
# interrupted-update probe) stat the checkout's git dir, which sits inside the real Hermes home
# when the suite runs from a developer worktree — the home-I/O guard is fixture-scoped and only
# active during a test, so the probe must happen here at collection time.
import gateway.run as gateway_run
from gateway.run_turn_runner import TurnRunner
from gateway.platforms.base import SendResult
from tools import approval as _approval
from tools import approval_gateway_wait as wait_mod

SESSION = "agent:main:telegram:dm:1"
APPROVAL = {"command": "rm -rf /tmp/x", "description": "recursive delete", "pattern_key": "k"}


def _clear_queues():
    with _approval._lock:
        _approval._gateway_queues.clear()
        _approval._gateway_notify_cbs.clear()


def test_timeout_settle_leaves_dispatch_and_settle_records(monkeypatch, caplog):
    """An unanswered prompt must log both the notifier handoff and how the wait ended."""
    _clear_queues()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(wait_mod, "_poll_event", lambda event, session_key, *, interrupt_log: "timeout")
    with caplog.at_level(logging.INFO, logger="tools.approval"):
        decision = wait_mod._await_gateway_decision(SESSION, lambda data: None, APPROVAL)

    assert decision == {"resolved": False, "choice": None, "reason": None}
    records = [r.message for r in caplog.records]
    dispatch = [m for m in records if m.startswith("Approval prompt handed to gateway notifier")]
    settle = [m for m in records if m.startswith("Approval wait ended")]
    assert len(dispatch) == 1 and f"session={SESSION}" in dispatch[0]
    assert len(settle) == 1 and "state=timeout, resolved=False, choice=None" in settle[0]
    # Bounded by contract: correlation fields only, never the command being approved.
    assert "rm -rf" not in " ".join(records)


def test_acknowledged_card_leaves_a_delivery_record(tmp_path, monkeypatch, caplog):
    """A platform-acked card must log its message_id — the acknowledgement entry #138133 lacked."""
    (tmp_path / "config.yaml").write_text("display: {}")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    class _ButtonAdapter:
        typed_command_prefix = "/"

        def pause_typing_for_chat(self, chat_id: str) -> None:
            return None

        async def send_exec_approval(self, *a: Any, **k: Any) -> SendResult:
            return SendResult(success=True, message_id="card-1")

        async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
            return SendResult(success=True, message_id="m2")

    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=_ButtonAdapter(), _status_chat_id="C1", _status_thread_metadata=None,
        session_key=SESSION,
    )

    class _Fut:
        def __init__(self, result): self._r = result
        def result(self, timeout=None): return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None

    data = dict(APPROVAL, request_id="req-1")
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        runner._approval_notify_sync(data)

    delivered = [r.message for r in caplog.records if r.message.startswith("Exec approval card delivered")]
    assert len(delivered) == 1
    assert "message_id=card-1" in delivered[0] and "request_id=req-1" in delivered[0]
    assert f"session={SESSION}" in delivered[0]
    assert "rm -rf" not in delivered[0]


def test_handoff_failure_still_logs_the_wait_side_boundary(monkeypatch, caplog):
    """A raising notifier keeps its existing warning; the handoff INFO must not fire for it."""
    _clear_queues()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)

    def _raising_notify(data):
        raise RuntimeError("boundary gone")

    with caplog.at_level(logging.INFO, logger="tools.approval"):
        decision = wait_mod._await_gateway_decision(SESSION, _raising_notify, APPROVAL)

    assert decision["notify_failed"] is True
    assert not [r for r in caplog.records if r.message.startswith("Approval prompt handed")]
    assert any(r.levelno == logging.WARNING and "notify failed" in r.message for r in caplog.records)
