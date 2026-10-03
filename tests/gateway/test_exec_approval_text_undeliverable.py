"""The text-fallback approval send must not swallow an undeliverable prompt.

``_approval_notify_sync`` is the ``notify_cb`` for ``_await_gateway_decision``. Its
BUTTON branch raises ``_ExecApprovalDeclined`` on a definitively failed card so the
caller drops the central queue entry and unblocks the waiting tool (see
``test_decline_fallback_suppression``). The TEXT branch, however, only logged the
exception and returned normally:

    except Exception as e:
        logger.error("Failed to send approval request: %s", e)

Returning normally told ``_await_gateway_decision`` that the notify succeeded, so the
entry stayed pending in the central approval queue and the tool sat blocked for the
full ``approvals.timeout`` (measured: 19 real timeouts, the tool returning
``BLOCKED: ... timed out without user response`` while the user never received a
prompt at all).

A ``SendResult(success=False)`` — an adapter that answers instead of raising — had
the same hole: the value was discarded, so a refused/failed text prompt also looked
delivered.

These tests pin the corrected contract:
* definitive text-send failure  -> raise, so the caller reports ``notify_failed``
* possibly-delivered (timeout)  -> return, the prompt may still be answered
* success                       -> return, and arm the timeout notice
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.platforms.base import SendResult
from tools import approval as _approval  # noqa: F401  (ensures the real module is loaded)

# ``_approval_notify_sync`` imports ``gateway.run`` lazily inside the method. Importing it
# here (at collection, like the ~186 other gateway test modules) prewarms that path so the
# first in-test import cannot trip the autouse real-home I/O guard.
import gateway.run  # noqa: E402,F401

SESSION = "agent:main:telegram:dm:1"
APPROVAL = {
    "command": "rm -rf /tmp/x",
    "description": "recursive delete",
    "pattern_key": "k",
    "request_id": "req-1",
}


class _TextAdapter:
    """Non-button adapter: the only approval lane is the plain-text prompt."""

    typed_command_prefix = "/"

    def __init__(self, *, result: Any = None, raises: BaseException | None = None) -> None:
        self._result = result
        self._raises = raises
        self.sends: list[str] = []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send(self, chat_id: str, message: str, **k: Any):
        self.sends.append(message)
        if self._raises is not None:
            raise self._raises
        return self._result


def _runner(adapter):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={},
        session_key=SESSION, source=SimpleNamespace(chat_id="C1", platform="telegram", session_key=SESSION),
    )

    class _Fut:
        def __init__(self, result): self._r = result
        def result(self, timeout=None): return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None
    return runner


def _unavailable_loop_schedule(coro, _label):
    """Real ``_schedule`` closes the coroutine when the loop is gone; mirror that."""
    coro.close()
    return None


class _ButtonAdapter:
    """Adapter that RENDERS approval buttons: the card lane is the only approval lane."""

    typed_command_prefix = "/"
    supports_exec_approval_buttons = True

    def __init__(self, *, send_raises: BaseException | None = None) -> None:
        self._send_raises = send_raises
        self.sends: list[str] = []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send_exec_approval(self, *a: Any, **k: Any) -> SendResult:
        return SendResult(success=True, message_id="card-1")

    async def send(self, chat_id: str, message: str, **k: Any):
        self.sends.append(message)
        return SendResult(success=True, message_id="m2")


def _button_runner(adapter):
    """A runner whose adapter class advertises native approval buttons."""
    runner = _runner(adapter)

    class _Buttoned(_ButtonAdapter):
        pass

    # ``_renders_exec_approval_buttons`` probes the CLASS; swap the ctx adapter's class so the
    # button branch is taken while the instance still records sends.
    runner._ctx._status_adapter.__class__ = _Buttoned
    return runner


def test_text_fallback_raise_is_reported_so_the_caller_reports_notify_failed():
    """A raising text send must propagate, not be swallowed into a normal return."""
    adapter = _TextAdapter(raises=RuntimeError("network down"))
    runner = _runner(adapter)

    with pytest.raises(Exception):
        runner._approval_notify_sync(dict(APPROVAL))


def test_text_fallback_failed_result_is_reported():
    """An adapter answering success=False is a definitive miss, same as a raise."""
    adapter = _TextAdapter(result=SendResult(success=False, error="Forbidden: bot was blocked by the user"))
    runner = _runner(adapter)

    with pytest.raises(Exception):
        runner._approval_notify_sync(dict(APPROVAL))


def test_text_fallback_success_arms_the_timeout_notice():
    """Control: a delivered text prompt must keep working (notice armed, no raise)."""
    adapter = _TextAdapter(result=SendResult(success=True, message_id="m1"))
    runner = _runner(adapter)

    runner._approval_notify_sync(dict(APPROVAL))

    assert len(adapter.sends) == 1


def test_text_fallback_scheduling_unavailable_is_reported():
    """No future = the loop is gone; the prompt cannot be delivered."""
    adapter = _TextAdapter(result=SendResult(success=True, message_id="m1"))
    runner = _runner(adapter)
    runner._schedule = _unavailable_loop_schedule

    with pytest.raises(Exception):
        runner._approval_notify_sync(dict(APPROVAL))


def test_scheduling_timeout_is_not_treated_as_possibly_delivered():
    """A TimeoutError from the SCHEDULER is not a 'card may still post' case.

    ``concurrent.futures.TimeoutError`` is the builtin ``TimeoutError`` on 3.11+, so a bare
    ``except`` around the whole block also swallows a timeout raised by ``_schedule`` itself
    (or by any other statement in the try). Only the send-wait deadline means "possibly
    delivered"; anything else is undeliverable and must reach the caller's notify_failed path.
    """
    adapter = _TextAdapter(result=SendResult(success=True, message_id="m1"))
    runner = _runner(adapter)

    def _schedule_raises(coro, _label):
        coro.close()
        raise TimeoutError("event loop scheduling timed out")

    runner._schedule = _schedule_raises

    with pytest.raises(Exception):
        runner._approval_notify_sync(dict(APPROVAL))


def test_ambiguous_button_send_also_arms_the_timeout_notice(monkeypatch):
    """An ambiguous card send must still tell the user when nobody answers.

    ``ambiguous`` keeps the registration alive for a late tap and sends no second card, which is
    right. But the card was never observed posted: if it never arrived, the tool waits out the whole
    approval window and the chat says nothing — indistinguishable, to the user, from the silent stall
    this fix is about. Arming the notice costs nothing when the card did post (the tap resolves the
    wait and the settle hook never fires for a timeout).
    """
    import gateway.run_turn_runner_approval_settle as settle_mod

    armed: list = []
    monkeypatch.setattr(settle_mod, "register_timeout_notice", lambda *a, **k: armed.append(a))

    adapter = _ButtonAdapter(send_raises=None)
    runner = _button_runner(adapter)
    monkeypatch.setattr(
        "gateway.run._approval_send_outcome", lambda fut, timeout: "ambiguous"
    )

    runner._approval_notify_sync(dict(APPROVAL))

    assert len(armed) == 1, "an ambiguous send must arm the timeout notice"


def test_failed_button_send_falls_through_to_text_and_still_reports(monkeypatch):
    """Control: a definitively failed CARD still reaches the user via the text lane."""
    adapter = _ButtonAdapter(send_raises=None)
    runner = _button_runner(adapter)
    monkeypatch.setattr("gateway.run._approval_send_outcome", lambda fut, timeout: "failed")

    runner._approval_notify_sync(dict(APPROVAL))

    assert len(adapter.sends) == 1
    assert "rm -rf /" in adapter.sends[0]


def test_text_fallback_raise_does_not_arm_the_settle_hook(monkeypatch):
    """A failed send must not also arm the settle hook that posts the timeout notice.

    ``register_timeout_notice`` is imported from its sibling module INSIDE the method,
    so the patch target is that module's attribute.
    """
    import gateway.run_turn_runner_approval_settle as settle_mod

    armed: list = []
    monkeypatch.setattr(settle_mod, "register_timeout_notice", lambda *a, **k: armed.append(a))

    adapter = _TextAdapter(raises=RuntimeError("network down"))
    runner = _runner(adapter)

    with pytest.raises(Exception):
        runner._approval_notify_sync(dict(APPROVAL))

    assert armed == []


def test_text_fallback_success_arms_the_settle_hook(monkeypatch):
    """Control: a delivered prompt DOES arm the notice, so an unanswered wait still reports."""
    import gateway.run_turn_runner_approval_settle as settle_mod

    armed: list = []
    monkeypatch.setattr(settle_mod, "register_timeout_notice", lambda *a, **k: armed.append(a))

    adapter = _TextAdapter(result=SendResult(success=True, message_id="m1"))
    runner = _runner(adapter)

    runner._approval_notify_sync(dict(APPROVAL))

    assert len(armed) == 1


def test_undeliverable_text_prompt_ends_the_wait_as_notify_failed(monkeypatch):
    """The end-to-end contract: an undelivered prompt must NOT wait out the approval window.

    Drives the real ``_await_gateway_decision`` with ``_approval_notify_sync`` as its notify_cb.
    Before the fix the callback swallowed the send failure and returned, so the central entry
    stayed pending and the tool blocked for the whole approval timeout. Now the callback raises,
    ``_await_gateway_decision`` drops the entry and answers ``notify_failed`` at once.
    """
    from tools import approval as _approval
    from tools.approval_gateway_wait import _await_gateway_decision

    # Keep the wait short so a REGRESSION fails fast instead of hanging for the real window.
    monkeypatch.setattr(
        "gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 0
    )

    adapter = _TextAdapter(raises=RuntimeError("network down"))
    runner = _runner(adapter)

    data = dict(APPROVAL)
    data["pattern_keys"] = ["k"]
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)
    try:
        result = _await_gateway_decision(SESSION, runner._approval_notify_sync, data)
    finally:
        with _approval._lock:
            _approval._gateway_queues.pop(SESSION, None)

    assert result.get("notify_failed") is True
