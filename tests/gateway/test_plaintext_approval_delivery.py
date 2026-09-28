"""Plain-text exec approval delivery must reach the central waiter's failure gate."""

from __future__ import annotations

import asyncio
import concurrent.futures
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.relay.egress import EGRESS_DECLINE_CODE
from tools import approval as approval_mod
from tools import approval_gateway_wait as wait_mod


SESSION = "approval-plain-text-delivery"
REQUEST = {"command": "rm -rf build", "description": "remove build", "pattern_key": "build"}


class TextAdapter:
    typed_command_prefix = "/"

    def __init__(self, result):
        self.result = result
        self.sends = []

    def pause_typing_for_chat(self, _chat):
        pass

    async def send(self, chat, message, **kwargs):
        self.sends.append((chat, message, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _runner(adapter, disposition="result"):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={},
        session_key=SESSION, source=SimpleNamespace(chat_id="C1", platform="test", session_key=SESSION),
    )
    runner._close_native_stream_boundary = lambda _: None

    def schedule(coro, _label):
        if disposition == "none":
            coro.close()
            return None
        if disposition == "timeout":
            coro.close()
            return concurrent.futures.Future()
        fut = concurrent.futures.Future()
        try:
            fut.set_result(asyncio.run(coro))
        except Exception as exc:
            fut.set_exception(exc)
        return fut

    runner._schedule = schedule
    return runner


@pytest.mark.parametrize("result,disposition", [
    (SendResult(success=False, error="session not ready"), "result"),
    (None, "none"),
    (RuntimeError("transport down"), "result"),
    (SendResult(success=False, error="declined", raw_response={"success": False, "code": EGRESS_DECLINE_CODE}), "result"),
])
def test_definitive_nondelivery_drops_queue_without_poll(monkeypatch, result, disposition):
    runner = _runner(TextAdapter(result), disposition)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda *a, **k: None)
    monkeypatch.setattr(wait_mod, "_poll_event", lambda *a, **k: pytest.fail("waited for unseen approval"))
    answer = wait_mod._await_gateway_decision(SESSION, runner._approval_notify_sync, REQUEST)
    assert answer == {"resolved": False, "choice": None, "notify_failed": True}
    assert SESSION not in approval_mod._gateway_queues
    assert len(runner._ctx._status_adapter.sends) == (0 if disposition == "none" else 1)


@pytest.mark.parametrize("result,disposition", [
    (SendResult(success=True, message_id="m1"), "result"),
    (None, "timeout"),
    (SendResult(success=False, raw_response={"ambiguous": True}), "result"),
])
def test_sent_or_ambiguous_stays_armed_once(monkeypatch, result, disposition):
    runner = _runner(TextAdapter(result), disposition)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda *a, **k: None)
    observed = []

    def poll(event, session_key, *, interrupt_log):
        observed.append(len(approval_mod._gateway_queues[session_key]))
        return "timeout"

    monkeypatch.setattr(wait_mod, "_poll_event", poll)
    answer = wait_mod._await_gateway_decision(SESSION, runner._approval_notify_sync, REQUEST)
    assert answer["resolved"] is False and not answer.get("notify_failed")
    assert observed == [1]
    sends = runner._ctx._status_adapter.sends
    # A confirmed send registers a timeout notice after the poll expires;
    # that notice is not a second approval prompt.
    prompts = [message for _, message, _ in sends if "wants to run a command" in message]
    assert len(prompts) == (0 if disposition == "timeout" else 1)
    assert SESSION not in approval_mod._gateway_queues
