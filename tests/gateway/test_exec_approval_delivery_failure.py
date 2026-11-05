"""Undelivered approval requests must fail closed, not wait for an absent user.

Exercise the actual notify -> gateway queue boundary, with no network or shell.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from tools import approval
from tools import approval_gateway_wait as wait

SESSION = "agent:test:telegram:dm:123:456"
DATA = {"command": "diagnostic-only", "description": "test request", "pattern_key": "test",
        "allow_session": False, "allow_permanent": False, "smart_denied": True}


class TextAdapter:
    typed_command_prefix = "/"

    def __init__(self, result):
        self.result = result
        self.sends = []

    def pause_typing_for_chat(self, chat_id):
        pass

    async def send(self, chat_id, message, **kwargs):
        self.sends.append((chat_id, message, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class ButtonAdapter(TextAdapter):
    async def send_exec_approval(self, **kwargs):
        return SendResult(success=False, error="card rejected")


def runner(adapter, *, no_loop=False):
    obj = object.__new__(TurnRunner)
    obj._approval_owner = None
    obj._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="123",
        _status_thread_metadata={"thread_id": "456"}, session_key=SESSION,
        source=SimpleNamespace(chat_id="123", platform="telegram", session_key=SESSION))
    obj._close_native_stream_boundary = lambda *a: None

    def schedule(coro, label):
        if no_loop:
            coro.close()
            return None
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return future

    obj._schedule = schedule
    return obj


@pytest.mark.parametrize("outcome", ["failed", "declined", "sent", "ambiguous", "retired"])
def test_owner_registers_only_after_answerable_delivery(monkeypatch, outcome):
    events = []
    results = {
        "failed": SendResult(success=False, error="offline"),
        "declined": SendResult(success=False, raw_response={"code": "egress_declined"}),
        "sent": SendResult(success=True, message_id="789"),
        "ambiguous": TimeoutError(),
        "retired": SendResult(success=True, message_id="789"),
    }
    adapter = TextAdapter(results[outcome])
    obj = runner(adapter)

    class Authority:
        def check_approval_generation(self, session_id, generation):
            assert (session_id, generation) == ("owned-session", 7)
            events.append("checked")
            if outcome == "retired":
                raise RuntimeError("retired generation")

        def register_approval(self, session_id, generation, session_key, data, **kwargs):
            assert (session_id, generation, session_key) == ("owned-session", 7, SESSION)
            assert len(adapter.sends) == 1
            assert data["request_id"]
            events.append("registered")

    obj._approval_owner = (Authority(), "owned-session", 7)
    monkeypatch.setattr("gateway.local_viewer_events.prompt_timeout_s", lambda *a: 30)

    def answer(*a, **kwargs):
        assert events == ["checked", "registered"]
        assert approval.resolve_gateway_approval(SESSION, "deny") == 1
        return "set"

    monkeypatch.setattr(wait, "_poll_event", answer)
    result = wait._await_gateway_decision(SESSION, obj._approval_notify_sync, dict(DATA))
    if outcome in {"sent", "ambiguous"}:
        assert result["resolved"] is True and result["choice"] == "deny"
        assert events == ["checked", "registered"]
    else:
        assert result.get("notify_failed") is True
        assert events == ["checked"]
    assert len(adapter.sends) == (0 if outcome == "retired" else 1)
    assert SESSION not in approval._gateway_queues


@pytest.mark.parametrize("lane", ["text", "buttons"])
@pytest.mark.parametrize("failure_site", ["adapter", "scheduler"])
def test_pre_future_failures_do_not_log_private_payloads(monkeypatch, caplog, lane, failure_site):
    marker = "PRIVATE_PRE_FUTURE_PAYLOAD"

    class SyncTextAdapter(TextAdapter):
        def send(self, *args, **kwargs):
            raise RuntimeError(marker)

    class SyncButtonAdapter(TextAdapter):
        def send_exec_approval(self, **kwargs):
            raise RuntimeError(marker)

    adapter_type = (SyncButtonAdapter if lane == "buttons" else SyncTextAdapter)
    if failure_site == "scheduler":
        adapter_type = ButtonAdapter if lane == "buttons" else TextAdapter
    adapter = adapter_type(SendResult(success=False))
    obj = runner(adapter)
    if failure_site == "scheduler":
        def schedule(coro, label):
            coro.close()
            raise RuntimeError(marker)
        obj._schedule = schedule
    polled = []
    monkeypatch.setattr(wait, "_poll_event", lambda *a, **kw: polled.append(True) or "timeout")
    with caplog.at_level("DEBUG"):
        result = wait._await_gateway_decision(SESSION, obj._approval_notify_sync, dict(DATA))
    assert result.get("notify_failed") is True
    assert not polled
    assert SESSION not in approval._gateway_queues
    assert marker not in caplog.text
    assert marker not in repr([record.__dict__ for record in caplog.records])


@pytest.fixture(autouse=True)
def isolated_queue(monkeypatch):
    monkeypatch.setattr(approval, "_gateway_queues", {})
    monkeypatch.setattr(wait._ctx, "_fire_approval_hook", lambda *a, **kw: None)


@pytest.mark.parametrize("adapter_type", [TextAdapter, ButtonAdapter])
@pytest.mark.parametrize("failure", [SendResult(success=False, error="offline"), RuntimeError("offline"), None])
def test_failed_delivery_never_waits_for_user(monkeypatch, adapter_type, failure):
    adapter = adapter_type(failure)
    polled = []
    monkeypatch.setattr(wait, "_poll_event", lambda *a, **kw: polled.append(True) or "timeout")
    result = wait._await_gateway_decision(SESSION, runner(adapter)._approval_notify_sync, dict(DATA))
    assert result.get("notify_failed") is True
    assert result["resolved"] is False and result["choice"] is None
    assert polled == [], "a definitive delivery failure must not start the human response timer"
    assert SESSION not in approval._gateway_queues


def test_unavailable_loop_does_not_leave_a_pending_request(monkeypatch):
    monkeypatch.setattr(wait, "_poll_event", lambda *a, **kw: "timeout")
    result = wait._await_gateway_decision(
        SESSION, runner(TextAdapter(None), no_loop=True)._approval_notify_sync, dict(DATA))
    assert result.get("notify_failed") is True
    assert SESSION not in approval._gateway_queues


def test_delivered_text_prompt_stays_answerable_in_the_original_topic(monkeypatch):
    adapter = TextAdapter(SendResult(success=True, message_id="789"))

    def answer(*a, **kw):
        assert approval.resolve_gateway_approval(SESSION, "once") == 1
        return "set"

    monkeypatch.setattr(wait, "_poll_event", answer)
    result = wait._await_gateway_decision(SESSION, runner(adapter)._approval_notify_sync, dict(DATA))
    assert result["resolved"] is True and result["choice"] == "once"
    assert len(adapter.sends) == 1
    chat, text, kwargs = adapter.sends[0]
    assert chat == "123" and kwargs["metadata"]["thread_id"] == "456"
    assert "/approve session" not in text and "/approve always" not in text
    assert SESSION not in approval._gateway_queues


@pytest.mark.parametrize("choice", ["once", "deny"])
def test_ambiguous_text_send_is_not_resent_and_stays_answerable(monkeypatch, choice):
    adapter = TextAdapter(TimeoutError())

    def answer(*a, **kw):
        assert approval.resolve_gateway_approval(SESSION, choice) == 1
        return "set"

    monkeypatch.setattr(wait, "_poll_event", answer)
    result = wait._await_gateway_decision(SESSION, runner(adapter)._approval_notify_sync, dict(DATA))
    assert result["resolved"] is True and result["choice"] == choice
    assert len(adapter.sends) == 1
    assert SESSION not in approval._gateway_queues


def test_delivery_log_carries_route_not_command(monkeypatch, caplog):
    adapter = TextAdapter(SendResult(success=True, message_id="789"))
    monkeypatch.setattr(wait, "_poll_event", lambda *a, **kw: "timeout")
    with caplog.at_level("INFO"):
        wait._await_gateway_decision(SESSION, runner(adapter)._approval_notify_sync, dict(DATA))
    assert "notifier=TurnRunner._approval_notify_sync" in caplog.text
    assert "chat=123 thread=456 message=789 lane=text" in caplog.text
    assert DATA["command"] not in caplog.text


@pytest.mark.parametrize("kind, outcome, category", [
    ("failed_result", "failed", "adapter_result"),
    ("exception", "failed", "adapter_exception"),
    ("structured_ambiguity", "ambiguous", "lost_ack"),
    ("structured_decline", "declined", "egress_declined"),
    ("legacy_decline", "declined", "egress_declined"),
])
def test_approval_send_classification_logs_only_safe_categories(
    monkeypatch, caplog, kind, outcome, category,
):
    # Exercise the actual adapter -> classifier -> queue boundary with real log
    # records, not a mocked logger. Adapter diagnostics may echo arbitrary secrets.
    markers = ("PRIVATE_COMMAND_MARKER", "PRIVATE_CREDENTIAL_MARKER", "PRIVATE_RESPONSE_MARKER")
    private_error = " ".join(markers) + "\nforged log line " + "x" * 4096
    raw: dict[str, object] = {"error": private_error, "detail": private_error}
    if kind == "exception":
        send_result = RuntimeError(private_error)
    else:
        if kind == "structured_ambiguity":
            raw.update(ambiguous=True, code="egress_declined")
        elif kind == "structured_decline":
            raw["code"] = "egress_declined"
        error = ("egress declined: " if kind == "legacy_decline" else "") + private_error
        send_result = SendResult(
            success=False, error=error,
            raw_response=None if kind == "legacy_decline" else raw,
        )
    adapter = TextAdapter(send_result)
    polled = []

    def answer(*a, **kw):
        polled.append(True)
        assert approval.resolve_gateway_approval(SESSION, "deny") == 1
        return "set"

    monkeypatch.setattr(wait, "_poll_event", answer)
    with caplog.at_level("DEBUG"):
        result = wait._await_gateway_decision(
            SESSION, runner(adapter)._approval_notify_sync, dict(DATA))
    if outcome == "ambiguous":
        assert result["resolved"] is True and result["choice"] == "deny"
        assert polled == [True]
    else:
        assert result.get("notify_failed") is True
        assert result["resolved"] is False and result["choice"] is None
        assert polled == []
    assert len(adapter.sends) == 1
    assert SESSION not in approval._gateway_queues
    for marker in markers:
        assert marker not in caplog.text
        assert marker not in repr([record.__dict__ for record in caplog.records])
    classified = [record for record in caplog.records if record.funcName == "_approval_send_outcome"]
    assert len(classified) == 1
    assert classified[0].getMessage() == f"Prompt send outcome={outcome} category={category}"
    assert not classified[0].exc_info and not classified[0].stack_info
