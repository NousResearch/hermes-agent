"""Inert completion reporting only: never execute an authority drain or turn."""
import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.session_authority import SessionAuthority


class PayloadError(Exception):
    def __str__(self):
        raise AssertionError("exception text must not be formatted")


class CompletedFuture(asyncio.Future):
    reads = 0

    def exception(self):
        self.reads += 1
        return super().exception()


@pytest.fixture
def loop():
    value = asyncio.new_event_loop()
    yield value
    value.close()


def reporter():
    from gateway.session_task_diagnostics import drain_task_reporter
    return drain_task_reporter(profile_id="profile-a", session_id="session-a", epoch=7)


def test_schedule_captures_reporter_without_running_drain(loop, monkeypatch, caplog):
    callbacks = []

    class RegisteredFuture(CompletedFuture):
        def add_done_callback(self, callback, *, context=None):
            callbacks.append(callback)

    completed = RegisteredFuture(loop=loop)
    completed.set_exception(PayloadError("private input and result"))
    marker = object()
    create = Mock(return_value=completed)
    monkeypatch.setattr("gateway.session_authority.asyncio.create_task", create)
    live = SimpleNamespace(task=None)
    waiter = loop.create_future()
    owner = SimpleNamespace(
        profile_id="profile-a", epoch=7, sessions={"session-a": live},
        _drain=Mock(return_value=marker), waiters={"admission": waiter},
        pending_results={"admission": {"private": "result"}},
        _schedule=Mock(side_effect=AssertionError("no rearm")),
    )
    ref = SimpleNamespace(session_id="session-a")
    try:
        SessionAuthority._schedule(owner, ref)
        create.assert_called_once_with(marker)
        owner._drain.assert_called_once_with(ref)
        assert live.task is completed
        assert len(callbacks) == 1
        replacement = loop.create_future()
        live.task = replacement
        owner.profile_id, owner.epoch, ref.session_id = "new-profile", 8, "new-session"
        with caplog.at_level(logging.ERROR):
            callbacks[0](completed)
            callbacks[0](completed)
        assert completed.reads == 1
        assert len(caplog.records) == 1
        assert json.loads(caplog.records[0].args[0]) == {
            "profile_id": "profile-a", "session_id": "session-a", "epoch": 7,
        }
        assert live.task is replacement
        assert owner.waiters == {"admission": waiter} and not waiter.done()
        assert owner.pending_results == {"admission": {"private": "result"}}
        owner._schedule.assert_not_called()
    finally:
        completed.exception()  # Consume even on the pre-fix RED assertion path.


def test_report_consumes_once_without_exception_chain_or_locals(loop, caplog):
    callback = reporter()
    future = CompletedFuture(loop=loop)
    private_input = "do-not-log-input"
    private_result = "do-not-log-result"
    try:
        try:
            raise ValueError(private_input)
        except ValueError as cause:
            raise PayloadError(private_result) from cause
    except PayloadError as error:
        future.set_exception(error)
    with caplog.at_level(logging.ERROR):
        callback(future)
        callback(future)
    assert future.reads == 1
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.args[1] == "PayloadError"
    assert "SessionAuthority._drain" in record.getMessage()
    assert private_input not in caplog.text and private_result not in caplog.text
    assert record.exc_info is None and record.stack_info is None
    assert all(not isinstance(arg, BaseException) for arg in record.args)


@pytest.mark.parametrize("cancelled", [False, True])
def test_success_and_cancellation_are_quiet(loop, caplog, cancelled):
    callback = reporter()
    future = CompletedFuture(loop=loop)
    if cancelled:
        future.cancel()
    else:
        future.set_result({"private input": "private result"})
    callback(future)
    callback(future)
    assert not caplog.records
    assert future.reads == (0 if cancelled else 1)


def test_pending_future_is_untouched_until_completed(loop, caplog):
    callback = reporter()
    future = CompletedFuture(loop=loop)
    callback(future)
    assert future.reads == 0 and not future.done()
    future.set_exception(ValueError("private"))
    with caplog.at_level(logging.ERROR):
        callback(future)
    assert future.reads == 1 and len(caplog.records) == 1


def test_owner_identifiers_are_escaped_for_one_line_logging(loop, caplog):
    from gateway.session_task_diagnostics import drain_task_reporter
    callback = drain_task_reporter(profile_id="profile\nnext", session_id="s\rnext", epoch=7)
    future = CompletedFuture(loop=loop)
    future.set_exception(ValueError("private"))
    with caplog.at_level(logging.ERROR):
        callback(future)
    message = caplog.records[0].getMessage()
    assert "\n" not in message and "\r" not in message
    assert "private" not in message


def test_sink_error_cannot_expose_future_through_callback_fallback(loop, monkeypatch):
    callback = reporter()
    future = CompletedFuture(loop=loop)
    future.set_exception(PayloadError("private input"))
    sink = Mock(side_effect=RuntimeError("private sink message"))
    monkeypatch.setattr("gateway.session_task_diagnostics.logger.error", sink)
    callback(future)
    callback(future)
    assert future.reads == 1
    sink.assert_called_once()
