"""Progress-aware deadline contracts; elapsed time alone is not a stall."""
from types import SimpleNamespace

import pytest

from agent.transports import codex_app_server_session as module
from agent.transports.codex_app_server_session import CodexAppServerSession
from tests.agent.transports.test_codex_app_server_session import FakeClient


def session_with_clock(monkeypatch, events):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    client = FakeClient()
    def take_notification(timeout=0):
        clock.now += 400
        return events.pop(0) if events else None
    client.take_notification = take_notification
    session = CodexAppServerSession(client_factory=lambda **kw: client)
    return session, client


def note(method, **params):
    return {"method": method, "params": {"threadId": "thread-fake-001", "turnId": "turn-fake-001", **params}}


def test_active_turn_can_outlive_old_total_deadline(monkeypatch):
    events = [note("item/commandExecution/outputDelta", delta="progress") for _ in range(3)]
    events += [note("turn/completed", turn={"id": "turn-fake-001", "status": "completed"})]
    session, client = session_with_clock(monkeypatch, events)
    result = session.run_turn("work", turn_timeout=0, idle_timeout=450)
    assert not result.error and not result.interrupted
    assert not any(m == "turn/interrupt" for m, _ in client.requests)


def test_silent_turn_still_stops_and_reports_incomplete(monkeypatch):
    session, client = session_with_clock(monkeypatch, [])
    result = session.run_turn("work", turn_timeout=0, idle_timeout=450)
    assert result.interrupted and result.should_retire
    assert "inactive" in result.error and "task incomplete" in result.error
    assert any(m == "turn/interrupt" for m, _ in client.requests)


def test_foreign_thread_cannot_keep_turn_alive(monkeypatch):
    events = [{"method": "item/commandExecution/outputDelta", "params": {"threadId": "other", "delta": "noise"}} for _ in range(10)]
    session, _ = session_with_clock(monkeypatch, events)
    result = session.run_turn("work", turn_timeout=0, idle_timeout=450)
    assert result.interrupted and "inactive" in result.error


def test_explicit_wall_clock_limit_remains_available(monkeypatch):
    events = [note("item/commandExecution/outputDelta", delta="progress") for _ in range(10)]
    session, _ = session_with_clock(monkeypatch, events)
    result = session.run_turn("work", turn_timeout=600, idle_timeout=450)
    assert result.interrupted and "timed out" in result.error


def test_cannot_disable_both_watchdogs():
    session = CodexAppServerSession(client_factory=lambda **kw: FakeClient())
    with pytest.raises(ValueError, match="timeout"):
        session.run_turn("work", turn_timeout=0, idle_timeout=0)
