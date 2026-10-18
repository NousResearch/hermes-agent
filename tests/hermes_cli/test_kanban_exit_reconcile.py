"""The one-shot exit funnel is the single reconciliation point.

`exit_single_query` is where every `-q` / `-Q` turn converges before `sys.exit`.
It must reconcile only when the caller supplies a settled turn result, and it
must never let a guard failure mask the real exit code.
"""
import sys

import pytest

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER, exit_single_query


def _guard(monkeypatch, action="recorded"):
    import agent.kanban_terminal_guard as guard

    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return action

    monkeypatch.setattr(guard, "reconcile_worker_exit", fake)
    return calls


def test_reconciles_when_turn_result_is_supplied(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    calls = _guard(monkeypatch)
    result = {"completed": False, "messages": []}
    with pytest.raises(SystemExit) as exc:
        exit_single_query(0, result=result)
    assert exc.value.code == 0
    assert calls == [{"completed": False, "result": result}]


def test_skips_reconcile_without_turn_result(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    calls = _guard(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        exit_single_query(1)
    assert exc.value.code == 1
    assert calls == []


def test_never_raises_when_guard_crashes(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    import agent.kanban_terminal_guard as guard

    def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(guard, "reconcile_worker_exit", boom)
    with pytest.raises(SystemExit) as exc:
        exit_single_query(0, result={"completed": False})
    assert exc.value.code == 0
    assert KANBAN_WORKER_EXIT_TRAILER in capsys.readouterr().err


def test_trailer_is_still_written(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    _guard(monkeypatch, action="skipped")
    with pytest.raises(SystemExit):
        exit_single_query(75, result={})
    assert f"{KANBAN_WORKER_EXIT_TRAILER}75" in capsys.readouterr().err
