"""CLI guard: ``hermes kanban complete`` must refuse a blocked card (#101785).

A ``blocked`` card is a deliberate stop — a ``needs_input`` human gate or
the dispatcher's failure circuit breaker. Completing it with a single
command silently erased that gate and promoted the children (``scheduled``
was already refused; ``blocked`` just was not covered by the guard). The
CLI now refuses the transition and names the block kind; ``--force`` keeps
the deliberate "the gate is satisfied, close it out" path available.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock


def _run_complete(monkeypatch, *, status="blocked", block_kind="needs_input",
                  force=False, goal_mode=False):
    # Resolve the CLI entrypoint at call time: earlier tests in a full-suite
    # run can reload hermes_cli modules, and a collection-time import would
    # patch a different module object than the one under test.
    from hermes_cli.kanban import _cmd_complete

    fake_task = SimpleNamespace(
        goal_mode=goal_mode,
        title="probe",
        body=None,
        status=status,
        block_kind=block_kind,
    )
    fake_conn = MagicMock()
    complete_calls: list[str] = []

    @contextmanager
    def fake_connect_closing():
        yield fake_conn

    def fake_complete_task(conn, tid, **kw):
        complete_calls.append(tid)
        return True

    monkeypatch.setattr("hermes_cli.kanban.kb.get_task", lambda conn, tid: fake_task)
    monkeypatch.setattr("hermes_cli.kanban.kb.complete_task", fake_complete_task)
    monkeypatch.setattr("hermes_cli.kanban.kb.connect_closing", fake_connect_closing)
    monkeypatch.setattr("hermes_cli.kanban._worker_run_id_for", lambda _: None)

    args = argparse.Namespace(
        task_ids=["t1"], summary="probe", result=None, metadata=None,
        force=force,
    )
    return _cmd_complete(args), complete_calls


class TestCompleteRefusesBlocked:
    def test_blocked_refused_without_force(self, monkeypatch):
        rc, complete_calls = _run_complete(monkeypatch)
        assert rc != 0, "completing a blocked card must fail"
        assert complete_calls == [], "complete_task must NOT run on a blocked card"

    def test_refusal_names_block_kind_and_force_hint(self, monkeypatch, capsys):
        _run_complete(monkeypatch, block_kind="needs_input")
        err = capsys.readouterr().err
        assert "needs_input" in err
        assert "--force" in err
        assert "unblock" in err

    def test_blocked_without_kind_falls_back_in_message(self, monkeypatch, capsys):
        _run_complete(monkeypatch, block_kind=None)
        err = capsys.readouterr().err
        assert "blocked: blocked" in err

    def test_force_completes_blocked_card(self, monkeypatch):
        rc, complete_calls = _run_complete(monkeypatch, force=True)
        assert rc == 0
        assert complete_calls == ["t1"]

    def test_ready_still_completes_without_force(self, monkeypatch):
        rc, complete_calls = _run_complete(monkeypatch, status="ready")
        assert rc == 0
        assert complete_calls == ["t1"]
