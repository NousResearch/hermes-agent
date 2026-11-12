"""Regression tests for the run-658 auto-execution incident.

A card blocked by the dispatcher's circuit breaker (``gave_up``) was reassigned to
another profile. ``assign_task`` reset ``consecutive_failures`` to zero on the
assignee change, which disarmed the breaker; the next ``recompute_ready`` pass then
promoted the card to ``ready``, the dispatcher claimed it and spawned a worker that
nobody asked for — it ran for minutes with production credentials after the card had
been blocked again.

These tests pin the four fixes:

* an assignment no longer clears the failure counter unless ``reset_breaker=True``;
* ``--clear-breaker`` (CLI) is the explicit way to do it;
* a promotion event now records why it happened;
* a block without ``--kind`` is recorded as ``needs_input`` so a human gate stays
  distinguishable from a breaker trip;
* worker termination signals the whole process group, not just the recorded pid.
"""

from __future__ import annotations

import argparse
import signal
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture()
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as c:
        yield c


def _make(conn, *, status: str = "blocked", assignee: str = "alpha") -> str:
    # ``create_task`` only accepts 'blocked'/'running' as initial status; the other
    # phases (todo/ready) are reached by a status flip, as the dispatcher does.
    tid = kb.create_task(
        conn, title="fixture", body="fixture", assignee=assignee, initial_status="blocked",
    )
    if status != "blocked":
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
    return tid


def _set_failures(conn, tid: str, failures: int) -> None:
    with kbc.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = ?, last_failure_error = 'x' WHERE id = ?",
            (failures, tid),
        )


def _events(conn, tid: str, kind: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id", (tid, kind)
    ).fetchall()


def _status(conn, tid: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()[0]


# --------------------------------------------------------------------------- #
# 1. Assignment must not rearm the breaker
# --------------------------------------------------------------------------- #

def test_assign_keeps_failure_counter(conn):
    tid = _make(conn)
    _set_failures(conn, tid, 7)

    assert kb.assign_task(conn, tid, "beta") is True

    failures = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()[0]
    assert failures == 7, "a plain assignment must not reset the circuit breaker"


def test_assign_clear_breaker_is_explicit(conn):
    tid = _make(conn)
    _set_failures(conn, tid, 3)

    assert kb.assign_task(conn, tid, "beta", reset_breaker=True) is True

    failures = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()[0]
    assert failures == 0


def test_breaker_blocked_card_is_not_promoted_by_reassignment(conn):
    """The run-658 scenario, end to end."""
    tid = _make(conn)
    _set_failures(conn, tid, 7)
    with kbc.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?,?,?,0)",
            (tid, "gave_up", '{"failures": 7, "effective_limit": 2}'),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?,?,?,0)",
            (tid, "unblocked", None),
        )

    kb.assign_task(conn, tid, "orchestrator")
    kb.recompute_ready(conn)

    assert _status(conn, tid) == "blocked"
    assert _events(conn, tid, "promoted") == []
    assert kb.claim_task(conn, tid, claimer="host:w1") is None, "no worker may be spawnable"


# --------------------------------------------------------------------------- #
# 2. Promotion events carry their cause
# --------------------------------------------------------------------------- #

def test_promoted_event_records_cause(conn):
    tid = _make(conn, status="todo")

    kb.recompute_ready(conn)

    assert _status(conn, tid) == "ready"
    payload = _events(conn, tid, "promoted")[0]["payload"]
    assert '"cause": "recompute_ready"' in (payload or "")


# --------------------------------------------------------------------------- #
# 3. block without --kind is a human gate
# --------------------------------------------------------------------------- #

def test_block_without_kind_defaults_to_needs_input(conn, monkeypatch):
    captured: dict = {}

    def fake_block(_conn, tid, *, reason=None, kind=None, expected_run_id=None):
        captured.update(task_id=tid, reason=reason, kind=kind)
        return True

    class _Landing:
        status = "blocked"
        block_kind = "needs_input"

    monkeypatch.setattr(kb, "block_task", fake_block)
    monkeypatch.setattr(kb, "get_task", lambda _conn, _tid: _Landing())
    monkeypatch.setattr(kb, "add_comment", lambda *a, **k: True)
    monkeypatch.setattr(kanban_cli, "_profile_author", lambda: "tester")

    rc = kanban_cli._cmd_block(
        argparse.Namespace(task_id="t_x", reason=["motif"], kind=None, ids=None)
    )

    assert rc == 0
    assert captured["kind"] == "needs_input"


def test_block_explicit_kind_is_respected(conn, monkeypatch):
    captured: dict = {}

    def fake_block(_conn, tid, *, reason=None, kind=None, expected_run_id=None):
        captured.update(kind=kind)
        return True

    class _Landing:
        status = "todo"
        block_kind = "dependency"

    monkeypatch.setattr(kb, "block_task", fake_block)
    monkeypatch.setattr(kb, "get_task", lambda _conn, _tid: _Landing())
    monkeypatch.setattr(kb, "add_comment", lambda *a, **k: True)
    monkeypatch.setattr(kanban_cli, "_profile_author", lambda: "tester")

    rc = kanban_cli._cmd_block(
        argparse.Namespace(task_id="t_x", reason=[], kind="dependency", ids=None)
    )

    assert rc == 0
    assert captured["kind"] == "dependency"


# --------------------------------------------------------------------------- #
# 4. Worker termination reaches the whole process group
# --------------------------------------------------------------------------- #

def test_signal_worker_tree_signals_group_and_pid(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(kbd.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(kbd.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(kbd.os, "killpg", lambda pg, sig: calls.append(("group", pg, sig)))

    delivered = kbd._signal_worker_tree(
        lambda pid, sig: calls.append(("pid", pid, sig)), 4242, signal.SIGTERM
    )

    assert delivered is True
    assert ("group", 4242, signal.SIGTERM) in calls
    assert ("pid", 4242, signal.SIGTERM) in calls


def test_signal_worker_tree_falls_back_to_pid(monkeypatch):
    calls: list[tuple] = []

    def boom(_pid):
        raise ProcessLookupError

    monkeypatch.setattr(kbd.os, "getpgid", boom)

    delivered = kbd._signal_worker_tree(
        lambda pid, sig: calls.append(("pid", pid, sig)), 4242, signal.SIGKILL
    )

    assert delivered is True
    assert calls == [("pid", 4242, signal.SIGKILL)]
