"""Regression tests for issue #113004: ``kanban show`` ownership fields
and ``task_runs.closed_by`` audit trail (the post-closure side of the issue;
the fence widening proposed alongside was declined by the maintainer — see
:mod:`tests.hermes_cli.test_kanban_complete_live_claim_guard` for the #111764
fence contract that pins the chosen behaviour).

The ``kanban show`` surface and the ``closed_by`` audit column together answer
the second half of the issue:

- ``show --json``'s task dict previously omitted ``claim_lock``,
  ``claim_expires``, ``worker_pid``, ``current_run_id``, and
  ``last_heartbeat_at`` — operators had no programmatic way to read who held
  a running card or which run was current.
- ``show`` text output had no live ``Ownership:`` section, so the same
  information was missing for human readers.
- ``task_runs`` recorded the original claimant (``profile``) on a closed row
  but not the caller that wrote the terminal row. A ``force=True`` close by
  a non-claimant left no audit trace of who actually closed the run.

The fix:
- ``task_runs.closed_by`` records the ``"<host>:<pid>"`` of the closer that
  wrote the terminal row; ``_end_run`` defaults the actor to ``_claimer_id()``
  so every freshly-closed run carries the trace.
- ``kanban show`` text output adds a dedicated ``Ownership:`` block on
  ``status=running`` tasks and marks a foreign close (``closed_by`` differs
  from the claim's host:pid) with ``⚠ foreign close``.
- ``kanban show --json`` exposes the live-claim fields on the task dict and
  ``closed_by`` on every runs[] entry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _claim_under(
    conn, claimer: str, *, live_worker_pid: int | None = None
) -> tuple[str, int]:
    """Claim a card under ``claimer``; optionally spawn a worker on it."""
    tid = kb.create_task(conn, title="pid-less", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=claimer) is not None
    if live_worker_pid is not None:
        kbd._set_worker_pid(conn, tid, live_worker_pid)
    return tid, kb._current_run_id(conn, tid)


class _Capturing:
    """Minimal write-buffering stdout stand-in for ``redirect_stdout``."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def write(self, s: str) -> int:
        if s:
            self._sink.append(s)
        return len(s)

    def flush(self) -> None:
        return None


def _capture_cmd_show(task_id: str, *, as_json: bool) -> str:
    buf: list[str] = []
    real = sys.stdout
    try:
        sys.stdout = _Capturing(buf)
        kcli._cmd_show(argparse.Namespace(task_id=task_id, json=as_json))
    finally:
        sys.stdout = real
    return "".join(buf)


# --- closed_by audit trail ---


def test_closed_by_records_self_on_default_close(conn):
    """Without an explicit closer, ``closed_by`` is the closing caller's
    ``_claimer_id()`` (host:pid)."""
    tid, run_id = _claim_under(conn, claimer=kb._claimer_id())
    kb.complete_task(conn, tid, result="own", expected_run_id=run_id)
    run = conn.execute(
        "SELECT closed_by FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert run["closed_by"] == kb._claimer_id()


def test_closed_by_records_operator_on_force(conn):
    """Force-override records the operator's id, not the claim's."""
    tid, _run_id = _claim_under(conn, claimer="lane-k-host:force")
    kb.complete_task(conn, tid, result="force", force=True)
    run = conn.execute(
        "SELECT closed_by, profile FROM task_runs WHERE task_id = ?", (tid,)
    ).fetchone()
    assert run["closed_by"] == kb._claimer_id()  # not lane-k-host:force
    assert run["profile"] == "coder"  # the original claimant survives


def test_show_text_marks_foreign_close(conn):
    """A closed run whose ``closed_by`` differs from its ``claim_lock`` is
    marked ``⚠ foreign close`` in ``show`` text output."""
    tid, run_id = _claim_under(conn, claimer="lane-k-host:owner")
    kb.complete_task(conn, tid, result="force", force=True)

    out = _capture_cmd_show(tid, as_json=False)
    assert "⚠ foreign close" in out
    assert "lane-k-host:owner" in out


def test_show_text_silent_on_self_close(conn):
    """A run closed by its own claimant prints ``closed_by`` with no marker —
    the same entity opened and closed the run."""
    tid, run_id = _claim_under(conn, claimer=kb._claimer_id())
    kb.complete_task(conn, tid, result="self", expected_run_id=run_id)

    out = _capture_cmd_show(tid, as_json=False)
    assert "closed_by" in out
    assert "⚠ foreign close" not in out


# --- show text + JSON output ---


def test_show_json_exposes_ownership_fields(conn):
    """``show --json`` carries ``claim_lock``, ``claim_expires``,
    ``worker_pid``, ``current_run_id``, ``last_heartbeat_at`` on the task
    dict (#113004 acceptance test)."""
    tid, run_id = _claim_under(
        conn, claimer="lane-k-host:show", live_worker_pid=os.getpid()
    )

    out = _capture_cmd_show(tid, as_json=True)
    task = json.loads(out)["task"]
    assert task["claim_lock"] == "lane-k-host:show"
    assert task["worker_pid"] == os.getpid()
    assert task["current_run_id"] == run_id
    assert task["last_heartbeat_at"] is None  # never heartbeated
    # runs[] exposes closed_by (None on a still-active run, but the key is
    # present so dashboards don't have to special-case the field).
    runs = json.loads(out)["runs"]
    assert all("closed_by" in r for r in runs)


def test_show_text_prints_ownership_section_for_running_card(conn):
    """``show`` text output adds an ``Ownership:`` block on ``status=running``
    tasks showing the claim, the worker, and the run id."""
    tid, run_id = _claim_under(
        conn, claimer="lane-k-host:text", live_worker_pid=os.getpid()
    )

    out = _capture_cmd_show(tid, as_json=False)
    assert "Ownership:" in out
    assert "lane-k-host:text" in out
    assert f"pid {os.getpid()}" in out
    assert f"#{run_id}" in out


def test_show_text_marks_pid_less_claim_lane(conn):
    """A pid-less claim (no worker spawned) prints the explanatory lane label."""
    tid, _run_id = _claim_under(conn, claimer="lane-k-host:lane", live_worker_pid=None)

    out = _capture_cmd_show(tid, as_json=False)
    assert "Ownership:" in out
    assert "pid-less claim" in out
    assert "lane-k-host:lane" in out


def test_show_text_omits_ownership_section_for_terminal_card(conn):
    """Done / ready / todo cards do not show an Ownership section — the
    fields would be NULL/0 and the block is for live claims only."""
    tid = kb.create_task(conn, title="ready", assignee="coder")
    out = _capture_cmd_show(tid, as_json=False)
    assert "Ownership:" not in out
