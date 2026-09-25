"""Behavior tests for the ``hermes kanban reopen-done`` CLI surface.

``kanban_db.reopen_done_task`` is the sanctioned route back from a
verified-bad ``done``/``archived`` card, but it had no caller: an operator
could not reach it. These tests drive the real
``hermes_cli.kanban._cmd_reopen_done`` handler against a temp board and
assert observable behavior (task status, exit code, comments, JSON payload),
never source text.

The fixture patches ``Path.home`` AND sets both ``HERMES_HOME`` and
``HERMES_KANBAN_HOME``: ``HERMES_HOME`` alone does not redirect
``kanban_db_path()``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from tests.hermes_cli._kanban_modules import KanbanModules

kb = KanbanModules()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


def _reopen_ns(task_id, *, ids=None, reason=None, as_json=False):
    return argparse.Namespace(
        task_id=task_id,
        reason=list(reason or []),
        ids=list(ids or []) or None,
        json=as_json,
    )


def _set_status(task_id: str, status: str) -> None:
    with kb.connect() as conn:
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()


def _status(task_id: str) -> str:
    with kb.connect() as conn:
        return kb.get_task(conn, task_id).status


def _make_task(title: str = "t", *, parents=None) -> str:
    with kb.connect() as conn:
        return kb.create_task(
            conn, title=title, assignee="setup", parents=list(parents or []),
        )


# ---------------------------------------------------------------------------
# 1 + 2: the terminal statuses both have a route back.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("terminal_status", ["done", "archived"])
def test_terminal_task_is_reopened_to_ready(kanban_home, capsys, terminal_status):
    tid = _make_task("verified bad")
    _set_status(tid, terminal_status)

    rc = kb_cli._cmd_reopen_done(_reopen_ns(tid))

    assert rc == 0
    assert _status(tid) == "ready"
    out = capsys.readouterr().out
    assert f"Reopened {tid} -> ready" in out


# ---------------------------------------------------------------------------
# 3: landing status is a RELATION to the parents' state, not a literal.
# ---------------------------------------------------------------------------

def test_landing_status_tracks_parent_completion(kanban_home, capsys):
    """Same card shape, two parent states -> two different landing statuses."""
    open_parent = _make_task("open parent")
    child_of_open = _make_task("child A", parents=[open_parent])
    _set_status(child_of_open, "done")

    closed_parent = _make_task("closed parent")
    child_of_closed = _make_task("child B", parents=[closed_parent])
    _set_status(child_of_closed, "done")
    _set_status(closed_parent, "done")

    assert _status(open_parent) not in ("done", "archived")
    assert _status(closed_parent) in ("done", "archived")

    assert kb_cli._cmd_reopen_done(_reopen_ns(child_of_open)) == 0
    assert kb_cli._cmd_reopen_done(_reopen_ns(child_of_closed)) == 0
    out = capsys.readouterr().out

    landed_open = _status(child_of_open)
    landed_closed = _status(child_of_closed)

    # The card with an unfinished parent must NOT be dispatchable, while the
    # one whose parents are satisfied must be. The difference is the point.
    assert landed_open != landed_closed
    assert landed_open == "todo"
    assert landed_closed == "ready"

    # The reported landing status must be the one the board actually holds,
    # per id — not a fixed string.
    assert f"Reopened {child_of_open} -> {landed_open}" in out
    assert f"Reopened {child_of_closed} -> {landed_closed}" in out


def test_json_reports_the_real_todo_landing_status(kanban_home, capsys):
    """A todo-landing reopen must not be reported as 'ready'."""
    parent = _make_task("open parent")
    child = _make_task("child", parents=[parent])
    _set_status(child, "done")

    assert kb_cli._cmd_reopen_done(_reopen_ns(child, as_json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    actual = _status(child)
    assert actual != "ready"  # the parent is still open
    assert payload["status"] == actual


# ---------------------------------------------------------------------------
# 4: non-terminal statuses are refused, loudly and without mutating.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("live_status", ["running", "review"])
def test_non_terminal_task_is_refused(kanban_home, capsys, live_status):
    tid = _make_task("still in flight")
    _set_status(tid, live_status)

    rc = kb_cli._cmd_reopen_done(_reopen_ns(tid))

    assert rc == 1
    assert _status(tid) == live_status
    captured = capsys.readouterr()
    assert f"cannot reopen {tid}" in captured.err
    assert live_status in captured.err


# ---------------------------------------------------------------------------
# 5: bulk --ids, partial progress is real.
# ---------------------------------------------------------------------------

def test_bulk_mixed_results_keeps_partial_progress(kanban_home, capsys):
    good_a = _make_task("good a")
    good_b = _make_task("good b")
    bad = _make_task("bad")
    _set_status(good_a, "done")
    _set_status(good_b, "archived")
    _set_status(bad, "running")

    rc = kb_cli._cmd_reopen_done(_reopen_ns(good_a, ids=[bad, good_b, good_a]))

    assert rc == 1
    assert _status(good_a) == "ready"
    assert _status(good_b) == "ready"
    assert _status(bad) == "running"
    captured = capsys.readouterr()
    assert f"cannot reopen {bad}" in captured.err
    # Deduped: the repeated positional id is reported exactly once.
    assert captured.out.count(f"Reopened {good_a} -> ready") == 1


# ---------------------------------------------------------------------------
# 6: --json shapes.
# ---------------------------------------------------------------------------

def test_json_single_id_is_a_flat_object(kanban_home, capsys):
    tid = _make_task("solo")
    _set_status(tid, "done")

    rc = kb_cli._cmd_reopen_done(_reopen_ns(tid, as_json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    assert payload["task_id"] == tid
    assert payload["reopened"] is True
    assert payload["status"] == _status(tid) == "ready"
    assert payload["error"] is None


def test_json_bulk_is_a_list_with_per_id_flags(kanban_home, capsys):
    ok_id = _make_task("ok")
    bad_id = _make_task("bad")
    _set_status(ok_id, "done")
    _set_status(bad_id, "running")

    rc = kb_cli._cmd_reopen_done(
        _reopen_ns(ok_id, ids=[bad_id], reason=["needs", "rework"], as_json=True)
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 2
    by_id = {r["task_id"]: r for r in payload}
    assert by_id[ok_id]["reopened"] is True
    assert by_id[ok_id]["status"] == _status(ok_id)
    assert by_id[ok_id]["reason"] == "needs rework"
    assert by_id[ok_id]["error"] is None
    assert by_id[bad_id]["reopened"] is False
    assert by_id[bad_id]["status"] is None
    assert by_id[bad_id]["error"]


# ---------------------------------------------------------------------------
# 7: the reason is durable.
# ---------------------------------------------------------------------------

def test_reason_lands_in_a_durable_comment(kanban_home):
    tid = _make_task("bad merge")
    _set_status(tid, "done")

    rc = kb_cli._cmd_reopen_done(
        _reopen_ns(tid, reason=["work", "never", "pushed"])
    )

    assert rc == 0
    with kb.connect() as conn:
        bodies = [c.body for c in kb.list_comments(conn, tid)]
    assert any("work never pushed" in b for b in bodies), bodies


# ---------------------------------------------------------------------------
# 8: the failure breaker is reset by this route.
# ---------------------------------------------------------------------------

def test_consecutive_failures_is_reset(kanban_home):
    tid = _make_task("flaky")
    _set_status(tid, "done")
    with kb.connect() as conn:
        conn.execute(
            "UPDATE tasks SET consecutive_failures=3 WHERE id=?", (tid,)
        )
        conn.commit()

    def _failures() -> int:
        with kb.connect() as conn:
            row = conn.execute(
                "SELECT consecutive_failures FROM tasks WHERE id=?", (tid,)
            ).fetchone()
        return int(row["consecutive_failures"])

    assert _failures() > 0
    assert kb_cli._cmd_reopen_done(_reopen_ns(tid)) == 0
    assert _failures() == 0


# ---------------------------------------------------------------------------
# 9: a delegated child agent cannot retract its own completion.
# ---------------------------------------------------------------------------

def _kanban_command_ns(action: str, **kw) -> argparse.Namespace:
    ns = argparse.Namespace(kanban_action=action, board=None, **kw)
    return ns


def test_delegated_child_cannot_reopen_done(kanban_home, capsys, monkeypatch):
    tid = _make_task("child's own card")
    _set_status(tid, "done")

    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    rc = kb_cli.kanban_command(
        _kanban_command_ns(
            "reopen-done", task_id=tid, reason=[], ids=None, json=False,
        )
    )

    assert rc == 1
    assert _status(tid) == "done"
    assert "cannot mutate Kanban tasks via the CLI" in capsys.readouterr().err


def test_operator_context_can_reopen_done_through_kanban_command(
    kanban_home, capsys,
):
    """The refusal above must be about delegation, not about the verb."""
    tid = _make_task("operator card")
    _set_status(tid, "done")

    rc = kb_cli.kanban_command(
        _kanban_command_ns(
            "reopen-done", task_id=tid, reason=[], ids=None, json=False,
        )
    )

    assert rc == 0
    assert _status(tid) == "ready"
    assert f"Reopened {tid} -> ready" in capsys.readouterr().out


def test_reopen_done_retracts_done_descendants(kanban_home):
    parent = _make_task("parent")
    child = _make_task("child", parents=[parent])
    _set_status(parent, "done")
    _set_status(child, "done")

    assert kb_cli._cmd_reopen_done(_reopen_ns(parent)) == 0

    assert _status(parent) == "ready"
    assert _status(child) == "todo"
