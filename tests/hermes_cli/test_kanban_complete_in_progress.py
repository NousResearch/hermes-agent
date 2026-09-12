"""Regression test: ``complete_task`` must accept every non-terminal VALID_STATUS.

Background
----------
``hermes_cli.kanban_db.complete_task`` (and the matching
``hermes_cli.kanban_pr_acceptance_store.prepare_acceptance``) gates the
write on a hard-coded whitelist of pre-terminal statuses:

    ('running', 'ready', 'blocked', 'review')

``VALID_STATUSES`` (``hermes_cli/kanban_db.py``) however includes
``triage``, ``scheduled`` and the legacy ``in_progress`` -- all valid
pre-terminal states that upstream routes to legitimately:

* ``triage`` lands there from :func:`block_task` when the unblock loop
  trips ``BLOCK_RECURRENCE_LIMIT`` (see ``_route_block``).
* ``scheduled`` lands there from :func:`schedule_task` (time-gated, not
  human-blocked).
* ``in_progress`` is a legacy status name not in upstream's current
  ``VALID_STATUSES`` set but may appear on cards persisted by older
  Hermes versions or third-party tools; the gate should treat unknown
  pre-terminal statuses as a soft "already terminal" rather than
  silently dropping them.

The previous whitelist rejected every such card with
"cannot complete <id> (unknown id or terminal state)", forcing operators
to manually unblock / re-route the card before it could be closed.

This test pins the fix: in_progress / scheduled / triage MUST complete
to ``done``, terminal states (done / archived) MUST NOT re-complete.

The 13 cards originally blocked (e.g. t_d0641768, t_bc07f803, ... plus
3 sandbox cards) all moved through ``complete_task`` once the gate was
extended.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME so ``init_db`` builds a fresh board in tmp_path."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _insert_task(conn, status: str, task_id: str = "t_test") -> None:
    """Insert a single task in the given status with no parents."""
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) VALUES (?, 'test', ?, 1)",
        (task_id, status),
    )
    conn.commit()


def test_complete_task_accepts_in_progress(kanban_home: Path) -> None:
    """Cards in ``in_progress`` (legacy / third-party reopen state) MUST complete."""
    with kbc.connect_closing() as conn:
        _insert_task(conn, "in_progress")

        ok = kb.complete_task(conn, "t_test", result="verified", summary="audit")
        assert ok is True, "complete_task should accept in_progress status"

        row = conn.execute(
            "SELECT status, result, completed_at FROM tasks WHERE id = ?", ("t_test",),
        ).fetchone()
        assert row["status"] == "done", f"status should be done, got {row['status']!r}"
        assert row["result"] == "verified"
        assert row["completed_at"] is not None, "completed_at must be set"


def test_complete_task_accepts_scheduled(kanban_home: Path) -> None:
    """Cards in ``scheduled`` (time-gated, not dispatchable) MUST complete when closed manually."""
    with kbc.connect_closing() as conn:
        _insert_task(conn, "scheduled")

        ok = kb.complete_task(conn, "t_test", result="forced complete")
        assert ok is True

        row = conn.execute("SELECT status FROM tasks WHERE id = ?", ("t_test",)).fetchone()
        assert row["status"] == "done"


def test_complete_task_accepts_triage(kanban_home: Path) -> None:
    """Cards in ``triage`` (block-loop-escalated, awaiting human) MUST complete."""
    with kbc.connect_closing() as conn:
        _insert_task(conn, "triage")

        ok = kb.complete_task(conn, "t_test", result="triaged")
        assert ok is True

        row = conn.execute("SELECT status FROM tasks WHERE id = ?", ("t_test",)).fetchone()
        assert row["status"] == "done"


def test_complete_task_still_rejects_terminal_states(kanban_home: Path) -> None:
    """``done`` and ``archived`` are terminal -- complete_task must NOT re-close them."""
    with kbc.connect_closing() as conn:
        _insert_task(conn, "done", task_id="t_terminal_done")
        _insert_task(conn, "archived", task_id="t_terminal_archived")

        ok_done = kb.complete_task(conn, "t_terminal_done", result="x")
        ok_arch = kb.complete_task(conn, "t_terminal_archived", result="x")

    assert ok_done is False, "should reject done (already terminal)"
    assert ok_arch is False, "should reject archived (already terminal)"


def test_prepare_acceptance_accepts_extended_pre_terminal_statuses(kanban_home: Path) -> None:
    """``prepare_acceptance`` (kanban_pr_acceptance_store) gates on the same whitelist;
    the matching fix must keep both gates aligned so a card in ``scheduled`` /
    ``triage`` / ``in_progress`` doesn't half-clear the kanban DB and then bounce
    off the PR-acceptance store."""
    from hermes_cli import kanban_pr_acceptance_store as accept_store

    with kbc.connect_closing() as conn:
        for status in ("in_progress", "scheduled", "triage"):
            tid = f"t_accept_{status}"
            _insert_task(conn, status, task_id=tid)
            # Insert a completion_contract so the gate isn't short-circuited by
            # the ``contract in (None, 'local-only')`` early return.
            conn.execute(
                "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                ("pr_required", tid),
            )
            conn.commit()

            outcome = accept_store.prepare_acceptance(
                conn, tid, expected_run_id=None, metadata={},
            )
            assert outcome is not False, (
                f"prepare_acceptance should accept {status!r} "
                f"(mirrors the complete_task whitelist); got {outcome!r}"
            )