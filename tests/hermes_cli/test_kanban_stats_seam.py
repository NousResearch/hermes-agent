"""Seam + behaviour tests for the kanban stats extraction (R5-S1).

Covers two things:

1. Seam identity: ``hermes_cli.kanban_db`` re-exports ``board_stats`` /
   ``task_age`` / ``_to_epoch`` from ``hermes_cli.stats_mixin`` so every
   module-access caller (``kanban.py``, dashboard plugin) resolves to the
   SAME function objects — no double definitions, no cycle.
2. Behaviour of the moved functions themselves, imported directly from
   the new module: ``board_stats`` aggregation (empty board, mixed
   statuses, archived exclusion, no-ready ``oldest_ready_age_seconds``)
   and ``task_age`` created/started/completed permutations.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import stats_mixin


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_status(conn, task_id: str, status: str) -> None:
    """Direct SQL status transition (existing test convention)."""
    with conn:
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


# ---------------------------------------------------------------------------
# Seam identity (shim)
# ---------------------------------------------------------------------------


def test_shim_reexports_same_objects():
    """kanban_db.<name> must BE stats_mixin.<name> — not a copy."""
    assert kb.board_stats is stats_mixin.board_stats
    assert kb.task_age is stats_mixin.task_age
    assert kb._to_epoch is stats_mixin._to_epoch


def test_shim_import_roundtrip():
    """Both modules import cleanly with no cycle (one-way edge only)."""
    import hermes_cli.kanban_db  # noqa: F401
    import hermes_cli.stats_mixin  # noqa: F401


# ---------------------------------------------------------------------------
# board_stats aggregation
# ---------------------------------------------------------------------------


def test_board_stats_empty_board(kanban_home):
    with kb.connect() as conn:
        stats = stats_mixin.board_stats(conn)
    assert stats["by_status"] == {}
    assert stats["by_assignee"] == {}
    assert stats["oldest_ready_age_seconds"] is None
    assert isinstance(stats["now"], int)


def test_board_stats_mixed_statuses_and_archived_excluded(kanban_home):
    with kb.connect() as conn:
        r1 = kb.create_task(conn, title="r1")
        r2 = kb.create_task(conn, title="r2")
        d1 = kb.create_task(conn, title="d1", assignee="alice")
        w1 = kb.create_task(conn, title="w1", assignee="alice")
        a1 = kb.create_task(conn, title="a1", assignee="bob")
        # Default create_task status is "ready" (no parents); transition the mix.
        _set_status(conn, r1, "ready")
        _set_status(conn, r2, "ready")
        _set_status(conn, d1, "done")
        _set_status(conn, w1, "running")
        # Archive a1.
        kb.archive_task(conn, a1)

        stats = stats_mixin.board_stats(conn)

    # Archived tasks are excluded from both rollups.
    assert stats["by_status"] == {"ready": 2, "done": 1, "running": 1}
    assert stats["by_assignee"] == {"alice": {"done": 1, "running": 1}}
    # At least one ready task exists -> oldest_ready_age is a real age.
    assert stats["oldest_ready_age_seconds"] is not None
    assert stats["oldest_ready_age_seconds"] >= 0


def test_board_stats_oldest_ready_age_absent_without_ready_tasks(kanban_home):
    with kb.connect() as conn:
        d1 = kb.create_task(conn, title="only-done")
        b1 = kb.create_task(conn, title="only-blocked", initial_status="blocked")
        _set_status(conn, d1, "done")
        stats = stats_mixin.board_stats(conn)
    assert stats["oldest_ready_age_seconds"] is None
    assert stats["by_status"] == {"done": 1, "blocked": 1}


# ---------------------------------------------------------------------------
# task_age permutations
# ---------------------------------------------------------------------------


def test_task_age_created_only(kanban_home):
    """No started/completed timestamps: only created age is present."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="fresh")
        task = kb.get_task(conn, tid)
        ages = stats_mixin.task_age(task)
    assert ages["created_age_seconds"] is not None
    assert ages["created_age_seconds"] >= 0
    assert ages["started_age_seconds"] is None
    assert ages["time_to_complete_seconds"] is None


def test_task_age_started_and_completed(kanban_home):
    """With started+completed set, all three metrics are populated and the
    time-to-complete is measured from the started timestamp."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ran")
        # Patch timestamps directly to control the arithmetic.
        now = int(time.time())
        with conn:
            conn.execute(
                "UPDATE tasks SET started_at = ?, completed_at = ? WHERE id = ?",
                (now - 60, now - 10, tid),
            )
        task = kb.get_task(conn, tid)
        ages = stats_mixin.task_age(task)
    assert ages["created_age_seconds"] is not None
    assert ages["started_age_seconds"] is not None
    assert ages["time_to_complete_seconds"] is not None
    assert ages["time_to_complete_seconds"] == 50


# ---------------------------------------------------------------------------
# _to_epoch normalisation (direct)
# ---------------------------------------------------------------------------


def test_to_epoch_normalises_inputs():
    now = int(time.time())
    assert stats_mixin._to_epoch(now) == now
    assert stats_mixin._to_epoch(float(now)) == now
    assert stats_mixin._to_epoch(str(now)) == now
    assert stats_mixin._to_epoch("  " + str(now) + "  ") == now
    assert stats_mixin._to_epoch(None) is None
    assert stats_mixin._to_epoch("") is None
    # ISO-8601 with Z suffix round-trips to epoch seconds.
    iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    assert stats_mixin._to_epoch(iso) == now
    # Garbage that is neither numeric nor ISO -> None (no crash).
    assert stats_mixin._to_epoch("not-a-timestamp") is None
