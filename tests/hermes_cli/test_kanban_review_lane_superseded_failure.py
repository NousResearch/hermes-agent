"""Regression: a successful handoff must retire the failure text it superseded.

A rate-limited (quota wall) requeue stamps ``tasks.last_failure_error`` without
counting a failure. The next run worked the card and handed it off for review —
a SUCCESS transition — but the stamp survived it, and
``check_respawn_guard(lane="review")`` matched that stale text against
``_RESPAWN_BLOCKER_RE`` on every dispatcher tick, returning ``blocker_auth``
forever: the reviewer was never spawned.

Observed live on the hermes-agent board 2026-11-13/14 (card t_690dfe6a):
t_ca7fa938 logged 254 ``respawn_guarded``/``blocker_auth`` events and
t_e15288d4 logged 277 over the same window, both parked in ``review`` with no
reviewer run until the column was cleared by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

_QUOTA_TEXT = (
    "pid 70000 exited rate-limited (quota wall) — requeued without counting a failure"
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


def _quota_wall_exit(conn, tid: str, pid: int) -> None:
    """Report the rate-limit sentinel exit for ``tid``'s LIVE worker and let the
    dispatcher's crash sweep book the requeue.

    The card must already be claimed (either lane): this is the exact sequence
    that produced the parked cards — a worker that hit the quota wall exits with
    ``KANBAN_RATE_LIMIT_EXIT_CODE`` and the sweep stamps
    ``tasks.last_failure_error`` without counting a failure.
    """
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (pid, tid))
    conn.commit()
    kbd._record_worker_exit(pid, _exited_status(kb.KANBAN_RATE_LIMIT_EXIT_CODE))
    kbd.detect_crashed_workers(conn)
    assert tid in getattr(kbd.detect_crashed_workers, "_last_rate_limited", [])


@pytest.fixture
def dead_workers(monkeypatch):
    """Make the crash sweep see this test's fake pids as dead."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")


def _cooldown_elapsed(conn, tid: str, seconds: int = 3600) -> None:
    """Backdate ``tid``'s ended runs so the rate-limit cooldown has elapsed.

    The parked cards sat for hours, so the transient cooldown was long gone and
    the guard's verdict was the permanent one (``blocker_auth``); without this
    the assertion would measure the cooldown instead of the defect.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - ? "
            "WHERE task_id = ? AND ended_at IS NOT NULL",
            (seconds, tid),
        )


def test_review_handoff_after_a_quota_wall_lets_the_reviewer_spawn(
    kanban_home, dead_workers,
):
    """The acceptance case: rate limit -> next run requests review -> the review
    lane releases the card instead of parking it on the requeue's own text."""
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="handoff after a quota wall", assignee="dev")

        assert kb.claim_task(conn, tid, claimer=f"{host}:w0") is not None
        _quota_wall_exit(conn, tid, 71000)
        _cooldown_elapsed(conn, tid)
        queued = kb.get_task(conn, tid)
        assert queued is not None
        assert queued.status == "ready"
        assert queued.consecutive_failures == 0, "a quota wall never spends retry budget"
        assert "pid 71000" in (queued.last_failure_error or "")
        assert "quota wall" in (queued.last_failure_error or "")
        # With the cooldown spent the READY lane releases the card (step 1) —
        # the park that follows is specific to the review lane, where the guard
        # no longer sees a rate-limit run and reads the text as a live blocker.
        assert kbd.check_respawn_guard(conn, tid) is None

        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w1")
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="ready for review", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        ) is True

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "review"
        # The acceptance case: the review lane releases the card ...
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None
        # ... and the handoff that got it there retired the text it superseded.
        assert task.last_failure_error is None, (
            "the successful handoff retires the failure text it superseded"
        )

        # Nothing is lost: the run that hit the wall keeps the text, and the
        # event log carries the requeue.
        runs = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["rate_limited", "review_requested"]
        assert "rate-limited" in runs[0]["error"]
        assert "rate_limited" in [e.kind for e in kb.list_events(conn, tid)]


def test_changes_requested_after_a_reviewer_quota_wall_releases_the_rework(
    kanban_home, dead_workers,
):
    """The same class one lane over: the REVIEWER's run hits the quota wall and a
    later reviewer run hands the card back. ``changes_requested`` lands the card
    in ``ready``; the stale text must not park the rework there."""
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="reviewer quota wall", assignee="dev")

        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w1")
        assert kb.request_review(
            conn, tid, summary="PR ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        ) is True

        assert kb.claim_review_task(conn, tid, claimer=f"{host}:r1") is not None
        _quota_wall_exit(conn, tid, 72000)
        _cooldown_elapsed(conn, tid)
        mid = kb.get_task(conn, tid)
        assert mid is not None
        assert mid.status == "review", "a reviewer run is requeued to its own lane"
        assert "rate-limited" in (mid.last_failure_error or "")

        rclaim = kb.claim_review_task(conn, tid, claimer=f"{host}:r2")
        assert rclaim is not None
        ok, implementer = kb.request_changes(
            conn, tid, reason="fix the tests", expected_run_id=rclaim.current_run_id,
        )
        assert (ok, implementer) == (True, "dev")

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert kbd.check_respawn_guard(conn, tid) is None
        assert task.last_failure_error is None


@pytest.mark.parametrize(
    "outcome, expected",
    [
        # A successful handoff outlived the stamp: the text is history.
        ("review_requested", None),
        ("changes_requested", None),
        ("completed", None),
        # A run that FAILED keeps the text live: a real quota / auth wall still
        # parks the card (the breaker is the only other stopping point).
        ("spawn_failed", "blocker_auth"),
        ("timed_out", "blocker_auth"),
        # No ended run at all: unchanged — see the curated-stem tests in
        # tests/hermes_cli/test_kanban_db.py.
        (None, "blocker_auth"),
    ],
)
def test_blocker_auth_only_reads_text_the_latest_run_still_owns(
    kanban_home, outcome, expected,
):
    """``blocker_auth`` must not trust failure text a later run superseded, even
    when the transition that wrote it never cleared the column (an install that
    predates the clearing, or a text written straight to the row)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=f"stale text / {outcome}", assignee="dev")
        if outcome is not None:
            with kb.write_txn(conn):
                conn.execute(
                    "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
                    "VALUES (?, 'dev', ?, ?, 5_000_000, 5_000_000)",
                    (tid, outcome, outcome),
                )
        conn.execute(
            "UPDATE tasks SET status = 'review', last_failure_error = ? WHERE id = ?",
            (_QUOTA_TEXT, tid),
        )
        conn.commit()

        assert kbd.check_respawn_guard(conn, tid, lane="review") == expected
