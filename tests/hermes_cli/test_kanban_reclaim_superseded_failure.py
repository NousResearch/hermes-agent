"""Regression: reclaim closes must not keep a superseded quota text live.

Follow-up to the review-lane fix (card t_690dfe6a, PR #10): a rate-limited
requeue stamps ``tasks.last_failure_error`` without counting a failure. When
the NEXT run is closed by a reclaim pass that never worked the card — a
TTL-expired claim (``release_stale_claims``), broken claim bookkeeping
(``reconcile_orphaned_running``) or a missing heartbeat
(``detect_stale_running``) — the latest ended run is ``stale``/``reclaimed``:
the rate-limit early return in ``check_respawn_guard`` no longer fires, and
the stale quota text matched the blocker pattern on every dispatcher tick,
parking the card ``blocker_auth`` forever.

The operator path (``reclaim_task``) already cleared the column via
``_clear_failure_counter``. The two automatic paths that close the run
WITHOUT booking their own error text (orphan reconcile, heartbeat-stale)
left the earlier stamp live; the TTL path replaces it with its own
``stale_lock=`` booking, which the guard must not read as a wall either.
A reclaim did not produce the text, so it does not own it: the column is
cleared on the close, and the read side treats ``stale``/``reclaimed``
latest runs as non-owners (a card already parked on an old install unsticks
on the next tick).

A REAL wall still holds: while the latest run IS the rate-limit run the
cooldown spaces the retries, and a run that genuinely failed with quota text
(``spawn_failed`` etc.) still parks ``blocker_auth`` — a re-manifesting wall
re-stamps on the next real attempt, and the breaker counts the reclaims.
"""

from __future__ import annotations

import time
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


@pytest.fixture
def dead_workers(monkeypatch):
    """Make the crash sweep see this test's fake pids as dead."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")


def _quota_wall_exit(conn, tid: str, pid: int) -> None:
    """Report the rate-limit sentinel exit for ``tid``'s LIVE worker and let the
    dispatcher's crash sweep book the requeue (stamps the quota text without
    counting a failure)."""
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (pid, tid))
    conn.commit()
    kbd._record_worker_exit(pid, _exited_status(kb.KANBAN_RATE_LIMIT_EXIT_CODE))
    kbd.detect_crashed_workers(conn)
    assert tid in getattr(kbd.detect_crashed_workers, "_last_rate_limited", [])


def _cooldown_elapsed(conn, tid: str, seconds: int = 3600) -> None:
    """Backdate ``tid``'s ended runs so the rate-limit cooldown has elapsed."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - ? "
            "WHERE task_id = ? AND ended_at IS NOT NULL",
            (seconds, tid),
        )


def _quota_wall_then_release(conn, tid: str, pid: int) -> str:
    """The shared setup: claim, hit the quota wall, let the cooldown elapse.

    Returns the host prefix for the second claim. Pins the REAL-wall side of
    the argument on the way through: while the latest run IS the rate-limit
    run the cooldown holds, and only once it elapses does the ready lane
    release the card for the (doomed) second attempt."""
    host = kb._claimer_id().split(":", 1)[0]
    assert kb.claim_task(conn, tid, claimer=f"{host}:w0") is not None
    _quota_wall_exit(conn, tid, pid)
    assert "quota wall" in (kb.get_task(conn, tid).last_failure_error or "")
    # A real wall still parks — transiently, while the latest run is the
    # rate-limit run and the cooldown runs.
    assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
    _cooldown_elapsed(conn, tid)
    assert kbd.check_respawn_guard(conn, tid) is None
    return host


def test_stale_claim_reclaim_does_not_park_on_superseded_quota_text(
    kanban_home, dead_workers,
):
    """The acceptance case: rate limit -> next run's claim expires -> the TTL
    reclaim closes it; with the cooldown elapsed the ready lane must release
    the card, not park it on the requeue's own text."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale claim after a quota wall", assignee="dev")
        host = _quota_wall_then_release(conn, tid, 71000)

        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w1")
        assert claimed is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(time.time()) - 3600, tid),
            )
        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        runs = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["rate_limited", "reclaimed"]
        # The TTL reclaim books its OWN failure (breaker still counts); the
        # earlier quota stamp no longer owns the column.
        assert task.consecutive_failures == 1
        assert "stale_lock=" in (task.last_failure_error or "")
        # The acceptance case: the guard releases the card.
        assert kbd.check_respawn_guard(conn, tid) is None
        # Nothing is lost: the run that hit the wall keeps the text.
        assert "rate-limited" in runs[0]["error"]


def test_orphan_reconcile_does_not_park_on_superseded_quota_text(
    kanban_home, dead_workers,
):
    """The same class through broken claim bookkeeping: the reconcile closes
    the run WITHOUT booking an error of its own, so it must clear the text it
    did not write instead of leaving the quota stamp live under a
    ``reclaimed`` latest run."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="orphan after a quota wall", assignee="dev")
        host = _quota_wall_then_release(conn, tid, 72000)

        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w1")
        assert claimed is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_lock = NULL, claim_expires = NULL "
                "WHERE id = ?",
                (tid,),
            )
        assert tid in kbd.reconcile_orphaned_running(conn)

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        runs = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["rate_limited", "reclaimed"]
        # The reconcile wrote no failure text: it clears the column it did
        # not write (the breaker counter is untouched — reconcile
        # deliberately does not book).
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        # The acceptance case: the guard releases the card.
        assert kbd.check_respawn_guard(conn, tid) is None
        # Nothing is lost: the run that hit the wall keeps the text, and the
        # event log carries both closes.
        assert "rate-limited" in runs[0]["error"]
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "rate_limited" in kinds and "reconciled" in kinds


def test_heartbeat_stale_reclaim_does_not_park_on_superseded_quota_text(
    kanban_home, dead_workers,
):
    """The ``stale`` outcome arm: a run with no heartbeat progress is closed
    ``stale`` without booking an error of its own — the same clear applies."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="heartbeat-stale after a quota wall", assignee="dev")
        host = _quota_wall_then_release(conn, tid, 73000)

        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w1")
        assert claimed is not None
        old = int(time.time()) - 1_000
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL "
                "WHERE id = ?",
                (old, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (old, claimed.current_run_id),
            )
        assert tid in kbd.detect_stale_running(conn, stale_timeout_seconds=1)

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        runs = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["rate_limited", "stale"]
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        # The acceptance case: the guard releases the card.
        assert kbd.check_respawn_guard(conn, tid) is None
        assert "rate-limited" in runs[0]["error"]


@pytest.mark.parametrize(
    "outcome, expected",
    [
        # Reclaim closes did not work the card: they own no text, so an
        # earlier quota stamp must not park under them (old-install rows
        # where the close never cleared the column unstick on the next tick).
        ("stale", None),
        ("reclaimed", None),
        # A run that genuinely FAILED with the quota text still owns it:
        # a real wall still parks (the breaker is the other stopping point).
        ("spawn_failed", "blocker_auth"),
        ("timed_out", "blocker_auth"),
    ],
)
def test_reclaim_outcomes_do_not_own_the_earlier_text(
    kanban_home, outcome, expected,
):
    """Read-side half: ``blocker_auth`` must not trust failure text a
    ``stale``/``reclaimed`` latest run did not produce, while a genuine
    failure with quota text still parks."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=f"reclaim ownership / {outcome}", assignee="dev")
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
        assert kbd.check_respawn_guard(conn, tid, lane="review") == expected
