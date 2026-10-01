"""Tests: the rate-limited requeue budget must actually bound the requeue loop.

t_02738474 added a bounded budget for the exit-75 (EX_TEMPFAIL) requeue path:
after ``_RATE_LIMIT_REQUEUE_FAILURE_LIMIT`` consecutive quota walls the card is
supposed to be blocked with an operator-readable message instead of being
released to ``ready`` forever (the 472x-respawn loop). These tests drive the
real ``detect_crashed_workers`` reclaim path end to end, so they fail if the
budget is unreachable in production wiring -- not merely wrong in isolation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _rewind(conn, tid):
    """Push started_at back so the post-fork launch grace window cannot skip us."""
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute("UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,))
    conn.commit()


def _one_rate_limited_cycle(conn, tid, host):
    """Claim the card, let the worker die on exit 75, reclaim it. One requeue."""
    kb.claim_task(conn, tid, claimer=f"{host}:w")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    _rewind(conn, tid)
    kbd._record_worker_exit(dead.pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)
    kbd.detect_crashed_workers(conn)
    row = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    return row["status"], row["consecutive_failures"]


def test_streak_counter_counts_consecutive_quota_walls(conn):
    """Unit-level sanity: the streak helper sees a wall trail and resets on
    any other outcome. (This part of the change is wired and must keep working.)"""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="streak", assignee="w")

    for _ in range(3):
        _one_rate_limited_cycle(conn, tid, host)
    assert kbd._rate_limit_streak(conn, tid) == 3, "3 consecutive quota walls must count as 3"

    # A non-rate-limited closed run breaks the streak.
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at) "
        "VALUES (?, 'crashed', 0, 1)",
        (tid,),
    )
    conn.commit()
    assert kbd._rate_limit_streak(conn, tid) == 0, "a real outcome must reset the quota streak"


def test_budget_blocks_card_after_repeated_quota_walls(conn):
    """REGRESSION for the 472x-respawn loop: after the documented budget the
    card must be blocked with the quota-wall message, not released to ``ready``."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="quota loop", assignee="w")
    limit = kbd._resolve_rate_limit_requeue_limit()
    assert limit > 0, "budget must be enabled by default"

    statuses = []
    for _ in range(limit + 2):
        statuses.append(_one_rate_limited_cycle(conn, tid, host)[0])
        if statuses[-1] == "blocked":
            break

    final = conn.execute(
        "SELECT status, last_failure_error, consecutive_failures FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert final["status"] == "blocked", (
        f"card was requeued {len(statuses)}x past a {limit}-wall budget without ever "
        f"being blocked; statuses={statuses!r} -- the requeue budget is unreachable "
        f"in the detect_crashed_workers wiring"
    )
    assert "rate-limited requeue budget exhausted" in (final["last_failure_error"] or ""), (
        f"blocked card must carry the operator message; got {final['last_failure_error']!r}"
    )
    # A quota wall is not a task failure: below the trip it must not consume the
    # unified failure budget either.
    assert final["consecutive_failures"] == 1, (
        "an exhausted quota-wall budget must cost exactly one unified failure "
        f"(the trip itself), got {final['consecutive_failures']}"
    )


def test_detect_crashed_workers_forwards_rate_limited_details_to_accounting(conn, monkeypatch):
    """ROOT CAUSE (t_1a94a63d): the sweep COLLECTS rate-limited requeues into
    ``rate_limited_details``, but ``detect_crashed_workers`` hands ONLY
    ``crash_details`` to ``_account_crashes``. A quota wall is deliberately kept
    out of ``crash_details``, so the ``elif dead.rate_limited:`` branch that
    t_02738474 added never executes and the requeue budget is dead code.

    Spy on the call so this pins the wiring contract (and passes the moment the
    call site is fixed) rather than restating the symptom.
    """
    seen: list[int] = []
    real = kbd._account_crashes

    def spy(connection, crash_details, *args, **kwargs):
        seen.append(len(crash_details))
        return real(connection, crash_details, *args, **kwargs)

    monkeypatch.setattr(kbd, "_account_crashes", spy)

    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="wiring", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:w")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    _rewind(conn, tid)
    kbd._record_worker_exit(dead.pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)

    kbd.detect_crashed_workers(conn)

    assert sum(seen) == 1, (
        "detect_crashed_workers must hand the quota wall to _account_crashes "
        f"(expected 1 entry forwarded, got {seen}) — rate_limited_details is "
        "collected but never passed, so the requeue budget cannot trip"
    )


def test_env_kill_switch_zero_restores_unbounded_requeue(conn, monkeypatch):
    """``HERMES_KANBAN_RATE_LIMIT_REQUEUE_LIMIT=0`` is the documented rollback
    lever: the card must keep requeueing forever."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_REQUEUE_LIMIT", "0")
    assert kbd._resolve_rate_limit_requeue_limit() == 0

    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="rollback lever", assignee="w")
    for _ in range(8):
        status, _ = _one_rate_limited_cycle(conn, tid, host)
        assert status != "blocked", "kill switch 0 must restore the unbounded requeue"
    final = conn.execute("SELECT consecutive_failures FROM tasks WHERE id=?", (tid,)).fetchone()
    assert final["consecutive_failures"] == 0, "quota walls never consume the failure budget"