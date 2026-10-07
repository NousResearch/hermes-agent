"""Forward-progress signal + worker-liveness classification.

The heartbeat answers "is the process making API traffic?"; these tests pin the
new signal that answers "is it getting anywhere?" — a repeated identical tool
call must not refresh ``last_progress_at`` even though ``last_heartbeat_at``
stays fresh, and the board-side classifier must name the resulting stall/loop.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_progress as kp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim_running(conn, *, title="job", assignee="worker"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    return tid


# ---------------------------------------------------------------------------
# tool_signature — stable, order-independent, argument-shape-sensitive
# ---------------------------------------------------------------------------

def test_signature_is_stable_and_key_order_independent():
    a = kp.tool_signature("terminal", {"cmd": "ls", "cwd": "/tmp"})
    b = kp.tool_signature("terminal", {"cwd": "/tmp", "cmd": "ls"})
    assert a == b, "canonical JSON must make key order irrelevant"


def test_signature_changes_with_arguments_and_name():
    base = kp.tool_signature("terminal", {"cmd": "ls"})
    assert base != kp.tool_signature("terminal", {"cmd": "pwd"})
    assert base != kp.tool_signature("read", {"cmd": "ls"})


def test_signature_never_raises_on_unserialisable_args():
    assert kp.tool_signature("t", {"x": object()})  # default=str


# ---------------------------------------------------------------------------
# ProgressTracker — a repeated call is not progress, a new call is
# ---------------------------------------------------------------------------

def test_repeated_identical_call_does_not_advance_progress():
    tracker = kp.ProgressTracker(now=1000)
    tracker.note("terminal", {"cmd": "ls"})
    first_progress = tracker.last_progress_at
    tracker.note("terminal", {"cmd": "ls"})
    tracker.note("terminal", {"cmd": "ls"})
    snap = tracker.snapshot()
    assert snap.progress_repeat_count == 3
    assert snap.tool_calls_total == 3
    assert snap.last_progress_at == first_progress, "repeats must not refresh progress"


def test_distinct_call_advances_progress_and_resets_repeats():
    tracker = kp.ProgressTracker(now=1000)
    tracker.note("terminal", {"cmd": "ls"})
    tracker.note("terminal", {"cmd": "ls"})
    tracker.note("read", {"path": "a.py"})
    snap = tracker.snapshot()
    assert snap.progress_repeat_count == 1
    assert snap.tool_calls_total == 3
    assert tracker.distinct_signatures == 2


# ---------------------------------------------------------------------------
# classify_liveness — the four states
# ---------------------------------------------------------------------------

def test_classify_working():
    v = kp.classify_liveness(
        now=10_000, status="running", pid_alive=True, started_at=9_000,
        last_heartbeat_at=9_990, last_progress_at=9_990, progress_repeat_count=1,
    )
    assert v.state == kp.WORKING


def test_classify_looping_beats_stall():
    v = kp.classify_liveness(
        now=10_000, status="running", pid_alive=True, started_at=9_000,
        last_heartbeat_at=9_990, last_progress_at=9_000, progress_repeat_count=50,
    )
    assert v.state == kp.LOOPING
    assert "50x" in v.reason


def test_classify_stalled_when_heartbeat_fresh_but_progress_old():
    v = kp.classify_liveness(
        now=10_000, status="running", pid_alive=True, started_at=8_000,
        last_heartbeat_at=9_990, last_progress_at=8_000, progress_repeat_count=1,
        stall_seconds=900,
    )
    assert v.state == kp.STALLED


def test_classify_zombie_on_dead_pid():
    v = kp.classify_liveness(now=10_000, status="running", pid_alive=False, started_at=9_000)
    assert v.state == kp.ZOMBIE


def test_classify_zombie_on_stale_heartbeat():
    v = kp.classify_liveness(
        now=100_000, status="running", pid_alive=True, started_at=9_000,
        last_heartbeat_at=9_000, last_progress_at=9_000,
        zombie_seconds=3600,
    )
    assert v.state == kp.ZOMBIE


def test_classify_unknown_without_signals():
    v = kp.classify_liveness(now=10_000, status="running")
    assert v.state == kp.UNKNOWN


def test_classify_not_running_is_unknown():
    v = kp.classify_liveness(
        now=10_000, status="done", pid_alive=True, started_at=9_000,
        last_heartbeat_at=9_990, last_progress_at=9_990,
    )
    assert v.state == kp.UNKNOWN


# ---------------------------------------------------------------------------
# heartbeat_worker — the progress rollup reaches the board
# ---------------------------------------------------------------------------

def test_heartbeat_persists_progress_on_task_and_run(kanban_home):
    conn = kbc.connect()
    try:
        tid = _claim_running(conn)
        run_id = kb._current_run_id(conn, tid)
        snap = kp.ProgressTracker(now=1234).snapshot()
        snap.last_progress_at = 1234
        snap.progress_repeat_count = 7
        snap.tool_calls_total = 9

        assert kbd.heartbeat_worker(conn, tid, progress=snap, expected_run_id=run_id)

        task = conn.execute(
            "SELECT last_progress_at, progress_repeat_count, tool_calls_total FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert (task["last_progress_at"], task["progress_repeat_count"], task["tool_calls_total"]) == (1234, 7, 9)

        run = conn.execute(
            "SELECT last_progress_at, progress_repeat_count, tool_calls_total FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert (run["last_progress_at"], run["progress_repeat_count"], run["tool_calls_total"]) == (1234, 7, 9)
    finally:
        conn.close()


def test_heartbeat_without_progress_leaves_signal_untouched(kanban_home):
    """An explicit ``kanban_heartbeat`` (no progress payload) must not zero a
    signal a previous auto-heartbeat wrote."""
    conn = kbc.connect()
    try:
        tid = _claim_running(conn)
        run_id = kb._current_run_id(conn, tid)
        assert kbd.heartbeat_worker(
            conn, tid, expected_run_id=run_id,
            progress={"last_progress_at": 55, "progress_repeat_count": 2, "tool_calls_total": 4},
        )
        assert kbd.heartbeat_worker(conn, tid, note="still here", expected_run_id=run_id)

        row = conn.execute(
            "SELECT last_progress_at, progress_repeat_count FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["last_progress_at"] == 55
        assert row["progress_repeat_count"] == 2
    finally:
        conn.close()


def test_legacy_board_gains_progress_columns_without_data_loss(kanban_home):
    """The additive migration must add the columns to an existing board."""
    conn = kbc.connect()
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert {"last_progress_at", "progress_repeat_count", "tool_calls_total"} <= cols
        run_cols = {r["name"] for r in conn.execute("PRAGMA table_info(task_runs)")}
        assert {"last_progress_at", "progress_repeat_count", "tool_calls_total"} <= run_cols
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Liveness tolerances — pinned, so the numbers cannot drift silently
# ---------------------------------------------------------------------------
#
# The progress signal only helps if you know how long the existing rails wait.
# Today's tolerances, and the blind spot they share, are:
#
#   - claim TTL          15 min   (kb.DEFAULT_CLAIM_TTL_SECONDS) — a live worker
#                                  gets its claim *extended*, not killed.
#   - claim heartbeat    1 h      (kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS)
#     max stale                    a live PID with a heartbeat older than this is
#                                  reclaimed anyway (#29747 gap 3).
#   - stale reclaim      >4 h run AND no heartbeat for 1 h
#                                  (config default 14400, kb_dispatch
#                                  _STALE_HEARTBEAT_GAP_SECONDS=3600).
#   - failed spawns      2        (kbd.DEFAULT_FAILURE_LIMIT) before the breaker
#                                  auto-blocks a card.
#   - runtime cap        opt-in   (tasks.max_runtime_seconds NULL by default;
#                                  ``--max-runtime`` on create).
#
# Blind spot: ``last_heartbeat_at`` is refreshed by the auto-heartbeat bridge for
# *any* ``_touch_activity`` tick — provider retries, stream reconnects, repeated
# model calls — so a worker looping on one tool call keeps every reaper above
# satisfied for hours. ``last_progress_at`` (kanban_progress) is the signal that
# does not, and classify_liveness() names the state in the meantime.

def test_liveness_tolerances_are_pinned():
    from hermes_cli import config_defaults

    assert kb.DEFAULT_CLAIM_TTL_SECONDS == 15 * 60
    assert kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS == 60 * 60
    assert kbd._STALE_HEARTBEAT_GAP_SECONDS == 3600
    assert kbd.DEFAULT_FAILURE_LIMIT == 2
    # Shipped default of the stale window the live dispatcher reads.
    assert config_defaults.DEFAULT_CONFIG["kanban"]["dispatch_stale_timeout_seconds"] == 14400


def test_max_runtime_is_opt_in_by_default(kanban_home):
    """A card created without ``--max-runtime`` carries no cap: the dispatcher's
    ``enforce_max_runtime`` cannot fire, which is why an estimate/default is needed."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="uncapped", assignee="worker")
        row = conn.execute("SELECT max_runtime_seconds FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["max_runtime_seconds"] is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Diagnostics surface — the dashboard drawer and `hermes kanban diagnostics`
# both render these, so one rule covers both display paths.
# ---------------------------------------------------------------------------

def test_diagnostic_flags_stalled_worker():
    from hermes_cli import kanban_diagnostics as diag

    row = {
        "id": "t_stall", "status": "running", "started_at": 8_000,
        "last_heartbeat_at": 9_990, "last_progress_at": 8_000,
        "progress_repeat_count": 1, "tool_calls_total": 5,
    }
    out = diag.compute_task_diagnostics(row, [], [], now=10_000)
    assert any(d.kind == "worker_stalled" for d in out), [d.kind for d in out]


def test_diagnostic_flags_looping_worker():
    from hermes_cli import kanban_diagnostics as diag

    row = {
        "id": "t_loop", "status": "running", "started_at": 9_000,
        "last_heartbeat_at": 9_990, "last_progress_at": 9_990,
        "progress_repeat_count": 50, "tool_calls_total": 60,
    }
    out = diag.compute_task_diagnostics(row, [], [], now=10_000)
    assert any(d.kind == "worker_looping" for d in out), [d.kind for d in out]


def test_diagnostic_quiet_for_working_worker():
    from hermes_cli import kanban_diagnostics as diag

    row = {
        "id": "t_ok", "status": "running", "started_at": 9_000,
        "last_heartbeat_at": 9_990, "last_progress_at": 9_990,
        "progress_repeat_count": 1, "tool_calls_total": 12,
    }
    out = diag.compute_task_diagnostics(row, [], [], now=10_000)
    assert not any(d.kind.startswith("worker_") for d in out)


def test_diagnostic_quiet_without_progress_signal():
    """A legacy row with no progress columns is unknown, not stalled."""
    from hermes_cli import kanban_diagnostics as diag

    row = {"id": "t_legacy", "status": "running", "started_at": 9_000, "last_heartbeat_at": 9_990}
    out = diag.compute_task_diagnostics(row, [], [], now=10_000)
    assert not any(d.kind.startswith("worker_") for d in out)
