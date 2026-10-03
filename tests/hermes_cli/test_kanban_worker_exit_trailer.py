"""A dead Kanban worker is booked the same way whichever process notices it.

``_recent_worker_exits`` is filled by ``os.waitpid`` and so only knows children of
the process running the sweep; a per-tick ``hermes kanban dispatch`` process finds it
empty. The worker's own exit trailer in its log is the durable witness the sweep reads
instead, and a tripped protocol-violation budget must hold the card until an operator
unblocks it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.active_sessions import MAX_CONCURRENT_SESSIONS, SESSION_COORDINATION_UNAVAILABLE
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER, exit_single_query


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _dead_worker_with_log(conn, tid: str, pid: int, rc: int) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its log — never reaped here."""
    _dead_worker_with_custom_log(
        conn, tid, pid,
        f"the model said something\n\nResume this session with:\n  hermes --resume x\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n",
    )


def _dead_worker_with_custom_log(conn, tid: str, pid: int, body: str) -> None:
    """Claim ``tid`` for a dead worker and write an exact log body."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(body)


@pytest.mark.parametrize(
    "rc, event, failure_counted",
    [(0, "protocol_violation", False), (kb.KANBAN_RATE_LIMIT_EXIT_CODE, "rate_limited", False)],
)
def test_fresh_process_sweep_books_the_logged_exit_code(kanban_home, rc, event, failure_counted):
    """Empty reap registry + exit trailer in the log: a clean exit is the protocol violation
    (marker, streak, no unified-budget hit) and a 75 is a rate-limit requeue — not a bare
    ``pid N not alive`` crash that counts a failure."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 70001, rc)

        kbd.detect_crashed_workers(conn)

        ev = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        run = conn.execute(
            "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,)).fetchone()
        task = kb.get_task(conn, tid)
        assert ev["kind"] == event
        assert "not alive" not in (run["error"] or "")
        assert task.status == "ready"
        assert task.consecutive_failures == (1 if failure_counted else 0)
        # The decoded rc lands in the run row so quota (75) vs crash stays tellable after
        # the fact even though the worker_output tail is trimmed (#113611).
        assert kb._json_dict(run["metadata"]).get("exit_code") == rc
        if rc == 0:
            assert kb._json_dict(run["metadata"]).get("protocol_violation") is True
            assert kbd._protocol_violation_streak(conn, tid) == 1
            assert KANBAN_WORKER_EXIT_TRAILER not in (run["error"] or "")
        else:
            assert run["outcome"] == "rate_limited"


def test_active_session_capacity_refusal_defers_without_counting_failure(kanban_home):
    """A worker refused by ``MAX_CONCURRENT_SESSIONS`` is host capacity: requeue,
    cooldown, and do not spend the card's failure budget."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="capacity", assignee="a")
        _dead_worker_with_custom_log(
            conn, tid, 72001,
            f"hermes-refusal-reason: {MAX_CONCURRENT_SESSIONS}\n"
            "Hermes is at the active session limit (4/4). Held by: desktop x3, cli.\n"
            f"\n{KANBAN_WORKER_EXIT_TRAILER}1\n",
        )

        assert kbd.detect_crashed_workers(conn) == []

        ev = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        run = conn.execute(
            "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,)).fetchone()
        task = kb.get_task(conn, tid)
        assert task is not None
        metadata = kb._json_dict(run["metadata"])
        assert ev["kind"] == "capacity_deferred"
        assert run["outcome"] == "capacity_deferred"
        assert metadata.get("refusal_reason") == MAX_CONCURRENT_SESSIONS
        assert metadata.get("exit_code") == 1
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert getattr(kbd.detect_crashed_workers, "_last_capacity_deferred") == [tid]
        assert kbd.check_respawn_guard(conn, tid) == "session_capacity_cooldown"


def test_stale_capacity_refusal_marker_does_not_hide_later_crash(kanban_home):
    """Append-only logs must classify the latest run, not reuse a prior capacity refusal."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale-capacity", assignee="a")
        _dead_worker_with_custom_log(
            conn,
            tid,
            72011,
            f"hermes-refusal-reason: {MAX_CONCURRENT_SESSIONS}\n"
            "Hermes is at the active session limit (4/4).\n"
            f"\n{KANBAN_WORKER_EXIT_TRAILER}1\n",
        )
        assert kbd.detect_crashed_workers(conn) == []

        _dead_worker_with_custom_log(
            conn,
            tid,
            72012,
            f"ordinary crash from a later run\n\n{KANBAN_WORKER_EXIT_TRAILER}1\n",
        )

        assert kbd.detect_crashed_workers(conn) == [tid]

        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,)
        ).fetchall()
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        task = kb.get_task(conn, tid)
        assert task is not None
        event_kinds = [row["kind"] for row in events]
        assert "capacity_deferred" in event_kinds
        assert event_kinds[-1] == "crashed"
        assert run["outcome"] == "crashed"
        metadata = kb._json_dict(run["metadata"])
        assert metadata.get("refusal_reason") is None
        assert metadata.get("exit_code") == 1
        assert "MAX_CONCURRENT_SESSIONS" not in metadata.get("worker_output", "")
        assert "ordinary crash" in metadata.get("worker_output", "")
        assert task.status == "ready"
        assert task.consecutive_failures == 1


def test_non_capacity_active_session_refusal_remains_a_crash(kanban_home):
    """Ownership/registry refusals are correctness failures, not capacity; the
    dispatcher must not bypass them by treating every refusal as retryable."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="registry-error", assignee="a")
        _dead_worker_with_custom_log(
            conn, tid, 72002,
            f"hermes-refusal-reason: {SESSION_COORDINATION_UNAVAILABLE}\n"
            "Hermes could not read the active-session registry.\n"
            f"\n{KANBAN_WORKER_EXIT_TRAILER}1\n",
        )

        assert kbd.detect_crashed_workers(conn) == [tid]

        ev = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,)).fetchone()
        task = kb.get_task(conn, tid)
        assert task is not None
        assert ev["kind"] == "crashed"
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"]).get("exit_code") == 1
        assert task.status == "ready"
        assert task.consecutive_failures == 1


def test_violation_budget_trip_holds_until_operator_unblock(kanban_home):
    """The third consecutive clean exit trips the violation budget and ``recompute_ready``
    must not promote the card back the same tick (``consecutive_failures`` is still below
    ``failure_limit``); ``unblock_task`` lifts the hold."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="loop", assignee="a")
        for i in range(kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT):
            _dead_worker_with_log(conn, tid, 71000 + i, 0)
            kbd.detect_crashed_workers(conn)
            kb.recompute_ready(conn, failure_limit=10)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures < 10

        kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        kb.recompute_ready(conn, failure_limit=10)
        assert kb.get_task(conn, tid).status == "ready"


def test_plain_budget_trip_still_auto_recovers(kanban_home):
    """A unified-budget trip carries no ``sticky`` marker, so the two recovery paths on main
    survive: raising the dispatcher ``failure_limit`` past the counter promotes the card, and
    ``assign_task`` to a fresh profile (counter reset by design) promotes it too."""
    with kbc.connect() as conn:
        tids = [kb.create_task(conn, title=t, assignee="a") for t in ("raise-limit", "reassign")]
        for tid in tids:
            for i in range(2):
                kbd._record_task_failure(
                    conn, tid, error=f"boom{i}", outcome="crashed", failure_limit=2,
                    release_claim=False, end_run=False,
                )
            assert kb.get_task(conn, tid).status == "blocked"
        assert kb.recompute_ready(conn, failure_limit=2) == 0

        assert kb.recompute_ready(conn, failure_limit=5) == 2
        assert kb.get_task(conn, tids[0]).status == "ready"

        for i in range(2):
            kbd._record_task_failure(
                conn, tids[1], error=f"again{i}", outcome="crashed", failure_limit=2,
                release_claim=False, end_run=False,
            )
        assert kb.get_task(conn, tids[1]).status == "blocked"
        kb.assign_task(conn, tids[1], "other-profile")
        assert kb.recompute_ready(conn, failure_limit=2) == 1
        assert kb.get_task(conn, tids[1]).status == "ready"


def test_exit_single_query_writes_trailer_only_for_kanban_workers(monkeypatch, capsys):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with pytest.raises(SystemExit) as exc:
        exit_single_query(1)
    assert exc.value.code == 1
    assert KANBAN_WORKER_EXIT_TRAILER not in capsys.readouterr().err

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    with pytest.raises(SystemExit) as exc:
        exit_single_query(kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert exc.value.code == kb.KANBAN_RATE_LIMIT_EXIT_CODE
    assert f"{KANBAN_WORKER_EXIT_TRAILER}{kb.KANBAN_RATE_LIMIT_EXIT_CODE}" in capsys.readouterr().err
