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


def _dead_worker_with_log(
    conn, tid: str, pid: int, rc: int, log_text: str | None = None
) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its log — never reaped here."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    text = (
        log_text
        if log_text is not None
        else (
            f"the model said something\n\nResume this session with:\n  hermes --resume x\n\n"
            f"{KANBAN_WORKER_EXIT_TRAILER}{rc}\n"
        )
    )
    with open(log, "a", encoding="utf-8") as f:
        f.write(text)


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


def test_clean_exit_on_a_transient_wall_requeues_without_counting(kanban_home):
    """A worker whose turn "completed" against a 429/503 wall exits 0 without a terminal
    board call; the wall is only in its log (#133795). The sweep must book the run
    ``rate_limited`` (requeue, no failure count, respawn cooldown) instead of charging
    the protocol-violation budget."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wall", assignee="a")
        _dead_worker_with_log(
            conn, tid, 70051, 0,
            log_text=(
                "Query: work kanban task\n"
                "Error code: 429 - {'type': 'rate_limit_error', 'msg': 'Too many requests'}\n"
                "giving up on this attempt\n\n"
                f"{KANBAN_WORKER_EXIT_TRAILER}0\n"
            ),
        )

        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,)).fetchone()
        task = kb.get_task(conn, tid)
        assert run["outcome"] == "rate_limited"
        assert kb._json_dict(run["metadata"]).get("transient_resource") is True
        assert task.status == "ready"
        assert task.consecutive_failures == 0  # a capacity wall spends no retry budget
        assert "transient provider wall" in (task.last_failure_error or "")
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"


def test_clean_exit_log_that_merely_mentions_the_words_stays_a_violation(kanban_home):
    """The transient-wall match is anchored to error shapes: a worker that only
    *talks about* rate limits while skipping the terminal call keeps the corrective
    protocol-violation booking, so a chatty log cannot park the card on the
    rate-limit cooldown forever."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="chat", assignee="a")
        _dead_worker_with_log(
            conn, tid, 70052, 0,
            log_text=(
                "the model explained that HTTP 429 is documented in the integration guide\n"
                "and finished the turn without a board call\n\n"
                f"{KANBAN_WORKER_EXIT_TRAILER}0\n"
            ),
        )

        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,)).fetchone()
        task = kb.get_task(conn, tid)
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"]).get("protocol_violation") is True
        assert kbd._protocol_violation_streak(conn, tid) == 1


def test_stale_wall_from_an_earlier_run_does_not_requeue_later_runs(kanban_home):
    """The worker log is append-mode across re-runs (size-rotated only): run 1 hit a
    real 429 and was requeued; run 2 worked, genuinely skipped the terminal board
    call and exited 0. Run 2's tail still holds run 1's wall line, so only the
    CURRENT run's segment — after the previous trailer — may be scanned: run 2
    books the protocol violation, not another rate-limit requeue (#133795)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale", assignee="a")
        _dead_worker_with_log(
            conn,
            tid,
            70061,
            0,
            log_text=(
                "Error code: 429 - {'type': 'rate_limit_error', 'msg': 'Too many requests'}\n\n"
                f"{KANBAN_WORKER_EXIT_TRAILER}0\n"
            ),
        )
        kbd.detect_crashed_workers(conn)
        run1 = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run1["outcome"] == "rate_limited"

        _dead_worker_with_log(
            conn,
            tid,
            70062,
            0,
            log_text=(
                "did the work, forgot the board call\n\n"
                f"{KANBAN_WORKER_EXIT_TRAILER}0\n"
            ),
        )
        kbd.detect_crashed_workers(conn)
        run2 = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run2["outcome"] == "crashed"
        assert kb._json_dict(run2["metadata"]).get("protocol_violation") is True


def test_recovered_midturn_wall_stays_a_violation(kanban_home):
    """A 429 the SDK retried past is followed by the rest of the turn's output, so
    the wall sits far from the trailer: only the tail of the current run's segment
    is scanned, and a recovered wall does not buy the neutral requeue (#133795)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="recovered", assignee="a")
        _dead_worker_with_log(
            conn,
            tid,
            70080,
            0,
            log_text=(
                "Error code: 429 - retrying\n"
                + "ordinary output line\n" * (kbd._TRANSIENT_EXIT_TAIL_LINES + 5)
                + f"{KANBAN_WORKER_EXIT_TRAILER}0\n"
            ),
        )

        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"]).get("protocol_violation") is True


def test_transient_wall_backstop_charges_violation_after_budget(kanban_home):
    """``rate_limited`` is neutral for both streaks and the breaker, so a log-based
    transient booking that keeps misreading would cycle one task through the
    rate-limit cooldown forever. Once the trailing streak of transient requeues
    reaches the protocol-violation budget, the next transient-looking clean exit
    is charged as a counted, operator-visible protocol violation (#133795)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="loop", assignee="a")
        wall = (
            "Error code: 429 - {'type': 'rate_limit_error', 'msg': 'Too many requests'}\n\n"
            f"{KANBAN_WORKER_EXIT_TRAILER}0\n"
        )
        for i in range(kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT):
            _dead_worker_with_log(conn, tid, 70070 + i, 0, log_text=wall)
            kbd.detect_crashed_workers(conn)
            run = conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            assert run["outcome"] == "rate_limited"

        # Budget spent: the same 429-looking exit now books a counted violation.
        _dead_worker_with_log(conn, tid, 70074, 0, log_text=wall)
        kbd.detect_crashed_workers(conn)
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"]).get("protocol_violation") is True
