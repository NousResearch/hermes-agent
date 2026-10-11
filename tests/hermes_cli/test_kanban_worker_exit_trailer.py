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


def _dead_worker_with_log(conn, tid: str, pid: int, rc: int) -> None:
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
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"the model said something\n\nResume this session with:\n  hermes --resume x\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n")


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


@pytest.mark.parametrize("max_retries", [1, 2])
def test_rate_limit_exit_spends_explicit_attempt_ceiling_before_dispatch(
    kanban_home, monkeypatch, all_assignees_spawnable, max_retries,
):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    observed = []

    def observe_exit(_hook, task_id, **_fields):
        if _hook != "on_kanban_worker_exited":
            return
        with kbc.connect() as committed:
            observed.append((task_id, kb.get_task(committed, task_id).status))

    monkeypatch.setattr(kb, "_kanban_observer_consumed", lambda hook: hook == "on_kanban_worker_exited")
    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", observe_exit)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="bounded quota attempt", assignee="a", max_retries=max_retries)
        for attempt in range(max_retries):
            _dead_worker_with_log(conn, tid, 72000 + attempt, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
            if attempt + 1 == max_retries:
                result = kbd.dispatch_once(conn, spawn_fn=lambda *_args: pytest.fail("blocked task spawned"))
                assert result.rate_limited == [tid]
                assert result.auto_blocked == [tid]
                assert result.crashed == []
            else:
                assert tid not in kbd.detect_crashed_workers(conn)
            task = kb.get_task(conn, tid)
            assert task.consecutive_failures == attempt + 1
            assert task.status == ("blocked" if attempt + 1 == max_retries else "ready")

        run = kb.latest_run(conn, tid)
        assert run.outcome == "rate_limited"
        assert "without counting a failure" not in (run.error or "")
        assert run.metadata["exit_code"] == kb.KANBAN_RATE_LIMIT_EXIT_CODE
        gave_up = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,),
        ).fetchone()
        assert gave_up is not None
        assert kb._json_dict(gave_up["payload"])["trigger_outcome"] == "rate_limited"

        spawned = []
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, _workspace: spawned.append(task.id))
        assert not result.spawned and not spawned
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "blocked"
        assert observed == [(tid, "ready")] * (max_retries - 1) + [(tid, "blocked")]


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_bounded_rate_limit_reclaim_is_durable_before_next_process(
    kanban_home, lane,
):
    """A dispatcher that dies as soon as reclaim commits must not lose the attempt."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="one quota attempt", assignee="a", max_retries=1)
        if lane == "review":
            implementation = kb.claim_task(conn, tid)
            assert implementation is not None
            assert kb.request_review(
                conn, tid, summary="ready for review", reviewer="reviewer",
                expected_run_id=implementation.current_run_id,
            )
            review = kb.claim_review_task(conn, tid)
            assert review is not None
            pid = 73002
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
                (pid, int(time.time()) - 120, tid),
            )
            conn.commit()
            log = kb.worker_log_path(tid)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"{KANBAN_WORKER_EXIT_TRAILER}{kb.KANBAN_RATE_LIMIT_EXIT_CODE}\n")
        else:
            _dead_worker_with_log(conn, tid, 73001, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

        # This is the old process's final call: no in-memory accounting survives it.
        sweep = kbd._reclaim_dead_workers(conn)
        assert sweep.rate_limited == [tid]

    with kbc.connect() as restarted:
        task = kb.get_task(restarted, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1
        assert kb.claim_task(restarted, tid) is None
        assert kb.claim_review_task(restarted, tid) is None
        run = kb.latest_run(restarted, tid)
        assert run.outcome == run.status == "rate_limited"
        assert run.metadata["exit_code"] == kb.KANBAN_RATE_LIMIT_EXIT_CODE
        gave_up = restarted.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,),
        ).fetchone()
        assert gave_up is not None
        assert kb._json_dict(gave_up["payload"])["retry_status"] == lane


@pytest.mark.parametrize("lane", ["ready", "review"])
@pytest.mark.parametrize("exit_code", [0, 1, kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE, "timeout"])
def test_failed_worker_accounting_survives_reclaim_process_exit(
    kanban_home, lane, exit_code,
):
    """One failed attempt and its closed run must commit together before the next dispatcher."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="one failed attempt", assignee="a", max_retries=1,
            max_runtime_seconds=1 if exit_code == "timeout" else None,
        )
        implementation = kb.claim_task(conn, tid)
        assert implementation is not None
        if lane == "review":
            assert kb.request_review(
                conn, tid, summary="ready for review", reviewer="reviewer",
                expected_run_id=implementation.current_run_id,
            )
            assert kb.claim_review_task(conn, tid) is not None
        pid = 74001
        started = int(time.time()) - 120
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
                (pid, started, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at=? WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
                (started, tid),
            )
        if exit_code == "timeout":
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None) == [tid]
        else:
            log = kb.worker_log_path(tid)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"{KANBAN_WORKER_EXIT_TRAILER}{exit_code}\n")
            assert kbd._reclaim_dead_workers(conn).crashed == [tid]

    with kbc.connect() as restarted:
        task = kb.get_task(restarted, tid)
        assert (task.status, task.consecutive_failures) == ("blocked", 1)
        assert kb.claim_task(restarted, tid) is None
        assert kb.claim_review_task(restarted, tid) is None
        run = restarted.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert run["outcome"] == ("timed_out" if exit_code == "timeout" else "crashed")
        gave_up = restarted.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,),
        ).fetchone()
        assert gave_up is not None
        assert kb._json_dict(gave_up["payload"])["retry_status"] == lane


@pytest.mark.parametrize("exit_code", [1, "timeout"])
def test_failed_worker_accounting_error_rolls_back_release_and_run(
    kanban_home, monkeypatch, exit_code,
):
    """A failure after closing the run cannot leave a released task with no attempt spent."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rollback failed attempt", assignee="a", max_retries=1,
            max_runtime_seconds=1 if exit_code == "timeout" else None,
        )
        assert kb.claim_task(conn, tid) is not None
        pid = 74002
        started = int(time.time()) - 120
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
                (pid, started, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at=? WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
                (started, tid),
            )
        if exit_code != "timeout":
            log = kb.worker_log_path(tid)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"{KANBAN_WORKER_EXIT_TRAILER}{exit_code}\n")
        original_append = kb._append_event

        def fail_gave_up(connection, task_id, kind, payload, **kwargs):
            if kind == "gave_up":
                raise RuntimeError("injected accounting failure")
            return original_append(connection, task_id, kind, payload, **kwargs)

        monkeypatch.setattr(kb, "_append_event", fail_gave_up)
        with pytest.raises(RuntimeError, match="injected accounting failure"):
            if exit_code == "timeout":
                kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None)
            else:
                kbd._reclaim_dead_workers(conn)

    with kbc.connect() as restarted:
        task = kb.get_task(restarted, tid)
        assert (task.status, task.consecutive_failures, task.worker_pid) == ("running", 0, pid)
        assert kb.latest_run(restarted, tid).ended_at is None
        kinds = [event.kind for event in kb.list_events(restarted, tid)]
        assert "gave_up" not in kinds
        assert "crashed" not in kinds
        assert "timed_out" not in kinds


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
