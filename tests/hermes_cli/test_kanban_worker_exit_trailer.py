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
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its run-tagged receipt — never reaped here."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()["current_run_id"]
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"the model said something\n\nResume this session with:\n  hermes --resume x\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc} run={int(run_id)}\n")


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


def _append_log(tid: str, text: str) -> None:
    """Append raw text to a task's worker log, like a worker's redirected stderr."""
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(text)


def _claim_dead_worker(conn, tid: str, pid: int) -> int:
    """Claim ``tid`` for a worker that is already dead; returns the new run id."""
    host = kb._claimer_id().split(":", 1)[0]
    claimed = kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    assert claimed is not None
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    task = kb.get_task(conn, tid)
    assert task is not None and task.current_run_id is not None
    return int(task.current_run_id)


def _last_event_and_run(conn, tid: str):
    ev = conn.execute(
        "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    run = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,)).fetchone()
    return ev["kind"], run


@pytest.mark.parametrize(
    "rc, stale_event",
    [(0, "protocol_violation"), (kb.KANBAN_RATE_LIMIT_EXIT_CODE, "rate_limited")],
)
def test_sigkilled_attempt_ignores_previous_attempt_trailer(kanban_home, rc, stale_event):
    """#121255: the worker log is append-mode across attempts and a SIGKILLed /
    OOM-killed worker writes no trailer — so the reap of the dead attempt must
    not book the PREVIOUS attempt's exit trailer. A stale rc=0 must not become
    a protocol violation and a stale rc=75 must not become a no-failure
    rate-limit requeue (which would keep the failure breaker from ever
    tripping on a worker that is killed every run)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        run1 = _claim_dead_worker(conn, tid, 70101)
        _append_log(tid, f"attempt one prose\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc} run={run1}\n")
        kbd.detect_crashed_workers(conn)
        assert _last_event_and_run(conn, tid)[0] == stale_event

        run2 = _claim_dead_worker(conn, tid, 70202)
        assert run2 != run1
        _append_log(tid, "attempt two heartbeat, then hung (SIGKILLed, no trailer)\n")
        kbd.detect_crashed_workers(conn)

        kind, run = _last_event_and_run(conn, tid)
        assert kind == "crashed"
        assert "not alive" in (run["error"] or "")
        assert kb._json_dict(run["metadata"]).get("exit_code") != rc
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.consecutive_failures == 1


@pytest.mark.parametrize("rc", [0, kb.KANBAN_RATE_LIMIT_EXIT_CODE])
def test_legacy_untagged_trailer_never_classifies_a_known_run(kanban_home, rc):
    """Upgrade window: attempt N ran the old writer and left an untagged trailer, then
    attempt N+1 — which HAS a ``current_run_id`` — is SIGKILLed before writing its own
    receipt. The legacy line is not evidence about N+1, so the kill stays a plain crash
    instead of re-booking N's rc=0 (protocol violation) or rc=75 (no-failure rate-limit
    requeue that would keep the breaker from tripping) (#121255)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _claim_dead_worker(conn, tid, 70301)
        _append_log(tid, f"legacy attempt prose\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n")
        kbd.detect_crashed_workers(conn)
        run2 = _claim_dead_worker(conn, tid, 70402)
        _append_log(tid, "attempt two heartbeat, then hung (SIGKILLed, no trailer)\n")
        kbd.detect_crashed_workers(conn)

        # The killed attempt's OWN booking: a following budget trip (``gave_up``) must not
        # hide what the reap of run 2 wrote.
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE run_id=? ORDER BY id", (run2,)).fetchall()
        booked = [(r["kind"], kb._json_dict(r["payload"])) for r in events
                  if r["kind"] in ("crashed", "protocol_violation", "rate_limited")]
        assert len(booked) == 1, [(r["kind"], r["payload"]) for r in events]
        kind, payload = booked[0]
        assert kind == "crashed", (kind, payload)
        assert payload.get("exit_code") is None, payload
        run = conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run2,)).fetchone()
        assert kb._json_dict(run["metadata"]).get("exit_code") is None


@pytest.mark.parametrize("rc", [0, kb.KANBAN_RATE_LIMIT_EXIT_CODE])
def test_untagged_receipt_only_counts_without_a_run_identity(kanban_home, monkeypatch, rc):
    """A pre-anchor receipt stays readable for a caller that has no run identity (a legacy
    row), and stops speaking for any run that does (#121255)."""
    monkeypatch.setattr(
        kb,
        "read_worker_log",
        lambda *_args, **_kwargs: f"prose\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n",
    )
    assert kbd._worker_log_exit_code("t_legacy") == rc
    assert kbd._worker_log_exit_code("t_legacy", run_id=7) is None


def test_current_attempt_trailer_still_books_its_own_exit(kanban_home):
    """The run anchor must not swallow a live signal: an attempt whose OWN
    trailer carries its own run id is still booked as its exit code says."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        run1 = _claim_dead_worker(conn, tid, 70101)
        _append_log(tid, f"old attempt\n\n{KANBAN_WORKER_EXIT_TRAILER}1 run={run1 - 1}\n")
        _append_log(tid, f"attempt one prose\n\n{KANBAN_WORKER_EXIT_TRAILER}0 run={run1}\n")
        kbd.detect_crashed_workers(conn)
        kind, run = _last_event_and_run(conn, tid)
        assert kind == "protocol_violation"
        assert kb._json_dict(run["metadata"]).get("exit_code") == 0


def test_exit_single_query_tags_trailer_with_run_id(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    with pytest.raises(SystemExit) as exc:
        exit_single_query(kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert exc.value.code == kb.KANBAN_RATE_LIMIT_EXIT_CODE
    err = capsys.readouterr().err
    assert f"{KANBAN_WORKER_EXIT_TRAILER}{kb.KANBAN_RATE_LIMIT_EXIT_CODE} run=42" in err

    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    with pytest.raises(SystemExit):
        exit_single_query(0)
    assert f"{KANBAN_WORKER_EXIT_TRAILER}0\n" in capsys.readouterr().err


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
