"""A model repetition-loop worker death is an INFRASTRUCTURE retry, not a coding miss.

The worker's reply degenerates, ``agent.repetition_guard`` interrupts it, and the
process dies with ``REPETITION_LOOP_INTERRUPTED`` in its last output. That death used
to fall into the generic crash branch and consume ``consecutive_failures`` — the
code-miss budget that routes genuine misses to Mack. It must instead get its own
bounded infra-retry streak: requeued without counting, stamped with the reason, and
at 3 consecutive loop deaths parked as a ROUTING FLAG (needs_input) that names the
Mack escalation. Genuine generic crashes and rate-limited requeues behave exactly as
before; the two counters never mix.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent.repetition_guard import REPETITION_LOOP_INTERRUPTED
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


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


def _dead_worker_with_log(conn, tid: str, pid: int, rc: int, last_line: str) -> None:
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
        f.write(f"{last_line}\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n")


def _loop_death(conn, tid: str, pid: int) -> None:
    _dead_worker_with_log(conn, tid, pid, 1, REPETITION_LOOP_INTERRUPTED)


def test_loop_death_requeues_without_counting(kanban_home):
    """One repetition-loop death: back to ready, ``consecutive_failures`` stays 0, the
    error is stamped for the retry worker, and the run carries the durable marker."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _loop_death(conn, tid, 72001)

        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        run = conn.execute(
            "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert REPETITION_LOOP_INTERRUPTED in (task.last_failure_error or "")
        assert kb._json_dict(run["metadata"]).get("repetition_loop") is True
        assert kbd._repetition_loop_streak(conn, tid) == 1


def test_three_loop_deaths_trip_the_mack_routing_flag(kanban_home):
    """Three consecutive loop deaths park the card blocked/needs_input with the Mack
    routing text — still WITHOUT ever touching ``consecutive_failures`` — and
    ``recompute_ready`` must not promote it back the same tick."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        for i in range(kbd._REPETITION_LOOP_FAILURE_LIMIT):
            _loop_death(conn, tid, 73000 + i)
            kbd.detect_crashed_workers(conn)
            if kb.get_task(conn, tid).status == "blocked":
                break
            kb.recompute_ready(conn, failure_limit=10)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert task.consecutive_failures == 0
        assert "3 infra retries (repetition-loop)" in (task.last_failure_error or "")
        assert "route to Mack (rework coder)" in (task.last_failure_error or "")

        kb.recompute_ready(conn, failure_limit=10)
        assert kb.get_task(conn, tid).status == "blocked"


def test_rate_limited_death_still_requeues_without_counting(kanban_home):
    """A quota-wall exit keeps today's booking: rate_limited outcome, no failure counted."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 74001, kb.KANBAN_RATE_LIMIT_EXIT_CODE, "working on it")

        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert run["outcome"] == "rate_limited"
        assert kbd._repetition_loop_streak(conn, tid) == 0


def test_genuine_generic_crash_still_counts_a_failure(kanban_home):
    """A plain crash (no repetition-loop marker) still consumes the code-miss budget."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 75001, 1, "something genuinely broke")

        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 1
        assert kbd._repetition_loop_streak(conn, tid) == 0
