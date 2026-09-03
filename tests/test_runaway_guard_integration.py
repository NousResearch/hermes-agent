"""Executable integration scenario for the standalone runaway-guard proposal."""
import os
import tempfile
import time
from pathlib import Path

home = Path(tempfile.mkdtemp(prefix="kanban-runaway-")) / ".hermes"
os.environ["HERMES_HOME"] = str(home)
os.environ["HERMES_KANBAN_CRASH_GRACE_SECONDS"] = "0"

import hermes_cli.kanban_db as kb


def expire(conn, task_id, pid):
    now = int(time.time())
    host = kb._claimer_id().split(":", 1)[0]
    assert kb.claim_task(conn, task_id, claimer=f"{host}:worker-{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, claim_expires=? WHERE id=?",
        (pid, now - 1, task_id),
    )
    conn.commit()


kb._pid_alive = lambda _pid: False
with kb.connect() as conn:
    # Scenario 1: two stale-lock reclaims consume max_retries=2 and stop.
    stale_task = kb.create_task(
        conn, title="stale reclaim bounded", assignee="coder", max_retries=2
    )
    expire(conn, stale_task, 70101)
    assert kb.release_stale_claims(conn, signal_fn=lambda *_: None) == 1
    first = kb.get_task(conn, stale_task)
    assert (first.status, first.consecutive_failures) == ("ready", 1)

    expire(conn, stale_task, 70102)
    assert kb.release_stale_claims(conn, signal_fn=lambda *_: None) == 1
    second = kb.get_task(conn, stale_task)
    assert (second.status, second.consecutive_failures) == ("blocked", 2)
    assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, stale_task).status == "blocked"

    # Scenario 2: independent total-run cap blocks even with a very high
    # failure budget. The reason carries both totals and block kind is capability.
    kb._runaway_limits = lambda: (2, 100_000)
    budget_task = kb.create_task(
        conn, title="lifetime budget", assignee="coder", max_retries=99
    )
    expire(conn, budget_task, 70201)
    kb.release_stale_claims(conn, signal_fn=lambda *_: None)
    expire(conn, budget_task, 70202)
    kb.release_stale_claims(conn, signal_fn=lambda *_: None)
    assert kb.get_task(conn, budget_task).status == "ready"
    budget_blocked = kb.enforce_runaway_limits(conn, signal_fn=lambda *_: None)
    assert budget_blocked == [budget_task], (
        f"run cap did not block: result={budget_blocked}, "
        f"totals={kb._task_run_totals(conn, budget_task, int(time.time()))}"
    )
    budget = kb.get_task(conn, budget_task)
    event = kb.list_events(conn, budget_task)[-1]
    assert budget.status == "blocked"
    assert budget.block_kind == "capability"
    assert event.kind == "blocked"
    assert event.payload["runaway"]["total_runs"] == 2
    assert "total runs=2/2" in event.payload["reason"]
    assert "total runtime=" in event.payload["reason"]
    assert kb.recompute_ready(conn) == 0

    # Scenario 3: the lifetime-runtime cap is independent from both the
    # failure budget and the total-run cap.
    kb._runaway_limits = lambda: (99, 2)
    runtime_task = kb.create_task(
        conn, title="runtime budget", assignee="coder", max_retries=99
    )
    assert kb.claim_task(conn, runtime_task)
    conn.execute(
        "UPDATE task_runs SET started_at=? WHERE task_id=? AND ended_at IS NULL",
        (int(time.time()) - 10, runtime_task),
    )
    conn.commit()
    runtime_blocked = kb.enforce_runaway_limits(conn, signal_fn=lambda *_: None)
    assert runtime_blocked == [runtime_task], (
        f"runtime cap did not block: result={runtime_blocked}, "
        f"totals={kb._task_run_totals(conn, runtime_task, int(time.time()))}"
    )
    runtime = kb.get_task(conn, runtime_task)
    runtime_event = kb.list_events(conn, runtime_task)[-1]
    assert runtime.status == "blocked"
    assert runtime.block_kind == "capability"
    assert runtime_event.payload["runaway"]["total_runs"] == 1
    assert runtime_event.payload["runaway"]["total_runtime_seconds"] >= 10
    assert kb.recompute_ready(conn) == 0

    # Scenario 4: Claude bridge's text is classified as a normal failed run.
    turns_task = kb.create_task(
        conn, title="max turns failure", assignee="coder", max_retries=2
    )
    assert kb.claim_task(conn, turns_task)
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (70301, turns_task))
    conn.commit()
    log_path = kb.worker_log_path(turns_task)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Error: Reached max turns (60)\n", encoding="utf-8")
    kb._record_worker_exit(70301, 1 << 8)
    assert kb.detect_crashed_workers(conn) == [turns_task]
    run = kb.latest_run(conn, turns_task)
    assert run.outcome == "crashed"
    assert "Reached max turns" in run.error
    assert kb.get_task(conn, turns_task).consecutive_failures == 1

print("PASS stale-reclaim: run 1 -> ready/failures=1")
print("PASS stale-reclaim: run 2 -> blocked/failures=2; recompute_ready=0")
print("PASS lifetime-cap: runs=2 -> blocked/capability; reason includes runs+runtime")
print("PASS runtime-cap: runtime >=10s -> blocked/capability despite runs=1/failures=0")
print("PASS max-turns: classified crashed and consumes failure budget")
