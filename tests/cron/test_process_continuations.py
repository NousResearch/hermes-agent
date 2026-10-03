"""Cron completion lifecycle (#110650), using real subprocesses and durable stores."""

import json
import subprocess
import sys

import pytest

from cron import continuations, executions, jobs, scheduler
from tools import process_registry as registry_module
from tools import terminal_tool as terminal


def test_process_continuation_keeps_runtime_data_context():
    from cron.scheduler_prompt import _build_job_prompt

    prompt = _build_job_prompt(
        {
            "id": "continuation",
            "prompt": "Inspect the result",
            "_process_continuation": {"result": "Process finished"},
        },
        runtime_data_prompt="Monitor status: healthy",
    )

    assert "## Completed Process" in prompt
    assert "Process finished" in prompt
    assert "## Run Context\nMonitor status: healthy" in prompt


@pytest.mark.parametrize("outcome", ["success", "failure", "oneshot", "paused", "deleted", "cancelled", "claim_crash"])
def test_completion_survives_producer_exit_and_is_consumed_once(tmp_path, monkeypatch, outcome):
    job = jobs.create_job(prompt="Build, then summarize the result",
                          schedule="1h" if outcome == "oneshot" else "every 1h")
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    command = "printf 'build result'; exit " + ("7" if outcome == "failure" else "0")
    if outcome == "cancelled":
        command = "sleep 30"
    # Exit the producer immediately after launch. A continuation's reader must
    # finish persisting its receipt even when the original cron worker exits.
    program = """
import sys
from tools.process_registry import ProcessRegistry
registry = ProcessRegistry()
session = registry.spawn_local(command=sys.argv[3], cwd=sys.argv[4],
    cron_continuation={'job_id': sys.argv[1], 'execution_id': sys.argv[2]})
print(session.id, flush=True)
if sys.argv[5] == 'cancelled':
    registry.kill_process(session.id)
"""
    spawned = subprocess.run([sys.executable, "-c", program, job["id"], execution["id"],
                              command, str(tmp_path), outcome], capture_output=True, text=True, timeout=30)
    assert spawned.returncode == 0, spawned.stderr
    process_id = spawned.stdout.strip()
    assert continuations.pending_jobs() == []  # Parent turn still owns the task.
    executions.finish_execution(execution["id"], success=True)
    jobs.mark_job_run(job["id"], True)
    before = jobs.get_job(job["id"])
    if outcome == "paused":
        jobs.pause_job(job["id"])
    elif outcome == "deleted":
        jobs.remove_job(job["id"])

    calls = []

    def run(job, **kwargs):
        calls.append(job)
        from cron.scheduler_prompt import _build_job_prompt
        prompt = _build_job_prompt(job)
        assert process_id in prompt
        assert "build result" in prompt
        scope = scheduler._CronRunScope(job, job["id"], kwargs["execution_id"])
        scope.enter()
        try:
            denied = json.loads(terminal._handle_terminal({
                "command": "exit 0", "background": True, "continue_on_complete": True,
            }, task_id=scope.task_id))
            assert "cannot create another" in denied["error"]
        finally:
            scope.exit()
        return True, "continued output", "build inspected", None

    monkeypatch.setattr(scheduler, "run_job", run)  # No model/network calls.
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    monkeypatch.setattr(scheduler, "_maybe_run_worktree_maintenance", lambda: None)
    monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: None)
    ready = continuations.pending_jobs()
    if outcome == "claim_crash":
        claim = subprocess.run([sys.executable, "-c", """
import json, sys
from cron import continuations, executions
job = json.load(sys.stdin)
job['execution_id'] = executions.create_execution(job['id'], source='continuation')['id']
assert continuations.claim_job(job)
"""], input=json.dumps(ready[0]), capture_output=True, text=True, timeout=30)
        assert claim.returncode == 0, claim.stderr
        assert executions.recover_interrupted_executions() == 1
    scheduler.tick(verbose=False)
    for candidate in ready:
        receipt = candidate["_process_continuation"]
        continuations.retain_completion(receipt, json.loads(receipt["result"]))
    scheduler.tick(verbose=False)
    expected = outcome in ("success", "failure", "oneshot")
    assert len(calls) == int(expected)
    assert continuations.pending_jobs() == []
    if expected:
        after = jobs.get_job(job["id"])
        assert after["next_run_at"] == before["next_run_at"]
        assert after["repeat"] == before["repeat"]
        rows = executions.list_executions(job_id=job["id"])
        assert [row["source"] for row in rows].count("continuation") == 1
        assert all(row["status"] == "completed" for row in rows)


def test_terminal_opt_in_is_scoped_and_fast_exit_is_durable(tmp_path, monkeypatch):
    from gateway.session_context import async_delivery_supported

    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    args = {"command": "printf 'done'", "background": True, "continue_on_complete": True}
    denied = json.loads(terminal._handle_terminal(args, task_id="interactive"))
    assert "owning cron run" in denied["error"]
    assert not registry._running
    job = jobs.create_job(prompt="Build then inspect", schedule="every 1h", workdir=str(tmp_path))
    execution = executions.create_execution(job["id"], source="builtin")
    scope = scheduler._CronRunScope(job, job["id"], execution["id"])
    scope.enter()
    try:
        assert not async_delivery_supported()
        denied = json.loads(terminal._handle_terminal(args, task_id="delegated-child"))
        assert "owning cron run" in denied["error"]
        result = json.loads(terminal._handle_terminal(args, task_id=scope.task_id))
        assert result.get("error") is None, result
        session = registry.get(result["session_id"])
        assert session._completion_event.wait(15)
        assert result["continue_on_complete"]
        assert not session.notify_on_complete
        assert registry.completion_queue.empty()
        assert registry.pending_watchers == []
        with executions._transaction() as conn:
            row = conn.execute("SELECT * FROM process_continuations").fetchone()
            assert row["process_id"] == session.id
            assert json.loads(row["result"])["output"].strip() == "done"
        # Completion replay cannot mint a second ready event.
        from tools.process_registry_results import save_completed_result
        save_completed_result(session)
        with executions._transaction() as conn:
            assert conn.execute("SELECT count(*) FROM process_continuations").fetchone()[0] == 1
    finally:
        registry.kill_all()
        scope.exit()
    assert continuations.run_context.get() is None
