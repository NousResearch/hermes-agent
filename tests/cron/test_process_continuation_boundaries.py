"""Failure and concurrency contracts for cron-owned completions (#110650)."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import sqlite3
import subprocess
import sys
import threading

import pytest

from cron import continuations as cont, executions, jobs, scheduler
from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_results import save_completed_result
from tools import terminal_tool as terminal


@contextmanager
def profile(home):
    token = set_hermes_home_override(home)
    try:
        with jobs.use_cron_store(home):
            yield
    finally:
        reset_hermes_home_override(token)


def ready_job(*, process_id="proc_ready", schedule="every 1h", code=0, reason="exited"):
    job = jobs.create_job(prompt="Inspect the finished build", schedule=schedule)
    parent = executions.create_execution(job["id"], source="builtin")
    executions.finish_execution(parent["id"], success=True)
    identity = {"job_id": job["id"], "execution_id": parent["id"],
                "profile_home": str(get_hermes_home())}
    result = {"id": process_id, "command": "build", "output": "build output",
              "exit_code": code, "completion_reason": reason}
    cont.retain_completion(identity, result)
    return job, identity, result


def candidate():
    job = cont.pending_jobs()[0]
    job["execution_id"] = executions.create_execution(job["id"], source="continuation")["id"]
    return job


def receipts():
    with executions._transaction() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM process_continuations")]


@pytest.fixture
def quiet_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    monkeypatch.setattr(scheduler, "_maybe_run_worktree_maintenance", lambda: None)
    monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: None)


@pytest.mark.parametrize("mode", ["foreground", "missing_id", "unknown_id", "wrong_job", "finished"])
def test_invalid_execution_cannot_launch_a_command(monkeypatch, mode):
    job = jobs.create_job(prompt="Build", schedule="every 1h")
    execution = executions.create_execution(job["id"], source="builtin")
    context = {"job_id": job["id"], "execution_id": execution["id"],
               "task_id": "owner", "continuation": False}
    if mode == "missing_id":
        context["execution_id"] = None
    elif mode == "unknown_id":
        context["execution_id"] = "missing"
    elif mode == "wrong_job":
        context["job_id"] = "someone-else"
    elif mode == "finished":
        executions.finish_execution(execution["id"], success=True)
    token = cont.run_context.set(context)
    launches = []
    monkeypatch.setattr(terminal, "_plan_execution", lambda *a, **kw: launches.append(a))
    try:
        result = json.loads(terminal._handle_terminal({"command": "must not run",
            "background": mode != "foreground", "continue_on_complete": True}, task_id="owner"))
        assert result["error"]
        assert not launches
    finally:
        cont.run_context.reset(token)


@pytest.mark.parametrize("status", ["claimed", "running", "completed", "failed", "unknown", "pruned"])
def test_only_terminal_parent_attempts_are_eligible(status):
    job, identity, result = ready_job()
    with executions._transaction() as conn:
        if status == "pruned":
            conn.execute("DELETE FROM executions WHERE id=?", (identity["execution_id"],))
        else:
            conn.execute("UPDATE executions SET status=? WHERE id=?", (status, identity["execution_id"]))
    assert bool(cont.pending_jobs()) == (status not in {"claimed", "running"})
    assert receipts()[0]["state"] == "ready"


@pytest.mark.parametrize("code,reason", [(None, "exited"), (-1, "lost"), (-1, "failed_start"),
                                         (-15, "killed"), (-1, "timed_out"), (-15, "exited"),
                                         ("0", "exited"), (True, "exited")])
def test_unknown_or_aborted_completion_is_permanently_skipped(code, reason):
    _, identity, result = ready_job(code=code, reason=reason)
    assert cont.pending_jobs() == []
    result.update(exit_code=0, completion_reason="exited")
    cont.retain_completion(identity, result)
    assert cont.pending_jobs() == []
    assert receipts()[0]["state"] == "skipped"


@pytest.mark.parametrize("change", ["pause", "disable", "delete"])
def test_state_is_rechecked_after_scan_and_before_claim(change):
    job, identity, result = ready_job()
    queued = candidate()
    if change == "pause":
        jobs.pause_job(job["id"])
    elif change == "disable":
        jobs.update_job(job["id"], {"enabled": False})
    else:
        jobs.remove_job(job["id"])
    assert not cont.claim_job(queued)
    assert cont.pending_jobs() == []
    if change != "delete":
        jobs.resume_job(job["id"])
        cont.retain_completion(identity, result)
        assert cont.pending_jobs() == []


def test_live_claim_defers_and_stale_candidate_cannot_be_reused():
    job, _, _ = ready_job()
    queued = candidate()
    other = jobs.claim_job_for_fire(job["id"], manual=True, return_job=True)
    assert other
    assert cont.claim_job(queued) is None
    assert receipts()[0]["state"] == "ready"
    jobs.mark_job_run(job["id"], True, expected_fire_owner=other["fire_claim"]["by"])
    claimed = cont.claim_job(queued)
    assert claimed
    jobs.mark_job_run(job["id"], True, expected_fire_owner=claimed["fire_claim"]["by"])
    assert cont.claim_job(queued) is None
    assert receipts()[0]["continuation_execution_id"] == queued["execution_id"]


def test_independent_processes_cannot_claim_the_same_completion():
    ready_job()
    queued = cont.pending_jobs()[0]
    start = threading.Barrier(3)
    program = """
import json, sys
from cron import continuations, executions
job = json.load(sys.stdin)
job['execution_id'] = executions.create_execution(job['id'], source='continuation')['id']
claimed = continuations.claim_job(job)
print(json.dumps({'claimed': bool(claimed), 'id': job['execution_id']}))
"""

    def compete():
        start.wait(timeout=20)
        completed = subprocess.run([sys.executable, "-c", program], input=json.dumps(queued),
                                   capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = [pool.submit(compete) for _ in range(2)]
        start.wait(timeout=20)
        results = [f.result(timeout=40) for f in attempts]
    assert sum(r["claimed"] for r in results) == 1
    assert receipts()[0]["continuation_execution_id"] == next(r["id"] for r in results if r["claimed"])


@pytest.mark.parametrize("boundary", ["before_claim", "after_claim", "after_job_save", "after_running"])
def test_abrupt_worker_exit_never_replays_a_claim(boundary):
    ready_job()
    queued = cont.pending_jobs()[0]
    program = """
import json, os, sys
from cron import continuations, executions, jobs
job = json.load(sys.stdin)
job['execution_id'] = executions.create_execution(job['id'], source='continuation')['id']
boundary = sys.argv[1]
if boundary == 'before_claim':
    os._exit(23)
save = jobs.save_jobs
def crash_save(records, *args, **kwargs):
    if boundary == 'after_claim':
        os._exit(23)
    save(records, *args, **kwargs)
    if boundary == 'after_job_save':
        os._exit(23)
jobs.save_jobs = crash_save
assert continuations.claim_job(job)
executions.mark_execution_running(job['execution_id'])
os._exit(23)
"""
    completed = subprocess.run([sys.executable, "-c", program, boundary], input=json.dumps(queued),
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 23, completed.stderr
    assert executions.recover_interrupted_executions() == 1
    assert executions.list_executions(job_id=queued["id"])[0]["status"] == "unknown"
    assert bool(cont.pending_jobs()) == (boundary == "before_claim")


def test_completion_and_receipt_stay_in_the_spawning_profile(tmp_path):
    original = get_hermes_home()
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    with profile(home_a):
        job, identity, result = ready_job()
        session = ProcessSession(id="proc_profile", command="build", exited=True,
                                 exit_code=0, output_buffer="PRIVATE_A", cron_continuation=identity)
    with profile(home_b):
        # A process-global registry can finish A's recovered session while B is
        # the current profile. Both persisted copies must still belong to A.
        save_completed_result(session)
        assert cont.pending_jobs() == []
        assert not (home_b / "logs/process-results/proc_profile.json").exists()
        assert get_hermes_home() == home_b
    with profile(home_a):
        assert len(cont.pending_jobs()) == 2
        assert (home_a / "logs/process-results/proc_profile.json").exists()
    assert get_hermes_home() == original


def test_storage_failure_restores_profile_and_does_not_acknowledge_completion(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    with profile(home_a):
        _, identity, _ = ready_job()
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_retry", command="build", exited=True, exit_code=0,
                             output_buffer="result", cron_continuation=identity)
    registry._running[session.id] = session
    with profile(home_b):
        with monkeypatch.context() as fault:
            fault.setattr(executions, "_connect", lambda: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")))
            with pytest.raises(sqlite3.OperationalError, match="disk full"):
                registry._move_to_finished(session)
            assert get_hermes_home() == home_b
            assert session.id in registry._running
            assert not session._completion_event.is_set()
        registry._move_to_finished(session)
        assert session._completion_event.is_set()
    with profile(home_a):
        assert len(cont.pending_jobs()) == 2


def test_claim_save_failure_is_audited_and_not_replayed(monkeypatch):
    ready_job()
    queued = candidate()
    with monkeypatch.context() as fault:
        fault.setattr(jobs, "save_jobs", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
        assert scheduler._process_due_job(queued, None, None, False) is False
    row = executions.get_execution(queued["execution_id"])
    assert row["status"] == "failed"
    assert "disk full" in row["error"]
    assert not cont.pending_jobs()


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"output":"missing status"}'])
def test_bad_receipt_does_not_block_other_completions(payload):
    first, _, _ = ready_job(process_id="proc_bad")
    second, _, _ = ready_job(process_id="proc_good")
    with executions._transaction() as conn:
        conn.execute("UPDATE process_continuations SET result=? WHERE process_id='proc_bad'", (payload,))
    assert [j["id"] for j in cont.pending_jobs()] == [second["id"]]
    assert next(r for r in receipts() if r["job_id"] == first["id"])["state"] == "skipped"


@pytest.mark.parametrize("failure", ["failure", "timeout", "exception", "unreachable", "delivery", "empty"])
def test_continuation_failure_is_final_without_changing_schedule(monkeypatch, quiet_scheduler, failure):
    job, _, _ = ready_job()
    before = jobs.get_job(job["id"])
    delivered = []

    def run(job, **kwargs):
        if failure == "exception":
            raise RuntimeError("continuation crashed")
        if failure == "unreachable":
            job["_model_unreachable"] = True
        if failure == "delivery":
            return True, "output", "summary", None
        if failure == "empty":
            return True, "output", "", None
        return False, "failure output", "", "TimeoutError" if failure == "timeout" else "model failure"

    def deliver(job, text, **kwargs):
        delivered.append(text)
        return "delivery refused" if failure == "delivery" else None

    monkeypatch.setattr(scheduler, "run_job", run)
    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    scheduler.tick(verbose=False)
    scheduler.tick(verbose=False)
    after = jobs.get_job(job["id"])
    assert after["next_run_at"] == before["next_run_at"]
    assert after["repeat"] == before["repeat"]
    assert after["last_status"] == ("delivery_failed" if failure == "delivery" else "error")
    assert after["failure_streak"] == (0 if failure == "delivery" else 1)
    rows = [r for r in executions.list_executions(job_id=job["id"]) if r["source"] == "continuation"]
    assert len(rows) == 1
    assert rows[0]["status"] == ("completed" if failure == "delivery" else "failed")
    assert len(delivered) == (0 if failure == "empty" else 1)
    assert cont.pending_jobs() == []


@pytest.mark.parametrize("gate", ["drain", "estop", "shutdown", "pool_failure"])
def test_dispatch_gate_preserves_ready_receipt_and_schedule(monkeypatch, quiet_scheduler, gate):
    job, _, _ = ready_job(schedule="1h")
    before = jobs.get_job(job["id"])
    if gate == "drain":
        assert scheduler.tick(verbose=False, can_dispatch=lambda: False) == 0
    elif gate == "estop":
        monkeypatch.setattr("agent.estop.check_paused", lambda *a: True)
        assert scheduler.tick(verbose=False) == 0
    else:
        queued = cont.pending_jobs()[0]
        if gate == "shutdown":
            monkeypatch.setattr(scheduler, "_interpreter_shutting_down", lambda *a: True)

        class BrokenPool:
            def submit(self, *a):
                raise RuntimeError("executor unavailable")

        assert scheduler._submit_with_guard(queued, BrokenPool(), lambda j: True) is None
    assert len(cont.pending_jobs()) == 1
    after = jobs.get_job(job["id"])
    assert after["next_run_at"] == before["next_run_at"]
    assert after.get("run_claim") == before.get("run_claim")


def test_result_is_redacted_bounded_and_pending_during_cache_pruning(monkeypatch):
    from tools import process_registry_results as results
    from agent import redact
    _, identity, _ = ready_job()
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    monkeypatch.setattr(results, "MAX_RETAINED_RESULTS", 0)
    secret = "sk-" + "aB2cD3eF4gH5iJ6kL7mN8pQ9rS0tU1vW2xY3zA4bC5dE6fG7"
    session = ProcessSession(id="proc_secret", command=f"echo {secret}", exited=True,
                             exit_code=0, output_buffer="x" * 15000 + "\n" + secret, cron_continuation=identity)
    save_completed_result(session)
    row = next(r for r in receipts() if r["process_id"] == session.id)
    assert secret not in row["result"]
    result = json.loads(row["result"])
    assert result["output_truncated"]
    assert len(result["output"]) <= 12000
    assert not (get_hermes_home() / "logs/process-results/proc_secret.json").exists()
    assert len(cont.pending_jobs()) == 2


def test_claim_preserves_manual_schedule_and_omits_collection_gates():
    job, _, _ = ready_job()
    jobs.update_job(job["id"], {"script": "collect.py", "context_from": ["upstream"]})
    queued = candidate()
    claimed = cont.claim_job(queued)
    assert "script" not in claimed and "context_from" not in claimed
    jobs.trigger_job(job["id"], "manual request arrived during continuation")
    before = copy.deepcopy(jobs.get_job(job["id"]))
    assert jobs.mark_job_run(job["id"], True, expected_fire_owner=claimed["fire_claim"]["by"])
    after = jobs.get_job(job["id"])
    for key in ("next_run_at", "repeat", "manual_run_at", "manual_run_prompt"):
        assert after.get(key) == before.get(key)


def test_async_ticks_do_not_duplicate_an_inflight_continuation(monkeypatch, quiet_scheduler):
    job, _, _ = ready_job()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def run(job, **kwargs):
        calls.append(job["execution_id"])
        entered.set()
        assert release.wait(20)
        return True, "output", "summary", None

    monkeypatch.setattr(scheduler, "run_job", run)
    try:
        assert scheduler.tick(verbose=False, sync=False) == 1
        assert entered.wait(15)
        key = scheduler._inflight_key(job["id"], get_hermes_home())
        future = scheduler._running_futures[key]
        assert scheduler.tick(verbose=False, sync=False) == 0
        assert job["id"] in scheduler.get_running_job_ids()
    finally:
        release.set()
    assert future.result(timeout=20)
    assert job["id"] not in scheduler.get_running_job_ids()
    assert len(calls) == 1


def test_execution_creation_failure_does_not_consume_the_completion(monkeypatch):
    job, _, _ = ready_job()
    with monkeypatch.context() as fault:
        fault.setattr(scheduler, "create_execution", lambda *a, **kw: (_ for _ in ()).throw(OSError("ledger unavailable")))
        assert scheduler._submit_with_guard(cont.pending_jobs()[0], None, None) is None
    assert job["id"] not in scheduler.get_running_job_ids()
    assert len(cont.pending_jobs()) == 1


@pytest.mark.parametrize("mutation", ["delete", "pause"])
def test_job_mutation_during_continuation_does_not_resurrect_job(monkeypatch, quiet_scheduler, mutation):
    job, _, _ = ready_job()
    delivered = []

    def run(job, **kw):
        if mutation == "delete":
            jobs.remove_job(job["id"])
        else:
            jobs.pause_job(job["id"])
        return True, "output", "summary", None

    monkeypatch.setattr(scheduler, "run_job", run)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **kw: delivered.append(a))
    scheduler.tick(verbose=False)
    if mutation == "delete":
        assert jobs.get_job(job["id"]) is None
        assert len(delivered) == 1
    else:
        assert not jobs.is_job_runnable(jobs.get_job(job["id"]))
    assert cont.pending_jobs() == []


def test_base_exception_finishes_attempt_without_retry(monkeypatch, quiet_scheduler):
    ready_job()
    queued = candidate()
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        scheduler._process_due_job(queued, None, None, False)
    assert executions.get_execution(queued["execution_id"])["status"] == "failed"
    assert cont.pending_jobs() == []


def test_real_watchdog_interrupts_continuation_and_records_timeout(monkeypatch, quiet_scheduler):
    job, _, _ = ready_job()
    stopped = threading.Event()

    class IdleAgent:
        def run_conversation(self, prompt, **kwargs):
            assert stopped.wait(20)
            return {"completed": True, "final_response": "interrupted"}

        def get_activity_summary(self):
            return {"seconds_since_activity": 100, "last_activity_desc": "model stalled"}

        def hard_interrupt(self, message):
            stopped.set()

    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 0.1)

    def run(job, **kwargs):
        scheduler._run_agent_with_watchdog(IdleAgent(), "continue", job, job["id"],
            "test", "owner", kwargs.get("cancel_event"))
        pytest.fail("watchdog should raise TimeoutError")

    monkeypatch.setattr(scheduler, "run_job", run)
    try:
        scheduler.tick(verbose=False)
        assert stopped.is_set()
        row = executions.latest_execution(job["id"])
        assert row["status"] == "failed"
        assert "idle for" in row["error"]
        assert cont.pending_jobs() == []
    finally:
        stopped.set()


def test_detached_worker_adopts_continuation_and_refuses_replay(tmp_path, monkeypatch):
    ready_job()
    queued = candidate()
    claimed = cont.claim_job(queued)
    assert executions.mark_execution_handoff_pending(claimed["execution_id"])
    payload, ack = tmp_path / "worker.json", tmp_path / "worker.ready"
    data = json.dumps({"job": claimed, "profile_home": str(get_hermes_home())})
    calls = []

    def run(job, **kwargs):
        calls.append(job["execution_id"])
        assert job["_process_continuation"]["process_id"] == "proc_ready"
        scope = scheduler._CronRunScope(job, job["id"], job["execution_id"])
        scope.enter()
        try:
            with pytest.raises(ValueError, match="cannot create another"):
                cont.validate_request(True, scope.task_id)
        finally:
            scope.exit()
        return True, "output", "summary", None

    monkeypatch.setattr(scheduler, "run_job", run)
    payload.write_text(data)
    assert scheduler._run_external_worker_payload(payload, ack)
    assert ack.exists()
    assert executions.get_execution(claimed["execution_id"])["status"] == "completed"
    payload.write_text(data)
    assert not scheduler._run_external_worker_payload(payload, ack)
    assert len(calls) == 1


@pytest.mark.parametrize("gate", ["script", "monitor_script", "monitor_url"])
def test_prompt_uses_completion_without_rerunning_collection_gate(monkeypatch, gate):
    job, _, _ = ready_job()
    value = "https://invalid.local/monitor" if gate == "monitor_url" else "collect.py"
    jobs.update_job(job["id"], {gate: value})
    claimed = cont.claim_job(candidate())
    invoked = []
    monkeypatch.setattr(scheduler, "_run_job_script_with_claim_heartbeat", lambda *a, **kw: invoked.append(a))
    monkeypatch.setattr("cron.monitor.check_monitor", lambda *a: invoked.append(a))
    early, prompt = scheduler._prepare_job_prompt(claimed, job["id"], "test", None, None)
    assert early is None
    assert "build output" in prompt
    assert not invoked


def test_auxiliary_receipt_failure_does_not_lose_durable_completion(monkeypatch):
    from tools import process_registry_results as results
    _, identity, _ = ready_job()
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_aux_failure", command="build", exited=True, exit_code=0,
                             output_buffer="result", cron_continuation=identity)
    registry._running[session.id] = session
    monkeypatch.setattr(results, "atomic_json_write", lambda *a, **kw: (_ for _ in ()).throw(OSError("receipt disk full")))
    registry._move_to_finished(session)
    assert session._completion_event.is_set()
    assert any(j["_process_continuation"]["process_id"] == session.id for j in cont.pending_jobs())
