"""Post-adoption failure visibility (#128509, #129254), not the unknown crash cause."""

import json
import sys
import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize(
    "death", ["hard_exit", "payload_exception", "concurrent_recovery",
              "timeout_hard_exit", "timeout_payload_exception"]
)
def test_adopted_worker_failure_is_visible(tmp_path, monkeypatch, manual, death):
    import cron.executions as executions
    import cron.scheduler as scheduler
    from cron.jobs import claim_job_for_fire, create_job, get_job, use_cron_store
    from tools.process_registry import GatewayChildDispatch

    print("SCHEDULER", scheduler.__file__, flush=True)
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "load_config_readonly", lambda: {})
    if death == "concurrent_recovery":
        terminalize = scheduler.terminalize_dead_owner

        def sweep_first(*args, **kwargs):
            executions.recover_interrupted_executions()
            return terminalize(*args, **kwargs)

        monkeypatch.setattr(scheduler, "terminalize_dead_owner", sweep_first)
    if death.startswith("timeout_"):
        wait_body = scheduler._wait_for_external_cron_worker_body

        def timeout_after_recovery(process, **kwargs):
            real_wait = process.wait

            def race_wait(timeout=None):
                # Force the ordering: timeout, child death, concurrent recovery,
                # then the waiter's ledger read. The child and stores are real.
                real_wait(timeout=30)
                assert executions.recover_interrupted_executions() == 1
                monkeypatch.setattr(process, "wait", real_wait)
                raise subprocess.TimeoutExpired(process.args, timeout)

            monkeypatch.setattr(process, "wait", race_wait)
            return wait_body(process, **kwargs)

        monkeypatch.setattr(scheduler, "_wait_for_external_cron_worker_body", timeout_after_recovery)
    delivered = []
    monkeypatch.setattr(
        scheduler, "_deliver_result", lambda *a, **k: delivered.append(True)
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json, os, sys, time
from pathlib import Path
# The stub below swaps the install recipe for this bare script, so do what the
# recipe's bootstrap does: put the checkout first on sys.path.
sys.path.insert(0, REPO_ROOT)
import cron.scheduler as scheduler
import cron.executions as executions
args = sys.argv
payload = Path(args[args.index("--external-worker-file") + 1])
ack = Path(args[args.index("--ack-file") + 1])
def fail(*args, **kwargs):
    time.sleep(2)
    raise RuntimeError("worker-probe-startup-sentinel")
if DEATH == "payload_exception":
    scheduler.run_one_job = fail
    scheduler._run_external_worker_payload(payload, ack)
else:
    data = json.loads(payload.read_text())
    eid = data["job"]["execution_id"]
    assert executions.adopt_claimed_execution(eid)
    ack.write_text(json.dumps({"pid": os.getpid(), "execution_id": eid}))
    time.sleep(2)
    sys.stderr.write("worker-probe-startup-sentinel\\n")
    sys.stderr.flush()
    os._exit(9)
""".replace("DEATH", repr(death.removeprefix("timeout_"))).replace(
            "REPO_ROOT", repr(str(Path(scheduler.__file__).resolve().parents[1]))
        ),
        encoding="utf-8",
    )

    def dispatch(command, **kwargs):
        flags = command[command.index("--external-worker-file") :]
        return GatewayChildDispatch("degraded", [sys.executable, str(worker), *flags])

    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv", dispatch
    )
    with use_cron_store(home):
        # A lost worker must obey the same bounded history as normal finishes.
        monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 2)
        for index in range(executions.MAX_TERMINAL_EXECUTIONS):
            old = executions.create_execution(f"old-{index}", source="direct")
            executions.finish_execution(old["id"], success=True)
        job = create_job(prompt="Reply OK", schedule="every 1h", deliver="local")
        claimed = claim_job_for_fire(job["id"], manual=True, return_job=True)
        assert claimed
        if not manual:
            execution = executions.create_execution(job["id"], source="builtin")
            claimed["execution_id"] = execution["id"]
        assert scheduler.run_one_job(claimed)
        record = get_job(job["id"])
        row = executions.get_execution(claimed["execution_id"])
        result = {
            "manual": manual,
            "death": death,
            "state": row["status"],
            "claim_released": record.get("fire_claim") is None,
            "job_error": record.get("last_error"),
            "ledger_error": row.get("error"),
            "notice_calls": len(delivered),
        }
        print("PROBE", json.dumps(result), flush=True)
        assert row["status"] == "unknown"
        if death != "concurrent_recovery" and not death.startswith("timeout_"):
            assert "Scheduler restarted" not in row["error"]
        assert result["claim_released"], result
        assert "worker-probe-startup-sentinel" in (result["job_error"] or ""), result
        assert len(delivered) == 1, result
        assert len(executions.list_executions()) <= executions.MAX_TERMINAL_EXECUTIONS
        from cron.scheduler_worker_failure import record_unknown_worker_outcome

        # Re-observing an old terminal row cannot notify twice or erase a newer fire.
        record_unknown_worker_outcome(claimed)
        assert len(delivered) == 1
        assert claim_job_for_fire(job["id"], manual=True)
        replacement = get_job(job["id"])["fire_claim"]
        record_unknown_worker_outcome(claimed)
        assert get_job(job["id"])["fire_claim"] == replacement
        assert len(delivered) == 1


def test_terminalizer_refuses_an_attempt_whose_worker_is_still_alive(
    tmp_path,
    monkeypatch,
):
    """The targeted terminalizer must not steal a LIVE worker's attempt: refusing a live
    owner is what keeps the fix from becoming a second way to lose side effects."""
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")

    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "import cron.executions as executions\n"
        f"executions.EXECUTIONS_FILE = Path({str(executions.EXECUTIONS_FILE)!r})\n"
        "assert executions.adopt_claimed_execution(sys.argv[1]) is not None\n"
        "sys.stdout.write('adopted\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    record = executions.create_execution("job-live", source="direct")
    assert executions.mark_execution_handoff_pending(record["id"]) is not None
    live_worker = subprocess.Popen(
        [sys.executable, "-c", script, record["id"]], stdout=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if executions.get_execution(record["id"])["status"] == "running":
                break
            time.sleep(0.05)
        assert executions.get_execution(record["id"])["status"] == "running"

        assert executions.terminalize_dead_owner(record["id"], reason="exit 9") is False
        assert executions.get_execution(record["id"])["status"] == "running"
    finally:
        live_worker.kill()
        live_worker.wait(timeout=30)
