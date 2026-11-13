"""An exit poll must reconcile ownership acquired after the acknowledgement check."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def handoff_exit(tmp_path, monkeypatch):
    from cron import executions, scheduler
    from tools.process_registry import GatewayChildDispatch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "load_config_readonly", dict)
    # A prior terminal run of the same job cannot establish this attempt's ownership.
    previous = executions.create_execution("job-exit", source="builtin")
    executions.finish_execution(previous["id"], success=True)
    row = executions.create_execution("job-exit", source="builtin")
    job = {"id": "job-exit", "execution_id": row["id"], "prompt": "work"}
    handoff_dir = tmp_path / "cron" / "external-workers"
    ack_path = handoff_dir / f"{row['id']}.ready"
    effects_path = tmp_path / "effects.jsonl"
    processes = []
    real_popen = subprocess.Popen
    real_exists = Path.exists

    def launch(outcome):
        worker = (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "from cron import executions\n"
            "sys.stdin.read()\n"
            f"eid = {row['id']!r}\n"
            f"outcome = {outcome!r}\n"
            "if outcome != 'unowned':\n"
            "    assert executions.adopt_claimed_execution(eid)\n"
            f"    with Path({str(effects_path)!r}).open('a') as effects:\n"
            "        effects.write(json.dumps({'execution_id': eid}) + '\\n')\n"
            "    if outcome in ('completed', 'failed'):\n"
            "        executions.finish_execution(eid, success=outcome == 'completed', "
            "error='worker-result')\n"
            f"    Path({str(ack_path)!r}).write_text(json.dumps("
            "{'execution_id': eid, 'pid': os.getpid()}))\n"
        )
        monkeypatch.setattr(
            "tools.process_registry.restart_safe_gateway_child_argv",
            lambda command, **kwargs: GatewayChildDispatch(
                "degraded", [sys.executable, "-c", worker]
            ),
        )

        def spawn(*args, **kwargs):
            kwargs["stdin"] = subprocess.PIPE
            process = real_popen(*args, **kwargs)
            processes.append(process)
            real_poll, real_wait = process.poll, process.wait

            def exit_after_ack_check():
                # The worker cannot adopt until the parent has observed no ack.
                assert not real_exists(ack_path)
                monkeypatch.setattr(process, "poll", real_poll)
                process.stdin.close()
                real_wait(timeout=30)
                return real_poll()

            monkeypatch.setattr(process, "poll", exit_after_ack_check)
            return process

        monkeypatch.setattr(scheduler.subprocess, "Popen", spawn)
        return scheduler._launch_external_cron_worker(job)

    yield launch, executions, row["id"], effects_path, handoff_dir, processes
    for process in processes:
        if process.returncode is None:
            process.kill()
        process.wait(timeout=30)


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_terminal_outcome_wins_acknowledgement_exit_race(handoff_exit, outcome):
    launch, executions, eid, effects_path, handoff_dir, processes = handoff_exit

    assert launch(outcome) is True

    row = executions.get_execution(eid)
    assert row["status"] == outcome
    assert row["error"] == (None if outcome == "completed" else "worker-result")
    assert effects_path.read_text().splitlines() == [json.dumps({"execution_id": eid})]
    assert len(processes) == 1
    assert not list(handoff_dir.iterdir())


@pytest.mark.parametrize("adopted", [False, True])
def test_clean_exit_requires_ownership_and_preserves_uncertainty(handoff_exit, adopted):
    from cron import scheduler

    launch, executions, eid, effects_path, handoff_dir, processes = handoff_exit
    error_type = scheduler._ExternalWorkerPostHandoffError if adopted else RuntimeError
    expected = "without writing a durable terminal state" if adopted else "before ownership acknowledgement"

    with pytest.raises(error_type, match=expected) as error:
        launch("adopted" if adopted else "unowned")

    assert isinstance(error.value, scheduler._ExternalWorkerPostHandoffError) is adopted
    row = executions.get_execution(eid)
    assert row["status"] == ("unknown" if adopted else "claimed")
    assert effects_path.exists() is adopted
    if adopted:
        assert effects_path.read_text().splitlines() == [json.dumps({"execution_id": eid})]
    assert len(processes) == 1
    assert not list(handoff_dir.iterdir())
