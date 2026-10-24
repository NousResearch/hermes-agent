"""A worker's CLI child must not destroy the claim needed for its final delivery."""

import contextlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("own_run", [True, False], ids=["worker-child", "external-cancel"])
def test_subprocess_removal_preserves_delivery_or_external_cancellation(
    tmp_path, monkeypatch, own_run,
):
    from cron import delivery_queue, executions, jobs, scheduler
    from tools.environments.local import build_subprocess_env

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "cron" / "deliveries.db")
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    delivered = []
    final = "Final report"
    children = []

    def run_agent(job, **kwargs):
        env = build_subprocess_env(dict(os.environ))
        env.update(
            HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(tmp_path),
            HERMES_RUNTIME_DIR=str(tmp_path / "runtime"), HERMES_DISABLE_LAZY_INSTALLS="1",
            TMPDIR=str(tmp_path), TMP=str(tmp_path), TEMP=str(tmp_path),
        )
        env.pop("_HERMES_CRON_EXTERNAL_WORKER", None)
        if own_run:
            # Environment survives a process boundary; the delivery ContextVar does not.
            env["_HERMES_CRON_EXTERNAL_WORKER"] = job["execution_id"]
        child = subprocess.run(
            [sys.executable, "-c",
             "import sys; from cron.jobs import remove_job; assert remove_job(sys.argv[1])",
             job["id"]],
            cwd=str(Path(jobs.__file__).resolve().parent.parent),
            env=env, text=True, capture_output=True, timeout=30, check=False,
        )
        children.append(child)
        if own_run and child.returncode != 0:
            retained = jobs.get_job(job["id"])
            assert retained["fire_claim"]["by"] == job["fire_claim"]["by"]
            assert output.read_text(encoding="utf-8") == "Earlier output"
            assert jobs.pause_job(job["id"])
        return True, "Run archive", final, None

    def deliver(job, content, for_failure):
        assert not for_failure
        delivered.append(content)

    monkeypatch.setattr(scheduler, "run_job", run_agent)
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="Produce a report", schedule="every 5m", deliver="telegram")
        assert jobs.claim_job_for_fire(job["id"])
        job = jobs.get_job(job["id"])
        output = jobs.save_job_output(job["id"], "Earlier output")
        assert scheduler.run_one_job(job)
        assert len(children) == 1
        child = children[0]
        execution = executions.get_execution(job["execution_id"])
        if own_run:
            assert child.returncode != 0, "worker child silently deleted its live delivery claim"
            assert "Pause" in child.stderr
            assert execution["status"] == "completed"
            assert execution["delivery_outcome"] == "queued"
            assert delivery_queue.get_status(job["execution_id"])["content"] == final
            assert delivery_queue.drain(deliver) == 1
            assert delivery_queue.drain(deliver) == 0
            assert delivered == [final]
            assert executions.get_execution(job["execution_id"])["delivery_outcome"] == "delivered"
            stored = jobs.get_job(job["id"])
            assert stored["state"] == "paused"
            assert stored["last_status"] == "ok"
            assert stored["fire_claim"] is None
        else:
            assert child.returncode == 0, child.stderr
            assert jobs.get_job(job["id"]) is None
            assert execution["status"] == "failed"
            assert "ownership lost" in execution["error"]
            assert delivery_queue.get_status(job["execution_id"]) is None
            assert delivery_queue.drain(deliver) == 0
            assert delivered == []


@pytest.mark.parametrize("identity", [
    "missing-ledger", "missing-row", "empty-marker", "corrupt-ledger",
    "wrong-home", "wrong-ledger", "wrong-store", "same-context", "other-job", "finished",
])
def test_worker_removal_requires_verifiable_profile_local_identity(tmp_path, monkeypatch, identity):
    from cron import executions, jobs

    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    with jobs.use_cron_store(home):
        job = jobs.create_job(prompt="Produce a report", schedule="every 5m")
        assert jobs.claim_job_for_fire(job["id"])
        before = jobs.get_job(job["id"])
        output = jobs.save_job_output(job["id"], "Earlier output")
        execution_id = "missing-execution"
        ledger = home / "cron" / "executions.db"
        if identity == "corrupt-ledger":
            ledger.write_text("not a database", encoding="utf-8")
        elif identity in {"other-job", "finished", "missing-row"}:
            execution = executions.create_execution(
                job["id"] if identity == "finished" else "other-job", source="test")
            assert executions.mark_execution_running(execution["id"])
            if identity != "missing-row":
                execution_id = execution["id"]
            if identity == "finished":
                assert executions.finish_execution(execution_id, success=True)
        elif identity in {"wrong-home", "wrong-ledger", "wrong-store"}:
            origin = tmp_path / "other-profile"
            with monkeypatch.context() as scoped:
                scoped.setenv("HERMES_HOME", str(origin))
                execution = executions.create_execution("other-job", source="test")
                execution_id = execution["id"]
                assert executions.mark_execution_running(execution_id)
            if identity == "wrong-ledger":
                monkeypatch.setattr(executions, "EXECUTIONS_FILE", origin / "cron" / "executions.db")
            elif identity == "wrong-home":
                monkeypatch.setenv("HERMES_HOME", str(origin))
        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "" if identity == "empty-marker" else execution_id)
        scope = (jobs.self_removal_delivery_scope(job["id"])
                 if identity == "same-context" else contextlib.nullcontext())
        with scope:
            if identity in {"same-context", "other-job", "finished"}:
                assert jobs.remove_job(job["id"])
                assert jobs.get_job(job["id"]) is None
                assert not output.exists()
                assert jobs.self_removal_delivery_allowed(job["id"]) == (identity == "same-context")
            else:
                with pytest.raises(ValueError, match="Cannot .*verif"):
                    jobs.remove_job(job["id"])
                assert jobs.get_job(job["id"]) == before
                assert output.read_text(encoding="utf-8") == "Earlier output"
