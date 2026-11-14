"""Error text persisted by cron (jobs.json, executions.db) is scrubbed before it lands on disk (#102700).

Provider / stderr error strings can carry credentials; jobs.json and executions.db survive restarts
and get backed up or synced, so every write of raw failure text goes through the redactor.
"""

import pytest

from cron.jobs import create_job, get_job, mark_job_run, note_fire_forward_failure

# Built at runtime so no token-shaped literal sits in the tree.
_KEY = "sk-" + "or-v1-" + "a1b2c3d4" * 8
_PROVIDER_ERROR = f"HTTP 401: invalid key {_KEY} (Key limit exceeded)"


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.executions.EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return tmp_path


def _jobs_json(tmp_path) -> str:
    return (tmp_path / "cron" / "jobs.json").read_text(encoding="utf-8")


def test_failed_run_error_is_redacted_in_jobs_json(tmp_cron_dir):
    job = create_job(prompt="watch prices", schedule="every 1h")
    assert mark_job_run(job["id"], success=False, error=_PROVIDER_ERROR) is True

    stored = get_job(job["id"])
    assert _KEY not in stored["last_error"]
    assert "Key limit exceeded" in stored["last_error"]
    assert _KEY not in stored["last_failure"]["detail"]
    assert _KEY not in _jobs_json(tmp_cron_dir)


def test_delivery_error_is_redacted_in_jobs_json(tmp_cron_dir):
    job = create_job(prompt="watch prices", schedule="every 1h")
    mark_job_run(job["id"], success=True, delivery_error=_PROVIDER_ERROR)

    assert _KEY not in get_job(job["id"])["last_delivery_error"]
    assert _KEY not in _jobs_json(tmp_cron_dir)


def test_fire_forward_failure_detail_is_redacted(tmp_cron_dir):
    job = create_job(prompt="watch prices", schedule="every 1h")
    assert note_fire_forward_failure(job["id"], _PROVIDER_ERROR) is True

    assert _KEY not in get_job(job["id"])["last_fire_error"]["detail"]


def test_ordinary_error_text_is_kept_verbatim(tmp_cron_dir):
    job = create_job(prompt="watch prices", schedule="every 1h")
    mark_job_run(job["id"], success=False, error="Connection timed out")

    assert get_job(job["id"])["last_error"] == "Connection timed out"


def test_persisted_error_is_length_bounded(tmp_cron_dir):
    from cron.jobs_error_text import MAX_PERSISTED_ERROR_CHARS

    job = create_job(prompt="watch prices", schedule="every 1h")
    mark_job_run(job["id"], success=False, error="x" * 50_000)

    assert len(get_job(job["id"])["last_error"]) == MAX_PERSISTED_ERROR_CHARS


def test_redactor_failure_stores_placeholder_not_raw_text(tmp_cron_dir, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("redactor exploded")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", _boom)
    job = create_job(prompt="watch prices", schedule="every 1h")
    assert mark_job_run(job["id"], success=False, error=_PROVIDER_ERROR) is True

    assert _KEY not in get_job(job["id"])["last_error"]


def test_finish_execution_redacts_error_in_ledger(tmp_cron_dir):
    from cron.executions import create_execution, finish_execution, get_execution

    record = create_execution("job-1", source="test")
    assert finish_execution(record["id"], success=False, error=_PROVIDER_ERROR) is not None

    assert _KEY not in get_execution(record["id"])["error"]


def test_url_credentials_in_error_are_redacted(tmp_cron_dir):
    job = create_job(prompt="watch prices", schedule="every 1h")
    mark_job_run(job["id"], success=False, error="GET https://api.example.com/v1?access_token=plainvalue123 -> 401")

    assert "plainvalue123" not in get_job(job["id"])["last_error"]


def test_receipt_recovery_redacts_error_in_ledger(tmp_cron_dir):
    from cron.executions import create_execution, get_execution, give_up_execution, recover_receipted_execution

    record = create_execution("job-1", source="test")
    assert give_up_execution(record["id"], reason="unverified admission") is True
    assert recover_receipted_execution(record["id"], "job-1", success=False, error=_PROVIDER_ERROR) is not None

    assert _KEY not in get_execution(record["id"])["error"]


def test_interrupted_run_delivery_error_is_redacted(tmp_cron_dir):
    from cron.executions import create_execution
    from cron.scheduler_bookkeeping import finish_interrupted_run

    job = create_job(prompt="watch prices", schedule="every 1h")
    finish_interrupted_run(job, create_execution(job["id"], source="test")["id"], _PROVIDER_ERROR)

    assert _KEY not in get_job(job["id"])["last_delivery_error"]


def test_queued_delivery_error_settled_onto_job_is_length_bounded(tmp_path, monkeypatch):
    from cron import delivery_queue, jobs, scheduler
    from cron.jobs_error_text import MAX_PERSISTED_ERROR_CHARS

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the result", None))
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="telegram:fixture")
    scheduler.run_one_job(job)
    assert jobs.get_job(job["id"])["last_status"] == "delivery_queued"

    delivery_queue.drain(lambda *_args: "refused: " + "x" * 50_000)

    saved = jobs.get_job(job["id"])
    assert saved["last_status"] == "delivery_failed"
    assert len(saved["last_delivery_error"]) == MAX_PERSISTED_ERROR_CHARS


def test_unknown_admission_reason_is_redacted_and_still_lifts_its_pause(tmp_path, monkeypatch):
    """CronExecutionUnknown text reaches jobs.json ``paused_reason`` and executions.db ``error``;
    both get the same scrubbed string, so a verified terminal receipt still lifts the pause."""
    from contextlib import asynccontextmanager
    import json

    from cron import executions, jobs, scheduler, scheduler_authority
    from hermes_cli import gateway_client

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")
        fire = executions.create_execution(job["id"], source="test")["id"]
        journal = scheduler_authority.journal_path(job["id"], fire)

        def run_job(job, **kwargs):
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text(json.dumps({"params": {"job_id": job["id"], "request_id": fire,
                                                      "extra_prompt": None}, "receipt": None}))
            raise scheduler_authority.CronExecutionUnknown(f"Cron execution unverified: {_PROVIDER_ERROR}")
        monkeypatch.setattr(scheduler, "run_job", run_job)
        assert scheduler._run_one_job_body(dict(jobs.get_job(job["id"]), execution_id=fire)) is False

        paused = jobs.get_job(job["id"])
        assert paused["state"] == "paused"
        assert _KEY not in paused["paused_reason"] and "unverified" in paused["paused_reason"]
        assert _KEY not in executions.get_execution(fire)["error"]

        class Peer:
            async def rpc(self, method, **params):
                return {"status": "terminal", "job": job, "result": [True, "document", "answer", None]}

        @asynccontextmanager
        async def connected():
            yield Peer()
        monkeypatch.setattr(gateway_client, "connect_gateway", connected)
        scheduler_authority.reconcile_pending()
        assert jobs.get_job(job["id"])["state"] == "scheduled"
