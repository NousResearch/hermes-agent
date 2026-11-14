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
    from cron.jobs import _MAX_PERSISTED_ERROR_CHARS

    job = create_job(prompt="watch prices", schedule="every 1h")
    mark_job_run(job["id"], success=False, error="x" * 50_000)

    assert len(get_job(job["id"])["last_error"]) == _MAX_PERSISTED_ERROR_CHARS


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
