"""Regression tests for issue #102700.

``cron/jobs.py`` persists ``last_error`` / ``last_delivery_error`` /
``last_fire_error`` into ``jobs.json``, which survives gateway restarts,
job edits, and profile backups. Provider error strings can embed a
key-management link (e.g. OpenRouter's
``https://openrouter.ai/workspaces/default/keys/<KEY_ID>``) or a vendor
prefix secret (``sk-…``, ``sk-or-v1-…``). Those must be sanitized before
the write, not after.
"""

import pytest

from cron.jobs import (
    _MAX_PERSISTED_ERROR_CHARS,
    _sanitize_persisted_error,
    create_job,
    get_job,
    mark_job_run,
    note_fire_forward_failure,
    update_job,
)


# Key id that does NOT match a vendor prefix (sk-/ghp_/…). Proves the
# /keys/ URL rewrite, not the prefix redactor, is what strips it.
_KEY_ID = "or_key_" + "abcdef1234567890"
_OPENROUTER_ERROR = (
    "HTTP 403: Key limit exceeded. Manage it using "
    f"https://openrouter.ai/workspaces/default/keys/{_KEY_ID}"
)
_SK_OR = "sk-or-v1-" + "a" * 48
_SK = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)


class TestSanitizePersistedError:
    def test_openrouter_keys_url_is_rewritten(self):
        sanitized = _sanitize_persisted_error(_OPENROUTER_ERROR)
        assert _KEY_ID not in sanitized
        assert "Key limit exceeded" in sanitized
        assert "/keys/[redacted]" in sanitized

    def test_sk_and_sk_or_v1_secrets_are_redacted(self):
        error = f"HTTP 401: Invalid API key {_SK_OR} (also {_SK})"
        sanitized = _sanitize_persisted_error(error)
        assert _SK_OR not in sanitized
        assert _SK not in sanitized
        assert "Invalid API key" in sanitized

    def test_benign_short_error_passes_through(self):
        assert _sanitize_persisted_error("Connection timed out") == (
            "Connection timed out"
        )

    def test_none_passthrough(self):
        assert _sanitize_persisted_error(None) is None

    def test_empty_string_passthrough(self):
        assert _sanitize_persisted_error("") == ""

    def test_bounds_length(self):
        sanitized = _sanitize_persisted_error("x" * 50_000)
        assert len(sanitized) == _MAX_PERSISTED_ERROR_CHARS


class TestMarkJobRunRedactsLastError:
    def test_last_error_redacted_on_persisted_job(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        assert mark_job_run(job["id"], success=False, error=_OPENROUTER_ERROR) is True

        stored = get_job(job["id"])
        assert _KEY_ID not in stored["last_error"]
        assert "Key limit exceeded" in stored["last_error"]

        from cron.jobs import JOBS_FILE
        raw = JOBS_FILE.read_text(encoding="utf-8")
        assert _KEY_ID not in raw

    def test_sk_prefix_redacted_on_persisted_job(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        error = f"provider rejected key {_SK_OR}"
        assert mark_job_run(job["id"], success=False, error=error) is True

        stored = get_job(job["id"])
        assert _SK_OR not in stored["last_error"]
        from cron.jobs import JOBS_FILE
        raw = JOBS_FILE.read_text(encoding="utf-8")
        assert _SK_OR not in raw

    def test_benign_error_unchanged_on_disk(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="Connection timed out")
        stored = get_job(job["id"])
        assert stored["last_error"] == "Connection timed out"

    def test_last_delivery_error_redacted_on_persisted_job(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        assert (
            mark_job_run(
                job["id"], success=True, delivery_error=_OPENROUTER_ERROR
            )
            is True
        )

        stored = get_job(job["id"])
        assert _KEY_ID not in stored["last_delivery_error"]
        from cron.jobs import JOBS_FILE
        raw = JOBS_FILE.read_text(encoding="utf-8")
        assert _KEY_ID not in raw


class TestNoteFireForwardFailureRedactsDetail:
    def test_last_fire_error_detail_redacted(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        assert note_fire_forward_failure(job["id"], _OPENROUTER_ERROR) is True

        detail = get_job(job["id"])["last_fire_error"]["detail"]
        assert _KEY_ID not in detail


class TestUpdateJobRedactsDeliveryError:
    def test_update_job_sanitizes_last_delivery_error(self, tmp_cron_dir):
        """Scheduler interrupt path writes delivery_error via update_job."""
        job = create_job(prompt="watch prices", schedule="every 1h")
        updated = update_job(job["id"], {"last_delivery_error": _OPENROUTER_ERROR})
        assert _KEY_ID not in updated["last_delivery_error"]
        from cron.jobs import JOBS_FILE
        raw = JOBS_FILE.read_text(encoding="utf-8")
        assert _KEY_ID not in raw


class TestExecutionsLedgerRedactsError:
    def test_finish_execution_redacts_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "cron.executions.EXECUTIONS_FILE", tmp_path / "executions.db"
        )
        from cron.executions import create_execution, finish_execution

        record = create_execution("job-1", source="test")
        result = finish_execution(
            record["id"], success=False, error=_OPENROUTER_ERROR
        )
        assert result is not None
        assert _KEY_ID not in (result.get("error") or "")
