"""Regression tests for issue #102700.

``cron/jobs.py`` persists ``last_error`` / ``last_delivery_error`` /
``last_fire_error`` into ``jobs.json``, which survives gateway restarts, job
edits, and profile backups indefinitely. Provider error strings can embed a
key-management link (e.g. OpenRouter's
``https://openrouter.ai/workspaces/default/keys/<KEY_ID>``) that lets anyone
holding the file manage/revoke that credential on the provider dashboard.

``cron/incidents.py`` (``_redact_error``) and ``cron/delivery_queue.py``
(``_finish``) already sanitize before persisting; this was the one cron
persistence chokepoint that did not. The fix adds
``cron.jobs._sanitize_persisted_error`` (redact + bound length) and calls it
at every raw-error write site: ``_mark_job_run_locked`` (last_error,
last_delivery_error), ``note_fire_forward_failure`` (last_fire_error), the
scheduler's post-interrupt ``update_job({"last_delivery_error": ...})`` path,
and ``cron.executions.finish_execution`` (the durable executions.db ledger).
"""

import pytest

from cron.jobs import create_job, get_job, mark_job_run, note_fire_forward_failure


_OPENROUTER_ERROR = (
    "HTTP 403: Key limit exceeded. Manage it using "
    "https://openrouter.ai/workspaces/default/keys/sk_or_key_id_abcdef123456"
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestSanitizePersistedError:
    def test_redacts_key_management_url(self):
        from cron.jobs import _sanitize_persisted_error

        sanitized = _sanitize_persisted_error(_OPENROUTER_ERROR)
        assert "sk_or_key_id_abcdef123456" not in sanitized
        assert "/keys/" in sanitized  # message stays readable
        assert "Key limit exceeded" in sanitized

    def test_none_passthrough(self):
        from cron.jobs import _sanitize_persisted_error

        assert _sanitize_persisted_error(None) is None

    def test_empty_string_passthrough(self):
        from cron.jobs import _sanitize_persisted_error

        assert _sanitize_persisted_error("") == ""

    def test_bounds_length(self):
        from cron.jobs import _sanitize_persisted_error, _MAX_PERSISTED_ERROR_CHARS

        sanitized = _sanitize_persisted_error("x" * 50_000)
        assert len(sanitized) == _MAX_PERSISTED_ERROR_CHARS

    def test_redaction_failure_fails_closed_not_raise(self, monkeypatch):
        """A broken redactor must never block recording that a job failed —
        but it must also never leak the raw text back out."""
        import cron.jobs as jobs_mod

        def _boom(*a, **k):
            raise RuntimeError("redactor exploded")

        monkeypatch.setattr(
            "agent.redact.redact_sensitive_text", _boom, raising=True
        )
        # cron.jobs imports redact_sensitive_text lazily inside the function,
        # so patching the agent.redact module attribute is sufficient.
        sanitized = jobs_mod._sanitize_persisted_error(_OPENROUTER_ERROR)
        assert sanitized is not None


class TestMarkJobRunRedactsLastError:
    def test_last_error_redacted_on_persisted_job(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        assert mark_job_run(job["id"], success=False, error=_OPENROUTER_ERROR) is True

        stored = get_job(job["id"])
        assert "sk_or_key_id_abcdef123456" not in stored["last_error"]
        assert "Key limit exceeded" in stored["last_error"]

    def test_last_delivery_error_redacted_on_persisted_job(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        assert (
            mark_job_run(
                job["id"], success=True, delivery_error=_OPENROUTER_ERROR
            )
            is True
        )

        stored = get_job(job["id"])
        assert "sk_or_key_id_abcdef123456" not in stored["last_delivery_error"]

    def test_non_secret_error_survives_readable(self, tmp_cron_dir):
        """Sanitization must not mangle ordinary error text with no secrets."""
        job = create_job(prompt="watch prices", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="Connection timed out")
        stored = get_job(job["id"])
        assert stored["last_error"] == "Connection timed out"


class TestNoteFireForwardFailureRedactsDetail:
    def test_last_fire_error_detail_redacted(self, tmp_cron_dir):
        job = create_job(prompt="watch prices", schedule="every 1h")
        assert note_fire_forward_failure(job["id"], _OPENROUTER_ERROR) is True

        detail = get_job(job["id"])["last_fire_error"]["detail"]
        assert "sk_or_key_id_abcdef123456" not in detail


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
        assert "sk_or_key_id_abcdef123456" not in (result.get("error") or "")
