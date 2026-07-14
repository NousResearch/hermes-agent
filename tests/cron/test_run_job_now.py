"""Tests for synchronous single-job execution (`hermes cron run <id> --wait`).

`run_job_now()` runs ONE job end-to-end (execute → save → deliver → mark) in the
calling thread and returns a structured result, without waiting for a scheduler
tick and regardless of whether the job is "due". This backs the CLI `--wait`
flag so slow jobs can be verified to completion in the foreground.
"""
import pytest

import cron.scheduler as scheduler


@pytest.fixture
def fake_job():
    return {"id": "abc123", "name": "slow-digest", "schedule": {"expr": "0 7 * * *"}}


def _patch_pipeline(monkeypatch, *, success=True, final_response="done", output="full-doc",
                    error=None, job=None):
    """Patch the scheduler's execute/save/deliver/mark seams and record calls."""
    calls = {"run_job": 0, "save": 0, "deliver": [], "deliver_success": [], "mark": []}

    def fake_run_job(j):
        calls["run_job"] += 1
        return success, output, final_response, error

    def fake_save(job_id, out):
        calls["save"] += 1
        return f"/tmp/{job_id}.md"

    def fake_deliver(j, content, success=True, adapters=None, loop=None):
        calls["deliver"].append(content)
        calls["deliver_success"].append(success)
        return None  # no delivery error

    def fake_mark(job_id, ok, err, delivery_error=None):
        calls["mark"].append((job_id, ok, err, delivery_error))

    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(scheduler, "save_job_output", fake_save)
    monkeypatch.setattr(scheduler, "_deliver_result", fake_deliver)
    monkeypatch.setattr(scheduler, "mark_job_run", fake_mark)
    monkeypatch.setattr(scheduler, "get_job", lambda jid: job, raising=False)
    return calls


class TestRunJobNow:
    def test_runs_job_synchronously_and_reports_success(self, monkeypatch, fake_job):
        calls = _patch_pipeline(monkeypatch, success=True, final_response="hello",
                                job=fake_job)
        result = scheduler.run_job_now("abc123", verbose=False)
        assert result["success"] is True
        assert result["final_response"] == "hello"
        assert result["job_id"] == "abc123"
        assert calls["run_job"] == 1
        assert calls["save"] == 1
        # success path delivers the real response
        assert calls["deliver"] == ["hello"]
        assert calls["mark"][0][:3] == ("abc123", True, None)

    def test_failure_surfaces_alert_and_marks_failed(self, monkeypatch, fake_job):
        calls = _patch_pipeline(monkeypatch, success=False, final_response="",
                                error="boom", job=fake_job)
        result = scheduler.run_job_now("abc123", verbose=False)
        assert result["success"] is False
        assert result["error"] == "boom"
        # Failed jobs still deliver. The error is now framed by
        # _summarize_cron_failure_for_delivery (a compact one-liner) rather than
        # handed raw to the wrapper — matching run_one_job's path. A non-transient
        # defect like "boom" is delivered (not suppressed) with a "failed:" frame.
        assert calls["deliver"], "failure should still deliver an alert"
        assert len(calls["deliver"]) == 1
        assert "boom" in calls["deliver"][0]
        assert "failed" in calls["deliver"][0].lower()
        assert calls["deliver_success"] == [False]
        assert calls["mark"][0][1] is False

    def test_unknown_job_id_returns_error_without_running(self, monkeypatch):
        calls = _patch_pipeline(monkeypatch, job=None)  # get_job → None
        result = scheduler.run_job_now("nope", verbose=False)
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert calls["run_job"] == 0

    def test_silent_marker_skips_delivery(self, monkeypatch, fake_job):
        calls = _patch_pipeline(monkeypatch, success=True,
                                final_response=scheduler.SILENT_MARKER, job=fake_job)
        result = scheduler.run_job_now("abc123", verbose=False)
        assert result["success"] is True
        assert calls["deliver"] == [], "[SILENT] should suppress delivery"

    def test_does_not_require_job_to_be_due(self, monkeypatch, fake_job):
        """run_job_now bypasses get_due_jobs entirely — a not-due job still runs."""
        called = {"due": 0}
        monkeypatch.setattr(scheduler, "get_due_jobs",
                            lambda: called.__setitem__("due", called["due"] + 1) or [])
        _patch_pipeline(monkeypatch, success=True, final_response="x", job=fake_job)
        scheduler.run_job_now("abc123", verbose=False)
        assert called["due"] == 0, "run_job_now must not consult the due-list"
