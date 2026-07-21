"""Tests for scripts/ci/live_comment.py — classify_jobs().

The poller's core logic is a pure function: take raw GitHub API job dicts
and split them into (completed, pending). The API wrapper + polling loop
are tested via E2E in CI, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "live_comment.py"
_spec = importlib.util.spec_from_file_location("live_comment", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load live_comment.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["live_comment"] = _mod
_spec.loader.exec_module(_mod)


def _job(name: str, status: str, conclusion: str | None = None, workflow: str = "") -> dict:
    """Build a raw API job dict."""
    j = {"name": name, "status": status, "conclusion": conclusion}
    if workflow:
        j["_workflow_name"] = workflow
    return j


def test_classify_empty():
    completed, pending, job_urls = _mod.classify_jobs([])
    assert completed == {}
    assert pending == []
    assert job_urls == {}


def test_classify_success():
    jobs = [_job("Python tests", "completed", "success")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "success"}
    assert pending == []


def test_classify_failure():
    jobs = [_job("Python tests", "completed", "failure")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "failure"}
    assert pending == []


def test_classify_skipped():
    jobs = [_job("Python tests", "completed", "skipped")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "skipped"}
    assert pending == []


def test_classify_in_progress():
    jobs = [_job("Python tests", "in_progress", None)]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {}
    assert pending == ["Python tests"]


def test_classify_queued():
    jobs = [_job("Python tests", "queued", None)]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {}
    assert pending == ["Python tests"]


def test_classify_waiting():
    jobs = [_job("Python tests", "waiting", None)]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {}
    assert pending == ["Python tests"]


def test_classify_mixed():
    jobs = [
        _job("Python tests", "completed", "success"),
        _job("Python lints", "completed", "failure"),
        _job("JS & TS checks", "in_progress", None),
        _job("Desktop E2E", "queued", None),
    ]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "success", "Python lints": "failure"}
    assert set(pending) == {"JS & TS checks", "Desktop E2E"}


def test_classify_infra_jobs_excluded():
    """Infra jobs (detect, all-checks-pass, comment-live) are never shown."""
    jobs = [
        _job("detect", "completed", "success"),
        _job("Detect affected areas", "completed", "success"),
        _job("all-checks-pass", "completed", "success"),
        _job("All required checks pass", "completed", "success"),
        _job("comment-live", "in_progress", None),
        _job("CI review comment (live)", "in_progress", None),
        _job("Python tests", "completed", "success"),
    ]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "success"}
    assert pending == []


def test_classify_sub_workflow_jobs_prefixed():
    """Sub-workflow jobs get 'Workflow / job' display names."""
    jobs = [
        _job("test", "completed", "success", workflow="Tests"),
        _job("check", "in_progress", None, workflow="JS Tests"),
    ]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert "Tests / test" in completed
    assert completed["Tests / test"] == "success"
    assert "JS Tests / check" in pending


def test_classify_captures_html_url():
    """The poller captures html_url per job for per-job log links."""
    jobs = [
        {**_job("Python tests", "completed", "failure"),
         "html_url": "https://github.com/repo/actions/runs/1/job/2"},
        _job("Python lints", "completed", "success"),
    ]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert job_urls["Python tests"] == "https://github.com/repo/actions/runs/1/job/2"
    # Jobs without html_url are simply absent from the dict
    assert "Python lints" not in job_urls


def test_classify_cancelled_treated_as_skipped():
    jobs = [_job("Python tests", "completed", "cancelled")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "skipped"}


def test_classify_timed_out_treated_as_failure():
    jobs = [_job("Python tests", "completed", "timed_out")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "failure"}


def test_classify_neutral_treated_as_skipped():
    jobs = [_job("Python tests", "completed", "neutral")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "skipped"}


def test_classify_action_required_treated_as_skipped():
    jobs = [_job("Python tests", "completed", "action_required")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "skipped"}


def test_classify_unknown_status_skipped():
    """Unknown status values are silently ignored, not crashed on."""
    jobs = [_job("weird-job", "unknown_status", None)]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {}
    assert pending == []
