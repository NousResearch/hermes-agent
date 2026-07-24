"""Tests for scripts/ci/live_comment.py — classify_jobs() + artifact helpers.

The poller's core logic is a pure function: take raw GitHub API job dicts
and split them into (completed, pending). Artifact parsing helpers are also
pure and tested here. The API wrapper + polling loop are tested via E2E
in CI, not here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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






def test_classify_queued():
    jobs = [_job("Python tests", "queued", None)]
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




def test_classify_timed_out_treated_as_failure():
    jobs = [_job("Python tests", "completed", "timed_out")]
    completed, pending, job_urls = _mod.classify_jobs(jobs)
    assert completed == {"Python tests": "failure"}








def test_commit_info_uses_present_tense_while_jobs_are_pending():
    info = "<sub>running on [abc1234](https://commit-url) — fix: thing</sub>"
    assert _mod._commit_info_for_state(info, ["Python tests"]) == info


def test_commit_info_uses_past_tense_after_jobs_complete():
    info = "<sub>running on [abc1234](https://commit-url) — fix: thing</sub>"
    assert _mod._commit_info_for_state(info, []) == (
        "<sub>ran on [abc1234](https://commit-url) — fix: thing</sub>"
    )


# ---------------------------------------------------------------------------
# Artifact parsing helpers
# ---------------------------------------------------------------------------

def test_parse_status_file_with_prefix():
    """GITHUB_OUTPUT format: review_status=<json>"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('review_status=[{"source":"test","results":[]}]')
        f.flush()
        statuses = _mod._parse_status_file(Path(f.name))
    assert len(statuses) == 1
    assert statuses[0]["source"] == "test"


def test_parse_status_file_without_prefix():
    """Raw JSON (no review_status= prefix) is also accepted."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('[{"source":"raw","results":[]}]')
        f.flush()
        statuses = _mod._parse_status_file(Path(f.name))
    assert len(statuses) == 1
    assert statuses[0]["source"] == "raw"


def test_parse_status_file_empty_array():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('review_status=[]')
        f.flush()
        statuses = _mod._parse_status_file(Path(f.name))
    assert statuses == []


def test_parse_status_file_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('review_status=not json')
        f.flush()
        statuses = _mod._parse_status_file(Path(f.name))
    assert statuses == []


def test_parse_status_file_nonexistent():
    assert _mod._parse_status_file(Path("/nonexistent/file.json")) == []


def test_parse_status_file_not_a_list():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('review_status={"not":"a list"}')
        f.flush()
        statuses = _mod._parse_status_file(Path(f.name))
    assert statuses == []


def test_merge_statuses_empty():
    assert _mod._merge_statuses([]) == ""


def test_merge_statuses_single():
    statuses = [{"source": "a", "results": []}]
    result = _mod._merge_statuses(statuses)
    assert json.loads(result) == statuses


def test_merge_statuses_multiple():
    statuses = [
        {"source": "a", "results": []},
        {"source": "b", "results": [{"kind": "warning"}]},
    ]
    result = _mod._merge_statuses(statuses)
    assert json.loads(result) == statuses


def test_review_status_artifact_prefix():
    """The prefix is used to filter artifacts from the API."""
    assert _mod._REVIEW_STATUS_ARTIFACT_PREFIX == "review-status-"
    # Artifacts with this prefix should be picked up
    assert "review-status-ci-timings".startswith(_mod._REVIEW_STATUS_ARTIFACT_PREFIX)
    assert "review-status-review-labels".startswith(_mod._REVIEW_STATUS_ARTIFACT_PREFIX)
    # Artifacts without it should not
    assert not "ci-timings-report".startswith(_mod._REVIEW_STATUS_ARTIFACT_PREFIX)
    assert not "playwright-report".startswith(_mod._REVIEW_STATUS_ARTIFACT_PREFIX)
