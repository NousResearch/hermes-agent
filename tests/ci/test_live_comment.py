"""Tests for scripts/ci/live_comment.py run selection.

The poller now reports on a run it is not part of, and merges jobs from
sibling runs of the same commit (the Docker image build, which left ci.yml
to stop holding the CI run open). ``select_watched_runs`` decides which
sibling runs count.
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

select_watched_runs = _mod.select_watched_runs
classify_jobs = _mod.classify_jobs

DOCKER = "Docker Build, Test, and Publish"


def _run(run_id: int, name: str, created_at: str) -> dict:
    return {"id": run_id, "name": name, "created_at": created_at}


def test_selects_only_named_workflows():
    runs = [
        _run(1, DOCKER, "2026-08-08T10:00:00Z"),
        _run(2, "Deploy site", "2026-08-08T10:00:00Z"),
        _run(3, "CI", "2026-08-08T10:00:00Z"),
    ]
    selected = select_watched_runs(runs, [DOCKER])
    assert [r["id"] for r in selected] == [1]


def test_keeps_newest_attempt_per_workflow():
    """A rerun makes a second run for the same commit; the old one is stale."""
    runs = [
        _run(1, DOCKER, "2026-08-08T10:00:00Z"),
        _run(2, DOCKER, "2026-08-08T11:30:00Z"),
    ]
    selected = select_watched_runs(runs, [DOCKER])
    assert [r["id"] for r in selected] == [2]


def test_excludes_the_ci_run_itself():
    runs = [_run(7, "CI", "2026-08-08T10:00:00Z")]
    assert select_watched_runs(runs, ["CI"], exclude_run_id="7") == []
    assert len(select_watched_runs(runs, ["CI"], exclude_run_id="8")) == 1


def test_no_watch_names_selects_nothing():
    runs = [_run(1, DOCKER, "2026-08-08T10:00:00Z")]
    assert select_watched_runs(runs, []) == []
    assert select_watched_runs(runs, [""]) == []


def test_watched_run_jobs_carry_the_workflow_name_into_the_comment():
    """A watched run's jobs must stay distinguishable from CI's own jobs."""
    jobs = [
        {"name": "build (amd64)", "status": "completed", "conclusion": "failure",
         "html_url": "https://example/1", "_workflow_name": DOCKER},
        {"name": "Python tests", "status": "completed", "conclusion": "success",
         "html_url": "https://example/2"},
    ]
    completed, pending, job_urls = classify_jobs(jobs)
    assert completed[f"{DOCKER} / build (amd64)"] == "failure"
    assert completed["Python tests"] == "success"
    assert pending == []
    assert job_urls[f"{DOCKER} / build (amd64)"] == "https://example/1"


def test_parse_watch_workflows_keeps_commas_inside_a_name():
    """Workflow names contain commas, so the list is newline-separated."""
    assert _mod.parse_watch_workflows("Docker Build, Test, and Publish\n") == [
        "Docker Build, Test, and Publish"
    ]
    assert _mod.parse_watch_workflows("A\nB\n\n  C  \n") == ["A", "B", "C"]
    assert _mod.parse_watch_workflows("") == []


# ─── runs_all_completed ───────────────────────────────────────────────


def test_runs_all_completed_true_only_when_every_run_finished():
    done = {"status": "completed"}
    running = {"status": "in_progress"}
    queued = {"status": "queued"}
    assert _mod.runs_all_completed([done])
    assert _mod.runs_all_completed([done, done])
    assert not _mod.runs_all_completed([done, running])
    assert not _mod.runs_all_completed([queued])


def test_runs_all_completed_empty_list_is_not_done():
    """No run info at all must not read as 'everything passed'."""
    assert not _mod.runs_all_completed([])


def test_runs_all_completed_missing_status_is_not_done():
    assert not _mod.runs_all_completed([{}])
