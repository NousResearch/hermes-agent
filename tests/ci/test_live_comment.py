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

import pytest

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


def test_workflow_watch_list_names_a_workflow_that_exists():
    """The names the workflow passes must match real workflow ``name:`` values.

    A name that matches nothing makes the poller silently drop that run
    from the comment, which no unit test on its own would notice.
    """
    yaml = pytest.importorskip("hermes_yaml")
    root = Path(__file__).resolve().parents[2]
    caller = yaml.safe_load(
        (root / ".github/workflows/ci-review-comment.yml").read_text(encoding="utf-8")
    )
    step = next(
        s for s in caller["jobs"]["comment"]["steps"]
        if "WATCH_WORKFLOWS" in (s.get("env") or {})
    )
    watched = _mod.parse_watch_workflows(step["env"]["WATCH_WORKFLOWS"])
    assert watched, "the poller is watching nothing"

    known = set()
    for path in (root / ".github/workflows").glob("*.yml"):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("name"), str):
            known.add(doc["name"])

    assert set(watched) <= known, f"unknown workflow names: {set(watched) - known}"


def test_poller_never_watches_its_own_workflow():
    """The poller's own run must never gate completion.

    ``runs_all_completed`` waits until every relevant run is completed.
    The poller's run is in progress for as long as it polls, so watching
    itself would make the loop wait for itself and only ever exit on
    timeout.
    """
    yaml = pytest.importorskip("hermes_yaml")
    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load(
        (root / ".github/workflows/ci-review-comment.yml").read_text(encoding="utf-8")
    )
    own_name = doc["name"]
    step = next(
        s for s in doc["jobs"]["comment"]["steps"]
        if "WATCH_WORKFLOWS" in (s.get("env") or {})
    )
    watched = _mod.parse_watch_workflows(step["env"]["WATCH_WORKFLOWS"])
    assert own_name not in watched


def test_the_merge_gate_never_shows_as_a_job_in_the_comment():
    """The gate's display name is versioned (bumped to void pre-change greens);
    every rename must reach the infra list, or the comment lists the gate as a job."""
    yaml = pytest.importorskip("hermes_yaml")
    root = Path(__file__).resolve().parents[2]
    ci = yaml.safe_load((root / ".github/workflows/ci.yaml").read_text(encoding="utf-8"))
    gate = ci["jobs"]["all-checks-pass"]["name"]
    completed, pending, _ = classify_jobs([{"name": gate, "status": "completed", "conclusion": "success"}])
    assert (completed, pending) == ({}, [])


GUARDS = "Repository guards / Repository guards"


def test_the_comment_names_each_red_step_of_a_failed_job():
    """repo-guards.yml and lint.yml run several checks as steps of one job: the comment names the
    red check, and a step its own review status covers is not repeated as a job failure."""
    url = "https://github.com/o/r/actions/runs/1/job/2"
    jobs = [{"name": GUARDS, "status": "completed", "conclusion": "failure", "html_url": url, "steps": [
        {"name": "Reject case-colliding filenames", "conclusion": "failure"},
        {"name": "Judge files added or modified against the base branch tip", "conclusion": "success"},
        {"name": "Fail on critical supply-chain findings", "conclusion": "skipped"},
        {"name": "Fail on unbounded deps (supply-chain policy)", "conclusion": "failure"}]}]
    completed, _, job_urls = classify_jobs(jobs)
    assert completed == {GUARDS: "failure"}, "the poller's job counts and transitions keep the job"
    by_check, check_urls = _mod.split_failed_steps(jobs, completed, job_urls)
    assert by_check == {f"{GUARDS} / Reject case-colliding filenames": "failure",
                        f"{GUARDS} / Fail on unbounded deps (supply-chain policy)": "failure"}
    items = _mod._import_assembler().collect_failed_jobs(
        _mod.json.dumps(by_check), "RUN", exclude_sources={"supply chain"}, job_urls=check_urls)
    assert [(item.title, item.job_url) for item in items] == [
        (f"{GUARDS} / Reject case-colliding filenames", url)]


def test_a_job_keeps_its_entry_without_a_red_step_or_a_failure():
    """A timeout fails the job with no failed step; a green job keeps its entry whatever an
    advisory step reported."""
    timed_out = {"name": "Python lints / Python lints", "status": "completed", "conclusion": "timed_out",
                 "steps": [{"name": "Run footgun checker", "conclusion": "success"}]}
    green = dict(timed_out, name=GUARDS, conclusion="success",
                 steps=[{"name": "Run ruff + ty on HEAD", "conclusion": "failure"}])
    completed, _, job_urls = classify_jobs([timed_out, green])
    assert _mod.split_failed_steps([timed_out, green], completed, job_urls)[0] == {
        "Python lints / Python lints": "failure", GUARDS: "success"}


def test_the_posted_comment_names_the_red_step(monkeypatch, capsys):
    jobs = [{"name": GUARDS, "status": "completed", "conclusion": "failure",
             "steps": [{"name": "Reject case-colliding filenames", "conclusion": "failure"}]}]
    monkeypatch.setattr(_mod, "collect_run_jobs", lambda *args, **kwargs: (jobs, True))
    monkeypatch.setattr(_mod, "fetch_all_review_statuses", lambda *args: [])
    monkeypatch.setattr(_mod.time, "sleep", lambda seconds: None)
    assert _mod.run("token", "o/r", "1", "7", "RUN", dry_run=True) == 0
    assert f"Job **{GUARDS} / Reject case-colliding filenames** failed." in capsys.readouterr().out


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
