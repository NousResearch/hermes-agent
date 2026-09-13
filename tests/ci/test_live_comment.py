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


def test_cancelled_job_is_not_reported_as_skipped():
    """A cancelled lane never ran its assertions, so it is not a pass.

    The comment shares its ``{job: result}`` dict with the merge gate,
    which blocks on anything outside success/skipped. Mapping
    ``cancelled`` onto ``skipped`` rendered a ✅ for the very lane the
    gate was failing on.
    """
    jobs = [
        {"name": "Python tests", "status": "completed", "conclusion": "cancelled",
         "html_url": "https://example/1"},
    ]
    completed, pending, _ = classify_jobs(jobs)
    assert completed["Python tests"] == "cancelled"
    assert pending == []


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
    yaml = pytest.importorskip("yaml")
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
    yaml = pytest.importorskip("yaml")
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


# ─── run-level conclusion authority ───────────────────────────────────


def _assembler():
    """Load the real assembler so tests assert the rendered comment body."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "assemble_review_comment.py"
    spec = importlib.util.spec_from_file_location("assemble_review_comment", path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load assemble_review_comment.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["assemble_review_comment"] = mod
    spec.loader.exec_module(mod)
    return mod


def _render(runs: list[dict], jobs: list[dict]) -> str:
    """Render the comment the poller would publish for this API state."""
    completed, pending, job_urls = classify_jobs(jobs)
    failed_runs, run_urls = _mod.classify_runs(runs, jobs)
    return _mod.build_comment_body(
        _assembler(), completed, pending, "https://example/run", job_urls, "",
        waiting=not _mod.runs_all_completed(runs),
        failed_runs=failed_runs, run_urls=run_urls,
    )


def _completed_run(conclusion: str, name: str = "CI", run_id: int = 33919406597) -> dict:
    return {
        "id": run_id, "name": name, "status": "completed", "conclusion": conclusion,
        "html_url": f"https://example/runs/{run_id}",
    }


@pytest.mark.parametrize("conclusion", ["action_required", "cancelled", "startup_failure", ""])
def test_completed_run_with_no_jobs_never_renders_all_good(conclusion):
    """A run can settle without passing while its jobs endpoint stays empty.

    Live witness on this repository: run 33919406597 was
    ``status=completed conclusion=action_required`` with ``total_count: 0``
    jobs. Judged from jobs alone that is indistinguishable from a green
    build, so the comment published "all good!" for a commit that ran no
    assertions at all.
    """
    body = _render([_completed_run(conclusion)], [])

    assert "all good!" not in body
    assert "CI (workflow run)" in body
    assert (conclusion or "no conclusion") in body


def test_a_zero_job_run_that_passed_still_renders_all_good():
    """The fix must not turn every quiet run into a false red."""
    body = _render([_completed_run("success")], [])
    assert "all good!" in body


def test_a_watched_sibling_run_is_held_to_the_same_conclusion_rule():
    """Docker/Nix are separate runs; a cancelled one must not vanish."""
    body = _render(
        [_completed_run("success"), _completed_run("cancelled", name=DOCKER, run_id=42)],
        [{"name": "Python tests", "status": "completed", "conclusion": "success",
          "_run_id": "33919406597"}],
    )
    assert "all good!" not in body
    assert f"{DOCKER} (workflow run)" in body


def test_a_run_still_queued_contributes_no_conclusion_item():
    """Only a settled run has a conclusion to judge."""
    running = {"id": 1, "name": "CI", "status": "in_progress", "conclusion": None}
    assert _mod.classify_runs([running], []) == ({}, {})


def test_a_failing_job_is_not_repeated_as_a_run_level_failure():
    """The run's red is already on the comment; don't double every build."""
    jobs = [{"name": "Python tests", "status": "completed", "conclusion": "failure",
             "_run_id": "33919406597"}]
    body = _render([_completed_run("failure")], jobs)

    assert "all good!" not in body
    assert "CI (workflow run)" not in body
    assert "Python tests" in body


def test_a_run_whose_jobs_are_all_green_still_reports_its_own_failure():
    """The gate job is infra and invisible; only the run records its red."""
    jobs = [{"name": "Python tests", "status": "completed", "conclusion": "success",
             "_run_id": "33919406597"}]
    body = _render([_completed_run("failure")], jobs)

    assert "all good!" not in body
    assert "CI (workflow run)" in body


# ─── job conclusion projection fails closed ───────────────────────────


@pytest.mark.parametrize(
    "conclusion", ["cancelled", "action_required", "neutral", "timed_out", "failure", "stale", ""],
)
def test_non_passing_job_conclusions_are_never_projected_onto_a_pass(conclusion):
    """Only success/skipped pass. Unknown conclusions fail closed.

    ``action_required`` and ``neutral`` used to map to ``skipped``, and any
    conclusion the map did not list defaulted to ``skipped`` — so a lane the
    gate blocks on rendered as a green ✅ in the comment.
    """
    completed, _, _ = classify_jobs(
        [{"name": "Python tests", "status": "completed", "conclusion": conclusion}]
    )
    assert completed["Python tests"] not in ("success", "skipped")

    body = _render([_completed_run("success")], [
        {"name": "Python tests", "status": "completed", "conclusion": conclusion,
         "_run_id": "33919406597"},
    ])
    assert "all good!" not in body
    assert "Python tests" in body


@pytest.mark.parametrize("conclusion", ["success", "skipped"])
def test_pass_like_job_conclusions_stay_pass_like(conclusion):
    completed, _, _ = classify_jobs(
        [{"name": "Python tests", "status": "completed", "conclusion": conclusion}]
    )
    assert completed["Python tests"] == conclusion


def test_the_comment_and_the_merge_gate_agree_on_what_passed():
    """Parity is the invariant: divergence is what produced the false ✅."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "gate_results.py"
    spec = importlib.util.spec_from_file_location("gate_results", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    assert set(_mod._PASSING_CONCLUSIONS) == set(gate.PASSING_RESULTS)
