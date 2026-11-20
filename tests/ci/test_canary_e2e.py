"""Canary E2E is manual-only now that Stable Release runs the canary suites directly;
Stable Release's own track-canary-red job updates the install-e2e-red tracker on every
canary completion (avoiding GitHub Actions' workflow_run suppression on GITHUB_TOKEN
dispatches), while install-e2e-red.yml remains manual-only for preview/replay.

The real workflow files are evaluated with the repo's Actions-expression evaluator
for each way a run starts, and the commit step is replayed in Bash.
"""
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from scripts.releases import lanes as lane_set
from tests.ci import _gha_expr as gha
from tests.ci import workflow_steps

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"
CANARY_COMMIT = "0123456789abcdef0123456789abcdef01234567"
MAIN_TIP = "fedcba9876543210fedcba9876543210fedcba98"
UPSTREAM = "NousResearch/hermes-agent"
RUN_ID = 4242


def load(name: str) -> dict:
    return YAML(typ="base").load((WORKFLOWS / name).read_text(encoding="utf-8"))


def dispatched(commit: str = "") -> dict:
    return {"github": {"event_name": "workflow_dispatch", "repository": UPSTREAM, "sha": MAIN_TIP, "run_id": RUN_ID,
                       "event": {}},
            "inputs": {"commit": commit}}


def resolved_commit(ctx: dict) -> str:
    step = next(s for s in load("canary-e2e.yml")["jobs"]["resolve"]["steps"] if s.get("id") == "commit")
    return workflow_steps.outputs(step, ctx)["commit"]


def test_canary_e2e_and_red_tracker_replay_are_manual_only():
    canary_e2e = load("canary-e2e.yml")
    # Daily canaries run ci, install-e2e, and windows-live inside Stable Release;
    # keeping workflow_run on canary-e2e.yml would run the ~1,400-minute set twice.
    assert set(canary_e2e["on"]) == {"workflow_dispatch"}
    # Canary Release dispatches Stable Release with GITHUB_TOKEN, so GitHub Actions
    # suppresses downstream workflow_run events; Stable Release runs track-canary-red
    # directly and install-e2e-red.yml stays manual-only for dry-run preview/replay.
    assert set(load("install-e2e-red.yml")["on"]) == {"workflow_dispatch"}


@pytest.mark.parametrize(("ctx", "commit"), [
    (dispatched(CANARY_COMMIT), CANARY_COMMIT),
    (dispatched(), MAIN_TIP),
])
def test_the_commit_under_test_is_the_dispatched_commit_or_head(ctx, commit):
    assert resolved_commit(ctx) == commit
    assert gha.render(load("canary-e2e.yml")["run-name"], ctx) == f"Canary E2E on {commit}"
    assert gha.render(load("canary-e2e.yml")["concurrency"]["group"], ctx) == f"canary-e2e-{RUN_ID}"
    assert load("canary-e2e.yml")["concurrency"]["cancel-in-progress"] == "false"


def test_a_dispatched_commit_must_be_a_full_sha():
    with pytest.raises(AssertionError, match="full 40-hex SHA"):
        resolved_commit(dispatched("main"))


def test_every_call_is_the_claims_call_on_the_dispatched_commit():
    canary_jobs = load("canary-e2e.yml")["jobs"]
    claim_jobs = load("stable-release.yml")["jobs"]
    calls = {name: job for name, job in canary_jobs.items() if "uses" in job}
    # The claim's whole E2E set: a dropped call is a suite manual dispatches stop covering.
    assert {job["uses"] for job in calls.values()} == {
        "./.github/workflows/ci.yaml",
        "./.github/workflows/install-e2e.yml",
        "./.github/workflows/windows-venv-e2e.yml",
    }
    for name, job in calls.items():
        assert job["needs"] == "resolve", name
        assert job["with"]["commit"] == "${{ needs.resolve.outputs.commit }}", name
        claim = [c for c in claim_jobs.values() if c.get("uses") == job["uses"]]
        assert len(claim) == 1, f"{name}: stable-release.yml calls {job['uses']} {len(claim)} times"
        # The claim tag is the only input a manual commit dispatch has no value for.
        expected = {k: v for k, v in claim[0]["with"].items() if k not in {"commit", "exclude-ref"}}
        assert {k: v for k, v in job["with"].items() if k != "commit"} == expected, name
        # A nested job asking for more than the caller grants fails the run at startup.
        assert job.get("permissions") == claim[0].get("permissions"), name
        assert job.get("secrets") == claim[0].get("secrets"), name


@pytest.mark.parametrize(("dry_run", "shared"), [(False, True), (True, False)])
def test_a_live_tracker_replay_queues_with_automatic_runs(dry_run, shared):
    """Two runs that can write the issue at once could both create one; a preview writes nothing."""
    ctx = {"github": {"event_name": "workflow_dispatch", "run_id": RUN_ID, "event": {}},
           "inputs": {"run-id": "1", "dry-run": dry_run}}
    group = gha.render(load("install-e2e-red.yml")["concurrency"]["group"], ctx)
    assert group == ("install-e2e-red-tracker-auto" if shared else f"install-e2e-red-tracker-{RUN_ID}")


def _call_closure(rel: str, seen: set[str]) -> None:
    if rel in seen:
        return
    seen.add(rel)
    for job in load(rel)["jobs"].values():
        uses = job.get("uses", "")
        if uses.startswith("./.github/workflows/"):
            _call_closure(uses.removeprefix("./.github/workflows/"), seen)


def test_artifacts_are_named_for_the_commit_under_test():
    """Under a commit-pinned dispatch github.sha is main's tip, so a name built from it labels the wrong commit."""
    called: set[str] = set()
    for job in load("canary-e2e.yml")["jobs"].values():
        if "uses" in job:
            _call_closure(job["uses"].removeprefix("./.github/workflows/"), called)
    ctx = {"github": {"sha": MAIN_TIP}, "inputs": {"commit": CANARY_COMMIT}}
    uploads = [(rel, step) for rel in sorted(called) for job in load(rel)["jobs"].values()
               for step in job.get("steps", []) if "upload-artifact@" in str(step.get("uses", ""))]
    assert uploads
    for rel, step in uploads:
        name = step["with"].get("name", "")  # unset: upload-artifact's default, "artifact"
        assert MAIN_TIP not in str(gha.render(name, ctx)), f"{rel}: {name}"


@pytest.mark.parametrize(("admit_result", "policy", "cancelled", "tracks"), [
    ("success", "canary", False, True),
    ("success", "stable", False, False),
    ("failure", "canary", False, False),
    ("skipped", "canary", False, False),
    ("success", "canary", True, False),
])
def test_stable_release_tracks_only_completed_admitted_canary_runs(admit_result, policy, cancelled, tracks):
    stable = load("stable-release.yml")
    job = stable["jobs"]["track-canary-red"]
    assert job["needs"] == ["admit", "complete"]
    assert str(job.get("continue-on-error")).lower() == "true"
    assert job["permissions"] == {"contents": "read", "actions": "read", "issues": "write"}
    assert job["concurrency"]["cancel-in-progress"] == "false"
    ctx = {
        "github": {"run_id": RUN_ID},
        "needs": {
            "admit": {"result": admit_result, "outputs": {"policy": policy}},
            "complete": {"result": "failure"},
        },
        "__status__": {
            "always": True,
            "cancelled": cancelled,
            "success": not cancelled and admit_result in {"success", "skipped"},
            "failure": not cancelled and admit_result == "failure",
        },
    }
    assert gha.condition(job["if"], ctx) is tracks
    # Canary tracking shares the exact lock group as live manual replays so both cannot open a tracker.
    expected_group = "install-e2e-red-tracker-auto" if policy == "canary" else f"install-e2e-red-tracker-{RUN_ID}"
    assert gha.render(job["concurrency"]["group"], ctx) == expected_group
    # Diagnostic calls must keep the "Build diagnostic " prefix so red-tracker.mjs ignores them.
    for call in lane_set.DIAGNOSTIC_CALLS:
        assert stable["jobs"][call]["name"].startswith("Build diagnostic "), call
