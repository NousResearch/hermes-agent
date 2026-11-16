"""Canary E2E runs a stable claim's E2E calls on the commit each successful canary tagged.

The real workflow files are evaluated with the repo's Actions-expression evaluator
for each way a run starts, and the commit step is replayed in Bash.
"""
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from tests.ci import _gha_expr as gha
from tests.ci import workflow_steps

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"
CANARY_COMMIT = "0123456789abcdef0123456789abcdef01234567"
MAIN_TIP = "fedcba9876543210fedcba9876543210fedcba98"
UPSTREAM = "NousResearch/hermes-agent"


def load(name: str) -> dict:
    return YAML(typ="base").load((WORKFLOWS / name).read_text(encoding="utf-8"))


def after_canary(conclusion: str = "success", branch: str = "main", repository: str = UPSTREAM) -> dict:
    """A run started by Canary Release finishing; github.sha is main's tip by then."""
    return {"github": {"event_name": "workflow_run", "repository": repository, "sha": MAIN_TIP,
                       "event": {"repository": {"default_branch": "main"},
                                 "workflow_run": {"conclusion": conclusion, "head_branch": branch,
                                                  "head_sha": CANARY_COMMIT}}},
            "inputs": {}}


def dispatched(commit: str = "") -> dict:
    return {"github": {"event_name": "workflow_dispatch", "repository": UPSTREAM, "sha": MAIN_TIP,
                       "event": {}},
            "inputs": {"commit": commit}}


def resolved_commit(ctx: dict) -> str:
    step = next(s for s in load("canary-e2e.yml")["jobs"]["resolve"]["steps"] if s.get("id") == "commit")
    return workflow_steps.outputs(step, ctx)["commit"]


def test_the_trigger_chain_names_real_workflows():
    canary_e2e = load("canary-e2e.yml")
    assert canary_e2e["on"]["workflow_run"]["workflows"] == [load("canary-release.yml")["name"]]
    assert load("install-e2e-red.yml")["on"]["workflow_run"]["workflows"] == [canary_e2e["name"]]


@pytest.mark.parametrize(("ctx", "runs"), [
    (after_canary(), True),
    (after_canary(conclusion="failure"), False),
    (after_canary(conclusion="cancelled"), False),
    (after_canary(branch="fix/some-branch"), False),
    (after_canary(repository="someone/hermes-agent"), False),
    (dispatched(), True),
])
def test_only_a_successful_upstream_canary_on_the_default_branch_starts_it(ctx, runs):
    assert gha.condition(load("canary-e2e.yml")["jobs"]["resolve"]["if"], ctx) is runs


@pytest.mark.parametrize(("ctx", "commit"), [
    (after_canary(), CANARY_COMMIT),
    (dispatched(CANARY_COMMIT), CANARY_COMMIT),
    (dispatched(), MAIN_TIP),
])
def test_the_commit_under_test_is_the_canarys_not_mains_tip(ctx, commit):
    assert resolved_commit(ctx) == commit


def test_a_dispatched_commit_must_be_a_full_sha():
    with pytest.raises(AssertionError, match="full 40-hex SHA"):
        resolved_commit(dispatched("main"))


def test_every_call_is_the_claims_call_on_the_canary_commit():
    canary_jobs = load("canary-e2e.yml")["jobs"]
    claim_jobs = load("stable-release.yml")["jobs"]
    calls = {name: job for name, job in canary_jobs.items() if "uses" in job}
    # The claim's whole E2E set: a dropped call is a suite the daily run stops covering.
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
        # The claim tag is the only input a canary has no value for.
        expected = {k: v for k, v in claim[0]["with"].items() if k not in {"commit", "exclude-ref"}}
        assert {k: v for k, v in job["with"].items() if k != "commit"} == expected, name
        # A nested job asking for more than the caller grants fails the run at startup.
        assert job.get("permissions") == claim[0].get("permissions"), name
        assert job.get("secrets") == claim[0].get("secrets"), name


@pytest.mark.parametrize(("started_by", "conclusion", "tracks"), [
    ("workflow_run", "failure", True),
    ("workflow_run", "success", True),
    ("workflow_run", "skipped", False),
    ("workflow_dispatch", "failure", False),
])
def test_the_tracker_follows_only_runs_a_canary_started(started_by, conclusion, tracks):
    ctx = {"github": {"event_name": "workflow_run",
                      "event": {"workflow_run": {"event": started_by, "conclusion": conclusion}}},
           "inputs": {}}
    assert gha.condition(load("install-e2e-red.yml")["jobs"]["track"]["if"], ctx) is tracks
