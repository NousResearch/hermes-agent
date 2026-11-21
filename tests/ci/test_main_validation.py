"""Main validation: the safety net that lets push CI stop running every lane.

`ci.yaml` currently runs the whole pipeline on every push to main because a
push used to have no diff to classify. `HERMES_CI_PUSH_DIFF` narrows that to
the lanes the diff touches, keeping a cheap ladder of guard lanes always on
(`classify_changes.PUSH_ALWAYS_ON`). The broad coverage has to go somewhere
real, so `main-validation.yml` dispatches the FULL pipeline on a schedule and
pins `ci.yaml`'s `commit` input to the revision it claims to validate.

These tests pin the properties that make that trade honest. If the net can
silently stop being taken, the narrowing is no longer safe and these fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_VALIDATION = _REPO / ".github" / "workflows" / "main-validation.yml"
_CI = _REPO / ".github" / "workflows" / "ci.yaml"
_CLASSIFIER = _REPO / "scripts" / "ci" / "classify_changes.py"


@pytest.fixture(scope="module")
def validation() -> dict:
    return yaml.safe_load(_VALIDATION.read_text(encoding="utf-8-sig"))


@pytest.fixture(scope="module")
def ci() -> dict:
    return yaml.safe_load(_CI.read_text(encoding="utf-8-sig"))


def _steps(workflow: dict) -> list[dict]:
    steps: list[dict] = []
    for job in (workflow.get("jobs") or {}).values():
        steps.extend(job.get("steps") or [])
    return steps


def _dispatch_step(workflow: dict) -> dict:
    for step in _steps(workflow):
        if "gh workflow run" in (step.get("run") or ""):
            return step
    raise AssertionError("no step dispatches a workflow")


def test_validation_runs_on_a_schedule_and_on_demand(validation):
    # `on:` parses to the boolean True under YAML 1.1 -- that is the key.
    triggers = validation.get("on", validation.get(True))
    assert triggers, "trigger block missing"
    schedules = triggers.get("schedule") or []
    assert schedules, "a scheduled run is the point: the net is taken without anyone remembering"
    crons = [entry.get("cron") for entry in schedules]
    assert all(crons), crons
    assert "workflow_dispatch" in triggers


def test_validation_never_cancels_itself(validation):
    """A cancelled full run proves nothing and still costs its compute."""
    concurrency = validation.get("concurrency") or {}
    assert concurrency.get("cancel-in-progress") is False


def test_dispatch_pins_the_revision_under_test(validation, ci):
    """The dispatched run must test THIS commit, not whatever the branch tip became.

    `gh workflow run --ref` takes a branch or tag, so a SHA cannot ride there.
    The pin is ci.yaml's `commit` input -- which only works because every job
    in ci.yaml honours it. Assert both halves.
    """
    step = _dispatch_step(validation)
    run = step["run"]
    assert "--field commit=" in run, "the commit input is the only way to pin a SHA"
    assert 'commit="$HEAD_SHA"' in run or "commit=${HEAD_SHA}" in run

    # ci.yaml must actually consume it everywhere, or the pin is decorative.
    text = _CI.read_text(encoding="utf-8-sig")
    assert "ref: ${{ inputs.commit }}" in text
    # Every `uses: ./.github/workflows/...` child must be handed the commit.
    children = re.findall(r"uses: \.\/\.github\/workflows\/([\w.-]+\.ya?ml)", text)
    assert children, "ci.yaml is expected to call sub-workflows"
    for child in sorted(set(children)):
        block = re.search(
            rf"uses: \./\.github/workflows/{re.escape(child)}\s*\n(?:\s+.*\n)*?(?=\s{{2}}[a-z]|\Z)",
            text,
        )
        assert block, child
        if "commit:" in block.group(0):
            continue
        # Not every child takes inputs; the ones that do must take `commit`.
        child_text = (_REPO / ".github" / "workflows" / child).read_text(encoding="utf-8-sig")
        has_commit_input = re.search(r"(?m)^\s+commit:\s*$", child_text)
        assert not has_commit_input, f"{child} accepts a commit input but ci.yaml does not pass it"


def test_validation_verifies_what_the_dispatch_tested(validation):
    """Trusting the dispatch is how an old green run becomes false evidence."""
    steps = _steps(validation)
    verify = [s for s in steps if s.get("id") == "dispatch" or "headSha" in (s.get("run") or "")]
    assert verify, "no step checks which revision the dispatched run actually tested"
    text = "\n".join((s.get("run") or "") for s in verify)
    assert "headSha" in text
    assert "HEAD_SHA" in text


def test_validation_dispatches_the_real_orchestrator(validation):
    """A parallel copy of the jobs could drift from what a release actually runs."""
    step = _dispatch_step(validation)
    assert 'WORKFLOW: ci.yaml' in "\n".join(f"{k}: {v}" for k, v in step["env"].items())
    assert "--field release=false" in step["run"], (
        "release=true would force a strict release acceptance run with every E2E lane; "
        "a scheduled main validation is not a release candidate"
    )


def test_validation_does_not_weaken_the_pr_path():
    """The net must not be bought by turning off required checks the PR relies on."""
    text = _CI.read_text(encoding="utf-8-sig")
    assert "cancel-in-progress" in text
    # The aggregate gate stays required.
    assert "All required checks pass" in text


def test_push_always_on_lanes_are_cheap_and_real():
    """A guard lane that costs real compute defeats the point of the ladder."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("cc_validation", _CLASSIFIER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lanes = set(mod.classify(["agent/deadline.py"]))
    assert mod.PUSH_ALWAYS_ON <= lanes, mod.PUSH_ALWAYS_ON - lanes
    expensive = {"python", "python_prod", "frontend", "rust", "docker", "nix", "e2e", "e2e_upgrade"}
    assert not (mod.PUSH_ALWAYS_ON & expensive), mod.PUSH_ALWAYS_ON & expensive
