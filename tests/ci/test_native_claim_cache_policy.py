"""Execute native claim cache gates from the anchor-expanded shared workflow.

The runner owns file hashing and the cache service; tests supply their outputs,
not a replacement save predicate. Claim kind comes from the payload tag even
when the legacy build_commit/channel caller flags are empty.
"""
from __future__ import annotations

import pytest

from tests.ci.desktop_release_roles import (
    BUILD_CACHE_ACTION, NATIVE_TARGETS, ROOT, SHA, admitted, evaluate, gate, load,
)

WORKFLOW = load()
NATIVE_JOBS = {
    name: job for name, job in WORKFLOW["jobs"].items()
    if any(step.get("uses") == BUILD_CACHE_ACTION for step in job.get("steps", []))
}
KINDS = ("stable", "canary", "commit", "channel")
SAVE_STEPS = [
    pytest.param(name, step, id=f"{name}-{step['name']}")
    for name, job in NATIVE_JOBS.items() if job["cache-mode"] == "write"
    for step in job["steps"]
    if step.get("uses", "").startswith("actions/cache/save@")
    or (step.get("uses") == BUILD_CACHE_ACTION and step["with"]["phase"] == "save")
]
PREPARATION_SAVES = {"Save dependency inputs before building", "Save the bundle-assembly caches"}
MAC_HASH_PATTERN = ".cache/mac-payload-signatures/**"


def claim_context(kind, job, *, prepare="success", digest="new-signatures", matched="old-key"):
    tag = "v0.28.0" if kind == "stable" else f"{kind}.20260818T101010Z-v0.28.0"
    inputs = {name: row.get("default", False if row["type"] == "boolean" else "")
              for name, row in WORKFLOW.get("on", WORKFLOW.get(True))["workflow_call"]["inputs"].items()}
    inputs.update(tag=tag, build_commit="", channel="", upload_release=False,
                  **{"claim-tag": "rc.1-v0.28.0" if kind == "stable" else tag,
                     "release-phase": "candidate"})
    needs = admitted(["validate"])
    needs["validate"]["outputs"].update({"sha": SHA, "channel-build": ""})
    return inputs, needs, {
        "matrix": {"target": job["strategy"]["matrix"]["target"][0]},
        "steps": {"prepare": {"outcome": prepare},
                  "mac-signatures": {"outputs": {"cache-matched-key": matched}}},
        "hash_files": {MAC_HASH_PATTERN: digest},
    }


def save_runs(step, inputs, needs, context, **status):
    if not gate(step.get("if", "success()"), inputs, needs, job_if=False, **context, **status):
        return False
    if step.get("uses") != BUILD_CACHE_ACTION:
        return True
    # Traverse the real composite save gate too, not just its caller's phase.
    action = load(ROOT / BUILD_CACHE_ACTION / "action.yml")
    (inner,) = [row for row in action["runs"]["steps"]
                if row.get("uses", "").startswith("actions/cache/save@")]
    return gate(inner["if"], step["with"], needs, job_if=False,
                steps={"describe": {"outcome": "success"}}, **status)


@pytest.mark.parametrize(("expression", "expected"), [
    ("matrix.target.label == 'darwin-arm64'", True),
    ("hashFiles('.cache/mac-payload-signatures/**') != ''", True),
    ("hashFiles('missing/**') != ''", False),
    ("format('mac-signatures-v1-{0}-{1}', matrix.target.label, "
     "hashFiles('.cache/mac-payload-signatures/**'))", "mac-signatures-v1-darwin-arm64-digest"),
])
def test_cache_gate_runner_context(expression, expected):
    assert evaluate(expression, {}, {}, job_if=False,
                    matrix={"target": {"label": "darwin-arm64"}},
                    hash_files={".cache/mac-payload-signatures/**": "digest", "missing/**": ""}) == expected


@pytest.mark.parametrize("kind", KINDS)
def test_claims_select_the_same_native_legs_and_keep_restores(kind):
    selected = []
    for name, job in NATIVE_JOBS.items():
        inputs, needs, context = claim_context(kind, job)
        if not gate(job["if"], inputs, needs):
            continue
        selected.append(context["matrix"]["target"]["label"])
        assert job["cache-mode"] == "write", name
        restores = [step for step in job["steps"]
                    if step.get("uses", "").startswith("actions/cache/restore@")
                    or (step.get("uses") == BUILD_CACHE_ACTION and step["with"]["phase"] == "restore")]
        assert restores, name
        prepare = next(step for step in job["steps"] if step.get("id") == "prepare")
        for step in [*restores, prepare]:
            assert gate(step.get("if", "success()"), inputs, needs, job_if=False, **context), step["name"]
    assert sorted(selected) == sorted(NATIVE_TARGETS)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("name", "step"), SAVE_STEPS)
def test_native_claim_cache_saves_follow_tag_policy(kind, name, step):
    inputs, needs, context = claim_context(kind, NATIVE_JOBS[name])
    assert gate(NATIVE_JOBS[name]["if"], inputs, needs)
    eligible_target = (step["name"] != "Save the bundle-assembly caches"
                       or context["matrix"]["target"]["label"] == "win32-x64")
    assert save_runs(step, inputs, needs, context) == (kind in ("stable", "canary") and eligible_target)


@pytest.mark.parametrize("kind", ("stable", "canary"))
@pytest.mark.parametrize(("name", "step"), SAVE_STEPS)
@pytest.mark.parametrize("state", ("cancelled", "failed", "unprepared", "empty", "unchanged", "dry"))
def test_native_save_prerequisites_are_preserved(kind, name, step, state):
    job = NATIVE_JOBS[name]
    target = job["strategy"]["matrix"]["target"][0]["label"]
    inputs, needs, context = claim_context(
        kind, job, prepare="failure" if state == "unprepared" else "success",
        digest="" if state == "empty" else "new-signatures",
        matched=f"mac-signatures-v1-{target}-new-signatures" if state == "unchanged" else "old-key")
    if state == "dry":
        inputs["release-phase"] = ""
    preparation_save = step["name"] in PREPARATION_SAVES
    mac_save = step["name"] == "Save verified macOS payload signatures"
    expected = not (
        state == "cancelled" or (state == "failed" and not preparation_save)
        or (state == "unprepared" and preparation_save)
        or (mac_save and state in ("empty", "unchanged", "dry"))
        or (step["name"] == "Save the bundle-assembly caches" and target != "win32-x64"))
    assert save_runs(step, inputs, needs, context,
                     cancelled=state == "cancelled", failed=state == "failed") == expected


def test_native_cache_audit_has_no_implicit_post_saves():
    """Walk local composites too: actions/cache@ can save after its step finishes."""
    pending = [step for job in NATIVE_JOBS.values() for step in job["steps"]]
    visited = set()
    found = set()
    while pending:
        step = pending.pop()
        uses = step.get("uses", "")
        assert not uses.startswith("actions/cache@"), step
        if uses.startswith("actions/cache/save@"):
            found.add(step["name"])
        if uses.startswith("./") and uses not in visited:
            visited.add(uses)
            pending.extend(load(ROOT / uses / "action.yml")["runs"]["steps"])
    assert found == {"Save the bundle-assembly caches", "Save verified payload signatures",
                     "Save verified macOS payload signatures",
                     "Save dependency candidates before compilation and signing"}
