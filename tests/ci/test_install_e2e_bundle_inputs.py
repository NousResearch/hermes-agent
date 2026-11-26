"""install-e2e's bundle-inputs job runs only when manifests or bundled routes are requested.

The real workflow and its callers are evaluated through the Actions expression
evaluator so release, canary, and source-only dispatches skip bundle validation
while manual manifest rehearsals, explicit bundled-route fail-fast checks, and
the downstream report job keep working.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from tests.ci import _gha_expr as gha

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_STATUS_FN_RE = re.compile(r"\b(?:always|success|failure|cancelled)\s*\(")
_OK_STATUS = {"always": True, "success": True, "failure": False, "cancelled": False}
_SKIPPED_NEED_STATUS = {"always": True, "success": False, "failure": False, "cancelled": False}


def _load(name: str) -> dict:
    return YAML(typ="base").load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _typed_inputs(trigger: str, overrides: dict[str, object] | None = None) -> dict[str, object]:
    spec = _load("install-e2e.yml")["on"][trigger]["inputs"]
    merged: dict[str, object] = {}
    for key, meta in spec.items():
        raw = overrides[key] if overrides and key in overrides else meta.get("default", "")
        if meta.get("type") == "boolean":
            merged[key] = raw if isinstance(raw, bool) else str(raw).lower() == "true"
        else:
            merged[key] = "" if raw is None else str(raw)
    return merged


def _job_runs(job: dict, inputs: dict[str, object], status: dict[str, bool] | None = None) -> bool:
    effective_status = status or _OK_STATUS
    expr = job.get("if")
    if expr is None:
        return effective_status["success"]
    if not _STATUS_FN_RE.search(str(expr)) and not effective_status["success"]:
        return False
    return gha.condition(expr, {"inputs": inputs, "__status__": effective_status})


@pytest.mark.parametrize(
    ("route", "windows_manifest", "macos_manifest", "runs"),
    [
        ("all", "", "", False),
        ("", "", "", False),
        ("both", "", "", False),
        ("update", "", "", False),
        ("installer", "", "", False),
        ("windows-desktop", "", "", False),
        ("macos-desktop", "", "", False),
        ("all", "https://example.invalid/win.json", "", True),
        ("all", "", "https://example.invalid/mac.json", True),
        ("all", "https://example.invalid/win.json", "https://example.invalid/mac.json", True),
        ("bundled", "", "", True),
        ("windows-bundled", "", "", True),
        ("macos-bundled", "", "", True),
        ("windows-bundled", "https://example.invalid/win.json", "", True),
        ("macos-bundled", "", "https://example.invalid/mac.json", True),
    ],
)
def test_bundle_inputs_runs_only_for_manifests_or_bundled_routes(route, windows_manifest, macos_manifest, runs):
    job = _load("install-e2e.yml")["jobs"]["bundle-inputs"]
    for trigger in ("workflow_dispatch", "workflow_call"):
        inputs = _typed_inputs(
            trigger,
            {
                "route": route,
                "windows-bundle-manifest": windows_manifest,
                "macos-bundle-manifest": macos_manifest,
            },
        )
        assert _job_runs(job, inputs) is runs, (trigger, route, windows_manifest, macos_manifest)


@pytest.mark.parametrize("workflow", ["stable-release.yml", "canary-e2e.yml"])
def test_release_and_canary_calls_skip_bundle_inputs(workflow):
    caller = _load(workflow)["jobs"]["install-e2e"]
    assert caller["uses"] == "./.github/workflows/install-e2e.yml"
    with_inputs = caller.get("with") or {}
    # An expression would be judged here as its literal text, never as a bundled route.
    gate_inputs = ("route", "windows-bundle-manifest", "macos-bundle-manifest")
    assert not [key for key in gate_inputs if "${{" in str(with_inputs.get(key, ""))], workflow
    inputs = _typed_inputs("workflow_call", with_inputs)
    assert not _job_runs(_load("install-e2e.yml")["jobs"]["bundle-inputs"], inputs), workflow


def test_downstream_jobs_handle_skipped_bundle_inputs_and_manifest_rehearsals():
    jobs = _load("install-e2e.yml")["jobs"]

    release_inputs = _typed_inputs("workflow_call", {"release": True, "route": "all"})
    assert not _job_runs(jobs["bundle-inputs"], release_inputs)
    assert not _job_runs(jobs["windows-bundled"], release_inputs, _SKIPPED_NEED_STATUS)
    assert not _job_runs(jobs["macos-bundled"], release_inputs, _SKIPPED_NEED_STATUS)
    assert _job_runs(jobs["source-gate"], release_inputs, _SKIPPED_NEED_STATUS)
    assert _job_runs(jobs["report"], release_inputs, _SKIPPED_NEED_STATUS)

    win_inputs = _typed_inputs(
        "workflow_dispatch",
        {"route": "all", "windows-bundle-manifest": "https://example.invalid/win.json"},
    )
    assert _job_runs(jobs["bundle-inputs"], win_inputs)
    assert _job_runs(jobs["windows-bundled"], win_inputs)
    assert not _job_runs(jobs["macos-bundled"], win_inputs)

    mac_inputs = _typed_inputs(
        "workflow_dispatch",
        {"route": "all", "macos-bundle-manifest": "https://example.invalid/mac.json"},
    )
    assert _job_runs(jobs["bundle-inputs"], mac_inputs)
    assert not _job_runs(jobs["windows-bundled"], mac_inputs)
    assert _job_runs(jobs["macos-bundled"], mac_inputs)
