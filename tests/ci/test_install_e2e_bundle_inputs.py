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


def test_every_workflow_caller_of_install_e2e_skips_bundle_inputs():
    install_e2e = _load("install-e2e.yml")
    bundle_inputs = install_e2e["jobs"]["bundle-inputs"]
    callers: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        workflow = _load(path.name)
        for job_id, caller_job in (workflow.get("jobs") or {}).items():
            if not isinstance(caller_job, dict):
                continue
            if caller_job.get("uses") != "./.github/workflows/install-e2e.yml":
                continue
            callers.append(f"{path.name}:{job_id}")
            inputs = _typed_inputs("workflow_call", caller_job.get("with") or {})
            assert not _job_runs(bundle_inputs, inputs), f"{path.name}:{job_id}"
    assert {"canary-e2e.yml:install-e2e", "stable-release.yml:install-e2e"} <= set(callers)


def test_downstream_jobs_handle_skipped_bundle_inputs_and_manifest_rehearsals():
    jobs = _load("install-e2e.yml")["jobs"]

    release_inputs = _typed_inputs("workflow_call", {"release": True, "route": "all"})
    assert not _job_runs(jobs["bundle-inputs"], release_inputs)
    assert not _job_runs(jobs["windows-bundled"], release_inputs, _SKIPPED_NEED_STATUS)
    assert not _job_runs(jobs["macos-bundled"], release_inputs, _SKIPPED_NEED_STATUS)
    assert "bundle-inputs" not in jobs["source-gate"]["needs"]
    assert _job_runs(jobs["source-gate"], release_inputs)
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
