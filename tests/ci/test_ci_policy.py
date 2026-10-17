"""Behavior contracts for CI orchestration policy."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "ci_policy.py"
_spec = importlib.util.spec_from_file_location("ci_policy", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load ci_policy.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ci_policy"] = _mod
_spec.loader.exec_module(_mod)


@pytest.mark.parametrize(
    ("event_name", "repository", "ref", "expected"),
    [
        ("pull_request", "mekenthompson/hermes-agent", "refs/pull/1/merge", True),
        ("pull_request", "NousResearch/hermes-agent", "refs/pull/1/merge", True),
        ("push", "mekenthompson/hermes-agent", "refs/heads/main", True),
        ("push", "mekenthompson/hermes-agent", "refs/heads/feature", False),
        ("push", "NousResearch/hermes-agent", "refs/heads/main", False),
        ("workflow_dispatch", "mekenthompson/hermes-agent", "refs/heads/main", False),
    ],
)
def test_cancel_in_progress_preserves_upstream_pushes(
    event_name: str, repository: str, ref: str, expected: bool
) -> None:
    assert _mod.should_cancel_in_progress(event_name, repository, ref) is expected


@pytest.mark.parametrize(
    ("needs", "selected", "expected_failed"),
    [
        ({}, {}, ["selected_jobs"]),
        ({"tests": {"result": "success"}}, {"tests": True, "lint": True}, ["lint"]),
        ({"tests": {"result": "skipped"}}, {"tests": True}, ["tests"]),
        ({"tests": {"result": "skipped"}}, {"tests": False}, []),
        ({"tests": {"result": "cancelled"}}, {"tests": False}, ["tests"]),
        ({"tests": {"result": "failure"}}, {"tests": True}, ["tests"]),
        ({"tests": {}}, {"tests": True}, ["tests"]),
        ({"tests": {"result": "skipped"}}, {"tests": "false"}, ["selected_jobs"]),
    ],
)
def test_evaluate_needs_requires_selected_success_and_explicit_non_applicability(
    needs: dict[str, dict[str, str]],
    selected: dict[str, bool],
    expected_failed: list[str],
) -> None:
    assert _mod.evaluate_needs(needs, selected).failed == expected_failed


def test_selection_uses_boolean_values_and_fork_scope() -> None:
    lanes = {
        key: "false"
        for key in (
            "python",
            "python_prod",
            "frontend",
            "os_tests",
            "installer",
            "bootstrap",
            "uv_lock",
            "ci_review",
            "mcp_catalog",
            "scan",
            "deps",
            "npm_lock",
            "docker_meta",
            "rust",
            "site",
            "binary_artifacts",
        )
    }
    selected = _mod.selected_jobs(lanes, "mekenthompson/hermes-agent", "pull_request")
    assert selected["tests"] is False
    assert selected["tests-os"] is False
    assert selected["installer-tests"] is False
    assert selected["history-check"] is True
    lanes["python"] = "true"
    lanes["os_tests"] = "true"
    assert (
        _mod.selected_jobs(lanes, "mekenthompson/hermes-agent", "pull_request")[
            "tests-os"
        ]
        is False
    )


@pytest.mark.parametrize("repository", ["mekenthompson/hermes-agent", "NousResearch/hermes-agent"])
@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("catalog", [False, True])
def test_catalog_lane_uses_shared_selection(repository, event, release, catalog) -> None:
    selected = _mod.selected_jobs(
        {"catalog": str(catalog).lower()}, repository, event, release=release)
    assert selected.get("catalog-tests") is catalog
    assert _mod.image_selected_jobs(selected).get("catalog-tests") is catalog
    import yaml

    jobs = yaml.safe_load((_PATH.parents[2] / ".github/workflows/ci.yaml").read_text())["jobs"]
    assert jobs["catalog-tests"]["if"] == "${{ fromJSON(needs.detect.outputs.selected_jobs)['catalog-tests'] }}"


@pytest.mark.parametrize("catalog", [False, True])
@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("result", ["success", "skipped", "failure", "cancelled", "missing", None])
def test_catalog_gate_requires_explicit_applicable_result(catalog, image, result) -> None:
    selected = _mod.selected_jobs(
        {"catalog": str(catalog).lower()}, "mekenthompson/hermes-agent", "push")
    if image:
        selected = _mod.image_selected_jobs(selected)
    needs = {name: {"result": "success" if required else "skipped"}
             for name, required in selected.items()}
    if result is not None:
        needs["catalog-tests"] = {"result": result}
    else:
        needs.pop("catalog-tests", None)
    expected = [] if result == "success" or (not catalog and result == "skipped") else ["catalog-tests"]
    assert _mod.evaluate_needs(needs, selected).failed == expected


def test_selection_covers_each_orchestrator_prerequisite() -> None:
    yaml = pytest.importorskip("hermes_yaml")
    workflow = yaml.safe_load(
        (_PATH.parents[2] / ".github/workflows/ci.yaml").read_text()
    )
    jobs = workflow["jobs"]
    expected = set(jobs) - {"all-checks-pass", "image-checks-pass", "ci-timings"}
    selected = _mod.selected_jobs({}, "mekenthompson/hermes-agent", "pull_request")
    assert set(selected) == expected == set(jobs["all-checks-pass"]["needs"])


def test_image_gate_keeps_runtime_failures_and_excludes_only_other_surfaces() -> None:
    import yaml

    jobs = yaml.safe_load((_PATH.parents[2] / ".github/workflows/ci.yaml").read_text())["jobs"]
    selected = {name: True for name in jobs["all-checks-pass"]["needs"]}
    image = _mod.image_selected_jobs(selected)
    assert set(image) == set(jobs["image-checks-pass"]["needs"])
    assert set(selected) - set(image) == {
        "native-install-tests",
        "tests-os", "installer-tests", "rust-tests", "bootstrap-installer",
        "e2e-desktop", "e2e-desktop-core", "e2e-desktop-update", "docs-site",
        "prompt-surface-diff", "blob-size-check",
    }
    needs = {name: {"result": "success"} for name in image}
    assert not _mod.evaluate_needs(needs, image).failed
    for name in image:
        for result in ("failure", "cancelled", "skipped", "missing"):
            broken = {**needs, name: {"result": result}}
            assert _mod.evaluate_needs(broken, image).failed == [name]
    # New lanes are required unless explicitly classified as another surface.
    expanded = _mod.image_selected_jobs({**selected, "new-runtime-check": True})
    assert _mod.evaluate_needs(needs, expanded).failed == ["new-runtime-check"]


def test_upgrade_applicability_reaches_real_upgrade_suite() -> None:
    yaml = pytest.importorskip("hermes_yaml")
    ci = yaml.safe_load((_PATH.parents[2] / ".github/workflows/ci.yaml").read_text())
    tests = yaml.safe_load(
        (_PATH.parents[2] / ".github/workflows/tests.yml").read_text()
    )
    assert (
        ci["jobs"]["tests"]["with"]["upgrade"]
        == "${{ github.repository != 'mekenthompson/hermes-agent' && needs.detect.outputs.upgrade == 'true' }}"
    )
    assert (
        ci["jobs"]["tests"]["with"]["e2e"]
        == "${{ github.repository != 'mekenthompson/hermes-agent' && needs.detect.outputs.e2e == 'true' }}"
    )
    assert tests[True]["workflow_call"]["inputs"]["upgrade"]["default"] is True
    assert tests["jobs"]["e2e-upgrade-plan"]["if"] == "inputs.upgrade"
    assert tests["jobs"]["e2e-upgrade"]["if"] == "inputs.upgrade"
    # No duplicate units/E2E in the native call; both partitions are required
    # for ordinary fork source merges, while releases retain the combined run.
    native = ci["jobs"]["native-install-tests"]
    assert native["with"] == {"commit": "${{ inputs.commit }}", "unit": False, "e2e": False, "upgrade": True, "upgrade_partition": "native"}
    assert native["if"] == "${{ fromJSON(needs.detect.outputs.selected_jobs).native-install-tests }}"
    assert ci["jobs"]["tests"]["with"]["upgrade_partition"] == "${{ github.repository == 'mekenthompson/hermes-agent' && inputs.release != true && 'runtime' || 'all' }}"
    assert tests[True]["workflow_call"]["inputs"]["unit"]["default"] is True
    assert tests[True]["workflow_call"]["inputs"]["upgrade_partition"]["default"] == "all"
    assert tests["jobs"]["test"]["if"] == "inputs.unit"
    assert "inputs.upgrade_partition" in tests["concurrency"]["group"]
    lanes = {"python": "true", "upgrade": "true"}
    for repo, release, expected in (("mekenthompson/hermes-agent", False, False),
                                    ("mekenthompson/hermes-agent", True, False),
                                    ("NousResearch/hermes-agent", False, False)):
        selected = _mod.selected_jobs(lanes, repo, "push", release=release)
        assert selected["native-install-tests"] is expected
        needs = {name: {"result": "success" if required else "skipped"} for name, required in selected.items()}
        assert not _mod.evaluate_needs(needs, selected).failed
        if expected:
            needs["native-install-tests"]["result"] = "failure"
            assert _mod.evaluate_needs(needs, selected).failed == ["native-install-tests"]


def test_review_label_gate_uses_critical_finding_result() -> None:
    needs = {"review-labels": {"result": "skipped"}}
    selected = {"review-labels": False}
    assert _mod.evaluate_needs(needs, selected).failed == []
    assert _mod.evaluate_needs(needs, selected, critical_findings=True).failed == [
        "review-labels"
    ]


def test_folded_guards_gate_detect_and_keep_release_checks() -> None:
    yaml = pytest.importorskip("hermes_yaml")
    jobs = yaml.safe_load((_PATH.parents[2] / ".github/workflows/ci.yaml").read_text())["jobs"]
    commands = {step.get("run"): step for step in jobs["detect"]["steps"]}
    for command in (
        "python3 scripts/check-case-collisions.py",
        "python3 scripts/ci/check_lazy_deps_imports.py",
    ):
        step = commands[command]
        assert step["if"] == "github.repository == 'mekenthompson/hermes-agent'"
        assert not step.get("continue-on-error", False)
    fork = _mod.selected_jobs({}, "mekenthompson/hermes-agent", "pull_request")
    upstream = _mod.selected_jobs({}, "NousResearch/hermes-agent", "pull_request")
    for job in ("case-collision-check", "lazy-deps-guard"):
        assert fork[job] is False
        assert upstream[job] is True
        assert jobs[job]["if"] == "github.repository != 'mekenthompson/hermes-agent'"
    assert fork["detect"] is True
    # A guard failure stops classification; its required parent cannot be skipped.
    assert _mod.evaluate_needs({"detect": {"result": "failure"}}, {"detect": True}).failed == ["detect"]


def test_policy_file_runs_as_the_aggregate_job_entrypoint(tmp_path: Path) -> None:
    """The checkout-provided script writes the output consumed by the workflow job."""
    output_path = tmp_path / "github-output"
    result = subprocess.run(
        [sys.executable, str(_PATH)],
        input=json.dumps({"tests": {"result": "success"}}),
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output_path),
            "SELECTED_JOBS": '{"tests":true}',
        },
        check=True,
    )

    assert "All selected checks passed" in result.stdout
    assert output_path.read_text(encoding="utf-8") == 'needs-json={"tests":"success"}\n'


def test_fork_sync_does_not_select_unshipped_surfaces() -> None:
    lanes = {key: "true" for key in (
        "python", "python_prod", "frontend", "os_tests", "installer", "bootstrap",
        "rust", "site", "upgrade", "e2e", "e2e_upgrade", "binary_artifacts",
        "uv_lock", "npm_lock", "scan", "deps", "docker_meta", "ci_review",
    )}
    selected = _mod.selected_jobs(lanes, "mekenthompson/hermes-agent", "push", release=True)
    for job in (
        "native-install-tests", "tests-os", "installer-tests", "rust-tests",
        "bootstrap-installer", "e2e-desktop", "e2e-desktop-core", "e2e-desktop-update",
        "docs-site", "contributor-check", "infographic-check", "icons-freshness-check",
    ):
        assert selected[job] is False
    assert selected["tests"] is True
    assert selected["lint"] is True
    assert selected["js-tests"] is True


def test_disabled_upstream_workflows_are_not_fork_selected_calls() -> None:
    yaml = pytest.importorskip("hermes_yaml")
    root = _PATH.parents[2]
    disabled = {
        line.split("#", 1)[0].strip()
        for line in (root / "scripts/ci/fork_disabled_workflows.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    }
    jobs = yaml.safe_load((root / ".github/workflows/ci.yaml").read_text())["jobs"]
    lanes = {key: "true" for key in ("python", "frontend", "uv_lock", "npm_lock", "scan", "deps", "docker_meta", "ci_review", "binary_artifacts")}
    selected = _mod.selected_jobs(lanes, "mekenthompson/hermes-agent", "pull_request")
    called = set()
    for name, job in jobs.items():
        if not selected.get(name):
            continue
        uses = job.get("uses") or ""
        if uses.startswith("./.github/workflows/"):
            called.add(uses.rsplit("/", 1)[-1])
    assert not (disabled & called)


def test_workflow_selection_output_is_boolean_json(tmp_path: Path) -> None:
    output_path = tmp_path / "selected"
    result = subprocess.run(
        [sys.executable, str(_PATH), "--select"],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output_path),
            "LANES": '{"python":"false","python_prod":"true","frontend":"false","upgrade":"false"}',
            "REPO": "mekenthompson/hermes-agent",
            "EVENT_NAME": "pull_request",
        },
    )
    line = output_path.read_text().strip()
    assert result.stdout.strip() == line
    selected = json.loads(line.removeprefix("selected_jobs="))
    assert selected["tests"] is False
    assert selected["e2e-desktop-core"] is False
    assert selected["e2e-desktop-update"] is False
    assert selected["native-install-tests"] is False
    assert selected["tests-os"] is False
