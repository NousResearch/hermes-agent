"""Static guards for supply-chain security workflows.

These tests intentionally validate CI configuration rather than runtime code.
They protect two high-signal, low-noise checks:

* zizmor for GitHub Actions static analysis
* OpenSSF Scorecard for repository security posture

Both workflows should remain advisory / scheduled by default so they improve
security posture without making every contributor wait on network-heavy scans.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> dict:
    path = WORKFLOWS / name
    assert path.exists(), f"Missing workflow: {path}"
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - assertion path
        pytest.fail(f"{name} is not valid YAML: {exc}")
    assert isinstance(parsed, dict), f"{name} must parse to a YAML mapping"
    return parsed


def _all_run_blocks(workflow: dict) -> list[str]:
    blocks: list[str] = []
    for job in workflow.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []):
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                blocks.append(step["run"])
    return blocks


def test_zizmor_workflow_runs_static_analysis_without_extra_permissions():
    """zizmor should scan GitHub Actions with least privilege."""
    workflow = _load_workflow("zizmor.yml")

    permissions = workflow.get("permissions", {})
    assert permissions.get("contents") == "read", "zizmor only needs read access"
    assert set(permissions) == {"contents"}, "do not grant broad zizmor permissions"

    joined_runs = "\n".join(_all_run_blocks(workflow))
    assert "zizmor" in joined_runs.lower(), "workflow must invoke zizmor"
    assert ".github/workflows" in joined_runs, "zizmor must scan workflow files"
    assert "uvx" in joined_runs or "uv tool run" in joined_runs, (
        "prefer uvx/uv tool execution instead of unpinned action wrappers"
    )


def test_zizmor_workflow_is_advisory_or_manual_not_required_on_every_push():
    """Keep the check useful without blocking all normal PRs by default."""
    workflow = _load_workflow("zizmor.yml")
    triggers = workflow.get(True) or workflow.get("on") or {}
    assert "workflow_dispatch" in triggers, "zizmor should be manually runnable"
    assert "schedule" in triggers, "zizmor should run on a schedule"
    assert "pull_request" not in triggers, (
        "do not make zizmor required on every PR until false positives are tuned"
    )


def test_scorecard_workflow_uses_official_action_with_minimal_permissions():
    """OpenSSF Scorecard should run with the documented narrow permissions."""
    workflow = _load_workflow("scorecard.yml")

    permissions = workflow.get("permissions", {})
    assert permissions.get("contents") == "read"
    assert permissions.get("security-events") == "write"
    assert "id-token" not in permissions, "OIDC is not needed for Scorecard SARIF upload"

    uses_lines = []
    for job in workflow.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []):
            if isinstance(step, dict) and isinstance(step.get("uses"), str):
                uses_lines.append(step["uses"])
    assert any(line.startswith("ossf/scorecard-action@") for line in uses_lines), (
        "scorecard.yml must use the official ossf/scorecard-action"
    )


def test_scorecard_workflow_is_scheduled_and_manual_only():
    """Scorecard is repo-posture telemetry; schedule/manual avoids PR noise."""
    workflow = _load_workflow("scorecard.yml")
    triggers = workflow.get(True) or workflow.get("on") or {}
    assert "workflow_dispatch" in triggers
    assert "schedule" in triggers
    assert "pull_request" not in triggers
    assert "push" not in triggers


def test_new_supply_chain_workflows_pin_external_actions_by_sha():
    """External actions in these workflows must use full commit SHAs.

    Existing Hermes CI pins actions by commit SHA instead of mutable tags. Keep
    that invariant for the new supply-chain posture checks too.
    """
    for workflow_name in ("zizmor.yml", "scorecard.yml"):
        workflow = _load_workflow(workflow_name)
        for job in workflow.get("jobs", {}).values():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps", []):
                if not isinstance(step, dict):
                    continue
                uses = step.get("uses")
                if not isinstance(uses, str) or uses.startswith("./"):
                    continue
                assert "@" in uses, f"{workflow_name}: action is not pinned: {uses}"
                ref = uses.rsplit("@", 1)[1].split()[0]
                assert len(ref) == 40 and all(ch in "0123456789abcdef" for ch in ref.lower()), (
                    f"{workflow_name}: action must be pinned to a full commit SHA, got {uses}"
                )


def test_verified_improvement_plan_is_documented_in_repo_docs():
    """The implementation slice should have an in-repo plan, not only local notes."""
    plan_path = REPO_ROOT / "docs" / "plans" / "2026-06-20-001-verified-supply-chain-observability-plan.md"
    assert plan_path.exists(), f"Missing verified improvement plan: {plan_path}"
    content = plan_path.read_text(encoding="utf-8")
    for required in ("Zizmor", "OpenSSF Scorecard", "Observability baseline", "Sandbox policy"):
        assert required in content, f"Plan is missing required section/topic: {required}"
