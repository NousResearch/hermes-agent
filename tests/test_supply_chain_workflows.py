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


def _load_dependabot() -> dict:
    """Load .github/dependabot.yml using the same YAML helper style as _load_workflow."""
    path = REPO_ROOT / ".github" / "dependabot.yml"
    assert path.exists(), f"Missing dependabot config: {path}"
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - assertion path
        pytest.fail(f"dependabot.yml is not valid YAML: {exc}")
    assert isinstance(parsed, dict), "dependabot.yml must parse to a YAML mapping"
    return parsed


def test_supply_chain_audit_workflow_static_contract():
    """High-signal static contract for supply-chain-audit.yml per batch 005.

    - run only on pull_request with opened/synchronize/reopened
    - permissions contents: read and pull-requests: write
    - checkout action pinned to a 40-char SHA
    - has jobs changes, scan, dep-bounds
    - scan job needs changes and gates with if containing changes.outputs...
    - critical patterns include .pth and base64+exec/subprocess
    - dep-bounds checks unbounded >= specifiers
    """
    workflow = _load_workflow("supply-chain-audit.yml")

    # trigger contract
    triggers = workflow.get(True) or workflow.get("on") or {}
    pr_trigger = triggers.get("pull_request") or {}
    types = pr_trigger.get("types", [])
    assert "opened" in types and "synchronize" in types and "reopened" in types, (
        "supply-chain-audit must include opened/synchronize/reopened for PR types"
    )
    assert "push" not in triggers
    assert "schedule" not in triggers
    assert "workflow_dispatch" not in triggers

    # permissions contract
    permissions = workflow.get("permissions", {})
    assert permissions.get("contents") == "read"
    assert permissions.get("pull-requests") == "write"

    # checkout pinned to 40-char SHA (non-brittle, parse based)
    for job in workflow.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []):
            if isinstance(step, dict):
                uses = step.get("uses", "")
                if isinstance(uses, str) and "checkout@" in uses:
                    ref = uses.split("@")[-1].split()[0]
                    assert len(ref) == 40 and all(
                        ch in "0123456789abcdef" for ch in ref.lower()
                    ), f"checkout must be pinned to 40-char SHA, got {uses}"

    # jobs contract
    jobs = workflow.get("jobs", {})
    assert "changes" in jobs
    assert "scan" in jobs
    assert "dep-bounds" in jobs

    # scan job needs + if gate (high-signal, matches documented contract)
    scan_job = jobs.get("scan", {})
    assert scan_job.get("needs") == "changes"
    if_cond = str(scan_job.get("if", "")).lower()
    assert "changes.outputs" in if_cond or "needs.changes.outputs" in if_cond
    assert "scan" in if_cond and "true" in if_cond

    # critical patterns (use existing _all_run_blocks helper, no line-order)
    run_blocks = _all_run_blocks(workflow)
    joined_runs = "\n".join(run_blocks).lower()
    assert ".pth" in joined_runs, "must detect .pth critical pattern"
    assert "base64" in joined_runs, "must detect base64 critical ingredient"
    assert "exec" in joined_runs or "eval" in joined_runs, (
        "must detect exec/eval critical ingredient"
    )
    assert "subprocess" in joined_runs, "must detect subprocess encoded critical pattern"
    assert "chr" in joined_runs or "\\\\x" in joined_runs or "base64" in joined_runs

    # dep-bounds checks unbounded >= specifiers
    dep_job = jobs.get("dep-bounds", {})
    assert dep_job.get("needs") == "changes"
    dep_steps = dep_job.get("steps", [])
    dep_runs = "\n".join(
        s.get("run", "") for s in dep_steps if isinstance(s, dict)
    ).lower()
    assert ">=" in dep_runs or "unbounded" in dep_runs, "dep-bounds must check unbounded >= specs"
    assert "pyproject.toml" in dep_runs


def test_dependabot_yml_static_contract():
    """dependabot.yml static contract per batch 005.

    - version 2
    - only ecosystem github-actions
    - directory /
    - weekly schedule
    - labels include dependencies and github-actions
    - open-pull-requests-limit is constrained
    - no pip/npm ecosystems
    """
    dependabot = _load_dependabot()

    assert dependabot.get("version") == 2

    updates = dependabot.get("updates", [])
    ecosystems = [
        u.get("package-ecosystem")
        for u in updates
        if isinstance(u, dict)
    ]
    assert "github-actions" in ecosystems
    assert "pip" not in ecosystems
    assert "npm" not in ecosystems

    for update in updates:
        if not isinstance(update, dict):
            continue
        if update.get("package-ecosystem") == "github-actions":
            assert update.get("directory") == "/"
            schedule = update.get("schedule", {})
            assert schedule.get("interval") == "weekly"
            labels = update.get("labels", [])
            assert "dependencies" in labels
            assert "github-actions" in labels
            assert "open-pull-requests-limit" in update
            limit = update.get("open-pull-requests-limit")
            assert isinstance(limit, int) and limit > 0, "open-pull-requests-limit must be constrained"


def test_batch_005_supply_chain_audit_dependabot_contract_plan_exists():
    """The batch 005 plan doc must exist and name the contract + no workflow behavior changes."""
    plan_path = REPO_ROOT / "docs" / "plans" / "2026-06-20-005-supply-chain-audit-contract.md"
    assert plan_path.exists(), f"Missing Batch 005 plan doc: {plan_path}"
    content = plan_path.read_text(encoding="utf-8")
    assert "Batch 005" in content or "batch 005" in content.lower()
    assert "supply-chain-audit/dependabot contract" in content
    assert "no workflow behavior changes" in content
    # optional cross-link to 001
    assert (
        "2026-06-20-001-verified-supply-chain-observability-plan.md" in content
        or "supply-chain-observability" in content
    )
