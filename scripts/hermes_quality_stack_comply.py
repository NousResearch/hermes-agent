#!/usr/bin/env python3
"""Numeric compliance report for the Hermes quality stack rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_ISSUES = [
    ("P0-01", "Baseline repo/runtime state", ["pyproject.toml", ".hermes/context.md"]),
    ("P0-02", "Local runtime verification path", ["scripts/run_tests.sh"]),
    ("P0-03", "VPS verification path", ["docs/hermes-agent-standalone/vps-team-workspace-runbook.md", "scripts/sync_vps_workspace.sh"]),
    ("P0-04", "Rollback path", ["docs/quality-stack/implementation-plan.md"]),
    ("P1-01", "Langfuse backend choice", ["plugins/observability/langfuse/plugin.yaml"]),
    ("P1-02", "Trace schema", ["plugins/observability/langfuse/__init__.py"]),
    ("P1-03", "Agent API request hooks", ["agent/conversation_loop.py"]),
    ("P1-04", "Tool call hooks", ["model_tools.py"]),
    ("P1-05", "Observability docs", ["plugins/observability/langfuse/README.md"]),
    ("P1-06", "Observability tests", ["tests/plugins/test_langfuse_plugin.py"]),
    ("P2-01", "Quality cases", ["tests/quality/fixtures/hermes_quality_cases.json"]),
    ("P2-02", "Quality eval harness", ["agent/quality_eval.py", "scripts/hermes_quality_eval.py"]),
    ("P2-03", "Owner approval regression", ["tests/quality/test_quality_eval.py"]),
    ("P2-04", "Quality report output", ["agent/quality_eval.py"]),
    ("P2-05", "Trace-compatible case identifiers", ["tests/quality/fixtures/hermes_quality_cases.json"]),
    ("P2-06", "Quality eval tests", ["tests/quality/test_quality_eval.py"]),
    ("P3-01", "Obsidian permission model", ["plugins/obsidian_safe_bridge/README.md"]),
    ("P3-02", "Safe bridge plugin", ["plugins/obsidian_safe_bridge/__init__.py"]),
    ("P3-03", "Plugin manifest", ["plugins/obsidian_safe_bridge/plugin.yaml"]),
    ("P3-04", "Audit log", ["plugins/obsidian_safe_bridge/__init__.py"]),
    ("P3-05", "Obsidian bridge tests", ["tests/plugins/test_obsidian_safe_bridge_plugin.py"]),
    ("P3-06", "Review-queue write rule", ["tests/plugins/test_obsidian_safe_bridge_plugin.py"]),
    ("P4-01", "OSV scanner workflow", [".github/workflows/osv-scanner.yml"]),
    ("P4-02", "Semgrep rules", [".semgrep/hermes-security.yml"]),
    ("P4-03", "Conservative Renovate config", ["renovate.json"]),
    ("P4-04", "Supply-chain audit", [".github/workflows/supply-chain-audit.yml"]),
    ("P4-05", "Security gate script", ["scripts/hermes_security_gate.py"]),
    ("P4-06", "Security gate tests", ["tests/scripts/test_hermes_quality_stack_scripts.py"]),
    ("P5-01", "Pilot candidate selection", ["scripts/hermes_stack_pilot.py"]),
    ("P5-02", "LangGraph pilot boundary", ["scripts/hermes_stack_pilot.py"]),
    ("P5-03", "Pydantic AI pilot boundary", ["scripts/hermes_stack_pilot.py"]),
    ("P5-04", "GraphRAG read-only boundary", ["scripts/hermes_stack_pilot.py"]),
    ("P5-05", "Keep/drop decision record", ["docs/quality-stack/implementation-plan.md"]),
    ("P5-06", "Pilot tests", ["tests/scripts/test_hermes_quality_stack_scripts.py"]),
]


def _issue_status(root: Path, issue_id: str, detail: str, paths: list[str]) -> dict[str, Any]:
    missing = [path for path in paths if not (root / path).exists()]
    done = 100 if not missing else 0
    return {
        "phase": issue_id.split("-")[0],
        "issue": issue_id,
        "detail": detail,
        "done": done,
        "remaining": 100 - done,
        "evidence": ", ".join(paths),
        "status": "pass" if done == 100 else "missing: " + ", ".join(missing),
    }


def build_compliance_report(root: Path) -> dict[str, Any]:
    issues = [_issue_status(root, issue_id, detail, paths) for issue_id, detail, paths in _ISSUES]
    done = 100 if all(issue["done"] == 100 for issue in issues) else 0
    phases: dict[str, dict[str, int]] = {}
    for issue in issues:
        bucket = phases.setdefault(issue["phase"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(issue["done"] == 100)
    phase_rows = []
    for phase, bucket in sorted(phases.items()):
        phase_done = int(round((bucket["passed"] / bucket["total"]) * 100))
        phase_rows.append({"phase": phase, "done": phase_done, "remaining": 100 - phase_done})
    return {
        "overall": {"done": done, "remaining": 100 - done},
        "phases": phase_rows,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report Hermes quality stack comply numbers")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_compliance_report(args.root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for issue in report["issues"]:
            print(
                f"{issue['phase']} {issue['issue']} {issue['done']} "
                f"{issue['remaining']} {issue['status']}"
            )
        print(f"PROJECT_TOTAL {report['overall']['done']} {report['overall']['remaining']}")
    return 0 if report["overall"]["done"] == 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
