#!/usr/bin/env python3
"""Deterministic workflow-guardrail scenario harness.

This is a low-cost Forge-style regression harness: no provider calls, no tools
executed. It feeds synthetic turns/tool successes into the pure workflow
controller and verifies expected final-gate decisions.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.workflow_guardrails import (  # noqa: E402
    WorkflowGuardrailConfig,
    WorkflowGuardrailController,
)


SCENARIOS = [
    {
        "name": "repo_review_missing_tests_advisory",
        "message": "다음 GitHub 레포지토리를 검토해줘 https://github.com/example/repo",
        "tools": ["read_file", "search_files"],
        "mode": "advisory",
        "expected_action": "advisory",
        "expected_missing": ["inspect_metadata", "verify"],
    },
    {
        "name": "devflow_complete_allows_final",
        "message": "이 버그를 수정하고 테스트까지 해줘",
        "tools": ["read_file", "patch", "terminal"],
        "mode": "advisory",
        "expected_action": "allow",
        "expected_missing": [],
    },
    {
        "name": "devflow_missing_verify_nudges_when_configured",
        "message": "구현 진행해서 완료해",
        "tools": ["search_files", "patch"],
        "mode": "nudge",
        "expected_action": "nudge",
        "expected_missing": ["verify"],
    },
    {
        "name": "disabled_mode_allows_final",
        "message": "구현 진행해서 완료해",
        "tools": [],
        "mode": "off",
        "expected_action": "allow",
        "expected_missing": [],
    },
]


def run_scenario(scenario: dict) -> dict:
    controller = WorkflowGuardrailController(
        WorkflowGuardrailConfig(enabled=True, final_gate_mode=scenario["mode"])
    )
    controller.reset_for_turn(scenario["message"])
    for tool in scenario["tools"]:
        controller.record_tool_result(tool, failed=False)
    decision = controller.evaluate_final_response("done")
    missing = [step.key for step in decision.missing_steps]
    passed = decision.action == scenario["expected_action"] and missing == scenario["expected_missing"]
    return {
        "name": scenario["name"],
        "passed": passed,
        "expected_action": scenario["expected_action"],
        "actual_action": decision.action,
        "expected_missing": scenario["expected_missing"],
        "actual_missing": missing,
        "decision": decision.to_metadata(),
    }


def main() -> int:
    results = [run_scenario(s) for s in SCENARIOS]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
