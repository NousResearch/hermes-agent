#!/usr/bin/env python3
"""Fail-closed checker for AION PR Intake Adapter v0.1 static packets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_FIXTURE_DIR = Path("tests/fixtures/aion-pr-intake-adapter-v0.1")
FORBIDDEN_SURFACE_FLAGS = (
    "runtime_enabled",
    "gateway_mutated",
    "cron_enabled",
    "required_mode_enabled",
    "live_scanner_enabled",
    "source_writeback_enabled",
    "production_access",
    "payment_access",
    "database_access",
    "webhook_access",
    "secret_access",
    "customer_data_access",
    "external_executor_real_run",
)


def _get(packet: dict[str, Any], path: str) -> Any:
    cur: Any = packet
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_packet(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if packet.get("version") != "aion_pr_intake_adapter_v0_1":
        errors.append("version must be aion_pr_intake_adapter_v0_1")

    if _get(packet, "github_pr_metadata.intake_mirror_only") is not True:
        errors.append("github_pr_metadata.intake_mirror_only must be true")
    if _get(packet, "github_pr_metadata.label_claims_execution") is not False:
        errors.append("GitHub label/PR metadata must not claim execution")
    if _get(packet, "github_pr_metadata.actions_business_flow_dispatch") is not False:
        errors.append("GitHub Actions business-flow dispatch must be false")

    if _get(packet, "execution_trigger.type") != "hermes_kanban_task_assignee_and_dispatcher_pickup":
        errors.append("execution trigger must be Hermes Kanban assignee + dispatcher pickup")
    if not _get(packet, "execution_trigger.kanban_task.assignee"):
        errors.append("execution trigger requires non-empty task.assignee")
    if _get(packet, "execution_trigger.dispatcher_pickup.picked_up") is not True:
        errors.append("execution trigger requires dispatcher pickup evidence")
    if not _get(packet, "execution_trigger.dispatcher_pickup.run_id"):
        errors.append("dispatcher pickup requires run_id")
    if not _get(packet, "execution_trigger.dispatcher_pickup.claimed_by"):
        errors.append("dispatcher pickup requires claimed_by")

    if _get(packet, "audit.verdict_required_before_merge") is not True:
        errors.append("audit verdict must be required before merge")
    if _get(packet, "audit.current_head_verdict") not in {"PASS", "CONDITIONAL_PASS", "BLOCK"}:
        errors.append("current-head audit verdict is missing or invalid")

    protected = packet.get("protected_surfaces")
    if not isinstance(protected, dict):
        errors.append("protected_surfaces must be an object with explicit false flags")
    else:
        for flag in FORBIDDEN_SURFACE_FLAGS:
            if protected.get(flag) is not False:
                errors.append(f"protected surface flag must be false: {flag}")

    if _get(packet, "non_claims.full_unattended_ready") is not False:
        errors.append("full_unattended_ready must be false")
    return errors


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def check_fixture_dir(fixture_dir: Path) -> dict[str, Any]:
    valid_files = sorted((fixture_dir / "valid").glob("*.json"))
    invalid_files = sorted((fixture_dir / "invalid").glob("*.json"))
    results: dict[str, Any] = {"valid": {}, "invalid": {}}
    failures: list[str] = []

    for path in valid_files:
        errors = validate_packet(load_json(path))
        results["valid"][str(path)] = errors
        if errors:
            failures.append(f"valid fixture failed {path}: {errors}")

    for path in invalid_files:
        errors = validate_packet(load_json(path))
        results["invalid"][str(path)] = errors
        if not errors:
            failures.append(f"invalid fixture unexpectedly passed {path}")

    results["verdict"] = "PASS" if not failures else "FAIL"
    results["failures"] = failures
    results["fixture_count"] = len(valid_files) + len(invalid_files)
    results["valid_fixture_count"] = len(valid_files)
    results["invalid_fixture_count"] = len(invalid_files)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--packet", type=Path, help="Validate one packet instead of the fixture suite")
    args = parser.parse_args()
    if args.packet:
        errors = validate_packet(load_json(args.packet))
        print(json.dumps({"verdict": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
        return 0 if not errors else 1
    result = check_fixture_dir(args.fixture_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
