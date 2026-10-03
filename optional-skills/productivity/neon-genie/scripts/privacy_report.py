#!/usr/bin/env python3
"""Privacy contract / telemetry / preflight status report (stdlib only).

Usage:
  python scripts/privacy_report.py
  python scripts/privacy_report.py --json
  python scripts/neon_genie.py do privacy [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import privacy_preflight  # noqa: E402

PRIVACY_CONTRACT_VERSION = "1.0.0"
ENVELOPE_SCHEMA_VERSION_EXPECTED = "1.1.0"
TELEMETRY_STATUS = "disabled"
CANNED_SECRET = "Bearer sk-testkeytestkeytestkey12"
CONTRACT_CANDIDATES = ("PRIVACY.md", "references/PRIVACY.md")


def existing_contract_paths(root: Path = SKILL_ROOT) -> list[str]:
    found: list[str] = []
    for rel in CONTRACT_CANDIDATES:
        if (root / rel).is_file():
            found.append(rel)
    return found


def run_preflight_self_test() -> str:
    """Pass only if canned secret is blocked (not safe_for_egress)."""
    result = privacy_preflight.preflight(CANNED_SECRET)
    if result.get("safe_for_egress") is True:
        return "fail"
    return "pass"


def build_report(root: Path = SKILL_ROOT) -> dict[str, Any]:
    return {
        "privacy_contract_version": PRIVACY_CONTRACT_VERSION,
        "contract_paths": existing_contract_paths(root),
        "telemetry_status": TELEMETRY_STATUS,
        "research_default": {
            "enabled": False,
            "note": "operator/host controlled at runtime",
        },
        "offline_enforcement": "requested_only",
        "redaction_available": True,
        "preflight_self_test": run_preflight_self_test(),
        "envelope_schema_version_expected": ENVELOPE_SCHEMA_VERSION_EXPECTED,
        "wayfinder_required": False,
        "unknowns": ["host_provider_retention", "model_training_policy"],
    }


def report_ok(report: dict[str, Any]) -> bool:
    """Exit 0 when contract present, preflight self-test pass, telemetry disabled."""
    if not report.get("contract_paths"):
        return False
    if report.get("preflight_self_test") != "pass":
        return False
    if report.get("telemetry_status") != "disabled":
        return False
    return True


def format_human(report: dict[str, Any]) -> str:
    lines = [
        "Neon Genie privacy report",
        f"  privacy_contract_version: {report['privacy_contract_version']}",
        f"  contract_paths: {', '.join(report['contract_paths']) or '(none)'}",
        f"  telemetry_status: {report['telemetry_status']}",
        f"  research_default.enabled: {report['research_default']['enabled']}",
        f"  offline_enforcement: {report['offline_enforcement']}",
        f"  redaction_available: {report['redaction_available']}",
        f"  preflight_self_test: {report['preflight_self_test']}",
        f"  envelope_schema_version_expected: {report['envelope_schema_version_expected']}",
        f"  wayfinder_required: {report['wayfinder_required']}",
        f"  unknowns: {', '.join(report['unknowns'])}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Privacy contract, telemetry, and preflight status",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    args = parser.parse_args(argv)

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_human(report))
        if report_ok(report):
            print("OK: privacy contract and preflight self-test")
        else:
            print("FAIL: privacy report checks failed", file=sys.stderr)

    return 0 if report_ok(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
