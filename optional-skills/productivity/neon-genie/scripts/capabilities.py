#!/usr/bin/env python3
"""Machine-readable Neon Genie capability surface for orchestrators.

Usage:
  python scripts/capabilities.py
  python scripts/capabilities.py --json
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
import lineage  # noqa: E402
import paths as ng_paths  # noqa: E402
from recipe_run import RECIPES  # noqa: E402

# Stable gate catalog (matches SKILL.md / anti-overclaim naming)
GATES: list[dict[str, str]] = [
    {"id": "P", "name": "public_gap_research", "severity": "blocking"},
    {"id": "Q", "name": "private_gap_data_request", "severity": "blocking"},
    {"id": "R", "name": "no_silent_observed_from_prior", "severity": "blocking"},
    {"id": "D", "name": "memetic_cannot_raise_promotion", "severity": "blocking"},
    {"id": "G", "name": "completion_proof_or_no_fiction", "severity": "blocking"},
    {"id": "C", "name": "buyer_beneficiary_separation", "severity": "blocking"},
    {"id": "B", "name": "no_forecast_as_fact", "severity": "blocking"},
    {"id": "AUTHORITY", "name": "advisory_boundary", "severity": "blocking"},
]

PACKET_TYPES = [
    "opportunity",
    "product",
    "fragmentation",
    "zero_option",
    "agentic",
    "commercial",
    "evidence",
    "memetic",
    "audit",
    "wayfinder",
    "receipt",
    "envelope",
]

CLI_JOBS = [
    "doctor",
    "privacy",
    "check",
    "run",
    "recipe",
    "route",
    "validate",
    "receipt",
    "envelope",
    "capabilities",
    "eval",
    "transcripts",
    "behavioral",
    "runtime",
    "dist",
    "learn",
    "reconcile",
    "release-check",
]


def build_capabilities() -> dict[str, Any]:
    try:
        manifest = json.loads(ng_paths.manifest_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        manifest = {}

    version = lineage.skill_version(SKILL_ROOT)
    profiles = list(manifest.get("profiles") or [])
    return {
        "skill": "neon-genie",
        "version": version,
        "authority": manifest.get("authority") or "advisory_only",
        "grants_execution": False,
        "claim_labels": list(manifest.get("claim_labels") or []),
        "modes": list(manifest.get("modes") or ["OPEN", "ALIGN", "ASCEND", "CLEAR", "SEAL"]),
        "profiles": profiles,
        "default_profiles": list(manifest.get("default_profiles") or ["core"]),
        "packet_types": PACKET_TYPES,
        "recipes": sorted(RECIPES.keys()),
        "gates": GATES,
        "cli_jobs": CLI_JOBS,
        "runtime_features": {
            "privacy_contract": "1.0.0",
            "privacy_default": "local_only",
            "privacy_purpose_bound_consent_only": True,
            "privacy_global_disable": False,
            "privacy_egress_decisions": [
                "ALLOW",
                "REDACT_THEN_ALLOW",
                "REQUEST_CONSENT",
                "BLOCK",
            ],
            "repository_telemetry": "disabled",

            "research": "host_dependent",
            "research_default": "proactive",
            "offline_opt_out": True,
            "repo_mutation": False,
            "publication": False,
            "spending": False,
            "run_envelope": True,
            "hub_mirrors": True,
            "behavioral_suite": True,
            "learning_ledger": True,
            "learning_auto_apply": False,
            "release_automation": True,
        },
        "gates_registry": "references/gates.yaml",
        "entry_points": {
            "skill_contract": "SKILL.md",
            "run_output": "run-envelope.json",
            "wayfinder_ingest": "run-envelope.json",
            "distribution": "distribution.yaml",
            "learning_ledger": "out/neon-genie/learning-ledger.jsonl",
            "adrs": "docs/adr/",
            "issue_templates": ".github/ISSUE_TEMPLATE/",
            "contributing": "CONTRIBUTING.md",
            "governance": "docs/GOVERNANCE.md",
        },
        "install": {
            "hub": "hermes skills install scrimshawlife-ctrl/NeonGenie/skills/neon-genie",
            "local": "./install.sh",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit Neon Genie capabilities")
    parser.add_argument("--json", action="store_true", help="JSON only (default)")
    parser.add_argument("--pretty", action="store_true", help="Human summary + JSON")
    args = parser.parse_args(argv)
    cap = build_capabilities()
    if args.pretty:
        print(f"neon-genie v{cap['version']} · authority={cap['authority']}")
        print(f"profiles: {len(cap['profiles'])} · recipes: {len(cap['recipes'])}")
        print(f"packets: {', '.join(cap['packet_types'])}")
        print(f"run entry: {cap['entry_points']['run_output']}")
        print("---")
    print(json.dumps(cap, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
