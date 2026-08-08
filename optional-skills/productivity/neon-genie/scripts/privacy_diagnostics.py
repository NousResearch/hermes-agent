#!/usr/bin/env python3
"""Report the resolved repository-owned privacy boundary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from privacy_runtime import (  # noqa: E402
    DECISIONS,
    default_privacy_context,
    resolve_privacy_config_from_brief,
)


def main(argv=None):
    p = argparse.ArgumentParser(description="Neon Genie privacy diagnostics")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default="out/neon-genie/example")
    p.add_argument(
        "--mode",
        default="local_only",
        choices=["local_only", "external_research_allowed", "custom"],
    )
    p.add_argument("--brief", type=Path, default=None, help="Resolve privacy from brief YAML")
    a = p.parse_args(argv)
    if a.brief is not None:
        c = resolve_privacy_config_from_brief(a.brief, artifact_path=a.out)
    else:
        c = default_privacy_context(a.out, a.mode)
    report = {
        "privacy_contract": c["contract_version"],
        "mode": c["mode"].upper(),
        "artifact_root": a.out,
        "repository_telemetry": "DISABLED",
        "external_research": "ENABLED" if c["egress"]["allowed"] else "DISABLED",
        "egress_policy": c["egress"],
        "egress_decisions": list(DECISIONS),
        "redaction_preflight": "AVAILABLE",
        "purpose_bound_consents_only": True,
        "global_privacy_disable": "FORBIDDEN",
        "known_host_providers": "NOT_COMPUTABLE",
        "provider_retention": "NOT_COMPUTABLE",
        "provider_processing": "NOT_COMPUTABLE",
        "offline_enforcement": "REPOSITORY_ENFORCED",
        "learning_ledger": c.get("learning_ledger"),
        "consents": c.get("consents") or [],
        "deletion_instructions": c["deletion_instructions"],
    }
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
