#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.hermes_pm.forge_issue_attestation import (
        FORGE_ISSUE_ATTESTATION_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        attest_forge_issue_creation,
        format_forge_issue_attestation_text,
    )
    from scripts.hermes_pm.validate_forge_approval import (
        ForgeApprovalInputError,
        _load_json,
    )
    from scripts.hermes_operator.operator_policy import redact_text
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    repo_root_for_import = Path(__file__).resolve().parents[2]
    if str(repo_root_for_import) not in sys.path:
        sys.path.insert(0, str(repo_root_for_import))
    from scripts.hermes_pm.forge_issue_attestation import (  # type: ignore[no-redef]
        FORGE_ISSUE_ATTESTATION_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        attest_forge_issue_creation,
        format_forge_issue_attestation_text,
    )
    from scripts.hermes_pm.validate_forge_approval import (  # type: ignore[no-redef]
        ForgeApprovalInputError,
        _load_json,
    )
    from scripts.hermes_operator.operator_policy import (  # type: ignore[no-redef]
        redact_text,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_attest_forge_issue_creation",
    "authority_class": "read",
    "schema_version": FORGE_ISSUE_ATTESTATION_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only attestation for the Hermes PM Checkpoint 6 one-issue "
            "forge-write rehearsal. This tool performs GET verification only."
        )
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-title", required=True)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--describe-authority",
        action="store_true",
        help="Print this tool's Hermes PM authority metadata as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe_authority:
        print(json.dumps(OPERATOR_AUTHORITY_METADATA, sort_keys=True))
        return 0
    try:
        evidence = _load_json(args.evidence, label="forge issue evidence")
        attestation = attest_forge_issue_creation(
            evidence=evidence,
            expected_title=args.expected_title,
            timeout=args.timeout,
        )
    except ForgeApprovalInputError as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")
    if args.format == "text":
        print(format_forge_issue_attestation_text(attestation))
    else:
        print(json.dumps(attestation, sort_keys=True))
    return 0 if attestation.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
