#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.forge_execution_gate import (
        FORGE_EXECUTION_GATE_SCHEMA_VERSION,
        build_forge_execution_gate,
        format_forge_execution_gate_text,
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
    from scripts.hermes_pm.forge_execution_gate import (  # type: ignore[no-redef]
        FORGE_EXECUTION_GATE_SCHEMA_VERSION,
        build_forge_execution_gate,
        format_forge_execution_gate_text,
    )
    from scripts.hermes_pm.validate_forge_approval import (  # type: ignore[no-redef]
        ForgeApprovalInputError,
        _load_json,
    )
    from scripts.hermes_operator.operator_policy import (  # type: ignore[no-redef]
        redact_text,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_check_forge_execution_gate",
    "authority_class": "propose",
    "schema_version": FORGE_EXECUTION_GATE_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    "calls_gitea_write_api": False,
    "mutation_executed": False,
    "creates_issues": False,
    "creates_labels": False,
    "mutates_projects": False,
    "comments": False,
    "starts_workflows": False,
    "starts_runners": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the dry-run Hermes PM forge execution gate for selected "
            "operations. This combines a forge-write plan, approval token, "
            "and optional read-only endpoint capabilities without executing "
            "or calling Gitea write APIs."
        )
    )
    parser.add_argument("--forge-write-plan", type=Path, required=True)
    parser.add_argument("--approval-token", type=Path, required=True)
    parser.add_argument("--gitea-capability-map", type=Path)
    parser.add_argument("--gitea-capabilities", type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--describe-authority",
        action="store_true",
        help="Print this tool's Hermes PM authority metadata as JSON.",
    )
    return parser


def _load_optional_json(path: Path | None, *, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_json(path, label=label)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe_authority:
        print(json.dumps(OPERATOR_AUTHORITY_METADATA, sort_keys=True))
        return 0
    try:
        plan = _load_json(args.forge_write_plan, label="forge write plan")
        token = _load_json(args.approval_token, label="approval token")
        capabilities = _load_optional_json(
            args.gitea_capability_map or args.gitea_capabilities,
            label="Gitea capability map",
        )
        gate = build_forge_execution_gate(
            forge_write_plan=plan,
            approval_token=token,
            gitea_capabilities=capabilities,
        )
    except ForgeApprovalInputError as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")

    if args.format == "text":
        print(format_forge_execution_gate_text(gate))
    else:
        print(json.dumps(gate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
