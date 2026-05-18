#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.forge_issue_executor import (
        FORGE_ISSUE_CREATION_RESULT_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        execute_forge_issue_create,
        format_forge_issue_result_text,
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
    from scripts.hermes_pm.forge_issue_executor import (  # type: ignore[no-redef]
        FORGE_ISSUE_CREATION_RESULT_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        execute_forge_issue_create,
        format_forge_issue_result_text,
    )
    from scripts.hermes_pm.validate_forge_approval import (  # type: ignore[no-redef]
        ForgeApprovalInputError,
        _load_json,
    )
    from scripts.hermes_operator.operator_policy import (  # type: ignore[no-redef]
        redact_text,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_execute_forge_issue_create",
    "authority_class": "forge_write",
    "schema_version": FORGE_ISSUE_CREATION_RESULT_SCHEMA_VERSION,
    "read_only": False,
    "mutation_capability": True,
    "allowed_operation_types": ["create_issue"],
    "max_operations": 1,
    "allowed_write_endpoint": "/api/v1/repos/preston/crypto_bot/issues",
    "forbidden_operation_types": [
        "create_label",
        "create_project_column",
        "create_project_card",
        "comment_on_issue",
        "update_issue",
        "request_pr_review",
        "create_status",
        "create_check",
        "create_release",
        "package_publish",
        "webhook_mutation",
        "trigger_workflow",
        "start_runner",
        "deploy",
        "runtime_admin",
        "financial",
        "secret",
    ],
    **NON_ACTION_BOOLEANS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the Hermes PM Checkpoint 6 one-issue forge-write "
            "rehearsal. The only mutation path is one POST to the local "
            "Gitea create-issue endpoint after exact approval and gate checks."
        )
    )
    parser.add_argument("--forge-write-plan", type=Path, required=True)
    parser.add_argument("--approval-token", type=Path, required=True)
    parser.add_argument("--gitea-capabilities", type=Path, required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--execution-payload", type=Path)
    parser.add_argument("--expected-execution-payload-sha256")
    parser.add_argument("--expected-title", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--i-understand-this-creates-one-gitea-issue",
        action="store_true",
        help=(
            "Required PM-6 confirmation for exactly one local Gitea issue "
            "creation."
        ),
    )
    parser.add_argument(
        "--describe-authority",
        action="store_true",
        help="Print this tool's Hermes PM authority metadata as JSON.",
    )
    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    plan = _load_json(args.forge_write_plan, label="forge write plan")
    token = _load_json(args.approval_token, label="approval token")
    capabilities = _load_json(args.gitea_capabilities, label="Gitea capabilities")
    execution_payload = (
        _load_json(args.execution_payload, label="execution payload")
        if args.execution_payload
        else {}
    )
    return plan, token, capabilities, execution_payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe_authority:
        print(json.dumps(OPERATOR_AUTHORITY_METADATA, sort_keys=True))
        return 0
    try:
        plan, token, capabilities, execution_payload = _load_inputs(args)
        result = execute_forge_issue_create(
            forge_write_plan=plan,
            approval_token=token,
            gitea_capabilities=capabilities,
            operation_id=args.operation_id,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_title=args.expected_title,
            confirmation=args.i_understand_this_creates_one_gitea_issue,
            expected_execution_payload_sha256=(
                args.expected_execution_payload_sha256
            ),
            execution_payload=execution_payload or None,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    except ForgeApprovalInputError as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")

    if args.format == "text":
        print(format_forge_issue_result_text(result))
    else:
        print(json.dumps(result, sort_keys=True))
    return 0 if result.get("created") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
