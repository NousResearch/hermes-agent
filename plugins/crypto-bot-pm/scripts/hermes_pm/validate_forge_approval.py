#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.forge_approval_token import (
        FORGE_APPROVAL_VALIDATION_SCHEMA_VERSION,
        format_forge_approval_validation_text,
        validate_forge_approval_token,
    )
    from scripts.hermes_pm.forge_write_plan import redact_secret_values
    from scripts.hermes_operator.operator_policy import redact_text
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    repo_root_for_import = Path(__file__).resolve().parents[2]
    if str(repo_root_for_import) not in sys.path:
        sys.path.insert(0, str(repo_root_for_import))
    from scripts.hermes_pm.forge_approval_token import (  # type: ignore[no-redef]
        FORGE_APPROVAL_VALIDATION_SCHEMA_VERSION,
        format_forge_approval_validation_text,
        validate_forge_approval_token,
    )
    from scripts.hermes_pm.forge_write_plan import (  # type: ignore[no-redef]
        redact_secret_values,
    )
    from scripts.hermes_operator.operator_policy import (  # type: ignore[no-redef]
        redact_text,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_validate_forge_approval",
    "authority_class": "propose",
    "schema_version": FORGE_APPROVAL_VALIDATION_SCHEMA_VERSION,
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

SENSITIVE_INPUT_SEGMENTS = {
    ".aws",
    ".gnupg",
    ".ssh",
    "credential",
    "credentials",
    "keychain",
    "keys",
    "private_keys",
    "secret",
    "secrets",
}
SENSITIVE_SUFFIXES = (
    ".asc",
    ".db",
    ".db-shm",
    ".db-wal",
    ".duckdb",
    ".env",
    ".gpg",
    ".key",
    ".log",
    ".p12",
    ".pem",
    ".pfx",
    ".sqlite",
    ".sqlite3",
)


class ForgeApprovalInputError(ValueError):
    """Raised when a local validation artifact cannot be read safely."""


def _ensure_safe_json_input(path: Path, *, label: str) -> Path:
    parts = [part.lower() for part in path.parts]
    for part in parts:
        if part in SENSITIVE_INPUT_SEGMENTS or part.startswith(".env"):
            raise ForgeApprovalInputError(
                f"Refusing {label}: path segment {part!r} is sensitive."
            )
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
        raise ForgeApprovalInputError(
            f"Refusing {label}: path suffix for {path.name!r} is forbidden."
        )
    return path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _ensure_safe_json_input(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ForgeApprovalInputError(f"Unable to read {label}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ForgeApprovalInputError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ForgeApprovalInputError(f"{label} JSON must be an object.")
    return redact_secret_values(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a scoped Hermes PM forge approval token against a "
            "dry-run forge-write plan. This tool never executes operations "
            "and never calls Gitea."
        )
    )
    parser.add_argument("--forge-write-plan", type=Path, required=True)
    parser.add_argument("--approval-token", type=Path, required=True)
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
        plan = _load_json(args.forge_write_plan, label="forge write plan")
        token = _load_json(args.approval_token, label="approval token")
        validation = validate_forge_approval_token(
            forge_write_plan=plan,
            approval_token=token,
        )
    except ForgeApprovalInputError as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")

    if args.format == "text":
        print(format_forge_approval_validation_text(validation))
    else:
        print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
