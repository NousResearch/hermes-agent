#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

try:
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
    from scripts.hermes_pm.issue_lifecycle_status import (
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        ISSUE_LIFECYCLE_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        capture_issue_lifecycle_status,
        format_issue_lifecycle_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        ISSUE_LIFECYCLE_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        capture_issue_lifecycle_status,
        format_issue_lifecycle_text,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_issue_lifecycle",
    "authority_class": "read",
    "schema_version": ISSUE_LIFECYCLE_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Hermes PM issue lifecycle attestation for the seed "
            "Gitea issue. The tool uses GET requests only and has no write mode."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_GITEA_BASE_URL)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--issue-index", type=int, default=DEFAULT_ISSUE_INDEX)
    parser.add_argument("--expected-title", default=EXPECTED_PM_SEED_ISSUE_TITLE)
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
        status = capture_issue_lifecycle_status(
            base_url=args.base_url,
            owner=args.owner,
            repo=args.repo,
            issue_index=args.issue_index,
            expected_title=args.expected_title,
            timeout=args.timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        parser.exit(2, f"error: {redact_text(str(exc))}\n")
    if args.format == "text":
        print(format_issue_lifecycle_text(status))
    else:
        print(json.dumps(status, sort_keys=True))
    return 0 if status.get("lifecycle_state") != "mismatch" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
