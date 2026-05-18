#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    from scripts.hermes_pm.gitea_forge_capabilities import (
        FORGE_CAPABILITIES_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        capture_gitea_forge_capabilities,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from gitea_forge_capabilities import (  # type: ignore[no-redef]
        FORGE_CAPABILITIES_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        capture_gitea_forge_capabilities,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_gitea_forge_capabilities",
    "authority_class": "read",
    "schema_version": FORGE_CAPABILITIES_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def format_capabilities_text(capabilities: dict[str, Any]) -> str:
    methods = ", ".join(capabilities.get("http_methods_used") or ["<none>"])
    lines = [
        "Hermes PM Gitea forge capabilities",
        f"Repo: {capabilities.get('owner')}/{capabilities.get('repo')}",
        f"Base: {capabilities.get('gitea_base_url')}",
        f"Auth used: {'yes' if capabilities.get('auth_used') else 'no'}",
        f"Methods: {methods}",
        f"Issue preview: {capabilities.get('issues_write_preview_supported')}",
        f"Label preview: {capabilities.get('labels_write_preview_supported')}",
        f"Project preview: {capabilities.get('projects_write_preview_supported')}",
        f"Comment preview: {capabilities.get('comments_write_preview_supported')}",
        f"Status read: {capabilities.get('statuses_read_supported')}",
        f"Blockers: {len(capabilities.get('blockers') or [])}",
        "Permission proof: no",
        "Gitea writes performed: no",
    ]
    return redact_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Gitea forge endpoint shape in read-only mode using GET/HEAD "
            "only. This is readiness evidence, not permission proof, and it "
            "never calls write APIs."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_GITEA_BASE_URL)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Ignore read-token environment variables and use unauthenticated GETs.",
    )
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
    capabilities = capture_gitea_forge_capabilities(
        base_url=args.base_url,
        owner=args.owner,
        repo=args.repo,
        timeout=args.timeout,
        no_token=args.no_token,
        limit=args.limit,
    )
    if args.format == "text":
        print(format_capabilities_text(capabilities))
    else:
        print(json.dumps(capabilities, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
