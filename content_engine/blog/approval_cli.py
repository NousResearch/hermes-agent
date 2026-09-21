#!/usr/bin/env python3
"""CLI entry point for handling Discord approval commands.

Usage:
  python -m blog.approval_cli handle "!approve my-slug"
  python -m blog.approval_cli handle "!reject my-slug Needs more data"
  python -m blog.approval_cli handle "!amend my-slug Add section"
  python -m blog.approval_cli list-pending
  python -m blog.approval_cli process-approved

Modes:
  handle <text>   — Parse a Discord command and execute it (approve/reject/amend)
  list-pending    — Print all items awaiting approval
  process-approved — Find items marked approved and publish them
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Ensure content_engine is on path.
_engine_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_engine_root))

from blog.blog_approval import (
    pending, approve, reject, amend, publish,
    parse_discord_command, handle_discord_command,
    batch_validation_issues,
)


def _cmd_list_pending() -> int:
    issues = batch_validation_issues()
    if issues:
        print("[Blog Approval Blocked]")
        print("Pending approval batch failed uniqueness validation:")
        for issue in issues:
            print(f"- {issue}")
        print("Amend/reject duplicates or mark an explicit series before posting approval requests.")
        return 1

    items = pending()
    if not items:
        print("No pending approvals.")
        return 0
    print(f"{len(items)} pending approval(s):\n")
    for item in items:
        from blog.blog_approval import summary
        print(summary(item))
        print()
    return 0


def _cmd_process_approved() -> int:
    """Find approved entries and publish them."""
    from blog.blog_approval import _read_tracker
    entries = _read_tracker()
    approved = [e for e in entries if e.get("status") == "approved"]
    if not approved:
        print("No approved items to process.")
        return 0
    results = []
    for entry in approved:
        slug = entry.get("slug", "")
        print(f"Processing approved: {slug}")
        result = publish(slug)
        results.append({"slug": slug, "result": result})
        print(f"  -> {result.get('status', '?')}: {result.get('message', '')}")
    return 0 if all(r["result"].get("status") == "ok" for r in results) else 1


def _cmd_handle(text: str) -> int:
    result = handle_discord_command(text)
    print(json.dumps(result, indent=2))
    return 0 if result.get("handled") else 1


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    if cmd == "list-pending":
        return _cmd_list_pending()
    elif cmd == "process-approved":
        return _cmd_process_approved()
    elif cmd == "handle" and len(sys.argv) >= 3:
        return _cmd_handle(" ".join(sys.argv[2:]))
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
