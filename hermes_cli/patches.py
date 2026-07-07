"""
Local patch ledger — tracks local source patches until upstream fixes land.

Each entry records a local branch, its touched files, the corresponding
upstream issue/PR, an optional verification command, and notes.  The ledger
is a small JSON file (``~/.hermes/patches.json``).

Commands:
    hermes patches add      Add a new patch entry (interactive)
    hermes patches list     List all tracked patches
    hermes patches remove   Remove a patch by id
    hermes patches check    Poll GitHub API for upstream PR status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.config import load_config
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── Ledger file ────────────────────────────────────────────────────────────────

PATCHES_FILENAME = "patches.json"


def _patches_path() -> Path:
    """Return the path to the patches.json ledger file."""
    return get_hermes_home() / PATCHES_FILENAME


def _load_patches() -> List[Dict[str, Any]]:
    """Load patches from the JSON ledger. Returns an empty list on any error."""
    path = _patches_path()
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("patches.json is not a list — resetting")
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load patches.json: %s", exc)
        return []


def _save_patches(patches: List[Dict[str, Any]]) -> None:
    """Write patches to the JSON ledger, creating parent directories if needed."""
    path = _patches_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically via temp-file + rename to avoid partial-write corruption
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(patches, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _next_id(patches: List[Dict[str, Any]]) -> str:
    """Return the next sequential patch id (e.g. ``P001``)."""
    existing = set()
    for p in patches:
        pid = p.get("id", "")
        if pid.startswith("P") and pid[1:].isdigit():
            existing.add(int(pid[1:]))
    n = 1
    while n in existing:
        n += 1
    return f"P{n:03d}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _info(text: str) -> None:
    print(f"  {text}")


def _success(text: str) -> None:
    print(f"  ✓ {text}")


def _warning(text: str) -> None:
    print(f"  ⚠ {text}")


def _error(text: str) -> None:
    print(f"  ✗ {text}")


# ── Commands ────────────────────────────────────────────────────────────────────

def cmd_patches_add(args: object) -> None:
    """Interactively add a new patch entry."""
    patches = _load_patches()

    print()
    print("  Add a new patch ledger entry")
    print("  ─────────────────────────────")
    print()

    local_branch = _prompt("Local branch name").strip()
    if not local_branch:
        _error("Branch name is required")
        return

    touched_files = _prompt("Touched files (comma-separated)", default="").strip()
    touched_list = [f.strip() for f in touched_files.split(",") if f.strip()] if touched_files else []

    upstream_issue = _prompt("Upstream issue URL").strip()
    if not upstream_issue:
        _error("Upstream issue URL is required")
        return

    upstream_pr = _prompt("Upstream PR URL (optional)", default="").strip()

    verification_cmd = _prompt("Verification command (optional)", default="").strip()

    notes = _prompt("Notes (optional)", default="").strip()

    entry: Dict[str, Any] = {
        "id": _next_id(patches),
        "local_branch": local_branch,
        "touched_files": touched_list,
        "upstream_issue_url": upstream_issue,
        "upstream_pr_url": upstream_pr or None,
        "verification_command": verification_cmd or None,
        "notes": notes or None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    patches.append(entry)
    _save_patches(patches)
    _success(f"Added patch {entry['id']} ({local_branch})")


def cmd_patches_list(args: object) -> None:
    """Show all tracked patches."""
    patches = _load_patches()

    if not patches:
        print()
        _info("No patches tracked.")
        print()
        _info("  Add one with:  hermes patches add")
        print()
        return

    print()
    print(f"  {'ID':<6} {'Branch':<30} {'Upstream Issue':<50} {'Upstream PR':<40} Status")
    print(f"  {'─'*5:<6} {'─'*29:<30} {'─'*49:<50} {'─'*39:<40} {'─'*20}")
    for p in patches:
        pid = p.get("id", "???")
        branch = p.get("local_branch", "?")[:28]
        issue_url = (p.get("upstream_issue_url", "") or "")[:48]
        pr_url = (p.get("upstream_pr_url", "") or "")[:38]
        # Basic status hint — look at the PR URL to guess state
        status = _guess_status(p)
        print(f"  {pid:<6} {branch:<30} {issue_url:<50} {pr_url:<40} {status}")
    print()


def cmd_patches_remove(args: argparse.Namespace) -> None:
    """Remove a patch by id."""
    patches = _load_patches()
    target = args.id.upper()

    for i, p in enumerate(patches):
        if p.get("id", "").upper() == target:
            removed = patches.pop(i)
            _save_patches(patches)
            _success(f"Removed patch {removed['id']} ({removed.get('local_branch', '?')})")
            return

    _error(f"Patch '{args.id}' not found")


def cmd_patches_check(args: argparse.Namespace) -> None:
    """Poll GitHub API for upstream PR status on all tracked patches."""
    patches = _load_patches()

    if not patches:
        print()
        _info("No patches to check.")
        print()
        return

    print()
    print("  Checking upstream PR status...")
    print()

    any_open = False
    for p in patches:
        pr_url = p.get("upstream_pr_url")
        if not pr_url:
            _info(f"  {p.get('id', '???')}: no PR URL — skipping")
            continue

        # Parse owner/repo/pull_number from a GitHub PR URL
        pr_info = _parse_github_pr_url(pr_url)
        if not pr_info:
            _warning(f"  {p.get('id', '???')}: could not parse PR URL: {pr_url}")
            continue

        owner, repo, pr_num = pr_info

        try:
            state, merged = _github_pr_state(owner, repo, pr_num)
        except Exception as exc:
            _warning(f"  {p.get('id', '???')}: API error — {exc}")
            continue

        if state == "closed" and merged:
            _success(f"{p.get('id', '???')}: PR #{pr_num} MERGED — patch can be retired!")
        elif state == "closed":
            _info(f"  {p.get('id', '???')}: PR #{pr_num} closed (not merged)")
            any_open = True
        else:
            _info(f"  {p.get('id', '???')}: PR #{pr_num} still open")
            any_open = True

        # If a verification command is set and the PR is merged, offer to run it
        if state == "closed" and merged:
            vcmd = p.get("verification_command")
            if vcmd:
                _info(f"  Verification command: {vcmd}")
                # We don't auto-run — that's too invasive.  Just surface it.

    print()
    if not any_open:
        _success("All patches with PR URLs have been merged upstream!")
    print()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _prompt(question: str, default: str = "") -> str:
    """Prompt the user for input via stdin."""
    try:
        if default:
            val = input(f"  {question} [{default}]: ").strip()
        else:
            val = input(f"  {question}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return default or ""
    if not val and default:
        return default
    return val


def _guess_status(patch: Dict[str, Any]) -> str:
    """Return a short status string based on PR URL presence and notes."""
    pr_url = patch.get("upstream_pr_url")
    if pr_url:
        # We can't check status without hitting the API unless the user
        # manually added notes. Just show a placeholder.
        notes = patch.get("notes") or ""
        if "merged" in notes.lower() or "retired" in notes.lower():
            return "retired"
        return "pending"
    return "no PR"


def _parse_github_pr_url(url: str) -> tuple[str, str, int] | None:
    """Extract (owner, repo, pr_number) from a GitHub PR URL.

    Supports:
        https://github.com/owner/repo/pull/123
        https://github.com/owner/repo/pull/123/files
        http://github.com/owner/repo/pull/123
        github.com/owner/repo/pull/123
    """
    import re

    m = re.match(
        r"(?:https?://)?github\.com/([^/]+)/([^/]+)/pull/(\d+)(?:/.*)?$",
        url.strip(),
    )
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _github_pr_state(
    owner: str, repo: str, pr_number: int
) -> tuple[str, bool]:
    """Query the GitHub API for a PR's state and merge status.

    Returns ``(state, merged)`` where ``state`` is ``"open"`` or ``"closed"``
    and ``merged`` is ``True`` if the PR was merged.

    Uses the ``gh`` CLI if available (authenticated), falling back to
    unauthenticated REST API (rate-limited to 60/hr).
    """
    # Try gh CLI first — it uses the user's auth
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--repo", f"{owner}/{repo}",
                "--json", "state,merged",
                "--jq", r'"\(.state) \(.merged)"',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            state = parts[0].lower()
            merged = parts[1].lower() == "true" if len(parts) > 1 else False
            return state, merged
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fallback: unauthenticated REST API
    import urllib.request

    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    # User-agent required by GitHub
    req.add_header("User-Agent", "hermes-agent-patch-ledger")

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    state = data.get("state", "unknown")
    merged = data.get("merged", False)
    return state, merged


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def patches_command(args: argparse.Namespace) -> None:
    """Main dispatcher for ``hermes patches`` subcommands."""
    action = getattr(args, "patches_action", None)

    handlers = {
        "add": cmd_patches_add,
        "list": cmd_patches_list,
        "ls": cmd_patches_list,
        "remove": cmd_patches_remove,
        "rm": cmd_patches_remove,
        "check": cmd_patches_check,
    }

    handler = handlers.get(action)
    if handler:
        handler(args)
    else:
        # No subcommand — show help
        print()
        print("  hermes patches — Local patch ledger")
        print()
        _info("  Usage:")
        _info("    hermes patches add        Add a new patch entry (interactive)")
        _info("    hermes patches list       List all tracked patches")
        _info("    hermes patches remove ID  Remove a patch by id")
        _info("    hermes patches check      Check upstream PR status via GitHub API")
        print()
