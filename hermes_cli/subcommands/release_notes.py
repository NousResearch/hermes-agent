"""``hermes release-notes`` subcommand — interactive GitHub Release viewer.

Implements upstream issue #64133: an interactive picker over the project's
GitHub Releases, printing the selected release's notes. ``--latest`` skips the
picker for non-interactive / scripted use.

Pure stdlib + httpx. Degrades gracefully on network errors or GitHub API rate
limit (hints the user to set ``GITHUB_TOKEN`` or ``gh auth login``).
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import httpx

from hermes_cli.curses_ui import curses_single_select

REPO = "NousResearch/hermes-agent"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases"


def _fetch_releases() -> list[dict]:
    """Fetch the GitHub Releases list. Raises on network/HTTP error."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = httpx.get(RELEASES_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict) and r.get("tag_name")]


def _hint_unavailable() -> int:
    """Print a friendly hint and return a non-zero exit code."""
    print(
        "Could not fetch release notes. Check your network, or set GITHUB_TOKEN "
        "(raises the rate limit) / run `gh auth login`.",
        file=sys.stderr,
    )
    return 1


def cmd_release_notes(args) -> int:
    """Show a GitHub Release's notes — interactive picker, or ``--latest``."""
    try:
        releases = _fetch_releases()
    except Exception:
        return _hint_unavailable()

    if not releases:
        print("No releases found.", file=sys.stderr)
        return 1

    if getattr(args, "latest", False):
        idx = 0
    else:
        items = [
            f"{r.get('tag_name', '?')} ({str(r.get('published_at', '?'))[:10]})"
            for r in releases
        ]
        try:
            idx = curses_single_select("Select version", items, default_index=0)
        except Exception:
            return _hint_unavailable()

    chosen = releases[idx]
    body = chosen.get("body") or "(no release notes)"
    print(f"# {chosen.get('tag_name', '?')}\n")
    print(body)
    return 0


def build_release_notes_parser(subparsers, *, cmd_release_notes: Callable) -> None:
    """Attach the ``release-notes`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "release-notes",
        help="Show release notes for a GitHub Release (interactive picker)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show the latest release without the interactive picker",
    )
    parser.set_defaults(func=cmd_release_notes)
