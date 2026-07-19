"""``hermes release-notes`` subcommand — interactive GitHub Release viewer.

Implements upstream issue #64133: an interactive picker over the project's
GitHub Releases, printing the selected release's notes. ``--latest`` skips the
picker for non-interactive / scripted use.

Output is a compact summary: markdown list items and headings are kept, prose
paragraphs dropped — for a scannable, changelog-like view. Falls back to the
raw body when too few structured lines are present.

Pure stdlib + httpx. Degrades gracefully on network errors or GitHub API rate
limit (hints the user to set ``GITHUB_TOKEN`` or ``gh auth login``).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Callable

import httpx

from hermes_cli.curses_ui import curses_single_select

REPO = "NousResearch/hermes-agent"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases"

# Compact summary: keep markdown list items and headings, drop prose.
_BULLET_OR_HEADING = re.compile(r"^\s*(?:[-*]\s+\S|#{1,6}\s+\S)")
# Fewer structured lines than this → fall back to the raw body so unusual
# formats aren't reduced to nothing.
_SUMMARY_MIN_LINES = 2


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


def _summarize_body(body: str) -> str:
    """Render release notes as a compact bullet/heading summary.

    Keeps markdown list items (``-``/``*``) and headings (``#``), drops prose
    paragraphs — for a concise, scannable view. Falls back to the raw body
    when too few structured lines are present.
    """
    if not body or not body.strip():
        return "(no release notes)"
    kept = [ln.rstrip() for ln in body.splitlines() if _BULLET_OR_HEADING.match(ln)]
    if len(kept) < _SUMMARY_MIN_LINES:
        return body.strip()
    return "\n".join(kept)


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
        if idx is None:  # ESC / Cancel
            print("Cancelled.", file=sys.stderr)
            return 1

    chosen = releases[idx]
    print(_summarize_body(chosen.get("body") or ""))
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
