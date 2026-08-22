"""
``hermes patches`` subcommand parser.

Manages a local patch ledger (``~/.hermes/patches.json``) that tracks local
source patches until upstream fixes land.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import argparse
from typing import Callable


def build_patches_parser(
    subparsers, *, cmd_patches: Callable
) -> None:
    """Attach the ``patches`` subcommand to ``subparsers``."""
    patches_parser = subparsers.add_parser(
        "patches",
        help="Manage local source patch ledger (track patches until upstream fixes land)",
        description=(
            "Manage a local patch ledger that tracks source patches against "
            "upstream issues and pull requests. Each entry records a local "
            "branch, touched files, the corresponding upstream issue/PR, an "
            "optional verification command, and notes.\n\n"
            "Use 'hermes patches add' to record a new patch, 'hermes patches list' "
            "to see all tracked patches, 'hermes patches remove <id>' to retire one, "
            "and 'hermes patches check' to poll the GitHub API for PR status."
        ),
    )
    patches_sub = patches_parser.add_subparsers(dest="patches_action")

    # ── add ─────────────────────────────────────────────────────────────────
    patches_sub.add_parser(
        "add",
        help="Add a new patch entry (interactive prompts)",
    )

    # ── list ────────────────────────────────────────────────────────────────
    patches_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List all tracked patches in the ledger",
    )

    # ── remove ─────────────────────────────────────────────────────────────
    remove_p = patches_sub.add_parser(
        "remove",
        aliases=["rm"],
        help="Remove a patch from the ledger by id",
    )
    remove_p.add_argument(
        "id",
        help="Patch id to remove (e.g. P001)",
    )

    # ── check ────────────────────────────────────────────────────────────────
    patches_sub.add_parser(
        "check",
        help="Poll GitHub API for upstream PR status on all tracked patches",
    )

    patches_parser.set_defaults(func=cmd_patches)
