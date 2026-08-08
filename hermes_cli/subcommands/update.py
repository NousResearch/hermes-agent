"""``hermes update`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_update_parser(subparsers, *, cmd_update: Callable) -> None:
    """Attach the ``update`` subcommand to ``subparsers``."""
    # =========================================================================
    # update command
    # =========================================================================
    update_parser = subparsers.add_parser(
        "update",
        help="Update Hermes Agent to the latest version",
        description="Pull the latest changes from git and reinstall dependencies",
    )
    update_parser.add_argument(
        "--gateway",
        action="store_true",
        default=False,
        help="Gateway mode: use file-based IPC for prompts instead of stdin (used internally by /update)",
    )
    update_parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Check whether an update is available without installing anything",
    )
    update_parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Skip ALL pre-update backups for this run (both the quick state snapshot and the full zip; overrides updates.pre_update_backup)",
    )
    update_parser.add_argument(
        "--backup",
        action="store_true",
        default=False,
        help="Force a FULL pre-update backup (quick state snapshot + HERMES_HOME zip) for this run, regardless of updates.pre_update_backup",
    )
    update_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Assume yes for interactive prompts (config migration, stash restore). API-key entry is skipped; run 'hermes config migrate' separately for those.",
    )
    update_parser.add_argument(
        "--branch",
        default=None,
        metavar="NAME",
        help=(
            "Update against this branch instead of the default (main). "
            "If the local checkout is on a different branch, hermes will "
            "switch to the requested branch first (auto-stashing any "
            "uncommitted changes)."
        ),
    )
    update_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Windows: proceed with the update even when another hermes.exe is detected. The concurrent process will likely cause WinError 32 warnings and may leave a reboot-deferred .exe replacement. Does NOT bypass the venv-process guard (see --force-venv).",
    )
    update_parser.add_argument(
        "--force-venv",
        action="store_true",
        default=False,
        help="Windows: mutate the venv even while other processes are running from its interpreter (desktop backend, gateway, terminals). Those processes keep native .pyd files locked, so the dependency sync will likely fail partway and strand the install half-updated. Use only if you know the detected holders are false positives.",
    )
    update_parser.add_argument(
        "--eject",
        action="store_true",
        default=False,
        help=(
            "Take control of updates for a desktop-bundled install. This "
            "option marks the checkout as source-managed (git updates with "
            "`hermes update`) and fetches the full git history. The desktop "
            "app continues to update itself, but it no longer touches the "
            "agent checkout. This option has no effect on installs that are "
            "already source-managed."
        ),
    )
    update_parser.add_argument(
        "--channel",
        default=None,
        choices=("stable", "main"),
        metavar="CHANNEL",
        help=(
            "With --eject: the releases that the ejected install tracks. Use "
            "'stable' for tagged releases or 'main' for the git main branch. "
            "The desktop cadence before the eject is 'stable'. The default "
            "is 'main'."
        ),
    )
    update_parser.set_defaults(func=cmd_update)
