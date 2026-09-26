"""``hermes update`` subcommand parser."""

from __future__ import annotations

import argparse
from typing import Callable


def build_update_parser(subparsers, *, cmd_update: Callable) -> None:
    """Attach the ``update`` subcommand to ``subparsers``."""
    from hermes_cli.release_channels import validate_name
    update_parser = subparsers.add_parser(
        "update", help="Update Hermes Agent to the latest version",
        description="Pull the latest changes from git and reinstall dependencies")
    update_parser.add_argument(
        "--gateway", action="store_true", default=False,
        help="Gateway mode: use file-based IPC for prompts instead of stdin (used internally by /update)",
    )
    update_parser.add_argument(
        "--check", action="store_true", default=False,
        help="Check whether an update is available without installing anything")
    update_parser.add_argument(
        "--plan", action="store_true", default=False,
        help="Show the update plan and exit without changing anything: install "
            "kind (git/docker/nix), every running Hermes service across all "
            "profiles with its supervisor and running code version, and how "
            "each will be restarted. Read-only; safe on a live fleet.")
    update_parser.add_argument(
        "--list-venv-holders", action="store_true", default=False,
        help="Print the processes the Windows venv-holder guard would refuse on as a JSON list "
            "[{pid, exe, argv, kind}] and exit: 0 when the venv is free, 3 when holders are present. "
            "Read-only; kind is gateway / backend (Desktop serve) / hermes:<subcommand> / python, so a "
            "scheduled update can stop exactly those PIDs instead of looping. Always [] off Windows.")
    update_parser.add_argument(
        "--no-backup", action="store_true", default=False,
        help="Skip ALL pre-update backups for this run (both the quick state snapshot and the full zip; overrides updates.pre_update_backup)",
    )
    update_parser.add_argument(
        "--backup", action="store_true", default=False,
        help="Force a FULL pre-update backup (quick state snapshot + HERMES_HOME zip) for this run, regardless of updates.pre_update_backup",
    )
    update_parser.add_argument(
        "--yes", "-y", action="store_true", default=False,
        help="Run without blocking on prompts: accepts the config-migration and stash-restore prompts, skips the fork-upstream prompt without adding a remote. API-key entry is skipped; run 'hermes config migrate' separately for those.",
    )
    update_parser.add_argument(
        "--keep-stash", action="store_true", default=False,
        help="Do NOT re-apply local changes after the update. Uncommitted "
            "changes are still stashed so the update can proceed, but they "
            "stay parked in git stash instead of being restored onto the "
            "updated code. Used by the desktop updater so local source edits "
            "never silently ride along across updates.")
    update_parser.add_argument(
        "--branch", default=None, metavar="NAME",
        help="Update against this branch instead of the default (main). "
            "If the local checkout is on a different branch, hermes will "
            "switch to the requested branch first (auto-stashing any "
            "uncommitted changes).")
    update_parser.add_argument(
        "--switch-branch", action="store_true", default=False,
        help="With updates.parked_branch_strategy: update_in_place configured, "
            "override it for this run: switch to the update target and update "
            "THERE instead of merging the target into the checked-out branch. "
            "The branch is left exactly as it was — no merge commit is written "
            "into its history. Use on long-lived feature branches where an "
            "update-driven merge commit would pollute the branch. No effect "
            "under the default strategy (switch), which already switches. "
            "Still refuses to touch a dirty tree.")
    update_parser.add_argument(
        "--force", action="store_true", default=False,
        help="Windows: proceed with the update even when another hermes.exe is detected. The concurrent process will likely cause WinError 32 warnings. Does NOT bypass the venv-process guard (see --force-venv).",
    )
    update_parser.add_argument(
        "--force-venv", action="store_true", default=False,
        help="Windows: mutate the venv even while other processes are running from its interpreter (desktop backend, gateway, terminals). Those processes keep native .pyd files locked, so the dependency sync will likely fail partway and strand the install half-updated. Use only if you know the detected holders are false positives.",
    )
    update_parser.add_argument(
        "--set-channel",
        default=None,
        type=validate_name,
        metavar="CHANNEL",
        help=(
            "Persist the update channel for THIS install (recorded per "
            "install in config.yaml under update.installs). Names are resolved "
            "from the release archive, not a built-in list. Source installs "
            "check out the published build's exact commit; main follows the "
            "source branch. Package channels are baked into their identities."
        ),
    )
    update_parser.add_argument(
        "--install-id",
        action="store_true",
        default=False,
        help=(
            "Print this install's id and path (the id keys its per-install "
            "channel record in config.yaml) and exit."
        ),
    )
    update_parser.add_argument(
        "--channel",
        default=None,
        type=validate_name,
        metavar="CHANNEL",
        help=(
            "Track CHANNEL for this run only (transient override; "
            "--set-channel persists). 'stable' and 'canary' select published "
            "releases, 'main' the branch tip. Source installs only."
        ),
    )
    update_parser.add_argument(
        "--update-id", default=None, metavar="ID", help=argparse.SUPPRESS)
    update_parser.add_argument(
        "--no-gateway-restart", action="store_true", default=False,
        help="Update code and dependencies but defer the fleet restart. Use for updates "
             "running inside a gateway cgroup, then restart gateways separately.",
    )
    update_parser.set_defaults(func=cmd_update)
