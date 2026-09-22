"""``hermes update`` subcommand parser."""

from __future__ import annotations

import argparse
from types import MethodType
from typing import Callable

from hermes_cli.update_target import parse_reviewed_source, validate_target_request


def build_update_parser(subparsers, *, cmd_update: Callable) -> None:
    """Attach the ``update`` subcommand to ``subparsers``."""
    update_parser = subparsers.add_parser(
        "update", help="Update Hermes Agent to the latest version",
        description="Pull the latest changes from git and reinstall dependencies",
    )
    update_parser.add_argument(
        "--revision", default=None, metavar="HEX",
        help="Pinned target revision (40 lowercase hexadecimal characters). "
            "Must be supplied with --expected-install-id and --expected-current-sha.")
    update_parser.add_argument(
        "--expected-install-id", default=None, metavar="HEX",
        help="Pinned installation identity (32 lowercase hexadecimal characters). "
            "Must be supplied with --revision and --expected-current-sha.")
    update_parser.add_argument(
        "--expected-current-sha", default=None, metavar="HEX",
        help="Pinned current checkout SHA (40 lowercase hexadecimal characters). "
            "Must be supplied with --revision and --expected-install-id.")
    update_parser.add_argument(
        "--reviewed-source", default=None, metavar="BASE64URL",
        help="Reviewed repository, origin, ref, target and applicable assurance binding "
             "as a base64url JSON record. Required for a pinned update.")
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
        "--no-gateway-restart", action="store_true", default=False,
        help="Update code and dependencies but skip the final gateway restart. "
            "Use for cron/automated updates that run inside the gateway process: "
            "the gateway would otherwise restart its own cgroup and kill the updater. "
            "Pair with a separate restart step (e.g. a cron that runs 10-15 min later).",
    )
    update_parser.add_argument(
        "--post-swap", default=None, metavar="FILE", help=argparse.SUPPRESS,
        # Internal: the pre-pull interpreter re-executes itself here after the code swap so the
        # rest of the update runs on the pulled code (hermes_cli/update_handoff.py).
    )
    # ``argparse`` routes a parent parser through the subparser's
    # ``parse_known_args`` method, so validate at that boundary rather than in
    # the mutating handler.  This makes --check/--plan fail before any
    # collaborator can be reached while preserving the normal handler seam.
    _parse_known_args = update_parser.parse_known_args
    _pinned_options = {
        "--revision",
        "--expected-install-id",
        "--expected-current-sha",
        "--reviewed-source",
    }

    # Reject only abbreviations of the new flags, at argparse's option
    # resolution boundary. Let argparse own values, '--', sys.argv and legacy
    # abbreviations rather than pre-scanning tokens with different semantics.
    _get_option_tuples = update_parser._get_option_tuples

    def _get_update_option_tuples(parser, option_string):
        matches = _get_option_tuples(option_string)
        if any(option in _pinned_options for _, option, *_ in matches):
            parser.error(
                f"unrecognized arguments: {option_string}; use exact pinned options: "
                "--revision, --expected-install-id, --expected-current-sha, --reviewed-source"
            )
        return matches

    update_parser._get_option_tuples = MethodType(
        _get_update_option_tuples, update_parser
    )

    def _parse_update_known_args(parser, args=None, namespace=None):
        parsed, remainder = _parse_known_args(args, namespace)
        try:
            source = (
                parse_reviewed_source(parsed.reviewed_source)
                if parsed.reviewed_source is not None else None
            )
            parsed.target_request = validate_target_request(
                parsed.revision,
                parsed.expected_install_id,
                parsed.expected_current_sha,
                source,
            )
            if parsed.target_request is not None and source is None:
                raise ValueError("source-binding-required")
            if parsed.target_request is None and source is not None:
                raise ValueError("reviewed-source-without-target")
            if parsed.target_request is not None and source.target_sha != parsed.target_request.revision:
                raise ValueError("reviewed-target-mismatch")
        except ValueError as exc:
            parser.error(
                f"{exc}; supply --revision, --expected-install-id and "
                "--expected-current-sha together as 40/32/40 lowercase hexadecimal "
                "characters respectively, or omit all three for a legacy update"
            )
        return parsed, remainder

    update_parser.parse_known_args = MethodType(_parse_update_known_args, update_parser)
    update_parser.set_defaults(func=cmd_update)
