"""``hermes profile-sync`` subcommand parser."""

from __future__ import annotations

from hermes_cli.profile_sync import build_parser

__all__ = ["build_profile_sync_parser"]


def build_profile_sync_parser(subparsers, *, cmd_profile_sync=None) -> None:
    """Attach the read-only profile-sync subcommand to ``subparsers``."""
    parser = build_parser(subparsers)
    if cmd_profile_sync is not None:
        # build_parser sets profile_sync.cmd_profile_sync directly on subcommands.
        # Keep the injected handler hook for consistency with other extracted
        # parser modules and tests that assert handler wiring.
        for action in getattr(parser, "_subparsers", None)._group_actions if getattr(parser, "_subparsers", None) else []:
            for subparser in getattr(action, "choices", {}).values():
                subparser.set_defaults(func=cmd_profile_sync)
