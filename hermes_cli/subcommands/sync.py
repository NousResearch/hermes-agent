"""``hermes sync`` subcommand parser (personal skill sync).

Cloned from ``hermes_cli/subcommands/cron.py`` — same injected-handler shape
(``func=cmd_sync``) so this module does not import ``main`` (cycle avoidance).

Commands:
  hermes sync status   -- show gate/opt-in/head state
  hermes sync pull      -- pull the owner's HEAD, materialize opted-in skills
  hermes sync push      -- push opted-in skills to the owner's HEAD
  hermes sync now       -- pull then push (full reconcile)
  hermes sync enable <skill>   -- opt a skill into sync
  hermes sync disable <skill>  -- opt a skill out of sync
  hermes sync device [--name]  -- show or set this device's sync label

This surface is PERSONAL sync only: it moves your own skills between your own
devices via ``refs/user/<owner>/HEAD``. Sharing a skill with an organisation
is a different operation with a different destination and an approval step —
see ``hermes skills propose``.

Sync is INERT unless the resolved Nous token carries the access-gate claim
AND a sync base URL is configured. The commands report that state rather than
failing opaquely.
"""

from __future__ import annotations

from typing import Callable


def build_sync_parser(subparsers, *, cmd_sync: Callable) -> None:
    """Attach the ``sync`` subcommand (and its sub-actions) to ``subparsers``."""
    import argparse

    sync_parser = subparsers.add_parser(
        "sync",
        help="Personal skill sync across your devices",
        description=(
            "Sync agent-created and user-authored skills across your own "
            "devices."
        ),
        epilog=(
            "Sharing with your team:\n"
            "  These commands cover your PERSONAL skills only. To share a "
            "skill with your\n"
            "  organisation, use `hermes skills propose <skill>` instead — it "
            "submits the\n"
            "  skill to your org's shared set (an admin approves it unless "
            "you are one).\n"
            "  Approved org skills arrive automatically and are read-only "
            "locally.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sync_sub = sync_parser.add_subparsers(dest="sync_command")

    sync_sub.add_parser("status", help="Show sync gate, opt-in, and head state")
    sync_sub.add_parser("pull", help="Pull the owner's HEAD and materialize opted-in skills")
    sync_sub.add_parser("push", help="Push opted-in skills to the owner's HEAD")
    sync_sub.add_parser("now", help="Reconcile now: pull then push")

    enable = sync_sub.add_parser("enable", help="Opt a skill into sync")
    enable.add_argument("skill", help="Skill name (frontmatter name / directory name)")

    disable = sync_sub.add_parser("disable", help="Opt a skill out of sync")
    disable.add_argument("skill", help="Skill name (frontmatter name / directory name)")

    device = sync_sub.add_parser(
        "device",
        help="Show or set this device's sync label (shown in the sync console)",
    )
    device.add_argument(
        "--name",
        dest="device_name",
        default=None,
        help="Set a human-friendly label for this device (e.g. \"Ben's Laptop\"). "
        "Omit to print the current label.",
    )

    sync_parser.set_defaults(func=cmd_sync)
