"""``hermes sync`` subcommand parser — Skill Sync.

Cloned from ``hermes_cli/subcommands/cron.py`` — same injected-handler shape
(``func=cmd_sync``) so this module does not import ``main`` (cycle avoidance).

Skill Sync covers two surfaces, both under this one command for launch:

  Personal — your own skills, across your own devices:
    hermes sync status                 show gate/opt-in/head state
    hermes sync pull                   pull and materialize opted-in skills
    hermes sync push                   push opted-in skills
    hermes sync now                    reconcile: pull then push
    hermes sync enable <skill>         opt a skill into sync
    hermes sync disable <skill>        opt a skill out of sync
    hermes sync device [--name]        show or set this device's label

  Organisation — skills shared with your team:
    hermes sync propose <skill>        share a skill with your organisation

Sync is INERT unless the resolved Nous token carries the access-gate claim
AND a sync base URL is configured. The commands report that state rather than
failing opaquely.
"""

from __future__ import annotations

import argparse
from typing import Callable


def build_sync_parser(subparsers, *, cmd_sync: Callable) -> None:
    """Attach the ``sync`` subcommand (and its sub-actions) to ``subparsers``."""
    sync_parser = subparsers.add_parser(
        "sync",
        help="Skill Sync — sync your skills across devices and with your team",
        description=(
            "Skill Sync keeps your skills with you. Personal sync moves your "
            "own skills between your devices; if you belong to an "
            "organisation, you also get its shared skills and can propose "
            "your own back to the team."
        ),
        epilog=(
            "Examples:\n"
            "  hermes sync status            what is synced, and from where\n"
            "  hermes sync enable my-skill   include a skill in your sync\n"
            "  hermes sync now               pull, then push\n"
            "  hermes sync propose my-skill  share a skill with your team\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sync_sub = sync_parser.add_subparsers(dest="sync_command")

    sync_sub.add_parser("status", help="Show what is synced, and from where")
    sync_sub.add_parser(
        "pull", help="Pull your synced skills (and your organisation's)"
    )
    sync_sub.add_parser("push", help="Push your opted-in skills")
    sync_sub.add_parser("now", help="Reconcile now: pull then push")

    enable = sync_sub.add_parser("enable", help="Include a skill in your sync")
    enable.add_argument("skill", help="Skill name (frontmatter name / directory name)")

    disable = sync_sub.add_parser("disable", help="Exclude a skill from your sync")
    disable.add_argument("skill", help="Skill name (frontmatter name / directory name)")

    device = sync_sub.add_parser(
        "device",
        help="Show or set this device's label (shown in the sync console)",
    )
    device.add_argument(
        "--name",
        dest="device_name",
        default=None,
        help="Set a human-friendly label for this device (e.g. \"Ben's Laptop\"). "
        "Omit to print the current label.",
    )

    # Org-shared skills. A member's submission becomes a proposal an admin
    # reviews; an admin's merges straight into the shared set. Accounts that
    # aren't in a shared organisation are told so plainly.
    propose = sync_sub.add_parser(
        "propose",
        help="Share a skill with your organisation",
        description=(
            "Submit one of your skills to your organisation's shared set. If "
            "you are an admin it is added directly; otherwise it becomes a "
            "proposal for an admin to review. Accounts that aren't part of a "
            "shared organisation don't have this workflow."
        ),
    )
    propose.add_argument("name", help="Skill name to share")
    propose.add_argument(
        "-m",
        "--message",
        default=None,
        help="Optional message describing the change",
    )

    sync_parser.set_defaults(func=cmd_sync)
