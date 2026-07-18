"""``hermes sync`` subcommand parser (HSP/1 personal skill sync).

Cloned from ``hermes_cli/subcommands/cron.py`` — same injected-handler shape
(``func=cmd_sync``) so this module does not import ``main`` (cycle avoidance).

Commands:
  hermes sync status   -- show gate/opt-in/head state
  hermes sync pull      -- pull the owner's HEAD, materialize opted-in skills
  hermes sync push      -- push opted-in skills to the owner's HEAD
  hermes sync now       -- pull then push (full reconcile)
  hermes sync enable <skill>   -- opt a skill into sync (M1-D)
  hermes sync disable <skill>  -- opt a skill out of sync

Sync is INERT unless the resolved Nous token carries the DEV-PHASE gate claim
(tool_gateway_admin) AND a sync base URL is configured. The commands report
that state rather than failing opaquely.
"""

from __future__ import annotations

from typing import Callable


def build_sync_parser(subparsers, *, cmd_sync: Callable) -> None:
    """Attach the ``sync`` subcommand (and its sub-actions) to ``subparsers``."""
    sync_parser = subparsers.add_parser(
        "sync",
        help="Personal skill sync (HSP/1)",
        description="Sync agent-created and user-authored skills across devices.",
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

    sync_parser.set_defaults(func=cmd_sync)
