"""hermes wisdom — share-candidate review commands (PRD 1, M0)."""

import argparse
from typing import Callable


def build_wisdom_parser(subparsers, *, cmd_wisdom: Callable) -> None:
    """Attach the ``wisdom`` subcommand (and its sub-actions) to ``subparsers``."""
    wisdom_parser = subparsers.add_parser(
        "wisdom",
        help="Wisdom — review and act on skill share candidates",
        description=(
            "The curator's weekly share pass scores your skills from usage "
            "analytics and nominates candidates for sharing. These commands "
            "let you review the current candidates, run a pass manually, and "
            "decline skills you don't want nominated again."
        ),
        epilog=(
            "Examples:\n"
            "  hermes wisdom candidates    show current share candidates\n"
            "  hermes wisdom run           force a scoring pass now\n"
            "  hermes wisdom approve foo   share skill 'foo' with your org\n"
            "  hermes wisdom decline foo   stop nominating skill 'foo'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    wisdom_sub = wisdom_parser.add_subparsers(dest="wisdom_command")

    wisdom_sub.add_parser(
        "candidates",
        help="Show the current share candidates with evidence",
    )
    wisdom_sub.add_parser(
        "run",
        help="Force a share-candidate scoring pass now (dry-run)",
    )

    decline = wisdom_sub.add_parser(
        "decline",
        help="Decline a share candidate (it won't be nominated again)",
    )
    decline.add_argument("skill", help="Skill name to stop nominating")

    approve = wisdom_sub.add_parser(
        "approve",
        help="Approve a share candidate (submits it to your organisation)",
    )
    approve.add_argument("skill", help="Skill name to share")
    approve.add_argument(
        "-m",
        "--message",
        default=None,
        help="Optional message describing the share",
    )
    approve.add_argument(
        "--collective",
        default=None,
        help="Share with a specific collective instead of the whole org",
    )

    wisdom_parser.set_defaults(func=cmd_wisdom)
