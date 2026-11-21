"""``hermes dream`` — consolidate memory across conversations, with the person's own words as proof."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def add_dream_command(memory_subparsers) -> None:
    """Attach ``dream`` under ``hermes memory``."""
    parser = memory_subparsers.add_parser(
        "dream",
        help="Consolidate memory across recent conversations (evidence required)",
        description="Read the last week of conversations together and keep the lasting patterns "
        "that no single conversation could have taught: how the person likes answers, how they "
        "work, what they keep correcting. Every entry must quote the person's own words from each "
        "conversation it relies on; an entry whose quote is not in the transcript, or that invents "
        "a number, path or URL, is dropped. Writes go through the memory store, so the same "
        "character budget applies, and each run is journalled for --revert.",
        epilog="Nightly use: run it from any scheduler, e.g. a script-only cron job "
        "(hermes cron create \"0 4 * * *\" --script ~/.hermes/dream.sh --no-agent) "
        "whose script calls `hermes memory dream`.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be kept; write nothing and no journal."
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=None,
        metavar="N",
        help="How far back to read (default: 168, one week).",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        metavar="N",
        help="At most this many conversations (default: 20).",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument(
        "--revert", metavar="JOURNAL", help="Undo one journalled run: a path from an earlier dream."
    )
    parser.set_defaults(func=cmd_dream)


def cmd_dream(args: argparse.Namespace) -> int:
    from agent.dream import format_report, revert_journal, run_dream
    from tools.memory_tool import load_on_disk_store

    if args.revert:
        path = Path(args.revert).expanduser()
        if not path.is_file():
            print(f"dream: no journal at {path}", file=sys.stderr)
            return 1
        result = revert_journal(path, load_on_disk_store())
        for failure in result["failures"]:
            print(f"dream: {failure}", file=sys.stderr)
        print(f"dream: restored {result['restored']} entr{'y' if result['restored'] == 1 else 'ies'}")
        return 1 if result["failures"] else 0

    from persistence.core.errors import StoreUnavailable

    try:
        report = run_dream(dry_run=args.dry_run, since_hours=args.since_hours, max_sessions=args.max_sessions)
    except StoreUnavailable:
        print("dream: no session store yet — nothing has been recorded on this profile", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"dream: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else format_report(report))
    return 0
