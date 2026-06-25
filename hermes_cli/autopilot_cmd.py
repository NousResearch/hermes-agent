"""`hermes autopilot` CLI — inspect and grow the deception dictionary.

Subcommands:
  harvest-deceptions   Scan the autopilot ADR logs for caught deceptions and
                       surface novel phrasings to promote into the dictionary.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_harvest(args: argparse.Namespace) -> int:
    from agent.autopilot import harvest

    root = Path(args.adr_root).expanduser() if getattr(args, "adr_root", None) else None
    report = harvest.harvest(root, top=getattr(args, "top", 25))
    if getattr(args, "json", False):
        import json
        print(json.dumps(report, indent=2))
    else:
        print(harvest.format_report(report))
    return 0


def register_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="autopilot_command", help="Autopilot maintenance commands")

    p_harvest = sub.add_parser(
        "harvest-deceptions",
        help="Mine ADR logs for caught deceptions + novel phrasings to grow the dictionary",
        description=(
            "Scan the autopilot decision logs (ADR) for kind='deception' records, "
            "tally which cheat categories the model reaches for most, and surface "
            "candidate NEW phrasings that aren't in the dictionary yet so you can "
            "promote the genuine excuses into deception_patterns.yaml (upstream) or "
            "~/.hermes/autopilot/deception-patterns.local.yaml (private)."
        ),
    )
    p_harvest.add_argument(
        "--adr-root",
        help="ADR file or directory (default: <workspace>/.hermes/autopilot/adr/)",
    )
    p_harvest.add_argument("--top", type=int, default=25, help="Max novel phrases to show")
    p_harvest.add_argument("--json", action="store_true", help="Emit the raw report as JSON")
    p_harvest.set_defaults(func=_cmd_harvest)

    # bare `hermes autopilot` → harvest
    parser.set_defaults(func=_cmd_harvest, adr_root=None, top=25, json=False)
