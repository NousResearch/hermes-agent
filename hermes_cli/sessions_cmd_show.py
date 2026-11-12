"""``hermes sessions show <id> [--json]``: one session through the public read-only projection
that plugins get as ``ctx.sessions`` (``hermes_cli.session_reader``).

``--json`` follows the CLI ``--json`` convention: one object on stdout with ``schema_version``
first; a failure prints nothing on stdout, reports on stderr and exits 2.
"""

from __future__ import annotations

import json
import sys

from hermes_constants import get_hermes_home
from hermes_cli.session_reader import SessionReader

SHOW_SCHEMA_VERSION = 1


def add_show_parser(sessions_subparsers) -> None:
    from hermes_cli.subcommands._shared import add_json_flag

    parser = sessions_subparsers.add_parser(
        "show", help="Show one session's metadata, lineage, subagents and usage")
    parser.add_argument("session_id", help="Session ID or unique prefix")
    add_json_flag(parser, "Emit one JSON object (the ctx.sessions projection)")


def _print_text(detail: dict) -> None:
    session, usage = detail["session"], detail["usage"]
    for key, value in session.items():
        if value not in (None, "", False):
            print(f"{key + ':':<20} {value}")
    if len(detail["lineage"]) > 1:
        print(f"{'lineage:':<20} {' -> '.join(detail['lineage'])}")
    if detail["subagents"]:
        print(f"{'subagents:':<20} {len(detail['subagents'])}")
    cost = usage["actual_cost_usd"] if usage["actual_cost_usd"] is not None else usage["estimated_cost_usd"]
    print(f"{'total tokens:':<20} in {usage['input_tokens']:,} / out {usage['output_tokens']:,}"
          f" across {len(usage['session_ids'])} session(s)" + (f", ${cost:.4f}" if cost is not None else ""))


def cmd_show(args) -> int | None:
    as_json = getattr(args, "json", False)
    detail = SessionReader(get_hermes_home()).detail(args.session_id)
    if detail is None:
        if as_json:
            print(f"hermes: error: session '{args.session_id}' not found", file=sys.stderr)
            return 2
        print(f"Session '{args.session_id}' not found.")
        return 1
    if as_json:
        document = {"schema_version": SHOW_SCHEMA_VERSION, **detail}
        print(json.dumps(document, ensure_ascii=False, indent=2, default=str))
    else:
        _print_text(detail)
    return None
