"""``hermes sessions outcome add|list|remove``: what happened to a session's work after it ended (#138653).

``list --json`` is the stable read contract for consumers outside the process (memory consolidation and
dreaming plugins, dashboards): ``{"version": 1, "outcomes": [...]}``, newest observed first, each object
carrying ``OUTCOME_COLUMNS``. Fields may be added within version 1; none are renamed or removed.
Exit codes: 0 ok, 2 invalid input, 3 unknown session or outcome.
"""
from __future__ import annotations

import json
import os
import sys

from hermes_state_outcomes import OUTCOME_SIGNALS, OUTCOME_VERDICTS

_BAD_INPUT, _NOT_FOUND = 2, 3


def add_outcome_parser(sessions_subparsers) -> None:
    from hermes_cli.subcommands._shared import add_json_flag
    outcome = sessions_subparsers.add_parser(
        "outcome", help="Record, list or remove external outcomes of sessions (merged, reverted, CI, review, ticket)",
        description="Attach what happened after a session to the session itself: a PR merged or reverted, CI "
            "passed or failed, a review accepted or rejected, a ticket closed or reopened. Outcomes are deleted "
            "with their session and show up in session_search results. `list --json` is the stable contract.")
    actions = outcome.add_subparsers(dest="outcome_action", metavar="{add,list,remove}", required=True)

    add = actions.add_parser("add", help="Record one outcome (re-recording the same signal/source/ref updates it)")
    add.add_argument("session", help="Session ID, unique prefix, or title")
    add.add_argument("--signal", required=True, choices=list(OUTCOME_SIGNALS), help="What happened")
    add.add_argument("--verdict", choices=OUTCOME_VERDICTS, default=None,
        help="good, bad or neutral; defaults to the signal's, required for custom")
    add.add_argument("--source", required=True, help="Who observed it: a lowercase label such as github or ci")
    add.add_argument("--ref", "--url", dest="ref", default="", help="PR/commit/ticket URL or id (max 512 chars)")
    add.add_argument("--note", default="", help="One-line note (max 500 chars)")
    add.add_argument("--observed-at", default=None, help="When it happened: a duration ago ('2h') or an ISO time "
        "(default: now)")
    add_json_flag(add, "Print the recorded outcome as JSON")

    lst = actions.add_parser("list", help="List outcomes, newest observed first")
    lst.add_argument("session", nargs="?", default=None, help="Only this session's lineage (ID, prefix, or title)")
    lst.add_argument("--signal", choices=list(OUTCOME_SIGNALS), default=None, help="Only this signal")
    lst.add_argument("--verdict", choices=OUTCOME_VERDICTS, default=None, help="Only this verdict")
    lst.add_argument("--since", default=None, help="Observed at/after: a duration ago ('7d') or an ISO time")
    lst.add_argument("--limit", type=int, default=200, help="Maximum rows (default: 200)")
    add_json_flag(lst, 'Emit {"version": 1, "outcomes": [...]} (the stable contract in the sessions guide)')

    rm = actions.add_parser("remove", help="Remove one outcome by id")
    rm.add_argument("outcome_id", type=int, help="Outcome id (from `outcome list`)")


def _recorder() -> str:
    """Provenance is core's call: a command run by an agent (HERMES_SESSION_ID set) records as 'agent'."""
    return "agent" if os.environ.get("HERMES_SESSION_ID", "").strip() else "cli"


def _fail(message: str, code: int = _BAD_INPUT) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return code


def _resolve(db, raw: str):
    return db.resolve_session_id(raw) or db.resolve_session_by_title(raw)


def _add(db, args) -> int:
    from hermes_cli.session_filters import parse_point_in_time
    if not (session_id := _resolve(db, args.session)):
        return _fail(f"No session '{args.session}'. Run: hermes sessions list to find the id.", _NOT_FOUND)
    try:
        observed_at = parse_point_in_time(args.observed_at, "--observed-at") if args.observed_at else None
        outcome = db.record_session_outcome(session_id, signal=args.signal, verdict=args.verdict,
            source=args.source, recorder=_recorder(), ref=args.ref, note=args.note, observed_at=observed_at)
    except ValueError as exc:
        return _fail(str(exc))
    if args.json:
        print(json.dumps(outcome, indent=2))
    else:
        print(f"Recorded outcome {outcome['id']}: {outcome['signal']} ({outcome['verdict']}) from "
              f"{outcome['source']} for session '{session_id}'.")
    return 0


def _list(db, args) -> int:
    from hermes_cli.session_filters import format_epoch, parse_point_in_time
    session_id = None
    if args.session and not (session_id := _resolve(db, args.session)):
        return _fail(f"No session '{args.session}'. Run: hermes sessions list to find the id.", _NOT_FOUND)
    try:
        since = parse_point_in_time(args.since, "--since") if args.since else None
        rows = db.list_session_outcomes(session_id, since=since, signal=args.signal, verdict=args.verdict,
                                        limit=args.limit)
    except ValueError as exc:
        return _fail(str(exc))
    if args.json:
        print(json.dumps({"version": 1, "outcomes": rows}, indent=2))
    elif not rows:
        print("No outcomes recorded. Add one with: hermes sessions outcome add <session> --signal merged --source github")
    else:
        print(f"{'ID':>5}  {'Observed':<17} {'Signal':<16} {'Verdict':<8} {'Source':<12} {'Recorder':<10} "
              f"{'Session':<28} Ref\n" + "─" * 110)
        for r in rows:
            print(f"{r['id']:>5}  {format_epoch(r['observed_at']):<17} {r['signal']:<16} {r['verdict']:<8} "
                  f"{r['source'][:12]:<12} {r['recorder'][:10]:<10} {r['session_id'][:28]:<28} {r['ref']}")
    return 0


def _remove(db, args) -> int:
    # An agent removes only what agents recorded; the user's own CLI may remove any row.
    recorder = _recorder()
    if not db.delete_session_outcome(args.outcome_id, recorder=None if recorder == "cli" else recorder):
        return _fail(f"No outcome {args.outcome_id} that this caller may remove. Run: hermes sessions outcome list",
                     _NOT_FOUND)
    print(f"Removed outcome {args.outcome_id}.")
    return 0


def cmd_outcome(db, args) -> int:
    return {"add": _add, "list": _list, "remove": _remove}[args.outcome_action](db, args)
