#!/usr/bin/env python3
"""CLI for the weekly Telegram-reaction feedback digest.

Reads reaction events captured by the Telegram adapter and prints either a
human-readable digest (default) or structured JSON. Designed to be used three
ways:

  1. Manual verification:
       python scripts/reaction_digest.py --since-days 7
  2. Cron data-collection step (agent rephrases + routes to Admin topic):
       python scripts/reaction_digest.py --json --no-mark
  3. Mark-as-reviewed weekly run (idempotent — only unreviewed rows):
       python scripts/reaction_digest.py --mark

Must run under the SAME Hermes profile as the gateway that captured the
reactions (default profile, unless overridden), since that determines which
``state.db`` is read. Pass --db to point at an explicit state.db.

Exit code is always 0 on success (including the empty case) so cron does not
flag an empty week as a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the repo root importable when run directly (scripts/ is one level down).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.reaction_review import build_digest, open_session_db, DEFAULT_SINCE_DAYS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since-days", type=int, default=DEFAULT_SINCE_DAYS,
        help=f"Look-back window in days (default {DEFAULT_SINCE_DAYS}).",
    )
    parser.add_argument(
        "--db", default=os.getenv("HERMES_REACTION_DB"),
        help="Explicit path to state.db (default: active profile's state.db).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit structured JSON instead of rendered text.",
    )
    parser.add_argument(
        "--mark", dest="mark", action="store_true",
        help="Mark surfaced reactions reviewed so the next run skips them.",
    )
    parser.add_argument(
        "--no-mark", dest="mark", action="store_false",
        help="Do not mark reviewed (default) — safe for preview/data-collection.",
    )
    parser.add_argument(
        "--all", dest="only_unreviewed", action="store_false",
        help="Include already-reviewed reactions in the window.",
    )
    parser.add_argument(
        "--quiet-empty", action="store_true",
        help="Print nothing (and exit 0) when there is nothing to review. "
             "Useful for a silent cron watchdog.",
    )
    parser.set_defaults(mark=False, only_unreviewed=True)
    args = parser.parse_args(argv)

    try:
        db = open_session_db(args.db)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"reaction_digest: could not open state.db: {exc}", file=sys.stderr)
        return 1

    try:
        digest = build_digest(
            db,
            since_days=args.since_days,
            only_unreviewed=args.only_unreviewed,
            mark_reviewed=args.mark,
        )
    finally:
        try:
            db.close()
        except Exception:
            pass

    if args.quiet_empty and digest.get("empty"):
        return 0

    if args.json:
        # Drop the rendered text from JSON to keep it machine-focused; callers
        # that want the text can render from buckets/followups themselves.
        out = {k: v for k, v in digest.items()}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(digest["text"])
        if digest["followups"]:
            print("\n--- targeted follow-ups ---")
            for q in digest["followups"]:
                print(f"• {q}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
