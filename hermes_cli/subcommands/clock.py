"""``hermes clock`` - what time it is, and whether that time was checked.

Every agent-sync action in this family stamps the instant it recorded plus the
source that instant came from.  This command is the way a seat checks rather
than assumes: it prints host time, authoritative time, the measured skew, and
the source a stamp would carry right now.
"""

from __future__ import annotations


def build_clock_parser(subparsers) -> None:
    """Attach the ``clock`` subcommand to ``subparsers``."""
    clock_parser = subparsers.add_parser(
        "clock",
        help="Check the shared clock: host time, authoritative time, measured skew",
        description=(
            "Print the clock every agent-sync action stamps with, and where that "
            "value came from. A reading that falls back to the host clock says "
            "DEGRADED rather than wearing the authority's clothes."
        ),
    )
    clock_parser.add_argument(
        "--json", action="store_true", help="Print the full reading as JSON"
    )
    clock_parser.add_argument(
        "--guard",
        action="store_true",
        help=(
            "Exit non-zero when the clock cannot be trusted: the authority is "
            "unreachable (DEGRADED) or the host is off it by more than "
            "HERMES_CLOCK_MAX_SKEW seconds (default 2)."
        ),
    )
    clock_parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the cached offset and re-measure against the authority",
    )
    clock_parser.set_defaults(func=cmd_clock)


def cmd_clock(args) -> int:
    import json as _json
    import sys

    from tools.agent_clock import probe

    reading = probe(force=getattr(args, "force", False))
    report = {
        "utc": reading.utc_iso,
        "source": reading.source,
        "degraded": reading.degraded,
        "offset_seconds": reading.offset_seconds,
        "host_utc": reading.host_utc_iso,
        "authority": reading.authority,
        "checked_at": reading.checked_at_iso,
        "check_age_seconds": reading.check_age_seconds,
        "stamp": reading.stamp(),
    }

    if getattr(args, "json", False):
        print(_json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"utc        {report['utc']}")
        print(f"source     {report['source']}")
        print(f"degraded   {str(report['degraded']).lower()}")
        print(f"host utc   {report['host_utc']}")
        authority = report["authority"] or "no authority reachable"
        print(f"skew       {report['offset_seconds']:+.1f}s against {authority}")
        print(f"stamp      {report['stamp']}")

    if getattr(args, "guard", False):
        import os

        if report["degraded"]:
            print(
                "clock guard: DEGRADED - the authority could not be reached, so a "
                "stamp taken now would carry host time it cannot vouch for.",
                file=sys.stderr,
            )
            return 1
        limit = float(os.environ.get("HERMES_CLOCK_MAX_SKEW", "2"))
        if abs(report["offset_seconds"]) >= limit:
            print(
                f"clock guard: SKEW - this host is {report['offset_seconds']:+.1f}s "
                f"off {report['authority']}; instants recorded here can invert an order.",
                file=sys.stderr,
            )
            return 1
    return 0
