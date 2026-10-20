"""Lidarr — music management.

This module imports NOTHING besides talaria.core.
"""

from . import core

NAME = "lidarr"
HELP = "Lidarr — music management (2 subs)"


def _lookup(args):
    results = core.api_get("lidarr", "/search", params={"term": args.query},
                           timeout=args.timeout, host_override=args.host)
    rows = []
    for r in results[:20]:
        rows.append([r.get("artistName") or r.get("artist") or "?",
                     r.get("albumTitle") or r.get("title") or "Artist",
                     r.get("albumType") or "Artist",
                     (r.get("releaseDate") or "N/A")[:4]])
    text = ("Found %d result(s):\n" % len(results)
            + core.table(["Artist", "Album", "Type", "Year"], rows))
    core.output(args, results, text)


def _queue(args):
    data = core.api_get("lidarr", "/queue", timeout=args.timeout,
                        host_override=args.host)
    records = data.get("records") if isinstance(data, dict) else data
    records = records or []
    rows = [[r.get("title") or "?", r.get("status") or "?",
             "%.0f MB left" % ((r.get("sizeleft") or 0) / 1048576),
             r.get("timeleft") or "?"] for r in records]
    text = ("Lidarr queue:\n" + (core.table(
        ["Title", "Status", "Progress", "ETA"], rows) if rows
        else "  (empty)"))
    core.output(args, data, text)


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("lookup", help="Search for music [GET /search]")
    p.add_argument("query")
    p.set_defaults(func=_lookup)
    p = s.add_parser("queue", help="Download queue [GET /queue] (P7)")
    p.set_defaults(func=_queue)
