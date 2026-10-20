"""Overseerr — media request management.

Auth: X-Api-Key header. Request body shapes: approve posts {} to
/request/{id}/approve, deny posts {message} to /request/{id}/decline.
PII gate: default output shows displayName only, never email addresses.
This module imports NOTHING besides talaria.core.
"""

from . import core

NAME = "overseerr"
HELP = "Overseerr — media request management (5 subs)"
MEDIA_STATUS = {2: "Pending", 3: "Processing", 4: "Downloading",
                5: "Available"}


def _requester(req):
    """PII gate: displayName or an opaque id — never the email."""
    who = req.get("requestedBy") or {}
    return who.get("displayName") or ("user#%s" % who.get("id", "?"))


def _list(args):
    params = {"take": 50, "skip": 0}
    if args.status != "all":
        params["filter"] = args.status
    data = core.api_get("overseerr", "/request", params=params,
                        timeout=args.timeout, host_override=args.host)
    results = data.get("results") or []
    rows = [[str(r.get("id") or ""),
             (r.get("media") or {}).get("title")
             or (r.get("media") or {}).get("name") or "?",
             str(r.get("type") or "?"),
             MEDIA_STATUS.get((r.get("media") or {}).get("status"),
                              "Unknown"),
             _requester(r)] for r in results]
    text = ("Requests (%s):\n" % args.status + (core.table(
        ["Id", "Title", "Type", "Status", "Requested by"],
        rows) if rows else "  No requests found")
        + "\nTotal: %s" % (data.get("pageInfo") or {}).get("results",
                                                           len(results)))
    core.output(args, data, text)


def _approve(args):
    out = core.api("overseerr", "POST", "/request/%s/approve" % args.id,
                   body={}, timeout=args.timeout, host_override=args.host)
    core.output(args, {"approved": int(args.id)},
                "Request %s approved." % args.id)


def _deny(args):
    out = core.api("overseerr", "POST", "/request/%s/decline" % args.id,
                   body={"message": args.reason},
                   timeout=args.timeout, host_override=args.host)
    core.output(args, {"denied": int(args.id), "reason": args.reason},
                "Request %s denied (reason: %s)." % (args.id, args.reason))


def _info(args):
    r = core.api_get("overseerr", "/request/%s" % args.id,
                     timeout=args.timeout, host_override=args.host)
    media = r.get("media") or {}
    lines = [
        "Title: %s" % (media.get("title") or media.get("name") or "Unknown"),
        "Type: %s" % str(r.get("type") or "?").upper(),
        "Status: %s" % MEDIA_STATUS.get(media.get("status"), "Unknown"),
        "Requested by: %s" % _requester(r),
        "Requested on: %s" % str(r.get("createdAt") or "")[:10],
        "TMDB/TVDB ID: %s" % (media.get("tmdbId") or media.get("tvdbId")),
    ]
    seasons = r.get("seasons")
    if seasons:
        lines.append("Seasons: %s" % ", ".join(
            str(s.get("seasonNumber")) for s in seasons))
    core.output(args, r, "\n".join(lines))


def _stats(args):
    data = core.api_get("overseerr", "/request", params={"take": 1000},
                        timeout=args.timeout, host_override=args.host)
    users = core.api_get("overseerr", "/user", timeout=args.timeout,
                         host_override=args.host)
    results = data.get("results") or []
    counts = {"total": len(results)}
    for status, label in MEDIA_STATUS.items():
        counts[label.lower()] = sum(
            1 for r in results if (r.get("media") or {}).get("status")
            == status)
    counts["movies"] = sum(1 for r in results if r.get("type") == "movie")
    counts["tv"] = sum(1 for r in results if r.get("type") == "tv")
    by_user = {}
    for r in results:
        by_user[_requester(r)] = by_user.get(_requester(r), 0) + 1
    top = sorted(by_user.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    rows = [[name, str(n)] for name, n in top]
    text = ("Request statistics:\n"
            "  Total: %d · Pending: %d · Processing: %d · Available: %d\n"
            "  Movies: %d · TV: %d\n"
            "  Users: %d\nTop requesters:\n"
            % (counts["total"], counts["pending"], counts["processing"],
               counts["available"], counts["movies"], counts["tv"],
               len(users))
            + (core.table(["User", "Requests"], rows) if rows else "  —"))
    core.output(args, {"counts": counts,
                       "top_requesters": dict(top),
                       "users": len(users)}, text)


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("list", help="List requests [GET /request]")
    p.add_argument("status", nargs="?", default="all",
                   choices=["all", "pending", "approved", "available"])
    p.set_defaults(func=_list)
    p = s.add_parser("approve", help="Approve a request "
                    "[POST /request/{id}/approve — write op]")
    p.add_argument("id")
    p.set_defaults(func=_approve)
    p = s.add_parser("deny", help="Deny a request "
                    "[POST /request/{id}/decline — write op]")
    p.add_argument("id")
    p.add_argument("--reason", default="No reason provided")
    p.set_defaults(func=_deny)
    p = s.add_parser("info", help="Request details [GET /request/{id}]")
    p.add_argument("id")
    p.set_defaults(func=_info)
    p = s.add_parser("stats", help="Request statistics "
                    "[GET /request · /user]")
    p.set_defaults(func=_stats)
