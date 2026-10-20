"""Radarr — movie management.

Request bodies match the services' documented API shapes.
This module imports NOTHING besides talaria.core.
"""

import datetime

from . import core

NAME = "radarr"
HELP = "Radarr — movie management (9 subs)"
LOOKUP_PATH = "/movie/lookup"
ITEM_PATH = "/movie"
ID_FIELD = "tmdbId"


def _lookup(args):
    results = core.api_get("radarr", LOOKUP_PATH,
                           params={"term": args.query},
                           timeout=args.timeout, host_override=args.host)
    rows = [[str(r.get(ID_FIELD) or ""), r.get("title") or "?",
             str(r.get("year") or "N/A"),
             (r.get("overview") or "")[:70]] for r in results[:20]]
    text = ("Found %d result(s):\n" % len(results)
            + core.table(["tmdbId", "Title", "Year", "Overview"], rows))
    core.output(args, results, text)


def _add(args):
    results = core.api_get("radarr", LOOKUP_PATH, params={"term": args.query},
                           timeout=args.timeout, host_override=args.host)
    if not results:
        raise core.ApiError("radarr: no lookup results for %r" % args.query)
    movie = results[0]
    profiles = core.api_get("radarr", "/qualityprofile",
                            timeout=args.timeout, host_override=args.host)
    quality = args.quality or (profiles[0]["id"] if profiles else 1)
    folders = core.api_get("radarr", "/rootfolder",
                           timeout=args.timeout, host_override=args.host)
    folder = args.folder or (folders[0].get("path") if folders else "")
    body = {                      # add-movie request body shape
        "title": movie.get("title"),
        ID_FIELD: movie.get(ID_FIELD),
        "year": movie.get("year"),
        "qualityProfileId": int(quality),
        "rootFolderPath": folder,
        "monitored": True,
        "minimumAvailability": "released",
        "addOptions": {"searchForMovie": True},
    }
    added = core.api("radarr", "POST", ITEM_PATH, body=body,
                     timeout=args.timeout, host_override=args.host)
    core.output(args, added,
                "Added: %s (%s) (quality profile %s, root %s)\n"
                "Search started automatically."
                % (movie.get("title"), movie.get("year"), quality, folder))


def _remove(args):
    item = core.api_get("radarr", "%s/%s" % (ITEM_PATH, args.id),
                        timeout=args.timeout, host_override=args.host)
    core.api("radarr", "DELETE", "%s/%s" % (ITEM_PATH, args.id),
             params={"deleteFiles": "false"},
             timeout=args.timeout, host_override=args.host)
    core.output(args, {"removed": item.get("id"), "deleteFiles": False},
                "Removed: %s (files kept — deleteFiles=false)"
                % item.get("title"))


def _wanted(args):
    data = core.api_get("radarr", "/wanted/missing",
                        params={"pageSize": args.limit, "sortKey": "title"},
                        timeout=args.timeout, host_override=args.host)
    records = data.get("records") or []
    rows = [["%s (%s)" % (r.get("title") or "?", r.get("year") or "N/A")]
            for r in records]
    text = ("Wanted movies (missing, monitored):\n"
            + core.table(["Movie"], rows)
            + "\nTotal missing: %s" % data.get("totalRecords", len(records)))
    core.output(args, data, text)


def _calendar(args):
    today = datetime.datetime.now(datetime.timezone.utc).date()
    end = today + datetime.timedelta(days=args.days)
    items = core.api_get("radarr", "/calendar",
                         params={"start": today.isoformat(),
                                 "end": end.isoformat()},
                         timeout=args.timeout, host_override=args.host)
    rows = [[str(i.get("digitalRelease") or i.get("physicalRelease")
                or "TBA")[:10],
             "%s (%s)" % (i.get("title") or "?", i.get("year") or "N/A")]
            for i in sorted(items, key=lambda x: x.get("title") or "")]
    text = ("Upcoming movies (next %d day(s)):\n" % args.days
            + (core.table(["Release", "Movie"], rows) if rows
               else "  No upcoming movies"))
    core.output(args, items, text)


def _history(args):
    data = core.api_get("radarr", "/history",
                        params={"pageSize": args.limit, "sortKey": "date",
                                "sortDirection": "descending"},
                        timeout=args.timeout, host_override=args.host)
    records = data.get("records") or []
    rows = [[str(r.get("date") or "")[:10], str(r.get("eventType") or ""),
             ((r.get("movie") or {}).get("title") or "?"),
             (r.get("quality", {}).get("quality", {}) or {}).get("name", "")]
            for r in records]
    text = ("History (last %d):\n" % args.limit
            + (core.table(["Date", "Event", "Movie", "Quality"], rows)
               if rows else "  No history"))
    core.output(args, data, text)


def _rename(args):
    out = core.api("radarr", "POST", "/command",
                   body={"name": "RenameMovie", "movieIds": [int(args.id)]},
                   timeout=args.timeout, host_override=args.host)
    core.output(args, out, "Rename command sent for movie %s." % args.id)


def _refresh(args):
    body = {"name": "RefreshMovie"}
    if args.id:
        body["movieId"] = int(args.id)
    out = core.api("radarr", "POST", "/command", body=body,
                   timeout=args.timeout, host_override=args.host)
    core.output(args, out, "Refresh command sent%s."
                % (" for movie %s" % args.id if args.id else " (all movies)"))


def _queue(args):
    data = core.api_get("radarr", "/queue", timeout=args.timeout,
                        host_override=args.host)
    records = data.get("records") or []
    rows = [[r.get("title") or "?", r.get("status") or "?",
             "%.0f MB left" % ((r.get("sizeleft") or 0) / 1048576),
             r.get("timeleft") or "?"] for r in records]
    text = ("Radarr queue:\n" + (core.table(
        ["Title", "Status", "Progress", "ETA"], rows) if rows
        else "  (empty)"))
    core.output(args, data, text)


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("lookup", help="Search for a movie [GET /movie/lookup]")
    p.add_argument("query")
    p.set_defaults(func=_lookup)
    p = s.add_parser("add", help="Add first lookup result and search "
                    "[POST /movie — write op]")
    p.add_argument("query")
    p.add_argument("--quality", type=int, metavar="N",
                   help="quality profile id (default: first profile)")
    p.add_argument("--folder", metavar="PATH",
                   help="root folder path (default: first root folder)")
    p.set_defaults(func=_add)
    p = s.add_parser("remove", help="Remove a movie, keep files "
                    "[DELETE /movie/{id}?deleteFiles=false — write op]")
    p.add_argument("id")
    p.set_defaults(func=_remove)
    p = s.add_parser("wanted", help="Missing monitored movies "
                    "[GET /wanted/missing]")
    p.add_argument("--limit", type=int, default=50, metavar="N")
    p.set_defaults(func=_wanted)
    p = s.add_parser("calendar", help="Upcoming movies [GET /calendar]")
    p.add_argument("--days", type=int, default=7, metavar="N")
    p.set_defaults(func=_calendar)
    p = s.add_parser("history", help="Recent task history [GET /history]")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.set_defaults(func=_history)
    p = s.add_parser("rename", help="Trigger rename scan "
                    "[POST /command RenameMovie — write op]")
    p.add_argument("id")
    p.set_defaults(func=_rename)
    p = s.add_parser("refresh", help="Refresh metadata "
                    "[POST /command RefreshMovie — write op]")
    p.add_argument("id", nargs="?")
    p.set_defaults(func=_refresh)
    p = s.add_parser("queue", help="Download queue [GET /queue]")
    p.set_defaults(func=_queue)
