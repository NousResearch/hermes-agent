"""Sonarr — TV series management.

Request bodies match the services' documented API shapes.
This module imports NOTHING besides talaria.core.
"""

import datetime

from . import core

NAME = "sonarr"
HELP = "Sonarr — TV series management (9 subs)"
LOOKUP_PATH = "/series/lookup"
ITEM_PATH = "/series"
ID_FIELD = "tvdbId"


def _episode_label(rec):
    """`Series - S01E02 - Title` from a calendar/wanted record."""
    series = (rec.get("series") or {}).get("title") or rec.get("seriesTitle") or "?"
    season = rec.get("seasonNumber")
    episode = rec.get("episodeNumber")
    if season is None or episode is None:
        return series
    return "%s - S%02dE%02d - %s" % (series, season, episode,
                                     rec.get("title") or "")


def _lookup(args):
    results = core.api_get("sonarr", LOOKUP_PATH,
                           params={"term": args.query},
                           timeout=args.timeout, host_override=args.host)
    rows = [[str(r.get(ID_FIELD) or ""), r.get("title") or "?",
             str(r.get("year") or "N/A"),
             (r.get("overview") or "")[:70]] for r in results[:20]]
    text = ("Found %d result(s):\n" % len(results)
            + core.table(["tvdbId", "Title", "Year", "Overview"], rows))
    core.output(args, results, text)


def _add(args):
    results = core.api_get("sonarr", LOOKUP_PATH, params={"term": args.query},
                           timeout=args.timeout, host_override=args.host)
    if not results:
        raise core.ApiError("sonarr: no lookup results for %r" % args.query)
    series = results[0]
    profiles = core.api_get("sonarr", "/qualityprofile",
                            timeout=args.timeout, host_override=args.host)
    quality = args.quality or (profiles[0]["id"] if profiles else 1)
    folders = core.api_get("sonarr", "/rootfolder",
                           timeout=args.timeout, host_override=args.host)
    folder = args.folder or (folders[0].get("path") if folders else "")
    body = {                      # add-series request body shape
        "title": series.get("title"),
        ID_FIELD: series.get(ID_FIELD),
        "qualityProfileId": int(quality),
        "rootFolderPath": folder,
        "seasonFolder": True,
        "monitored": True,
        "addOptions": {"searchForMissingEpisodes": True},
    }
    added = core.api("sonarr", "POST", ITEM_PATH, body=body,
                     timeout=args.timeout, host_override=args.host)
    core.output(args, added,
                "Added: %s (quality profile %s, root %s)\n"
                "Search for missing episodes started."
                % (series.get("title"), quality, folder))


def _remove(args):
    item = core.api_get("sonarr", "%s/%s" % (ITEM_PATH, args.id),
                        timeout=args.timeout, host_override=args.host)
    core.api("sonarr", "DELETE", "%s/%s" % (ITEM_PATH, args.id),
             params={"deleteFiles": "false"},
             timeout=args.timeout, host_override=args.host)
    core.output(args, {"removed": item.get("id"), "deleteFiles": False},
                "Removed: %s (files kept — deleteFiles=false)"
                % item.get("title"))


def _wanted(args):
    data = core.api_get("sonarr", "/wanted/missing",
                        params={"pageSize": args.limit, "sortKey": "airDateUtc"},
                        timeout=args.timeout, host_override=args.host)
    records = data.get("records") or []
    rows = [[_episode_label(r)] for r in records]
    text = ("Wanted episodes (missing, monitored):\n"
            + core.table(["Episode"], rows)
            + "\nTotal missing: %s" % data.get("totalRecords", len(records)))
    core.output(args, data, text)


def _calendar(args):
    today = datetime.datetime.now(datetime.timezone.utc).date()
    end = today + datetime.timedelta(days=args.days)
    items = core.api_get("sonarr", "/calendar",
                         params={"start": today.isoformat(),
                                 "end": end.isoformat()},
                         timeout=args.timeout, host_override=args.host)
    rows = [[str(i.get("airDateUtc") or "")[:10], _episode_label(i)]
            for i in sorted(items, key=lambda x: x.get("airDateUtc") or "")]
    text = ("Upcoming episodes (next %d day(s)):\n" % args.days
            + (core.table(["Air date", "Episode"], rows) if rows
               else "  No upcoming episodes"))
    core.output(args, items, text)


def _history(args):
    data = core.api_get("sonarr", "/history",
                        params={"pageSize": args.limit, "sortKey": "date",
                                "sortDirection": "descending"},
                        timeout=args.timeout, host_override=args.host)
    records = data.get("records") or []
    rows = [[str(r.get("date") or "")[:10], str(r.get("eventType") or ""),
             ((r.get("series") or {}).get("title") or "?"),
             (r.get("quality", {}).get("quality", {}) or {}).get("name", "")]
            for r in records]
    text = ("History (last %d):\n" % args.limit
            + (core.table(["Date", "Event", "Series", "Quality"], rows)
               if rows else "  No history"))
    core.output(args, data, text)


def _rename(args):
    out = core.api("sonarr", "POST", "/command",
                   body={"name": "RenameSeries", "seriesIds": [int(args.id)]},
                   timeout=args.timeout, host_override=args.host)
    core.output(args, out, "Rename command sent for series %s." % args.id)


def _refresh(args):
    body = {"name": "RefreshSeries"}
    if args.id:
        body["seriesId"] = int(args.id)
    out = core.api("sonarr", "POST", "/command", body=body,
                   timeout=args.timeout, host_override=args.host)
    core.output(args, out, "Refresh command sent%s."
                % (" for series %s" % args.id if args.id else " (all series)"))


def _queue(args):
    data = core.api_get("sonarr", "/queue", timeout=args.timeout,
                        host_override=args.host)
    records = data.get("records") or []
    rows = [[r.get("title") or "?", r.get("status") or "?",
             "%.0f MB left" % ((r.get("sizeleft") or 0) / 1048576),
             r.get("timeleft") or "?"] for r in records]
    text = ("Sonarr queue:\n" + (core.table(
        ["Title", "Status", "Progress", "ETA"], rows) if rows
        else "  (empty)"))
    core.output(args, data, text)


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("lookup", help="Search for a series [GET /series/lookup]")
    p.add_argument("query")
    p.set_defaults(func=_lookup)
    p = s.add_parser("add", help="Add first lookup result and search "
                    "[POST /series — write op]")
    p.add_argument("query")
    p.add_argument("--quality", type=int, metavar="N",
                   help="quality profile id (default: first profile)")
    p.add_argument("--folder", metavar="PATH",
                   help="root folder path (default: first root folder)")
    p.set_defaults(func=_add)
    p = s.add_parser("remove", help="Remove a series, keep files "
                    "[DELETE /series/{id}?deleteFiles=false — write op]")
    p.add_argument("id")
    p.set_defaults(func=_remove)
    p = s.add_parser("wanted", help="Missing monitored episodes "
                    "[GET /wanted/missing]")
    p.add_argument("--limit", type=int, default=50, metavar="N")
    p.set_defaults(func=_wanted)
    p = s.add_parser("calendar", help="Upcoming episodes [GET /calendar]")
    p.add_argument("--days", type=int, default=7, metavar="N")
    p.set_defaults(func=_calendar)
    p = s.add_parser("history", help="Recent task history [GET /history]")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.set_defaults(func=_history)
    p = s.add_parser("rename", help="Trigger rename scan "
                    "[POST /command RenameSeries — write op]")
    p.add_argument("id")
    p.set_defaults(func=_rename)
    p = s.add_parser("refresh", help="Refresh metadata "
                    "[POST /command RefreshSeries — write op]")
    p.add_argument("id", nargs="?")
    p.set_defaults(func=_refresh)
    p = s.add_parser("queue", help="Download queue [GET /queue]")
    p.set_defaults(func=_queue)
