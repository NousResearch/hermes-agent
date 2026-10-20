"""Jellyfin — first-class media server.

Serves the media-server role via the Emby-lineage REST API
(research/jellyfin-api.md, spec 12.2.0). Auth: Authorization header with
the MediaBrowser scheme — the only future-proof channel on 12.x;
the legacy channels (the emby-token header, key-bearing query params, the
legacy emby route prefix) are NEVER emitted — see tests/test_jellyfin.py
for the negative traps that would catch them. Keys never ride URLs
. All /Items queries set enableImages=false for lean CLI
payloads and ALWAYS carry userId: live 12.2.0 answers 400 for
/Items/Latest and for user-scoped /Items queries (userdata sorts,
recursive+IsPlayed) without user context — pinned by the mock's
CONTRACT_400S (17/10 live sweep). Default --user is the first user from
GET /Users.
Five analytics subs (approx) derive stats from item play-state, NOT session
history — their output carries the mandatory approximation banner and
--json carries "approx": true (P8). peak-hours is DROPPED (P9).
PII gate: sessions render no remote end-points.
This module imports NOTHING besides talaria.core.
"""

from . import core

NAME = "jellyfin"
HELP = "Jellyfin — media server sessions, libraries, analytics (14 subs)"


def _user_id(args, explicit=None):
    """Explicit --user name/id, else the first user from GET /Users."""
    wanted = (explicit if explicit is not None
              else getattr(args, "user", None))
    if wanted is None:
        users = core.api_get("jellyfin", "/Users", timeout=args.timeout,
                             host_override=args.host)
        if not users:
            raise core.ApiError("jellyfin: no users found on the server")
        return users[0].get("Id") or users[0].get("id")
    # a name or an id — resolve via /Users when it is not id-shaped
    if len(wanted) == 32 and all(c in "0123456789abcdefABCDEF"
                                 for c in wanted):
        return wanted
    users = core.api_get("jellyfin", "/Users", timeout=args.timeout,
                         host_override=args.host)
    for u in users:
        if wanted in (u.get("Name") or u.get("name"),
                      u.get("Id") or u.get("id")):
            return u.get("Id") or u.get("id")
    raise core.ApiError("jellyfin: user %r not found" % wanted)


def _sessions(args):
    data = core.api_get("jellyfin", "/Sessions", timeout=args.timeout,
                        host_override=args.host)
    rows = []
    for s in data:
        item = s.get("NowPlayingItem") or s.get("nowPlayingItem")
        user = s.get("UserName") or s.get("userName") or "?"
        client = "%s (%s)" % (s.get("Client") or s.get("client") or "?",
                              s.get("DeviceName")
                              or s.get("deviceName") or "?")
        if not item:
            rows.append([user, client, "idle", "", "", ""])
            continue
        play = s.get("PlayState") or s.get("playState") or {}
        trans = s.get("TranscodingInfo") or s.get("transcodingInfo") or {}
        pos = (play.get("PositionTicks") or play.get("positionTicks") or 0)
        total = (item.get("RunTimeTicks") or item.get("runTimeTicks") or 1)
        pct = int(100.0 * pos / max(total, 1))
        state = "paused" if play.get("IsPaused") or play.get("isPaused") \
            else "playing"
        method = play.get("PlayMethod") or play.get("playMethod") \
            or ("transcode" if trans else "direct")
        rows.append([user, client,
                     "%s — %s" % (item.get("Name") or item.get("name") or "?",
                                  item.get("Type") or item.get("type") or "?"),
                     "%d%% %s" % (pct, state), method,
                     str(trans.get("Bitrate") or trans.get("bitrate") or "")])
    active = sum(1 for s in data
                 if (s.get("NowPlayingItem") or s.get("nowPlayingItem")))
    text = ("Active streams: %d of %d session(s)\n" % (active, len(data))
            + (core.table(["User", "Client", "Item", "Position", "Method",
                           "Bitrate"], rows) if rows else "  No sessions"))
    core.output(args, data, text)


def _libraries(args):
    data = core.api_get("jellyfin", "/Library/VirtualFolders",
                        timeout=args.timeout, host_override=args.host)
    rows = [[v.get("Name") or v.get("name") or "?",
             v.get("CollectionType") or v.get("collectionType") or "-",
             str(len(v.get("Locations") or v.get("locations") or []))]
            for v in data]
    text = ("Libraries:\n" + core.table(["Name", "Type", "Locations"], rows))
    core.output(args, data, text)


_COUNT_KINDS = ["movieCount", "seriesCount", "episodeCount", "artistCount",
                "albumCount", "songCount", "musicVideoCount", "boxSetCount",
                "bookCount", "itemCount"]


def _counts(args):
    params = {}
    if args.user:
        params["userId"] = _user_id(args)
    data = core.api_get("jellyfin", "/Items/Counts", params=params or None,
                        timeout=args.timeout, host_override=args.host)
    # Live 17/10 (v12.2.0): the server serializes ItemCounts with the
    # schema's native PascalCase keys (MovieCount…) under the default
    # application/json profile; camelCase only arrives via Accept-profile
    # negotiation we do not send. Read PascalCase-first with a camelCase
    # fallback — the same convention as every other reader here. Display
    # labels stay normalized camelCase regardless of source casing.
    rows = []
    for kind in _COUNT_KINDS:
        value = data.get(kind[0].upper() + kind[1:])
        if value is None:
            value = data.get(kind)
        if value:
            rows.append([kind, str(value)])
    text = ("Item counts:\n" + (core.table(["Kind", "Count"], rows)
                                if rows else "  No items"))
    core.output(args, data, text)


def _latest(args):
    # Live 12.2.0 (17/10): /Items/Latest 400s unless userId rides the
    # query — latest-media is per-user state. Always send user context;
    # default is the first user, same as every other user-scoped sub.
    params = {"userId": _user_id(args), "limit": args.limit,
              "enableImages": "false"}
    if args.types:
        params["includeItemTypes"] = args.types
    data = core.api_get("jellyfin", "/Items/Latest", params=params,
                        timeout=args.timeout, host_override=args.host)
    rows = [["%s — %s" % (i.get("Name") or i.get("name") or "?",
                          i.get("Type") or i.get("type") or "?"),
             str(i.get("ProductionYear") or i.get("productionYear")
                 or "")] for i in data]
    text = ("Latest items:\n" + (core.table(["Item", "Year"], rows)
                                 if rows else "  Nothing yet"))
    core.output(args, data, text)


def _resume(args):
    params = {"userId": _user_id(args), "limit": args.limit,
              "enableImages": "false"}
    data = core.api_get("jellyfin", "/UserItems/Resume", params=params,
                        timeout=args.timeout, host_override=args.host)
    items = data if isinstance(data, list) else (data.get("Items") or [])
    rows = [[i.get("Name") or i.get("name") or "?",
             core.ticks_mmss((i.get("UserData")
                              or i.get("userData") or {}).get(
                                  "PlaybackPositionTicks")
             or (i.get("UserData") or i.get("userData") or {}).get(
                 "playbackPositionTicks") or 0),
             core.ticks_mmss(i.get("RunTimeTicks")
                             or i.get("runTimeTicks") or 0)]
            for i in items]
    text = ("Continue watching:\n" + (core.table(
        ["Item", "Position", "Runtime"], rows) if rows else "  Nothing "
        "in progress"))
    core.output(args, data, text)


def _nextup(args):
    params = {"userId": _user_id(args), "limit": args.limit}
    data = core.api_get("jellyfin", "/Shows/NextUp", params=params,
                        timeout=args.timeout, host_override=args.host)
    items = data if isinstance(data, list) else (data.get("Items")
                                                 or data.get("items") or [])
    rows = [[i.get("SeriesName") or i.get("seriesName") or "?",
             i.get("Name") or i.get("name") or "?"] for i in items]
    text = ("Next up:\n" + (core.table(["Series", "Episode"], rows)
                            if rows else "  Nothing next up"))
    core.output(args, data, text)


def _users(args):
    data = core.api_get("jellyfin", "/Users", timeout=args.timeout,
                        host_override=args.host)
    rows = [[u.get("Name") or u.get("name") or "?",
             str(u.get("LastLoginDate") or u.get("lastLoginDate")
                 or "")[:10]] for u in data]
    text = ("Users:\n" + core.table(["Name", "Last login"], rows))
    core.output(args, data, text)


def _activity(args):
    data = core.api_get("jellyfin", "/System/ActivityLog/Entries",
                        params={"startIndex": 0, "limit": args.limit},
                        timeout=args.timeout, host_override=args.host)
    entries = data.get("Items") or data.get("items") or []
    rows = [[str(e.get("Date") or e.get("date") or "")[:19].replace("T", " "),
             e.get("Name") or e.get("name") or "?",
             e.get("Severity") or e.get("severity") or "Info",
             (e.get("ShortOverview")
              or e.get("shortOverview") or "")[:60]] for e in entries]
    text = ("Activity log (last %d):\n" % args.limit + core.table(
        ["When", "Event", "Severity", "Detail"], rows))
    core.output(args, data, text)


def _refresh(args):
    if args.item:
        core.api("jellyfin", "POST", "/Items/%s/Refresh" % args.item,
                 params={"metadataRefreshMode": "Default",
                         "imageRefreshMode": "Default"},
                 timeout=args.timeout, host_override=args.host)
        core.output(args, {"refreshed": args.item},
                    "Refresh triggered for item %s." % args.item)
        return
    core.api("jellyfin", "POST", "/Library/Refresh", body={},
             timeout=args.timeout, host_override=args.host)
    core.output(args, {"refreshed": "all"},
                "Library refresh triggered for all libraries.")


def _items_query(args, extra):
    params = {"enableImages": "false"}
    # Live 12.2.0 (17/10): bare /Items 400s when user-scoped params ride
    # the query without user context (sortBy=DatePlayed/PlayCount,
    # recursive+filters=IsPlayed). Always send userId — an explicit id in
    # extra still wins (user-stats aggregates per-user).
    if "userId" not in extra:
        params["userId"] = _user_id(args)
    params.update(extra)
    return core.api_get("jellyfin", "/Items", params=params,
                        timeout=args.timeout, host_override=args.host)


def _played_items(args, extra=None):
    data = _items_query(args, {"filters": "IsPlayed", **(extra or {})})
    return data.get("Items") or data.get("items") or [], data


def _history(args):
    items, data = _played_items(args, {
        "sortBy": "DatePlayed", "sortOrder": "Descending",
        "limit": args.limit, "fields": "UserData"})
    rows = []
    for i in items:
        ud = i.get("UserData") or i.get("userData") or {}
        rows.append([
            str(ud.get("LastPlayedDate") or ud.get("lastPlayedDate")
                or "")[:10],
            i.get("Name") or i.get("name") or "?",
            i.get("Type") or i.get("type") or "?",
            str(ud.get("PlayCount") or ud.get("playCount") or 0)])
    text = ("Recently played (item-grain, newest first):\n" + core.table(
        ["Played", "Item", "Type", "Plays"], rows))
    core.output(args, data, text, approx=True)


def _most_watched(args):
    items, data = _played_items(args, {
        "sortBy": "PlayCount", "sortOrder": "Descending",
        "limit": args.limit})
    rows = []
    for i in items:
        ud = i.get("UserData") or i.get("userData") or {}
        plays = ud.get("PlayCount") or ud.get("playCount") or 0
        if not plays:
            continue
        rows.append([i.get("Name") or i.get("name") or "?",
                     i.get("Type") or i.get("type") or "?", str(plays),
                     core.ticks_mmss(i.get("RunTimeTicks")
                                     or i.get("runTimeTicks") or 0)])
    text = ("Most watched (by item play count):\n" + core.table(
        ["Item", "Type", "Plays", "Runtime"], rows))
    core.output(args, data, text, approx=True)


def _user_stats(args):
    users = core.api_get("jellyfin", "/Users", timeout=args.timeout,
                         host_override=args.host)
    rows = []
    per_user = {}
    for u in users:
        name = u.get("Name") or u.get("name") or "?"
        uid = u.get("Id") or u.get("id")
        items, _ = _played_items(
            args, {"userId": uid, "limit": 1000}) if uid else ([], {})
        plays = sum((i.get("UserData") or i.get("userData") or {}).get(
            "PlayCount") or (i.get("UserData") or i.get("userData")
                             or {}).get("playCount") or 0 for i in items)
        hours = sum(
            ((i.get("UserData") or i.get("userData") or {}).get(
                "PlayCount") or (i.get("UserData")
                                 or i.get("userData") or {}).get(
                                     "playCount") or 0)
            * (i.get("RunTimeTicks") or i.get("runTimeTicks") or 0)
            for i in items) / 3.6e10
        rows.append([name, str(len(items)), str(plays),
                     "%.1f h" % round(hours, 1)])
        per_user[name] = {"played_items": len(items), "plays": int(plays),
                          "approx_watch_hours": round(hours, 1)}
    text = ("Per-user play activity:\n" + core.table(
        ["User", "Played items", "Plays", "≈ Watch time"], rows))
    core.output(args, per_user, text, approx=True)


def _play_totals(args):
    libs = core.api_get("jellyfin", "/Library/VirtualFolders",
                        timeout=args.timeout, host_override=args.host)
    params = {"recursive": "true", "filters": "IsPlayed", "limit": 1000}
    if args.user:
        params["userId"] = _user_id(args)
    rows = []
    total = 0.0
    per_lib = {}
    for lib in libs:
        lid = lib.get("ItemId") or lib.get("itemId")
        if not lid:
            continue
        data = _items_query(args, dict(params, parentId=lid))
        items = data.get("Items") or data.get("items") or []
        hours = sum(
            ((i.get("UserData") or i.get("userData") or {}).get(
                "PlayCount") or (i.get("UserData")
                                 or i.get("userData") or {}).get(
                                     "playCount") or 0)
            * (i.get("RunTimeTicks") or i.get("runTimeTicks") or 0)
            for i in items) / 3.6e10
        total += hours
        per_lib[lib.get("Name") or lib.get("name") or "?"] = round(hours, 1)
        rows.append([lib.get("Name") or lib.get("name") or "?",
                     str(len(items)), "%.1f h" % round(hours, 1)])
    text = ("Watch-time totals (plays × runtime, per library):\n"
            + core.table(["Library", "Played items", "≈ Watch time"], rows)
            + "\nTotal: %.1f h" % round(total, 1))
    core.output(args, {"per_library": per_lib,
                       "total_hours": round(total, 1)}, text, approx=True)


def _popular_genres(args):
    items, data = _played_items(args, {"fields": "Genres", "limit": 1000})
    counts = {}
    for i in items:
        plays = (i.get("UserData") or i.get("userData") or {}).get(
            "PlayCount") or (i.get("UserData") or i.get("userData")
                             or {}).get("playCount") or 0
        for genre in i.get("Genres") or i.get("genres") or []:
            counts[genre] = counts.get(genre, 0) + plays
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    rows = [[genre, str(plays)] for genre, plays in top]
    text = ("Popular genres (weighted by plays):\n" + core.table(
        ["Genre", "≈ Plays"], rows))
    core.output(args, {"genres": dict(top)}, text, approx=True)


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("sessions", help="Live sessions and streams "
                    "[GET /Sessions]")
    p.set_defaults(func=_sessions)
    p = s.add_parser("libraries", help="Libraries [GET /Library/VirtualFolders]")
    p.set_defaults(func=_libraries)
    p = s.add_parser("counts", help="Item counts [GET /Items/Counts]")
    p.add_argument("--user", metavar="U", help="user name or id "
                   "(default: first user)")
    p.set_defaults(func=_counts)
    p = s.add_parser("latest", help="Recently added items [GET /Items/Latest]")
    p.add_argument("--types", default="Movie,Episode", metavar="T",
                   help="comma-separated item types")
    p.add_argument("--user", metavar="U", help="user name or id "
                   "(default: first user)")
    p.add_argument("--limit", type=int, default=10, metavar="N")
    p.set_defaults(func=_latest)
    p = s.add_parser("resume", help="Continue watching "
                    "[GET /UserItems/Resume] (new, P10)")
    p.add_argument("--user", metavar="U")
    p.add_argument("--limit", type=int, default=10, metavar="N")
    p.set_defaults(func=_resume)
    p = s.add_parser("nextup", help="Next episodes "
                    "[GET /Shows/NextUp] (new, P10)")
    p.add_argument("--user", metavar="U")
    p.add_argument("--limit", type=int, default=10, metavar="N")
    p.set_defaults(func=_nextup)
    p = s.add_parser("users", help="Users [GET /Users] (new, P10)")
    p.set_defaults(func=_users)
    p = s.add_parser("activity", help="Server activity feed "
                    "[GET /System/ActivityLog/Entries] (new, P10)")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.set_defaults(func=_activity)
    p = s.add_parser("refresh", help="Refresh library or one item "
                    "[POST /Library/Refresh · /Items/{id}/Refresh — "
                    "write op] (new, P10)")
    p.add_argument("--item", metavar="ID")
    p.set_defaults(func=_refresh)
    p = s.add_parser("history", help="Recently played (≈ from play-state) "
                    "[GET /Items filters=IsPlayed]")
    p.add_argument("--user", metavar="U")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.set_defaults(func=_history)
    p = s.add_parser("most-watched", help="Most played items "
                    "(≈ from play-state) [GET /Items sortBy=PlayCount]")
    p.add_argument("--user", metavar="U")
    p.add_argument("--limit", type=int, default=10, metavar="N")
    p.set_defaults(func=_most_watched)
    p = s.add_parser("user-stats", help="Per-user play aggregation (≈)")
    p.set_defaults(func=_user_stats)
    p = s.add_parser("play-totals", help="Watch-time totals per library (≈)")
    p.add_argument("--user", metavar="U")
    p.set_defaults(func=_play_totals)
    p = s.add_parser("popular-genres", help="Genres weighted by plays (≈)")
    p.set_defaults(func=_popular_genres)
