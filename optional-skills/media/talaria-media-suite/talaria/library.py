"""library — cross-arr library analytics.

All calls go through core against sonarr/radarr/lidarr directly (no
cross-module imports; core is the only shared surface). Facet maths is
plain int/Counter arithmetic.
This module imports NOTHING besides talaria.core.
"""

import datetime

from . import core

NAME = "library"
HELP = "Cross-arr library analytics (10 subs)"
GB = 1073741824


def _movies(args):
    return core.api_get("radarr", "/movie", timeout=args.timeout,
                        host_override=args.host)


def _series(args):
    return core.api_get("sonarr", "/series", timeout=args.timeout,
                       host_override=args.host)


def _artists(args):
    return core.api_get("lidarr", "/artist", timeout=args.timeout,
                        host_override=args.host)


def _series_gaps(series):
    """[(title, gap)] for monitored series, gap = totalEpisodeCount -
    episodeFileCount from /series statistics. Single source of truth for
    `library stats` and `library missing` so the two can never disagree.
    (17/10 live bug: stats previously fetched bare GET /api/v3/episode,
    which real Sonarr rejects — 400 "seriesId or episodeIds must be
    provided". /series statistics are the aggregate the API provides.)"""
    gaps = []
    for s in series:
        st = s.get("statistics") or {}
        gap = (st.get("totalEpisodeCount") or 0) \
            - (st.get("episodeFileCount") or 0)
        if s.get("monitored") and gap > 0:
            gaps.append((s.get("title"), gap))
    return gaps


def _stats(args):
    sections = []

    def radarr_section():
        data = _movies(args)
        total = len(data)
        monitored = sum(1 for m in data if m.get("monitored"))
        downloaded = sum(1 for m in data if m.get("hasFile"))
        missing = sum(1 for m in data
                      if m.get("monitored") and not m.get("hasFile"))
        size = sum(m.get("sizeOnDisk") or 0 for m in data
                   if m.get("hasFile"))
        sections.append(("radarr", {
            "movies": total, "monitored": monitored,
            "downloaded": downloaded, "missing": missing,
            "disk_gb": round(size / GB, 2)}))

    def sonarr_section():
        series = _series(args)
        stats = [s.get("statistics") or {} for s in series]
        size = sum(st.get("sizeOnDisk") or 0 for st in stats)
        sections.append(("sonarr", {
            "series": len(series),
            "monitored_series": sum(1 for s in series if s.get("monitored")),
            "episodes": sum(st.get("totalEpisodeCount") or 0 for st in stats),
            "downloaded_episodes": sum(st.get("episodeFileCount") or 0
                                       for st in stats),
            "missing_episodes": sum(g for _, g in _series_gaps(series)),
            "disk_gb": round(size / GB, 2)}))

    def lidarr_section():
        data = _artists(args)
        size = sum((a.get("statistics") or {}).get("sizeOnDisk") or 0
                   for a in data)
        sections.append(("lidarr", {
            "artists": len(data),
            "monitored_artists": sum(1 for a in data if a.get("monitored")),
            "albums": sum((a.get("statistics") or {}).get("albumCount") or 0
                          for a in data),
            "disk_gb": round(size / GB, 2)}))

    runners = {"radarr": radarr_section, "sonarr": sonarr_section,
               "lidarr": lidarr_section}
    if args.app == "all":
        order = ["radarr", "sonarr", "lidarr"]
    else:
        order = [args.app]
    for name in order:
        try:
            runners[name]()
        except core.TalariaError as exc:
            sections.append((name, {"error": str(exc)}))
    lines = []
    for name, stat in sections:
        lines.append("%s:" % name)
        lines.extend("  %s: %s" % (k, v) for k, v in stat.items())
    core.output(args, dict(sections), "\n".join(lines))


def _quality(args):
    profiles = core.api_get(args.app, "/qualityprofile",
                            timeout=args.timeout, host_override=args.host)
    items = _movies(args) if args.app == "radarr" else _series(args)
    counts = {}
    for it in items:
        pid = it.get("qualityProfileId")
        counts[pid] = counts.get(pid, 0) + 1
    rows = [[p.get("name") or "?", str(counts.get(p.get("id"), 0))]
            for p in profiles]
    noun = "movies" if args.app == "radarr" else "series"
    text = ("Quality profile breakdown (%s):\n" % args.app + core.table(
        ["Profile", noun.title()], rows))
    core.output(args, {p.get("name"): counts.get(p.get("id"), 0)
                       for p in profiles}, text)


def _missing(args):
    lines = []
    payload = {}
    if args.app in ("all", "radarr"):
        data = _movies(args)
        miss = [m for m in data if m.get("monitored")
                and not m.get("hasFile")]
        payload["radarr"] = {"missing_movies": len(miss)}
        lines.append("Radarr missing movies: %d" % len(miss))
        lines.extend("  %s (%s)" % (m.get("title"), m.get("year"))
                     for m in miss[:20])
        if len(miss) > 20:
            lines.append("  ... and %d more" % (len(miss) - 20))
    if args.app in ("all", "sonarr"):
        miss = _series_gaps(_series(args))
        payload["sonarr"] = {"missing_episodes": sum(g for _, g in miss)}
        lines.append("Sonarr missing episodes: %d"
                     % sum(g for _, g in miss))
        lines.extend("  %s: %d missing" % (t, g) for t, g in miss[:20])
    core.output(args, payload, "\n".join(lines))


def _unmonitored(args):
    data = _movies(args) if args.app == "radarr" else _series(args)
    unmon = [i for i in data if not i.get("monitored")]
    rows = [["%s (%s)" % (i.get("title"), i.get("year"))
             if args.app == "radarr" else i.get("title") or "?"]
            for i in unmon[:20]]
    text = ("Unmonitored (%s): %d\n" % (args.app, len(unmon))
            + core.table(["Title"], rows))
    core.output(args, {"unmonitored": len(unmon)}, text)


def _recent(args):
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(days=args.days)).strftime(
                  "%Y-%m-%dT%H:%M:%SZ")
    data = _movies(args) if args.app == "radarr" else _series(args)
    added = [i for i in data if (i.get("added") or "") >= cutoff]
    added.sort(key=lambda i: i.get("added") or "", reverse=True)
    rows = [[str(i.get("added") or "")[:10],
             "%s (%s)" % (i.get("title"), i.get("year"))
             if args.app == "radarr" else i.get("title") or "?"]
            for i in added[:20]]
    text = ("Recently added (last %d day(s), %s):\n" % (args.days, args.app)
            + (core.table(["Added", "Title"], rows) if rows
               else "  Nothing in that window"))
    core.output(args, {"cutoff": cutoff, "count": len(added)}, text)


def _facet(args, field, label):
    data = _movies(args) if args.app == "radarr" else _series(args)
    counts = {}
    for i in data:
        value = i.get(field)
        if value in (None, ""):
            value = "Unknown"
        if isinstance(value, list):
            for v in value:
                counts[str(v)] = counts.get(str(v), 0) + 1
        else:
            counts[str(value)] = counts.get(str(value), 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    rows = [[k, str(v)] for k, v in top]
    text = ("%s distribution (%s):\n" % (label, args.app) + core.table(
        [label, "Count"], rows))
    core.output(args, dict(top), text)


def _studios(args):
    data = _movies(args)
    counts = {}
    for m in data:
        studio = m.get("studio") or "Unknown"
        counts[studio] = counts.get(studio, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    rows = [[k, str(v)] for k, v in top]
    text = ("Studio distribution (radarr):\n" + core.table(
        ["Studio", "Movies"], rows))
    core.output(args, dict(top), text)


def _nofiles(args):
    lines = []
    payload = {}
    if args.app in ("radarr",):
        data = _movies(args)
        nofile = [m for m in data if m.get("monitored")
                  and not m.get("hasFile")]
        payload["radarr"] = len(nofile)
        lines.append("Radarr monitored movies with no files: %d"
                     % len(nofile))
        lines.extend("  %s (%s)" % (m.get("title"), m.get("year"))
                     for m in nofile[:30])
    else:
        data = _series(args)
        nofile = [s for s in data if s.get("monitored")
                  and ((s.get("statistics") or {}).get("episodeFileCount")
                       or 0) == 0]
        payload["sonarr"] = len(nofile)
        lines.append("Sonarr monitored series with no files: %d"
                     % len(nofile))
        lines.extend("  %s" % (s.get("title") or "?")
                     for s in nofile[:30])
    core.output(args, payload, "\n".join(lines))


def _disk(args):
    lines = []
    payload = {}
    targets = (["radarr", "sonarr", "lidarr"] if args.app == "all"
               else [args.app])
    for app in targets:
        try:
            folders = core.api_get(app, "/rootfolder", timeout=args.timeout,
                                   host_override=args.host)
        except core.TalariaError as exc:
            lines.append("%s: unavailable (%s)" % (app, exc))
            continue
        payload[app] = [{"path": f.get("path"),
                         "free_gb": int((f.get("freeSpace") or 0) / GB),
                         "total_gb": int((f.get("totalSpace") or 0) / GB)}
                        for f in folders]
        lines.append("%s:" % app)
        lines.extend("  %s: %d GB free / %s"
                     % (f.get("path"),
                        int((f.get("freeSpace") or 0) / GB),
                        ("%d GB total" % int((f.get("totalSpace") or 0) / GB))
                        if (f.get("totalSpace") or 0) > 0 else "total unknown")
                     for f in folders)
    core.output(args, payload, "\n".join(lines))


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("stats", help="Library statistics "
                    "[GET /movie · /series · /artist]")
    p.add_argument("app", nargs="?", default="all",
                   choices=["all", "radarr", "sonarr", "lidarr"])
    p.set_defaults(func=_stats)
    p = s.add_parser("quality", help="Quality profile breakdown "
                    "[GET /qualityprofile + /movie|/series]")
    p.add_argument("app", nargs="?", default="radarr",
                   choices=["radarr", "sonarr"])
    p.set_defaults(func=_quality)
    p = s.add_parser("missing", help="Monitored content with no files "
                    "[GET /movie|/series]")
    p.add_argument("app", nargs="?", default="all",
                   choices=["all", "radarr", "sonarr"])
    p.set_defaults(func=_missing)
    p = s.add_parser("unmonitored", help="Unmonitored content "
                    "[GET /movie|/series]")
    p.add_argument("app", nargs="?", default="radarr",
                   choices=["radarr", "sonarr"])
    p.set_defaults(func=_unmonitored)
    p = s.add_parser("recent", help="Recently added [GET /movie|/series]")
    p.add_argument("app", nargs="?", default="radarr",
                   choices=["radarr", "sonarr"])
    p.add_argument("--days", type=int, default=7, metavar="N")
    p.set_defaults(func=_recent)
    p = s.add_parser("genres", help="Genre distribution "
                    "[GET /movie|/series]")
    p.add_argument("app", nargs="?", default="radarr",
                   choices=["radarr", "sonarr"])
    p.set_defaults(func=lambda a: _facet(a, "genres", "Genre"))
    p = s.add_parser("years", help="Year distribution "
                    "[GET /movie|/series]")
    p.add_argument("app", nargs="?", default="radarr",
                   choices=["radarr", "sonarr"])
    p.set_defaults(func=lambda a: _facet(a, "year", "Year"))
    p = s.add_parser("studios", help="Studio distribution, radarr only "
                    "[GET /movie]")
    p.add_argument("app", nargs="?", default="radarr",
                   choices=["radarr"])
    p.set_defaults(func=_studios)
    p = s.add_parser("nofiles", help="Monitored content with no files "
                    "(detailed) [GET /movie|/series]")
    p.add_argument("app", nargs="?", default="radarr",
                   choices=["radarr", "sonarr"])
    p.set_defaults(func=_nofiles)
    p = s.add_parser("disk", help="Disk usage by root folder "
                    "[GET /rootfolder × sonarr/radarr/lidarr]")
    p.add_argument("app", nargs="?", default="all",
                   choices=["all", "radarr", "sonarr", "lidarr"])
    p.set_defaults(func=_disk)
