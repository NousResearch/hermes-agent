"""Bazarr — subtitle management.

Request body shapes: episodes search posts {seriesId}, movies search posts
{radarrId}. Bazarr api base is /api. This module imports
NOTHING besides talaria.core.
"""

from . import core

NAME = "bazarr"
HELP = "Bazarr — subtitle management (4 subs)"


def _fmt_langs(value):
    """Render missing-subtitle langs: live payloads send a list of language
    dicts ([{"name": "English", "code2": "en", ...}]) — display names, never
    the raw repr. Plain strings / counts pass through."""
    if isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, dict):
                names.append(str(item.get("name") or item.get("code2") or "?"))
            else:
                names.append(str(item))
        return ", ".join(names) if names else "0"
    return str(value if value is not None else 0)


def _wanted(args):
    episodes = core.api_get("bazarr", "/episodes/wanted",
                            params={"start": 0, "length": 20},
                            timeout=args.timeout, host_override=args.host)
    movies = core.api_get("bazarr", "/movies/wanted",
                          params={"start": 0, "length": 20},
                          timeout=args.timeout, host_override=args.host)
    ep_rows = [[e.get("seriesTitle") or "?",
                _fmt_langs(e.get("missing_subtitles") if e.get("missing_subtitles") is not None else e.get("missingCount"))]
               for e in (episodes.get("data") or [])]
    mv_rows = [[m.get("title") or m.get("movieTitle") or "?",
                _fmt_langs(m.get("missing_subtitles") if m.get("missing_subtitles") is not None else m.get("missingCount"))]
               for m in (movies.get("data") or [])]
    text = ("Missing subtitles — episodes:\n" + (core.table(
        ["Series", "Missing langs"], ep_rows) if ep_rows
        else "  No missing episode subtitles")
        + "\n\nMissing subtitles — movies:\n" + (core.table(
            ["Movie", "Missing langs"], mv_rows) if mv_rows
            else "  No missing movie subtitles"))
    core.output(args, {"episodes": episodes, "movies": movies}, text)


def _history(args):
    data = core.api_get("bazarr", "/history", params={"length": args.limit},
                        timeout=args.timeout, host_override=args.host)
    rows = [[str(h.get("timestamp") or "")[:19].replace("T", " "),
             str(h.get("action") or "?").upper(),
             h.get("seriesTitle") or h.get("title") or "?",
             h.get("language") or "?", h.get("provider") or "?"]
            for h in (data.get("data") or [])]
    text = ("Recent subtitle downloads (last %d):\n" % args.limit + core.table(
        ["When", "Action", "Item", "Language", "Provider"], rows))
    core.output(args, data, text)


def _search(args):
    if args.episode is None and args.movie is None:
        raise core.ConfigError(
            "bazarr search needs --episode ID or --movie ID")
    if args.episode is not None:
        body = {"seriesId": int(args.episode)}   # request body shape
        path = "/episodes/search"
        label = "series %s" % args.episode
    else:
        body = {"radarrId": int(args.movie)}     # request body shape
        path = "/movies/search"
        label = "movie %s" % args.movie
    out = core.api("bazarr", "POST", path, body=body,
                   timeout=args.timeout, host_override=args.host)
    core.output(args, {"triggered": label},
                "Subtitle search triggered for %s." % label)


def _languages(args):
    data = core.api_get("bazarr", "/system/settings", timeout=args.timeout,
                        host_override=args.host)
    general = ((data.get("settings") or {}).get("general") or {})
    langs = general.get("enabled_languages") or []
    rows = [[lang] for lang in langs]
    text = ("Configured languages:\n" + core.table(["Language"], rows)
            + "\nSeries sync: %s · Movie sync: %s · Single language: %s"
            % (general.get("serie_default_enabled"),
               general.get("movie_default_enabled"),
               general.get("single_language")))
    core.output(args, general, text)


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("wanted", help="Missing subtitles "
                    "[GET /episodes/wanted · /movies/wanted]")
    p.set_defaults(func=_wanted)
    p = s.add_parser("history", help="Recent subtitle downloads "
                    "[GET /history]")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.set_defaults(func=_history)
    p = s.add_parser("search", help="Manual subtitle search "
                    "[POST /episodes/search · /movies/search — write op]")
    p.add_argument("--episode", type=int, metavar="ID",
                   help="series id (posts seriesId)")
    p.add_argument("--movie", type=int, metavar="ID",
                   help="movie id (posts radarrId)")
    p.set_defaults(func=_search)
    p = s.add_parser("languages", help="Configured languages "
                    "[GET /system/settings]")
    p.set_defaults(func=_languages)
