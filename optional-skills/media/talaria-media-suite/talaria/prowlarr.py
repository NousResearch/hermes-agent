"""Prowlarr — indexer management.

add-app derives the target app's URL and API key from that service's env
var (keys are never CLI args, never argv — the
`add-app <type> <url> <key>` argv-key form is not supported).
sync posts /command {ApplicationsIndexerSync}.
This module imports NOTHING besides talaria.core.
"""

from . import core

NAME = "prowlarr"
HELP = "Prowlarr — indexer management (9 subs)"
IMPLEMENTATIONS = {"sonarr": "Sonarr", "radarr": "Radarr", "lidarr": "Lidarr"}


def _indexers(args):
    data = core.api_get("prowlarr", "/indexer", timeout=args.timeout,
                        host_override=args.host)
    rows = [[str(i.get("id") or ""), i.get("name") or "?",
             i.get("protocol") or "?",
             "enabled" if i.get("enable") else "disabled"]
            for i in data]
    text = ("Indexers: %d total, %d enabled\n" % (
        len(data), sum(1 for i in data if i.get("enable")))
        + core.table(["Id", "Name", "Protocol", "State"], rows))
    core.output(args, data, text)


def _test(args):
    if args.id:
        indexer = core.api_get("prowlarr", "/indexer/%s" % args.id,
                               timeout=args.timeout, host_override=args.host)
        body = {k: indexer[k] for k in ("enable", "name", "fields",
                                        "implementationName", "implementation",
                                        "configContract", "protocol",
                                        "priority") if k in indexer}
        core.api("prowlarr", "POST", "/indexer/test", body=body,
                 timeout=args.timeout, host_override=args.host)
        core.output(args, {"tested": [int(args.id)]},
                    "Test passed for indexer %s (%s)."
                    % (args.id, indexer.get("name") or "?"))
        return
    core.api("prowlarr", "POST", "/indexer/testall", body={},
             timeout=args.timeout, host_override=args.host)
    core.output(args, {"tested": "all"}, "Test-all triggered for indexers.")


def _stats(args):
    indexers = core.api_get("prowlarr", "/indexer", timeout=args.timeout,
                            host_override=args.host)
    data = core.api_get("prowlarr", "/indexerstats", timeout=args.timeout,
                        host_override=args.host)
    rows = [[i.get("indexerName") or "?",
             str(i.get("averageResponseTime") or 0),
             str(i.get("queries") or 0), str(i.get("grabs") or 0),
             str(i.get("failedQueries") or 0)]
            for i in (data.get("indexers") or [])]
    text = ("Indexer stats (indexers: %d enabled / %d total):\n" % (
        sum(1 for i in indexers if i.get("enable")), len(indexers))
        + core.table(["Indexer", "Resp ms", "Queries", "Grabs", "Failed"],
                     rows))
    core.output(args, data, text)


def _search(args):
    params = {"query": args.query}
    if args.indexers:
        params["indexerIds"] = [int(x) for x in args.indexers.split(",")
                                if x.strip()]
    data = core.api_get("prowlarr", "/search", params=params,
                        timeout=args.timeout, host_override=args.host)
    rows = [[r.get("indexer") or "?", r.get("title") or "?",
             "%.0f MB" % ((r.get("size") or 0) / 1048576),
             r.get("protocol") or "?"] for r in data[:20]]
    text = ("Found %d result(s):\n" % len(data) + core.table(
        ["Indexer", "Title", "Size", "Protocol"], rows))
    core.output(args, data, text)


def _apps(args):
    data = core.api_get("prowlarr", "/applications", timeout=args.timeout,
                        host_override=args.host)
    rows = [[str(a.get("id") or ""), a.get("name") or "?",
             a.get("implementation") or "?", a.get("syncLevel") or "?"]
            for a in data]
    text = ("App sync targets:\n" + (core.table(
        ["Id", "Name", "Implementation", "Sync"], rows) if rows
        else "  No apps configured. Add Sonarr/Radarr in Prowlarr "
             "settings (prowlarr add-app)."))
    core.output(args, data, text)


def _add_app(args):
    impl = IMPLEMENTATIONS[args.type]
    target = args.type                      # sonarr | radarr | lidarr
    target_url = core.base_url(target, host_override=args.host)
    target_key = core.headers(target)["X-Api-Key"]   # from env, never argv
    body = {                                  # add-app request body shape
        "name": impl,
        "implementation": impl,
        "implementationName": impl,
        "configContract": impl + "Settings",
        "syncLevel": "fullSync",
        "fields": [
            {"name": "prowlarrUrl",
             "value": core.base_url("prowlarr", host_override=args.host)},
            {"name": "baseUrl", "value": target_url},
            {"name": "apiKey", "value": target_key},
            {"name": "syncCategories", "value": []},
        ],
    }
    created = core.api("prowlarr", "POST", "/applications", body=body,
                       timeout=args.timeout, host_override=args.host)
    core.output(args, {"id": created.get("id"), "name": created.get("name"),
                       "implementation": created.get("implementation"),
                       "baseUrl": target_url},
                "Added %s sync target (id: %s) pointing at %s."
                % (impl, created.get("id"), target_url))


def _sync(args):
    out = core.api("prowlarr", "POST", "/command",
                   body={"name": "ApplicationsIndexerSync"},
                   timeout=args.timeout, host_override=args.host)
    core.output(args, out, "Indexer sync to all apps triggered.")


def _status(args):
    health = core.api_get("prowlarr", "/health", timeout=args.timeout,
                          host_override=args.host)
    status = core.api_get("prowlarr", "/system/status", timeout=args.timeout,
                          host_override=args.host)
    lines = []
    if health:
        lines.extend("  issue [%s] %s" % (h.get("type") or "?",
                                          h.get("message") or "")
                     for h in health)
    else:
        lines.append("  No issues")
    lines.append("  Version: %s (branch %s)" % (status.get("version") or "?",
                                                status.get("branch") or "?"))
    core.output(args, {"health": health, "status": status},
                "Prowlarr health:\n" + "\n".join(lines))


def _logs(args):
    data = core.api_get("prowlarr", "/log",
                        params={"pageSize": args.limit,
                                "sortDirection": "descending",
                                "sortKey": "time"},
                        timeout=args.timeout, host_override=args.host)
    rows = [[str(r.get("time") or "")[:10], r.get("level") or "?",
             (r.get("message") or "")[:80]]
            for r in (data.get("records") or [])]
    text = ("Prowlarr logs (last %d):\n" % args.limit + core.table(
        ["Date", "Level", "Message"], rows))
    core.output(args, data, text)


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("indexers", help="List indexers [GET /indexer]")
    p.set_defaults(func=_indexers)
    p = s.add_parser("test", help="Test one indexer or all "
                    "[POST /indexer/test · /indexer/testall]")
    p.add_argument("id", nargs="?")
    p.set_defaults(func=_test)
    p = s.add_parser("stats", help="Indexer statistics [GET /indexerstats]")
    p.set_defaults(func=_stats)
    p = s.add_parser("search", help="Search across indexers [GET /search]")
    p.add_argument("query")
    p.add_argument("--indexers", metavar="IDS",
                   help="comma-separated indexer ids (default: all)")
    p.set_defaults(func=_search)
    p = s.add_parser("apps", help="List sync targets [GET /applications]")
    p.set_defaults(func=_apps)
    p = s.add_parser("add-app", help="Add a Sonarr/Radarr/Lidarr sync target "
                    "using that service's env key [POST /applications — "
                    "write op]")
    p.add_argument("--type", required=True, dest="type",
                   choices=sorted(IMPLEMENTATIONS))
    p.set_defaults(func=_add_app)
    p = s.add_parser("sync", help="Trigger indexer sync to apps "
                    "[POST /command ApplicationsIndexerSync — write op]")
    p.set_defaults(func=_sync)
    p = s.add_parser("status", help="Health + version "
                    "[GET /health · /system/status]")
    p.set_defaults(func=_status)
    p = s.add_parser("logs", help="Recent logs [GET /log]")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.set_defaults(func=_logs)
