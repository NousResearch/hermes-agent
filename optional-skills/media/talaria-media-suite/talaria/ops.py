"""ops — setup, discover, status, diagnose, queue.

setup: arr keys auto-fetch via the public GET /initialize.json probe;
Jellyfin has NO anonymous key endpoint so setup prints
create-a-key instructions and verifies JELLYFIN_KEY from the ENVIRONMENT
when present (keys are never CLI args — the earlier
--jellyfin-key argv is deliberately not implemented). discover: jellyfin
via the unauthenticated GET /System/Info/Public. diagnose: the
docker branch is DROPPED — read-only queue/rootfolder/history
heuristics only. queue covers sonarr+radarr+lidarr (P7). This module
imports NOTHING besides talaria.core.
"""

import os

from . import core

NAME = "ops"
HELP = "Stack operations: status, setup, discover, diagnose, queue (5 subs)"

# per-service unauthenticated reachability probe paths (root-relative:
# /initialize.json lives at the server root, outside every api base)
PROBE_PATHS = {
    "sonarr": "/initialize.json", "radarr": "/initialize.json",
    "lidarr": "/initialize.json", "prowlarr": "/initialize.json",
    "bazarr": "/api/system/status", "overseerr": "/api/v1/status",
    "jellyfin": "/System/Info/Public",
}
HEALTH_PATHS = {
    "sonarr": "/health", "radarr": "/health", "lidarr": "/health",
    "prowlarr": "/health", "bazarr": "/system/health",
    "overseerr": "/status", "jellyfin": "/System/Info",
}


def _probe(name, host_override, timeout):
    """Unauthenticated reachability probe. Returns (found, payload):
    found=True when ANY HTTP status came back (200 or 401/403 both mean
    the service is running)."""
    try:
        payload = core.api(name, "GET", PROBE_PATHS[name], auth=False,
                           root=True, timeout=timeout,
                           host_override=host_override)
        return True, payload
    except core.AuthError:
        return True, {}                      # server answered: auth needed
    except core.ApiError:
        return True, {}                      # server answered: odd status
    except (core.NetworkError, core.TimeoutError):
        return False, None


def _status(args):
    rows = []
    payload = {}
    for name in core.SERVICES:
        if not os.environ.get(core.SERVICES[name].key_env):
            rows.append([name, "—", "no key (export %s)"
                         % core.SERVICES[name].key_env])
            payload[name] = {"state": "no-key"}
            continue
        try:
            if name == "jellyfin":
                info = core.api_get(name, HEALTH_PATHS[name],
                                    timeout=args.timeout,
                                    host_override=args.host)
                rows.append([name, "UP", "v%s"
                             % (info.get("Version")
                                or info.get("version") or "?")])
                payload[name] = {"state": "up",
                                 "version": info.get("Version")
                                 or info.get("version")}
                continue
            health = core.api_get(name, HEALTH_PATHS[name],
                                  timeout=args.timeout,
                                  host_override=args.host)
            issues = [h for h in (health if isinstance(health, list)
                                  else []) if h.get("type") != "info"]
            if not issues:
                rows.append([name, "UP", "healthy"])
                payload[name] = {"state": "up"}
            else:
                rows.append([name, "ISSUES",
                             "; ".join("%s: %s" % (h.get("type"),
                                                   h.get("message"))
                                       for h in issues)])
                payload[name] = {"state": "issues", "count": len(issues)}
        except core.TalariaError as exc:
            rows.append([name, "DOWN", str(exc)])
            payload[name] = {"state": "down", "error": str(exc)}
    text = ("Service status:\n" + core.table(["Service", "State", "Detail"],
                                             rows))
    core.output(args, payload, text)


def _setup(args):
    lines = ["Probing %s for the 7 services…" % args.host_value]
    found = []
    keys = {}
    for name in core.SERVICES:
        ok, payload = _probe(name, args.host_value, args.timeout)
        if not ok:
            lines.append("  ·  %s — not found on port %d"
                         % (name, core.port_for(name, None)))
            continue
        found.append(name)
        key = None
        if isinstance(payload, dict):
            key = payload.get("apiKey")
        if key:
            keys[name] = key
            lines.append("  ✓ %s — up (API key recovered via %s)"
                         % (name, PROBE_PATHS[name]))
        elif name == "jellyfin":
            lines.append("  ✓ %s — up (Jellyfin %s)"
                         % (name, (payload or {}).get("Version")
                            or (payload or {}).get("version") or "?"))
        else:
            lines.append("  ✓ %s — up (needs API key: Settings → General "
                         "→ API Key)" % name)
    if not found:
        raise core.NetworkError(
            "no services detected at %s — check host, ports and firewall"
            % args.host_value)
    lines.append("Found %d service(s): %s" % (len(found), ", ".join(found)))

    # verify each recovered arr key with an authenticated status call
    for name, key in keys.items():
        try:
            os.environ.update({core.SERVICES[name].key_env: key})
            status = core.api_get(name, "/system/status", auth=True,
                                  timeout=args.timeout,
                                  host_override=args.host_value)
            lines.append("  ✓ %s v%s — key verified"
                         % (name, status.get("version") or "?"))
        except core.TalariaError as exc:
            lines.append("  ! %s — key verification failed: %s"
                         % (name, exc))

    # Jellyfin key: env-only (never argv, the design)
    jf_key = os.environ.get("JELLYFIN_KEY")
    if jf_key:
        try:
            info = core.api_get("jellyfin", "/System/Info",
                                timeout=args.timeout,
                                host_override=args.host_value)
            lines.append("  ✓ jellyfin v%s — JELLYFIN_KEY verified"
                         % (info.get("Version") or info.get("version")
                            or "?"))
        except core.TalariaError as exc:
            lines.append("  ! jellyfin — JELLYFIN_KEY verification failed: "
                         "%s" % exc)
    else:
        lines.append("  Jellyfin API key: create one in Dashboard → API "
                     "Keys, then:")
        lines.append("    export JELLYFIN_KEY=<key>")

    lines.append("")
    lines.append("Export these to use talaria:")
    lines.append("  export TALARIA_HOST=%s" % args.host_value)
    for name in ("sonarr", "radarr", "lidarr", "prowlarr"):
        if name in keys:
            lines.append("  export %s=%s"
                         % (core.SERVICES[name].key_env, keys[name]))
    core.output(args, {"host": args.host_value, "found": found,
                       "recovered_keys": sorted(keys)}, "\n".join(lines))


def _discover(args):
    lines = ["Scanning %s for the 7 services…" % args.host_value]
    found = []
    for name in core.SERVICES:
        ok, payload = _probe(name, args.host_value, args.timeout)
        if ok:
            found.append(name)
            if name == "jellyfin" and isinstance(payload, dict):
                lines.append("  ✓ jellyfin — http://%s:%d (%s %s)"
                             % (args.host_value, core.port_for(name, None),
                                payload.get("ProductName")
                                or payload.get("productName") or "Jellyfin",
                                payload.get("Version")
                                or payload.get("version") or "?"))
            elif isinstance(payload, dict) and payload.get("apiKey"):
                lines.append("  ✓ %s — http://%s:%d (API key available "
                             "via %s)" % (name, args.host_value,
                                          core.port_for(name, None),
                                          PROBE_PATHS[name]))
            else:
                lines.append("  ✓ %s — http://%s:%d (auth required)"
                             % (name, args.host_value,
                                core.port_for(name, None)))
        else:
            lines.append("  ✗ %s — not detected on port %d"
                         % (name, core.port_for(name, None)))
    lines.append("Found %d service(s)" % len(found))
    if found:
        lines.append("Next: python3 -m talaria setup %s" % args.host_value)
    text = "\n".join(lines)
    if not found:
        core.output(args, {"found": []}, text)
        return
    core.output(args, {"found": found, "host": args.host_value}, text)


def _diagnose(args):
    lines = ["Diagnostics:"]
    payload = {}
    for app in ("radarr", "sonarr"):
        if not os.environ.get(core.SERVICES[app].key_env):
            continue
        lines.append("%s:" % app)
        sect = {}
        try:
            queue = core.api_get(app, "/queue", timeout=args.timeout,
                                 host_override=args.host)
            records = queue.get("records") or []
            warnings = [r for r in records if r.get("status")
                        in ("warning", "failed")]
            sect["queue_warnings"] = len(warnings)
            if warnings:
                for w in warnings:
                    msgs = (w.get("statusMessages") or [{}])[0]
                    lines.append("  ⚠ queue: %s — %s"
                                 % (w.get("title"),
                                    (msgs.get("messages")
                                     or [w.get("status")])[0]))
            else:
                lines.append("  ✓ queue clean")
        except core.TalariaError as exc:
            sect["queue_error"] = str(exc)
            lines.append("  ! queue unavailable: %s" % exc)
        try:
            history = core.api_get(app, "/history",
                                   params={"pageSize": 20, "eventType": 3},
                                   timeout=args.timeout,
                                   host_override=args.host)
            failures = [r for r in (history.get("records") or [])
                        if r.get("eventType") in (3, "downloadFailed")]
            sect["recent_failures"] = len(failures)
            if failures:
                for f in failures[:5]:
                    lines.append("  ✗ failed import: %s"
                                 % (f.get("sourceTitle") or "?"))
            else:
                lines.append("  ✓ no recent failed imports")
        except core.TalariaError as exc:
            sect["history_error"] = str(exc)
            lines.append("  ! history unavailable: %s" % exc)
        try:
            folders = core.api_get(app, "/rootfolder", timeout=args.timeout,
                                   host_override=args.host)
            for f in folders:
                free = int((f.get("freeSpace") or 0) / 1073741824)
                lines.append("  · %s — %d GB free" % (f.get("path"), free))
            sect["rootfolders"] = len(folders)
        except core.TalariaError as exc:
            sect["rootfolder_error"] = str(exc)
        payload[app] = sect
    lines.append("")
    lines.append("If imports stall: check remote path mappings "
                 "(Settings → Download Clients), the download-client "
                 "category, and indexer health in Prowlarr.")
    core.output(args, payload, "\n".join(lines))


def _queue(args):
    lines = []
    payload = {}
    for app in ("radarr", "sonarr", "lidarr"):
        if not os.environ.get(core.SERVICES[app].key_env):
            continue
        try:
            data = core.api_get(app, "/queue", timeout=args.timeout,
                                host_override=args.host)
        except core.TalariaError as exc:
            lines.append("%s: unavailable (%s)" % (app, exc))
            continue
        records = data.get("records") if isinstance(data, dict) else data
        records = records or []
        payload[app] = len(records)
        lines.append("%s queue: %d item(s)" % (app, len(records)))
        lines.extend("  • %s — %s (%.0f MB left)"
                     % (r.get("title"), r.get("status"),
                        (r.get("sizeleft") or 0) / 1048576)
                     for r in records)
    if not lines:
        lines.append("No API keys configured. Export SONARR_KEY, "
                     "RADARR_KEY and/or LIDARR_KEY.")
    core.output(args, payload, "\n".join(lines))


def register(group):
    s = group.add_subparsers(dest="sub", metavar="<sub>")
    p = s.add_parser("status", help="Health badges for the 7 services "
                    "[per-service /health · /System/Info]")
    p.set_defaults(func=_status)
    p = s.add_parser("setup", help="Probe a host, recover arr keys via "
                    "public /initialize.json, print the export block "
                    "[GET /initialize.json · /System/Info]")
    p.add_argument("host_value", metavar="host")
    p.set_defaults(func=_setup)
    p = s.add_parser("discover", help="Port-scan a host for the 7 services "
                    "jellyfin via GET /System/Info/Public (no auth)")
    p.add_argument("host_value", metavar="host")
    p.set_defaults(func=_discover)
    p = s.add_parser("diagnose", help="Read-only troubleshooting heuristics "
                    "[GET /queue · /history?eventType=3 · /rootfolder]")
    p.set_defaults(func=_diagnose)
    p = s.add_parser("queue", help="Cross-app download queue snapshot "
                    "sonarr+radarr+lidarr (P7) [GET /queue]")
    p.set_defaults(func=_queue)


def register_promoted(top_subparsers):
    """Also expose the ops subs at top level. Handlers are shared."""
    for name, help_text in (
            ("status", "alias of: ops status — health badges"),
            ("setup", "alias of: ops setup — probe host + recover keys"),
            ("discover", "alias of: ops discover — port-scan a host"),
            ("diagnose", "alias of: ops diagnose — troubleshooting report"),
            ("queue", "alias of: ops queue — cross-app queue snapshot")):
        p = top_subparsers.add_parser(name, help=help_text)
        if name in ("setup", "discover"):
            p.add_argument("host_value", metavar="host")
        p.set_defaults(func=_HANDLERS[name])


_HANDLERS = {
    "status": _status,
    "setup": _setup,
    "discover": _discover,
    "diagnose": _diagnose,
    "queue": _queue,
}
