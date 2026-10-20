"""dashboard — self-contained dark-theme HTML dashboard.

Panels: 7-service status badges · jellyfin active streams (/Sessions) ·
library counts (/Items/Counts + VirtualFolders) · latest adds
(/Items/Latest) · arr wanted counts. stdlib-only generation via string
templates with inline SVG; math is plain int arithmetic; no fixed-name
temp-dir scratch —
the output path is written directly. Every panel degrades independently:
a down service renders a "down" badge instead of failing the artifact.
This module imports NOTHING besides talaria.core.
"""

import html
import os

from . import core

NAME = "dashboard"
HELP = "Generate a self-contained HTML dashboard"


def _panel(title, body):
    return ('<div class="panel"><h2>%s</h2>%s</div>'
            % (html.escape(title), body))


def _badge(name, state, detail=""):
    color = {"up": "#2ecc71", "issues": "#f39c12", "down": "#e74c3c",
             "no-key": "#7f8c8d"}.get(state, "#7f8c8d")
    return ('<div class="badge" style="border-color:%s"><b>%s</b>'
            '<span>%s</span><small>%s</small></div>'
            % (color, html.escape(name), state, html.escape(detail)))


def _svg_bars(pairs, unit=""):
    """Inline SVG bar chart from [(label, value)] — int math only."""
    if not pairs:
        return "<p class='muted'>no data</p>"
    width, height, pad = 560, 200, 30
    n = len(pairs)
    slot = (width - 2 * pad) // max(n, 1)
    bar_w = max(slot - 18, 6)
    top = max(v for _, v in pairs) or 1
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
             'height="%d" role="img">' % (width, height + 40)]
    for i, (label, value) in enumerate(pairs):
        h = int((height - 2 * pad) * value // max(top, 1))
        x = pad + i * slot
        y = height - pad - h
        parts.append('<rect x="%d" y="%d" width="%d" height="%d" '
                     'fill="#3498db"/>' % (x, y, bar_w, max(h, 1)))
        parts.append('<text x="%d" y="%d" class="tick">%d%s</text>'
                     % (x, y - 6, value, unit))
        parts.append('<text x="%d" y="%d" class="lbl">%s</text>'
                     % (x, height - pad + 16,
                        html.escape(str(label)[:14])))
    parts.append("</svg>")
    return "".join(parts)


def _status_panel(args):
    badges = []
    for name in core.SERVICES:
        if not os.environ.get(core.SERVICES[name].key_env):
            badges.append(_badge(name, "no-key", "no key"))
            continue
        try:
            if name == "jellyfin":
                info = core.api_get(name, "/System/Info",
                                    timeout=args.timeout,
                                    host_override=args.host)
                badges.append(_badge(name, "up", "v%s" % (
                    info.get("Version") or info.get("version") or "?")))
                continue
            health = core.api_get(name, {
                "sonarr": "/health", "radarr": "/health",
                "lidarr": "/health", "prowlarr": "/health",
                "bazarr": "/system/health", "overseerr": "/status",
            }[name], timeout=args.timeout, host_override=args.host)
            issues = [h for h in (health if isinstance(health, list)
                                  else []) if h.get("type") != "info"]
            badges.append(_badge(
                name, "up" if not issues else "issues",
                "healthy" if not issues else "%d issue(s)" % len(issues)))
        except core.TalariaError as exc:
            badges.append(_badge(name, "down", str(exc)[:60]))
    return _panel("Services", "".join(badges))


def _streams_panel(args):
    try:
        sessions = core.api_get("jellyfin", "/Sessions",
                                timeout=args.timeout,
                                host_override=args.host)
    except core.TalariaError as exc:
        return _panel("Active streams", "<p class='muted'>jellyfin: %s</p>"
                      % html.escape(str(exc)[:80]))
    active = [s for s in sessions
              if s.get("NowPlayingItem") or s.get("nowPlayingItem")]
    rows = []
    for s in active:
        item = s.get("NowPlayingItem") or s.get("nowPlayingItem") or {}
        user = s.get("UserName") or s.get("userName") or "?"
        rows.append("<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (html.escape(str(user)),
                       html.escape(str(item.get("Name")
                                      or item.get("name") or "?")),
                       html.escape(str(item.get("Type")
                                      or item.get("type") or "?"))))
    body = ("<p class='muted'>%d active stream(s) of %d session(s)</p>"
            % (len(active), len(sessions)))
    if rows:
        body += ("<table><tr><th>User</th><th>Item</th><th>Type</th></tr>"
                 + "".join(rows) + "</table>")
    return _panel("Active streams", body)


def _counts_panel(args):
    try:
        counts = core.api_get("jellyfin", "/Items/Counts",
                              timeout=args.timeout, host_override=args.host)
    except core.TalariaError as exc:
        return _panel("Jellyfin library", "<p class='muted'>%s</p>"
                      % html.escape(str(exc)[:80]))
    # Live 17/10 (v12.2.0): ItemCounts arrives with schema-native
    # PascalCase keys under the default application/json profile;
    # camelCase only via Accept-profile negotiation we never send.
    # Read PascalCase-first with a camelCase fallback — same as
    # jellyfin._counts (this module imports nothing besides core).
    pairs = []
    for k in ("movieCount", "seriesCount", "episodeCount",
              "artistCount", "albumCount", "songCount"):
        v = counts.get(k[0].upper() + k[1:], counts.get(k))
        if v:
            pairs.append((k, v))
    try:
        libs = core.api_get("jellyfin", "/Library/VirtualFolders",
                            timeout=args.timeout, host_override=args.host)
        lib_names = ", ".join("%s (%s)" % (
            l.get("Name") or l.get("name") or "?",
            l.get("CollectionType") or l.get("collectionType") or "-")
            for l in libs)
    except core.TalariaError:
        lib_names = "libraries unavailable"
    return _panel("Jellyfin library",
                  _svg_bars([(k, v) for k, v in pairs])
                  + "<p class='muted'>%s</p>" % html.escape(lib_names))


def _latest_panel(args):
    try:
        # Live 12.2.0 (17/10): /Items/Latest requires userId — resolve the
        # first user, same default as `jellyfin latest`.
        uid = None
        try:
            users = core.api_get("jellyfin", "/Users", timeout=args.timeout,
                                 host_override=args.host)
            for u in users if isinstance(users, list) else []:
                if isinstance(u, dict) and (u.get("Id") or u.get("id")):
                    uid = u.get("Id") or u.get("id")
                    break
        except core.TalariaError:
            pass                            # /Users down -> let Latest say it
        params = {"includeItemTypes": "Movie,Episode",
                  "limit": 10, "enableImages": "false"}
        if uid:
            params["userId"] = uid
        items = core.api_get("jellyfin", "/Items/Latest",
                             params=params,
                             timeout=args.timeout, host_override=args.host)
    except core.TalariaError as exc:
        return _panel("Latest additions", "<p class='muted'>%s</p>"
                      % html.escape(str(exc)[:80]))
    rows = ["<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (html.escape(str(i.get("Name") or i.get("name") or "?")),
               html.escape(str(i.get("Type") or i.get("type") or "?")),
               html.escape(str(i.get("ProductionYear")
                               or i.get("productionYear") or "")))
            for i in items]
    body = ("<table><tr><th>Item</th><th>Type</th><th>Year</th></tr>"
            + "".join(rows) + "</table>") if rows else \
        "<p class='muted'>nothing yet</p>"
    return _panel("Latest additions", body)


def _wanted_panel(args):
    cells = []
    for app in ("sonarr", "radarr"):
        if not os.environ.get(core.SERVICES[app].key_env):
            cells.append("<div class='badge'><b>%s</b><span>no key"
                         "</span><small></small></div>" % app)
            continue
        try:
            data = core.api_get(app, "/wanted/missing",
                                params={"pageSize": 1, "sortKey": "title"},
                                timeout=args.timeout, host_override=args.host)
            total = data.get("totalRecords", len(data.get("records")
                                                 or []))
            cells.append("<div class='badge' style='border-color:#e74c3c'>"
                         "<b>%s</b><span>wanted</span><small>%s</small>"
                         "</div>" % (app, total))
        except core.TalariaError as exc:
            cells.append("<div class='badge' style='border-color:#e74c3c'>"
                         "<b>%s</b><span>unavailable</span><small>%s"
                         "</small></div>" % (app, html.escape(str(exc)[:40])))
    return _panel("Wanted (arr)", "".join(cells))


_STYLE = """
body{background:#111518;color:#d8dee4;font-family:system-ui,sans-serif;
margin:0;padding:24px}
h1{font-size:20px;margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,
1fr));gap:16px}
.panel{background:#1a2026;border:1px solid #2a323a;border-radius:10px;
padding:14px}
h2{font-size:14px;margin:0 0 10px;color:#9fb3c8;text-transform:uppercase}
.badge{display:inline-block;border:2px solid;border-radius:8px;
padding:6px 10px;margin:3px;min-width:86px}
.badge b{display:block;font-size:13px}
.badge span{font-size:11px;color:#9fb3c8}
.badge small{font-size:10px;color:#7f8c8d}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{padding:3px 6px;border-bottom:1px solid #2a323a;text-align:left}
.muted{color:#7f8c8d;font-size:12px}
.tick{fill:#d8dee4;font-size:11px}
.lbl{fill:#7f8c8d;font-size:10px}
"""


def generate(args):
    panels = [_status_panel(args), _streams_panel(args), _counts_panel(args),
              _latest_panel(args), _wanted_panel(args)]
    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           "<title>talaria dashboard</title><style>%s</style></head>"
           "<body><h1>talaria — media stack dashboard</h1>"
           "<div class='grid'>%s</div></body></html>"
           % (_STYLE, "".join(panels)))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    core.output(args, {"written": os.path.abspath(args.out),
                       "panels": len(panels)},
                "Dashboard written: %s" % os.path.abspath(args.out))


def register_promoted(top_subparsers):
    p = top_subparsers.add_parser(
        NAME, help="Generate a self-contained HTML dashboard "
        "[jellyfin /Sessions · /Items/Counts · /Items/Latest + arr status]")
    p.add_argument("out", metavar="out.html",
                   help="output file path (written directly)")
    p.set_defaults(func=generate)
