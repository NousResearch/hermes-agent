"""macOS user-level probes: browser, comms_work.

Registration only at import time. Shared helpers and registration wrappers live in darwin_apps.
"""
from __future__ import annotations

import collections
import glob
import os
import re
import time

from . import browser_files as bf
from . import browser_files_comms as bf_comms
from . import browser_files_content as bf_content
from .darwin_apps import (BROWSER, COMMS, MAC_EPOCH, _AS, _LIB, _U, _app, _bundles, _day, _ex,
    _fda_meta, _is_op, _isdir, _ls, _memo, _mirror, _mp, _newest_mtime, _plist, _read_json)

# ================================================================== browser

_CHROMIUM = [("chrome", ("Google", "Chrome")), ("chrome_beta", ("Google", "Chrome Beta")),
             ("chrome_canary", ("Google", "Chrome Canary")), ("chromium", ("Chromium",)),
             ("brave", ("BraveSoftware", "Brave-Browser")), ("brave_beta", ("BraveSoftware", "Brave-Browser-Beta")),
             ("brave_nightly", ("BraveSoftware", "Brave-Browser-Nightly")), ("edge", ("Microsoft Edge",)),
             ("edge_beta", ("Microsoft Edge Beta",)), ("edge_dev", ("Microsoft Edge Dev",)), ("vivaldi", ("Vivaldi",)),
             ("opera", ("com.operasoftware.Opera",)), ("arc", ("Arc", "User Data")), ("dia", ("Dia", "User Data")),
             ("comet", ("Comet",)), ("helium", ("net.imput.helium",)), ("thorium", ("Thorium",)),
             ("aside", ("Aside",)), ("atlas", ("com.openai.atlas", "browser-data", "host")), ("sidekick", ("Sidekick",)),
             ("yandex", ("Yandex", "YandexBrowser"))]


def _gecko_roots(h):
    return [("firefox", _AS(h, "Firefox")), ("zen", _AS(h, "zen")), ("librewolf", _AS(h, "librewolf")),
            ("floorp", _AS(h, "Floorp")), ("waterfox", _AS(h, "Waterfox")), ("tor", _AS(h, "TorBrowser-Data"))]


def _catalog_mac(h):
    found_c, found_g = {}, {}
    n = 0
    for bid, rel in _CHROMIUM:
        n += 1
        p = _AS(h, *rel)
        if _isdir(p) and os.path.exists(os.path.join(p, "Local State")):
            found_c[bid] = p
    for bid, root in _gecko_roots(h):
        n += 1
        ok = _isdir(root) if bid == "tor" else os.path.isfile(os.path.join(root, "profiles.ini"))
        if ok:
            found_g[bid] = root
    return {"chromium": found_c, "gecko": found_g, "checked": n}


def _prime_browser(h):
    bf._cached(h, "catalog", _catalog_mac)


SAFARI_HISTORY = ("Library", "Safari", "History.db")


@_mp("browser.catalog", level="L1", family=BROWSER, tier="T0", collect="core")
def browser_catalog(h, facts):
    """Chromium (~/Library/Application Support: Chrome, Brave, Edge, Arc, Dia, Vivaldi, Opera, ...), Gecko and Safari roots."""
    c = bf._cached(h, "catalog", _catalog_mac)
    saf = _fda_meta(os.path.join(_U(h), *SAFARI_HISTORY))
    return {"present": bool(c["chromium"] or c["gecko"] or saf.get("present")), "chromium": sorted(c["chromium"]),
            "gecko": sorted(c["gecko"]), "safari_history": saf, "checked": c["checked"] + 1}


@_mp("browser.brave.present", level="L1", family=BROWSER, tier="T0", collect="core")
def brave_present(h, facts):
    """Brave user-data dir exists."""
    p = _AS(h, "BraveSoftware", "Brave-Browser")
    if not _isdir(p):
        return {"present": False}
    return {"present": True, "local_state_mtime": bf._iso(h.meta(os.path.join(p, "Local State")).get("mtime"))}


@_mp("browser.edge.present", level="L1", family=BROWSER, tier="T0", collect="core")
def edge_present(h, facts):
    """Microsoft Edge user-data dir with a Local State (on macOS Edge is never preinstalled)."""
    p = _AS(h, "Microsoft Edge")
    if not os.path.exists(os.path.join(p, "Local State")):
        return {"present": False, "dir_only": _isdir(p)}
    return {"present": True, "local_state_mtime": bf._iso(h.meta(os.path.join(p, "Local State")).get("mtime"))}


def _ls_handlers(h):
    """LaunchServices handler table: {scheme or content type: bundle id} from launchservices.secure.plist."""
    def build():
        d = _plist(_LIB(h, "Preferences", "com.apple.LaunchServices", "com.apple.launchservices.secure.plist")) or {}
        out = {}
        for e in d.get("LSHandlers", []) if isinstance(d, dict) else []:
            if not isinstance(e, dict):
                continue
            key = e.get("LSHandlerURLScheme") or e.get("LSHandlerContentType")
            val = e.get("LSHandlerRoleAll") or e.get("LSHandlerRoleViewer") or e.get("LSHandlerRoleEditor")
            if key and val and key not in out:
                out[str(key).lower()] = str(val)
        return out
    return _memo(h, "ls_handlers", build)


_BROWSER_IDS = [("brave", r"com\.brave\.browser"), ("chrome", r"com\.google\.chrome"), ("chromium", r"org\.chromium"),
                ("firefox", r"org\.mozilla\.firefox"), ("edge", r"com\.microsoft\.edgemac"), ("vivaldi", r"vivaldi"),
                ("opera", r"operasoftware"), ("arc", r"company\.thebrowser\.browser"), ("dia", r"company\.thebrowser\.dia"),
                ("safari", r"com\.apple\.safari"), ("zen", r"zen"), ("comet", r"perplexity|comet"), ("aside", r"aside"),
                ("atlas", r"com\.openai\.atlas"), ("orion", r"kagi"), ("helium", r"helium")]


def _browser_of(bid):
    d = (bid or "").lower()
    return next((b for b, rx in _BROWSER_IDS if re.search(rx, d)), bid)


@_mp("browser.default", level="L1", family=BROWSER, tier="T0", collect="core")
def browser_default(h, facts):
    """Default browser from the LaunchServices handler for https/http (Safari when no user handler is set)."""
    m = _ls_handlers(h)
    pid = m.get("https") or m.get("http") or m.get("public.html")
    return {"present": True, "browser": _browser_of(pid) if pid else "safari", "prog_id": pid or "com.apple.safari",
            "set_by_user": bool(pid), "handlers": {k: _browser_of(m[k]) for k in ("https", "http", "public.html") if k in m}}


@_mp("defaults.mailto_pdf_media", level="L1", family=BROWSER, tier="T0", collect="core")
def defaults_mailto_pdf_media(h, facts):
    """User-set default handlers (bundle ids) for mailto, PDF, video, audio, images, text, source code."""
    m = _ls_handlers(h)
    keys = {"mailto": "mailto", ".pdf": "com.adobe.pdf", ".mp4": "public.mpeg-4", ".mkv": "org.matroska.mkv",
            ".mp3": "public.mp3", ".jpg": "public.jpeg", ".png": "public.png", ".txt": "public.plain-text",
            ".md": "net.daringfireball.markdown", ".py": "public.python-script", "source": "public.source-code",
            "ssh": "ssh", "vscode": "vscode"}
    got = {k: m[v] for k, v in keys.items() if v in m}
    return {"present": bool(got), "handlers": got, "user_handlers_total": len(m)}


@_mp("browser.registered", level="L2", family=BROWSER, tier="T0", collect="core", gate="apps.uninstall")
def browser_registered(h, facts):
    """Installed browser app bundles (by bundle id)."""
    names = sorted({_browser_of(b["bundle_id"]) for b in _bundles(h)
                    if b["bundle_id"] and _browser_of(b["bundle_id"]) != b["bundle_id"]})
    return {"present": bool(names), "browsers": names}


_T3_CHROMIUM = ("Cookies", "Network/Cookies", "Login Data", "Login Data For Account", "Web Data", "Account Web Data")
_T3_GECKO = ("cookies.sqlite", "logins.json", "key4.db", "formhistory.sqlite")


@_mp("browser.t3_presence", level="L1", family=BROWSER, tier="T3", collect="core")
def browser_t3_presence(h, facts):
    """Credential/cookie stores per profile (Chromium, Gecko, Safari): stat only (size), never opened."""
    _prime_browser(h)
    rows = {}
    for bid, ud, pdir, _i in bf._profiles(h):
        base = os.path.join(ud, pdir) if pdir else ud
        sizes = {f.replace("Network/", ""): m["bytes"] for f in _T3_CHROMIUM
                 for m in [h.meta(os.path.join(base, f))] if m.get("present")}
        if sizes:
            rows[bf._pkey(bid, pdir)] = sizes
    for bid, root in bf._cached(h, "catalog", _catalog_mac)["gecko"].items():
        for pd in glob.glob(os.path.join(root, "Profiles", "*"))[:30]:
            sizes = {f: m["bytes"] for f in _T3_GECKO for m in [h.meta(os.path.join(pd, f))] if m.get("present")}
            if sizes:
                rows[f"{bid}:{len(rows)}"] = sizes
    saf = {f: m["bytes"] for f, p in (("Cookies.binarycookies", _LIB(h, "Containers", "com.apple.Safari", "Data", "Library",
                                                                    "Cookies", "Cookies.binarycookies")),
                                      ("Cookies.binarycookies_legacy", _LIB(h, "Cookies", "Cookies.binarycookies")))
           for m in [h.meta(p)] if m.get("present")}
    if saf:
        rows["safari"] = saf
    return {"present": bool(rows), "profiles": rows}


@_mp("browser.generic_sweep", level="L1", family=BROWSER, tier="T0", collect="extended")
def browser_generic_sweep(h, facts):
    """Chromium-style user-data dirs under ~/Library/Application Support not in the catalog (Local State + Default
    profile). Electron apps have a Local State but no Default profile and are not counted."""
    known = {os.path.realpath(p) for p in bf._cached(h, "catalog", _catalog_mac)["chromium"].values()}
    unknown = []
    for pat in (_AS(h, "*", "Local State"), _AS(h, "*", "*", "Local State"), _AS(h, "*", "*", "*", "Local State")):
        for p in glob.glob(pat)[:300]:
            d = os.path.dirname(p)
            if os.path.realpath(d) not in known and os.path.isfile(os.path.join(d, "Default", "History")) \
                    and _isdir(os.path.join(d, "Default", "Extensions")) and not _is_op(h, d) \
                    and not re.search(r"/Caches/|/htmlcache", d):
                unknown.append(os.path.relpath(d, _U(h)))
    return {"present": True, "unknown_forks": len(unknown), "roots": unknown[:10]}


for _id, _fn in (("browser.local_state", bf.browser_local_state), ("browser.tab_stats", bf.browser_tab_stats),
                 ("browser.profile_prefs", bf.browser_profile_prefs), ("browser.extensions", bf.browser_extensions),
                 ("browser.bookmarks", bf.browser_bookmarks), ("browser.history.copy", bf.history_copy),
                 ("browser.history.stats", bf.history_stats), ("browser.history.hour_weekday", bf.history_hour_weekday),
                 ("browser.history.top_domains", bf.history_top_domains),
                 ("browser.history.top_domains_named", bf.history_top_domains_named),
                 ("browser.history.localhost", bf.history_localhost), ("browser.history.downloads", bf.history_downloads),
                 ("browser.history.search_terms", bf.history_search_terms),
                 ("browser.history.github_repos", bf.history_github_repos), ("browser.firefox.history", bf.firefox_history),
                 ("l3.browser_category_share", bf.l3_browser_category_share), ("l3.primary_browser", bf.l3_primary_browser),
                 ("l3.browser_rhythm", bf.l3_browser_rhythm), ("l3.browser_persona_hints", bf.l3_browser_persona_hints),
                 ("browser.history.workspaces", bf_comms.history_workspaces), ("browser.history.work_hosts", bf_comms.history_work_hosts)):
    _mirror(_id, _fn, _prime_browser)


@_mp("browser.safari.history", level="L2", family=BROWSER, tier="T1", collect="extended", gate="browser.catalog",
     timeout_ms=5000)
def safari_history(h, facts):
    """Safari History.db aggregates: visits, URLs, first/last visit, active days, hour histogram, domain-category
    shares (sensitive categories folded). No URLs, titles or domains are emitted. Needs Full Disk Access."""
    p = os.path.join(_U(h), *SAFARI_HISTORY)
    m = _fda_meta(p)
    if not m.get("present"):
        return {"present": False}
    if not m.get("readable"):
        return {"present": False, "needs_fda": True, "bytes": m.get("bytes")}
    rows = h.sqlite(p, "select v.visit_time, i.url from history_visits v join history_items i on i.id = v.history_item",
                    timeout_ms=4000)
    if rows is None:
        return {"present": False, "error": "query_failed"}
    urls = h.sqlite(p, "select count(*) from history_items") or [[0]]
    cats, days, hours = collections.Counter(), collections.Counter(), [0] * 24
    ts = []
    for vt, url in rows:
        if vt is None:
            continue
        t = vt + MAC_EPOCH
        ts.append(t)
        lt = time.localtime(t)
        days[time.strftime("%Y-%m-%d", lt)] += 1
        hours[lt.tm_hour] += 1
        host, _ = bf._host_of(url or "")
        c = bf._categorize(bf_content._reg_domain(host)) if host else "other"
        cats["sensitive" if c in bf.SENSITIVE_CATS else c] += 1
    if not ts:
        return {"present": False}
    tot = sum(cats.values()) or 1
    return {"present": True, "visits": len(ts), "urls": urls[0][0], "first_visit": bf._iso(min(ts)),
            "last_visit": bf._iso(max(ts)), "active_days": len(days),
            "visits_per_active_day": round(len(ts) / len(days), 1), "hours": hours,
            "category_share": {k: round(v / tot, 3) for k, v in cats.most_common()}, "via": h.last_via}


# ================================================================== comms_work

def _app_row(h, apps, data_dirs, activity=()):
    """installed/launched/last-used for one app. activity = files whose mtime means the app ran."""
    b = _app(h, *apps) if apps else None
    data = [p for p in data_dirs if _isdir(p)]
    if not (b or data):
        return None
    last = _newest_mtime([a for a in activity if a] + data)
    return {"installed": bool(b), "version": (b or {}).get("version"), "launched": bool(data), "last_used": _day(last),
            "recent_use": bool(last and time.time() - last < 30 * 86400)}


def _comms_cands(h):
    return {
        "slack": (["Slack"], [_AS(h, "Slack")], [_AS(h, "Slack", "logs"), _AS(h, "Slack", "Local Storage")]),
        "telegram": (["Telegram", "Telegram Desktop"], [_LIB(h, "Group Containers", "6N38VWS5BX.ru.keepcoder.Telegram"),
                                                        _AS(h, "Telegram Desktop")], []),
        "signal": (["Signal"], [_AS(h, "Signal")], [_AS(h, "Signal", "logs")]),
        "whatsapp": (["WhatsApp"], [_LIB(h, "Group Containers", "group.net.whatsapp.WhatsApp.shared")], []),
        "zoom": (["zoom.us"], [_AS(h, "zoom.us")], [_AS(h, "zoom.us", "data")]),
        "teams": (["Microsoft Teams", "Microsoft Teams (work or school)"], [_LIB(h, "Containers", "com.microsoft.teams2")], []),
        "element": (["Element"], [_AS(h, "Element")], []),
        "beeper": (["Beeper Desktop", "Beeper"], [_AS(h, "BeeperTexts")], []),
        "outlook": (["Microsoft Outlook"], [_LIB(h, "Group Containers", "UBF8T346G9.Office", "Outlook")], []),
        "apple_mail": (["Mail"], [_LIB(h, "Mail")], []),
        "messages": ([], [_LIB(h, "Messages")], []),
        "facetime": (["FaceTime"], [_AS(h, "FaceTime")], []),
        "spark": (["Spark", "Spark Desktop"], [_LIB(h, "Group Containers", "3L68KQB4HG.group.com.readdle.smartemail")], []),
        "superhuman": (["Superhuman"], [_AS(h, "Superhuman")], []),
        "webex": (["Webex"], [_AS(h, "Cisco Spark")], []),
        "around": (["Around"], [_AS(h, "Around")], []),
        "linear": (["Linear"], [_AS(h, "Linear")], []),
        "mattermost": (["Mattermost"], [_AS(h, "Mattermost")], []),
        "skype": (["Skype"], [_AS(h, "Microsoft", "Skype for Desktop")], []),
    }


@_mp("comms.native_apps", level="L1", family=COMMS, tier="T0", collect="core")
def comms_native_apps(h, facts):
    """Native comms/mail apps: installed (bundle), launched (data dir exists), last-used day from data-dir mtimes."""
    rows = {}
    for app, (inst, data, act) in _comms_cands(h).items():
        r = _app_row(h, inst, data, act)
        if r:
            rows[app] = r
    if not rows:
        return {"present": False}
    return {"present": True, **rows}


@_mp("comms.teams_launched", level="L2", family=COMMS, tier="T0", collect="core", gate="comms.native_apps")
def comms_teams_launched(h, facts):
    """New Teams for Mac actually started: com.microsoft.teams2 container with WebView profile dirs; tfw = work,
    tfl = personal."""
    b = _LIB(h, "Containers", "com.microsoft.teams2", "Data", "Library", "Application Support", "Microsoft", "MSTeams")
    if not _isdir(b):
        return {"present": False}
    prof = h.list_dir(os.path.join(b, "EBWebView"), 100) + h.list_dir(os.path.join(b, "WebView"), 100)
    work = sum(1 for p in prof if "tfw" in p.lower())
    return {"present": True, "work_profile": work > 0, "work_profiles": work,
            "personal_profiles": sum(1 for p in prof if "tfl" in p.lower()),
            "log_files": max(h.count_dir(os.path.join(b, "Logs"), 5000), 0)}


@_mp("office.c2r", level="L1", family=COMMS, tier="T0", collect="core")
def office_c2r(h, facts):
    """Microsoft 365 for Mac: Word/Excel/PowerPoint/Outlook/OneNote bundles and versions (licence kind is not
    readable without opening the Office licensing store)."""
    rows = {}
    for n in ("Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint", "Microsoft Outlook", "Microsoft OneNote"):
        b = _app(h, n)
        if b:
            rows[n.split()[-1].lower()] = b["version"]
    if not rows:
        return {"present": False}
    return {"present": True, "products": ",".join(sorted(rows)), "versions": rows, "platform": "mac",
            "mas": any(b["mas"] for b in _bundles(h) if b["name"].startswith("Microsoft ")), "license_kind": None}


def _discord_dirs(h):
    return [d for d in (_AS(h, "discord"), _AS(h, "discordcanary"), _AS(h, "discordptb")) if _isdir(d)]


@_mp("discord.present", level="L1", family=COMMS, tier="T0", collect="core")
def discord_present(h, facts):
    """Discord desktop: app bundle, profile dir, version dir."""
    b = _app(h, "Discord", "Discord Canary", "Discord PTB")
    dirs = _discord_dirs(h)
    if not (b or dirs):
        return {"present": False}
    vers = sorted(n for d in dirs for n in (_ls(d) or []) if re.fullmatch(r"\d+\.\d+\.\d+", n))
    return {"present": True, "installed": bool(b), "profile_dir": bool(dirs),
            "version": (b or {}).get("version") or (vers[-1] if vers else None)}


@_mp("discord.usage", level="L2", family=COMMS, tier="T1", collect="extended", gate="discord.present", timeout_ms=3000)
def discord_usage(h, facts):
    """Signed-in and voice days from logs/renderer_js.log tags; last use from Local Storage mtime."""
    dirs = _discord_dirs(h)
    if not dirs:
        return {"present": False}
    days, gw, rtc = set(), set(), set()
    size = 0
    rx = re.compile(r"^\[(\d{4}-\d\d-\d\d) ([\d:.]+)\] \[\w+\]\s+\[?([A-Za-z_]+)")
    for d in dirs:
        for log in glob.glob(os.path.join(d, "logs", "renderer_js*.log"))[:5]:
            size += os.path.getsize(log)
            with open(log, encoding="utf-8", errors="replace") as fh:
                if os.path.getsize(log) > 16_000_000:
                    fh.seek(os.path.getsize(log) - 16_000_000)
                for line in fh:
                    m = rx.match(line)
                    if not m:
                        continue
                    dd, _tm, tag = m.groups()
                    days.add(dd)
                    if tag == "GatewaySocket":
                        gw.add(dd)
                    elif tag.startswith("RTC"):
                        rtc.add(dd)
    ls = _newest_mtime([os.path.join(d, "Local Storage", "leveldb") for d in dirs])
    used = bool(gw or rtc) or bool(ls and time.time() - ls < 90 * 86400)
    return {"present": used, "log_days": len(days), "first_day": min(days) if days else None,
            "last_day": max(days) if days else None, "gateway_days": len(gw), "rtc_days": len(rtc),
            "local_storage_last": _day(ls), "log_bytes": size}


MESSAGES_DB = ("Library", "Messages", "chat.db")
NOTES_DB = ("Library", "Group Containers", "group.com.apple.notes", "NoteStore.sqlite")


@_mp("comms.imessage", level="L2", family=COMMS, tier="T1", collect="extended", gate="comms.native_apps", timeout_ms=5000)
def comms_imessage(h, facts):
    """Messages chat.db record counts only: messages, chats, sent share, by service, first/last date, per-month
    counts. No text, no handles, no attachments. Needs Full Disk Access."""
    p = os.path.join(_U(h), *MESSAGES_DB)
    m = _fda_meta(p)
    if not m.get("present"):
        return {"present": False}
    if not m.get("readable"):
        return {"present": False, "needs_fda": True, "bytes": m.get("bytes")}
    r = h.sqlite(p, "select count(*), sum(is_from_me), min(date), max(date) from message", timeout_ms=4000)
    if not r:
        return {"present": False, "error": "query_failed"}
    n, mine, a, z = r[0]
    scale = 1e9 if (z or 0) > 1e12 else 1
    chats = (h.sqlite(p, "select count(*) from chat") or [[None]])[0][0]
    svc = dict(h.sqlite(p, "select coalesce(service,'?'), count(*) from message group by 1") or [])
    cut = (time.time() - MAC_EPOCH - 30 * 86400) * scale
    n30 = (h.sqlite(p, "select count(*) from message where date >= ?", (cut,)) or [[None]])[0][0]
    months = dict(h.sqlite(p, f"select strftime('%Y-%m', date / {scale} + {MAC_EPOCH}, 'unixepoch'), count(*) "
                              "from message group by 1") or [])
    return {"present": True, "messages": n, "sent_share": round((mine or 0) / n, 3) if n else None, "chats": chats,
            "by_service": svc, "messages_30d": n30, "first": bf._iso(a / scale + MAC_EPOCH) if a else None,
            "last": bf._iso(z / scale + MAC_EPOCH) if z else None, "by_month": dict(sorted(months.items())[-24:]),
            "via": h.last_via}


def _notes_cands(h):
    U = _U(h)
    return {
        "obsidian": (["Obsidian"], [_AS(h, "obsidian")], [_AS(h, "obsidian", "obsidian.json")]),
        "notion": (["Notion"], [_AS(h, "Notion")], []),
        "logseq": (["Logseq"], [os.path.join(U, ".logseq"), _AS(h, "Logseq")], []),
        "apple_notes": (["Notes"], [_LIB(h, "Group Containers", "group.com.apple.notes")], []),
        "bear": (["Bear"], [_LIB(h, "Group Containers", "9K33E3U3T4.net.shinyfrog.bear")], []),
        "craft": (["Craft"], [_LIB(h, "Containers", "com.lukilabs.lukiapp")], []),
        "joplin": (["Joplin"], [os.path.join(U, ".config", "joplin-desktop")], []),
        "evernote": (["Evernote"], [_LIB(h, "Containers", "com.evernote.Evernote")], []),
        "todoist": (["Todoist"], [_AS(h, "Todoist")], []),
        "zotero": (["Zotero"], [_AS(h, "Zotero"), os.path.join(U, "Zotero")], []),
        "granola": (["Granola"], [_AS(h, "Granola")], []),
        "anytype": (["Anytype"], [_AS(h, "anytype")], []),
    }


@_mp("productivity.notes_apps", level="L1", family=COMMS, tier="T0", collect="core")
def notes_apps(h, facts):
    """Notes apps installed or launched: Obsidian, Notion, Logseq, Apple Notes, Bear, Craft, Zotero, ..."""
    rows = {}
    for app, (inst, data, act) in _notes_cands(h).items():
        r = _app_row(h, inst, data, act)
        if r:
            rows[app] = r
    return {"present": bool(rows), "apps": sorted(rows), "detail": rows}


@_mp("obsidian.vaults", level="L2", family=COMMS, tier="T1", collect="extended", gate="productivity.notes_apps")
def obsidian_vaults(h, facts):
    """Obsidian vault count and note counts (vault names and paths are not emitted)."""
    j = _read_json(_AS(h, "obsidian", "obsidian.json"))
    vs = ((j or {}).get("vaults") or {}) if isinstance(j, dict) else {}
    if not vs:
        return {"present": False}
    notes, files, newest = 0, 0, 0

    def on_file(e, s, depth):
        nonlocal notes, newest
        if e.name.lower().endswith(".md"):
            notes += 1
        newest = max(newest, s.st_mtime)
    for v in list(vs.values())[:10]:
        p = v.get("path") if isinstance(v, dict) else None
        if p and _isdir(p) and p.startswith(_U(h)):
            st = bf._walk(p, max_depth=6, max_entries=30000, budget_s=1.5, on_file=on_file,
                          skip_dir=lambda n, _p: n in (".obsidian", ".git", ".trash", "node_modules"))
            files += st["files"]
    return {"present": True, "vaults": len(vs), "open": sum(1 for v in vs.values() if isinstance(v, dict) and v.get("open")),
            "notes_md": notes, "files": files, "newest_note": bf._iso(newest) if newest else None}


@_mp("productivity.apple_notes", level="L2", family=COMMS, tier="T1", collect="extended", gate="productivity.notes_apps",
     timeout_ms=4000)
def apple_notes(h, facts):
    """Apple Notes NoteStore.sqlite record counts only: notes, folders, accounts, created/modified range. No titles
    or bodies. Needs Full Disk Access."""
    p = os.path.join(_U(h), *NOTES_DB)
    m = _fda_meta(p)
    if not m.get("present"):
        return {"present": False}
    if not m.get("readable"):
        return {"present": False, "needs_fda": True, "bytes": m.get("bytes")}
    cols = {r[1] for r in (h.sqlite(p, "pragma table_info(ZICCLOUDSYNCINGOBJECT)") or [])}
    if not cols:
        return {"present": False, "error": "schema"}
    crt = "ZCREATIONDATE3" if "ZCREATIONDATE3" in cols else "ZCREATIONDATE1" if "ZCREATIONDATE1" in cols else None
    mod = "ZMODIFICATIONDATE1" if "ZMODIFICATIONDATE1" in cols else None
    note_where = "ZNOTEDATA is not null" if "ZNOTEDATA" in cols else "ZTITLE1 is not null"
    q = f"select count(*), min({crt or 'null'}), max({mod or 'null'}) from ZICCLOUDSYNCINGOBJECT where {note_where}"
    if "ZMARKEDFORDELETION" in cols:
        q += " and coalesce(ZMARKEDFORDELETION, 0) = 0"
    r = h.sqlite(p, q) or [[0, None, None]]
    folders = (h.sqlite(p, "select count(*) from ZICCLOUDSYNCINGOBJECT where ZTITLE2 is not null") or [[None]])[0][0] \
        if "ZTITLE2" in cols else None
    accts = (h.sqlite(p, "select count(*) from ZICCLOUDSYNCINGOBJECT where ZACCOUNTTYPE is not null") or [[None]])[0][0] \
        if "ZACCOUNTTYPE" in cols else None
    n, a, z = r[0]
    n30 = None
    if mod:
        n30 = (h.sqlite(p, f"select count(*) from ZICCLOUDSYNCINGOBJECT where {note_where} and {mod} >= ?",
                        (time.time() - MAC_EPOCH - 30 * 86400,)) or [[None]])[0][0]
    return {"present": bool(n), "notes": n, "folders": folders, "accounts": accts, "modified_30d": n30,
            "first_created": bf._iso(a + MAC_EPOCH) if a else None, "last_modified": bf._iso(z + MAC_EPOCH) if z else None,
            "via": h.last_via}


_EDR = {"crowdstrike": ["/Applications/Falcon.app", "/Library/CS"], "sentinelone": ["/Library/Sentinel", "/Applications/SentinelOne"],
        "defender": ["/Applications/Microsoft Defender.app"], "jamf_protect": ["/Applications/JamfProtect.app"],
        "jamf": ["/usr/local/jamf", "/Library/Application Support/JAMF"], "kandji": ["/Library/Kandji"],
        "mosyle": ["/Library/Application Support/Mosyle"], "addigy": ["/Library/Addigy"],
        "intune_agent": ["/Library/Intune", "/Applications/Company Portal.app"], "zscaler": ["/Applications/Zscaler"],
        "netskope": ["/Library/Application Support/Netskope"], "globalprotect": ["/Applications/GlobalProtect.app"],
        "cisco_secure_client": ["/opt/cisco/secureclient", "/opt/cisco/anyconnect"], "osquery": ["/private/var/osquery", "/opt/osquery"],
        "kolide": ["/usr/local/kolide-k2"], "fleetd": ["/opt/orbit"], "santa": ["/Applications/Santa.app"],
        "sophos": ["/Library/Sophos Anti-Virus"], "carbonblack": ["/Applications/VMware Carbon Black Cloud"],
        "tanium": ["/Library/Tanium"], "lulu": ["/Applications/LuLu.app"], "little_snitch": ["/Applications/Little Snitch.app"],
        "blockblock": ["/Applications/BlockBlock Helper.app"]}


@_mp("work.security_agents", level="L1", family=COMMS, tier="T0", collect="core")
def work_security_agents(h, facts):
    """EDR/MDM/ZTNA agents and personal firewalls by install path (no launchctl queries)."""
    hits = sorted(k for k, ps in _EDR.items() if _ex(*ps))
    personal = {"lulu", "little_snitch", "blockblock"}
    return {"present": True, "agents": [a for a in hits if a not in personal],
            "personal_security_tools": [a for a in hits if a in personal], "checked": len(_EDR)}


@_mp("work.mdm", level="L1", family=COMMS, tier="T0", collect="core")
def work_mdm(h, facts):
    """MDM enrollment: DEP activation record and enrollment markers under /var/db/ConfigurationProfiles, managed
    preference domains under /Library/Managed Preferences (counts only; no `profiles` spawn)."""
    cp = "/private/var/db/ConfigurationProfiles"
    dep = _ex(os.path.join(cp, "Settings", ".cloudConfigHasActivationRecord"), os.path.join(cp, "Settings", ".cloudConfigRecordFound"))
    enrolled = _ex(os.path.join(cp, "Settings", ".profilesAreInstalled"), os.path.join(cp, "Store", "ConfigProfiles.binary"))
    managed = [n for n in (_ls("/Library/Managed Preferences") or []) if n.endswith(".plist")]
    real = ["mdm"] if (enrolled and (dep or managed)) else []
    return {"present": True, "real_enrollments": len(real), "enrollments": real, "dep_activation_record": dep,
            "profiles_marker": enrolled, "managed_pref_domains": len(managed)}


@_mp("work.join_state", level="L1", family=COMMS, tier="T0", collect="core")
def work_join_state(h, facts):
    """Directory binding: Active Directory plugin config, Kerberos SSO / Platform SSO extension presence."""
    ad_cfg = _ex("/Library/Preferences/OpenDirectory/Configurations/Active Directory")
    ad_bound = bool(_ls("/Library/Preferences/OpenDirectory/Configurations/Active Directory"))
    psso = _ex("/Library/Preferences/com.apple.extensiblesso.plist", "/Library/Managed Preferences/com.apple.extensiblesso.plist")
    return {"present": True, "state": "joined" if ad_bound else "workgroup", "domain_joined": ad_bound,
            "ad_plugin_config": ad_cfg, "azure_ad_joined": psso, "platform_sso": psso}


@_mp("work.policies", level="L1", family=COMMS, tier="T0", collect="core")
def work_policies(h, facts):
    """Managed app policy plists (Chrome, Edge, Brave, Firefox, Office) under /Library/Managed Preferences: counts."""
    doms = {"chrome": "com.google.Chrome", "edge": "com.microsoft.Edge", "brave": "com.brave.Browser",
            "firefox": "org.mozilla.firefox", "office": "com.microsoft.office", "safari": "com.apple.Safari"}
    root = "/Library/Managed Preferences"
    names = _ls(root) or []
    user = _ls(os.path.join(root, os.environ.get("USER", ""))) or []
    configured = {k: {"values": sum(1 for n in names + user if n.startswith(d)), "subkeys": 0} for k, d in doms.items()
                  if any(n.startswith(d) for n in names + user)}
    return {"present": True, "app_policies": configured, "app_policy_count": len(configured)}


@_mp("onedrive.accounts", level="L1", family=COMMS, tier="T0", collect="core")
def onedrive_accounts(h, facts):
    """OneDrive for Mac: app bundle and File Provider roots under ~/Library/CloudStorage (OneDrive-Personal /
    OneDrive-TENANT). Folder names are not emitted."""
    b = _app(h, "OneDrive")
    roots = [n for n in (_ls(_LIB(h, "CloudStorage")) or []) if n.startswith("OneDrive")]
    if not (b or roots):
        return {"present": False}
    return {"present": True, "installed": bool(b), "accounts": len(roots), "signed_in": len(roots),
            "business": sum(1 for n in roots if n != "OneDrive-Personal")}
