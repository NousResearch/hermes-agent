"""Linux user-level probes: browser, comms_work.

Registration only at import time. Shared helpers and registration wrappers live in linux_apps.
"""
from __future__ import annotations

import glob
import os
import re
import time

from userscan.specs import browser_files as bf
from userscan.specs import browser_files_comms as bf_comms
from userscan.specs.linux_apps import (BROWSER, COMMS, _U, _cfg, _data, _day, _desktop, _ex, _first, _flat,
    _is_op, _isdir, _lp, _ls, _mirror, _newest_mtime, _open_text, _read_json, _snapu)

# ================================================================== browser

_CHROMIUM = [("chrome", "google-chrome"), ("chrome_beta", "google-chrome-beta"), ("chrome_dev", "google-chrome-unstable"),
             ("chromium", "chromium"), ("brave", "BraveSoftware/Brave-Browser"), ("brave_beta", "BraveSoftware/Brave-Browser-Beta"),
             ("brave_nightly", "BraveSoftware/Brave-Browser-Nightly"), ("edge", "microsoft-edge"), ("edge_beta", "microsoft-edge-beta"),
             ("edge_dev", "microsoft-edge-dev"), ("vivaldi", "vivaldi"), ("opera", "opera"), ("thorium", "thorium"),
             ("yandex", "yandex-browser"), ("helium", "net.imput.helium")]
_CHROMIUM_EXTRA = [("chromium", "snap", ("chromium", "common", "chromium")),
                   ("brave", "snap", ("brave", "current", ".config", "BraveSoftware", "Brave-Browser")),
                   ("chrome", "flatpak", ("com.google.Chrome", "config", "google-chrome")),
                   ("brave", "flatpak", ("com.brave.Browser", "config", "BraveSoftware", "Brave-Browser")),
                   ("chromium", "flatpak", ("org.chromium.Chromium", "config", "chromium")),
                   ("edge", "flatpak", ("com.microsoft.Edge", "config", "microsoft-edge")),
                   ("vivaldi", "flatpak", ("com.vivaldi.Vivaldi", "config", "vivaldi"))]


def _gecko_roots(h):
    U = _U(h)
    return [("firefox", os.path.join(U, ".mozilla", "firefox")), ("firefox", _snapu(h, "firefox", "common", ".mozilla", "firefox")),
            ("firefox", _flat(h, "org.mozilla.firefox", ".mozilla", "firefox")), ("librewolf", os.path.join(U, ".librewolf")),
            ("librewolf", _flat(h, "io.gitlab.librewolf-community", ".librewolf")), ("zen", os.path.join(U, ".zen")),
            ("floorp", os.path.join(U, ".floorp")), ("waterfox", os.path.join(U, ".waterfox")),
            ("thunderbird_gecko", os.path.join(U, ".thunderbird")), ("tor", os.path.join(U, ".local", "share", "torbrowser"))]


def _catalog_lx(h):
    found_c, found_g = {}, {}
    n = 0
    for bid, rel in _CHROMIUM:
        n += 1
        p = _cfg(h, *rel.split("/"))
        if _isdir(p) and os.path.exists(os.path.join(p, "Local State")) and bid not in found_c:
            found_c[bid] = p
    for bid, kind, parts in _CHROMIUM_EXTRA:
        n += 1
        p = _snapu(h, *parts) if kind == "snap" else _flat(h, *parts)
        key = bid if bid not in found_c else f"{bid}_{kind}"
        if _isdir(p) and os.path.exists(os.path.join(p, "Local State")):
            found_c[key] = p
    for bid, root in _gecko_roots(h):
        n += 1
        if bid == "thunderbird_gecko":
            continue
        ok = _isdir(root) if bid == "tor" else os.path.isfile(os.path.join(root, "profiles.ini"))
        if ok:
            found_g[bid if bid not in found_g else f"{bid}_{len(found_g)}"] = root
    return {"chromium": found_c, "gecko": found_g, "checked": n}


def _prime_browser(h):
    bf._cached(h, "catalog", _catalog_lx)


@_lp("browser.catalog", level="L1", family=BROWSER, tier="T0", collect="core")
def browser_catalog(h, facts):
    """Chromium (native/snap/flatpak under ~/.config, ~/snap, ~/.var) and Gecko (~/.mozilla...) user-data roots."""
    c = bf._cached(h, "catalog", _catalog_lx)
    return {"present": bool(c["chromium"] or c["gecko"]), "chromium": sorted(c["chromium"]), "gecko": sorted(c["gecko"]),
            "checked": c["checked"]}


@_lp("browser.brave.present", level="L1", family=BROWSER, tier="T0", collect="core")
def brave_present(h, facts):
    """Brave user-data dir exists (native, snap or flatpak)."""
    for p in (_cfg(h, "BraveSoftware", "Brave-Browser"), _snapu(h, "brave", "current", ".config", "BraveSoftware", "Brave-Browser"),
              _flat(h, "com.brave.Browser", "config", "BraveSoftware", "Brave-Browser")):
        if _isdir(p):
            m = h.meta(os.path.join(p, "Local State"))
            return {"present": True, "local_state_mtime": bf._iso(m.get("mtime"))}
    return {"present": False}


@_lp("browser.edge.present", level="L1", family=BROWSER, tier="T0", collect="core")
def edge_present(h, facts):
    """Microsoft Edge user-data dir exists (native or flatpak); on Linux it is never preinstalled."""
    for p in (_cfg(h, "microsoft-edge"), _flat(h, "com.microsoft.Edge", "config", "microsoft-edge")):
        if _isdir(p):
            return {"present": True, "local_state_mtime": bf._iso(h.meta(os.path.join(p, "Local State")).get("mtime"))}
    return {"present": False}


def _mimeapps(h):
    """Merged [Default Applications] from the XDG mimeapps.list chain (user files first). {mime: (desktop_id, scope)}."""
    files = [("user", _cfg(h, "kde-mimeapps.list")), ("user", _cfg(h, "mimeapps.list")),
             ("user", _data(h, "applications", "mimeapps.list")), ("system", "/etc/xdg/kde-mimeapps.list"),
             ("system", "/etc/xdg/mimeapps.list"), ("system", "/usr/share/applications/kde-mimeapps.list"),
             ("system", "/usr/share/applications/mimeapps.list"), ("system", "/usr/share/applications/defaults.list")]
    out = {}
    for scope, p in files:
        t = _open_text(p, 2_000_000)
        if not t:
            continue
        sec = None
        for line in t.splitlines():
            line = line.strip()
            if line.startswith("["):
                sec = line
                continue
            if sec in ("[Default Applications]",) and "=" in line:
                k, v = line.split("=", 1)
                first = next((x for x in v.split(";") if x), None)
                if first and k.strip() not in out:
                    out[k.strip()] = (first, scope)
    return out


_BROWSER_IDS = [("brave", r"brave"), ("chrome", r"google-chrome|com\.google\.chrome"), ("chromium", r"chromium"),
                ("firefox", r"firefox"), ("edge", r"microsoft-edge"), ("vivaldi", r"vivaldi"), ("opera", r"opera"),
                ("librewolf", r"librewolf"), ("zen", r"zen"), ("floorp", r"floorp"), ("falkon", r"falkon"),
                ("konqueror", r"konqueror"), ("epiphany", r"epiphany")]


def _browser_of(desktop_id):
    d = (desktop_id or "").lower()
    return next((b for b, rx in _BROWSER_IDS if re.search(rx, d)), desktop_id)


@_lp("browser.default", level="L1", family=BROWSER, tier="T0", collect="core")
def browser_default(h, facts):
    """Default browser from mimeapps.list (x-scheme-handler/https, http, text/html) or KDE BrowserApplication."""
    m = _mimeapps(h)
    got = {k: m[k] for k in ("x-scheme-handler/https", "x-scheme-handler/http", "text/html", "application/pdf") if k in m}
    kde = None
    t = _open_text(_cfg(h, "kdeglobals"), 2_000_000) or ""
    km = re.search(r"^BrowserApplication=(.+)$", t, re.M)
    if km:
        kde = km.group(1).strip().lstrip("!")
    pid = (got.get("x-scheme-handler/https") or got.get("x-scheme-handler/http") or got.get("text/html") or (None,))[0] or kde
    if not pid:
        return {"present": False}
    src = got.get("x-scheme-handler/https") or got.get("x-scheme-handler/http") or got.get("text/html")
    return {"present": True, "browser": _browser_of(pid), "prog_id": pid,
            "set_by_user": bool(src and src[1] == "user") or bool(kde),
            "handlers": {k: _browser_of(v[0]) for k, v in got.items()}, "kde_browser_application": _browser_of(kde) if kde else None}


@_lp("defaults.mailto_pdf_media", level="L1", family=BROWSER, tier="T0", collect="core")
def defaults_mailto_pdf_media(h, facts):
    """Default handlers (desktop ids) for mailto, PDF, video, audio, images, text."""
    m = _mimeapps(h)
    keys = {"mailto": "x-scheme-handler/mailto", ".pdf": "application/pdf", ".mp4": "video/mp4", ".mkv": "video/x-matroska",
            ".mp3": "audio/mpeg", ".jpg": "image/jpeg", ".png": "image/png", ".txt": "text/plain", ".md": "text/markdown",
            ".py": "text/x-python"}
    got = {k: m[v][0] for k, v in keys.items() if v in m}
    return {"present": bool(got), "handlers": got}


@_lp("browser.registered", level="L2", family=BROWSER, tier="T0", collect="core", gate="apps.dpkg")
def browser_registered(h, facts):
    """Browsers with a launchable .desktop entry that handles x-scheme-handler/http(s)."""
    names = sorted({_browser_of(e["id"]) for e in _desktop(h) if "x-scheme-handler/http" in e["mime"]
                    and ("WebBrowser" in e["categories"] or "Network" in e["categories"])})
    return {"present": bool(names), "browsers": names}


_T3_CHROMIUM = ("Cookies", "Network/Cookies", "Login Data", "Login Data For Account", "Web Data", "Account Web Data")
_T3_GECKO = ("cookies.sqlite", "logins.json", "key4.db", "formhistory.sqlite")


@_lp("browser.t3_presence", level="L1", family=BROWSER, tier="T3", collect="core")
def browser_t3_presence(h, facts):
    """Credential/cookie stores per profile: stat only (size), never opened."""
    _prime_browser(h)
    rows = {}
    for bid, ud, pdir, _i in bf._profiles(h):
        base = os.path.join(ud, pdir) if pdir else ud
        sizes = {f.replace("Network/", ""): m["bytes"] for f in _T3_CHROMIUM
                 for m in [h.meta(os.path.join(base, f))] if m.get("present")}
        if sizes:
            rows[bf._pkey(bid, pdir)] = sizes
    for bid, root in bf._cached(h, "catalog", _catalog_lx)["gecko"].items():
        for pd in glob.glob(os.path.join(root, "*"))[:30]:
            sizes = {f: m["bytes"] for f in _T3_GECKO for m in [h.meta(os.path.join(pd, f))] if m.get("present")}
            if sizes:
                rows[f"{bid}:{len(rows)}"] = sizes
    return {"present": bool(rows), "profiles": rows}


@_lp("browser.generic_sweep", level="L1", family=BROWSER, tier="T0", collect="extended")
def browser_generic_sweep(h, facts):
    """Chromium-style user-data dirs under ~/.config not in the catalog (Local State + Default)."""
    known = {os.path.realpath(p) for p in bf._cached(h, "catalog", _catalog_lx)["chromium"].values()}
    unknown = []
    for pat in (_cfg(h, "*", "Local State"), _cfg(h, "*", "*", "Local State")):
        for p in glob.glob(pat)[:200]:
            d = os.path.dirname(p)
            if os.path.realpath(d) not in known and _isdir(os.path.join(d, "Default")) and not _is_op(h, d):
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


# ================================================================== comms_work

def _app_row(h, installs, data_dirs, activity):
    """installed/launched/last-used for one app. activity = files whose mtime means the app ran."""
    inst = [p for p in installs if os.path.exists(p)]
    data = [p for p in data_dirs if _isdir(p)]
    if not (inst or data):
        return None
    last = _newest_mtime([a for a in activity if a] + data)
    return {"installed": bool(inst), "launched": bool(data), "last_used": _day(last),
            "recent_use": bool(last and time.time() - last < 30 * 86400)}


def _comms_cands(h):
    U = _U(h)
    return {
        "slack": (["/usr/bin/slack", "/usr/lib/slack", "/snap/slack", "/var/lib/flatpak/app/com.slack.Slack"],
                  [_cfg(h, "Slack"), _snapu(h, "slack", "current", ".config", "Slack"), _flat(h, "com.slack.Slack", "config", "Slack")],
                  [_cfg(h, "Slack", "logs"), _cfg(h, "Slack", "Local Storage")]),
        "telegram": (["/usr/bin/telegram-desktop", "/opt/telegram", "/snap/telegram-desktop", "/var/lib/flatpak/app/org.telegram.desktop"],
                     [_data(h, "TelegramDesktop"), _flat(h, "org.telegram.desktop", "data", "TelegramDesktop"),
                      _snapu(h, "telegram-desktop", "current", ".local", "share", "TelegramDesktop")],
                     [_data(h, "TelegramDesktop", "tdata", "settingss")]),
        "signal": (["/usr/bin/signal-desktop", "/opt/Signal", "/var/lib/flatpak/app/org.signal.Signal"],
                   [_cfg(h, "Signal"), _flat(h, "org.signal.Signal", "config", "Signal")],
                   [_cfg(h, "Signal", "logs"), _cfg(h, "Signal", "sql")]),
        "zoom": (["/usr/bin/zoom", "/opt/zoom", "/var/lib/flatpak/app/us.zoom.Zoom"],
                 [os.path.join(U, ".zoom"), _flat(h, "us.zoom.Zoom", ".zoom")],
                 [_cfg(h, "zoomus.conf"), os.path.join(U, ".zoom", "logs")]),
        "teams_for_linux": (["/usr/bin/teams-for-linux", "/opt/teams-for-linux"], [_cfg(h, "teams-for-linux")], []),
        "element": (["/usr/bin/element-desktop", "/opt/Element"], [_cfg(h, "Element")], []),
        "whatsapp": (["/usr/bin/whatsapp-for-linux", "/usr/bin/zapzap"], [_cfg(h, "whatsapp-for-linux"), _flat(h, "com.rtosta.zapzap")], []),
        "thunderbird": (["/usr/bin/thunderbird", "/snap/thunderbird"], [os.path.join(U, ".thunderbird"),
                                                                        _snapu(h, "thunderbird", "common", ".thunderbird")], []),
        "evolution": (["/usr/bin/evolution"], [_cfg(h, "evolution"), _data(h, "evolution")], []),
        "kmail": (["/usr/bin/kmail"], [_data(h, "kmail2"), _cfg(h, "kmail2rc")], [_cfg(h, "kmail2rc")]),
        "mattermost": (["/opt/Mattermost"], [_cfg(h, "Mattermost")], []),
        "skype": (["/usr/bin/skypeforlinux", "/snap/skype"], [_cfg(h, "skypeforlinux")], []),
    }


@_lp("comms.native_apps", level="L1", family=COMMS, tier="T0", collect="core")
def comms_native_apps(h, facts):
    """Native comms/mail apps: installed, launched (profile dir exists), last-used day from profile mtimes."""
    rows = {}
    for app, (inst, data, act) in _comms_cands(h).items():
        r = _app_row(h, inst, data, act)
        if r:
            rows[app] = r
    if not rows:
        return {"present": False}
    return {"present": True, **rows}


def _discord_dirs(h):
    return [d for d in (_cfg(h, "discord"), _cfg(h, "discordcanary"), _cfg(h, "discordptb"),
                        _flat(h, "com.discordapp.Discord", "config", "discord"), _snapu(h, "discord", "current", ".config", "discord"))
            if _isdir(d)]


@_lp("discord.present", level="L1", family=COMMS, tier="T0", collect="core")
def discord_present(h, facts):
    """Discord desktop: install dirs, profile dir, version dir, autostart entry."""
    inst = _first("/usr/share/discord", "/opt/discord", "/usr/lib/discord", "/var/lib/flatpak/app/com.discordapp.Discord", "/snap/discord")
    dirs = _discord_dirs(h)
    if not (inst or dirs):
        return {"present": False}
    vers = sorted(n for d in dirs for n in (_ls(d) or []) if re.fullmatch(r"\d+\.\d+\.\d+", n))
    auto = any("discord" in n.lower() for n in (_ls(_cfg(h, "autostart")) or []))
    return {"present": True, "installed": bool(inst), "profile_dir": bool(dirs), "version": vers[-1] if vers else None,
            "autostart": auto}


@_lp("discord.usage", level="L2", family=COMMS, tier="T1", collect="extended", gate="discord.present", timeout_ms=3000)
def discord_usage(h, facts):
    """Signed-in and voice days from logs/renderer_js.log tags; last use from Local Storage mtime."""
    dirs = _discord_dirs(h)
    if not dirs:
        return {"present": False}
    days, gw, rtc = set(), set(), set()
    size = 0
    rx = re.compile(r"^\[(\d{4}-\d\d-\d\d) ([\d:.]+)\] \[\w+\]\s+\[?([A-Za-z_]+)")
    for d in dirs:
        log = os.path.join(d, "logs", "renderer_js.log")
        if not os.path.isfile(log):
            continue
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


def _notes_cands(h):
    U = _U(h)
    return {
        "obsidian": (["/usr/bin/obsidian", "/opt/Obsidian", "/snap/obsidian", "/var/lib/flatpak/app/md.obsidian.Obsidian"],
                     [_cfg(h, "obsidian"), _flat(h, "md.obsidian.Obsidian", "config", "obsidian")],
                     [_cfg(h, "obsidian", "obsidian.json")]),
        "notion": (["/opt/Notion", "/usr/bin/notion-app"], [_cfg(h, "Notion"), _cfg(h, "notion-app")], []),
        "logseq": (["/opt/Logseq", "/var/lib/flatpak/app/com.logseq.Logseq"], [os.path.join(U, ".logseq"), _cfg(h, "Logseq")], []),
        "joplin": ([os.path.join(U, ".joplin")], [_cfg(h, "joplin-desktop"), _cfg(h, "Joplin")], []),
        "standard_notes": ([], [_cfg(h, "Standard Notes")], []),
        "zim": (["/usr/bin/zim"], [_cfg(h, "zim")], []),
        "knotes": (["/usr/bin/knotes"], [_data(h, "knotes")], []),
        "anytype": ([], [_cfg(h, "anytype")], []),
        "trilium": ([], [_cfg(h, "trilium-data"), os.path.join(U, "trilium-data")], []),
    }


@_lp("productivity.notes_apps", level="L1", family=COMMS, tier="T0", collect="core")
def notes_apps(h, facts):
    """Notes apps installed or launched: Obsidian, Notion, Logseq, Joplin, Zim, KNotes, ..."""
    rows = {}
    for app, (inst, data, act) in _notes_cands(h).items():
        r = _app_row(h, inst, data, act)
        if r:
            rows[app] = r
    return {"present": bool(rows), "apps": sorted(rows), "detail": rows}


@_lp("obsidian.vaults", level="L2", family=COMMS, tier="T1", collect="extended", gate="productivity.notes_apps")
def obsidian_vaults(h, facts):
    """Obsidian vault count and note counts (vault names and paths are not emitted)."""
    j = _read_json(_cfg(h, "obsidian", "obsidian.json")) or _read_json(_flat(h, "md.obsidian.Obsidian", "config", "obsidian", "obsidian.json"))
    vs = (j or {}).get("vaults") or {} if isinstance(j, dict) else {}
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
            "notes_md": notes, "files": files, "newest_note": bf._iso(newest)}


_EDR = {"crowdstrike": ["/opt/CrowdStrike"], "sentinelone": ["/opt/sentinelone"], "defender_mdatp": ["/opt/microsoft/mdatp", "/etc/opt/microsoft/mdatp"],
        "tanium": ["/opt/Tanium"], "wazuh": ["/var/ossec"], "elastic_agent": ["/opt/Elastic"], "osquery": ["/etc/osquery", "/opt/osquery"],
        "sophos": ["/opt/sophos-spl"], "carbonblack": ["/opt/carbonblack"], "qualys": ["/usr/local/qualys"],
        "rapid7": ["/opt/rapid7"], "trendmicro": ["/opt/ds_agent"], "jamf_protect": ["/opt/jamf"], "kolide": ["/usr/local/kolide-k2"],
        "zscaler": ["/opt/zscaler"], "globalprotect": ["/opt/paloaltonetworks"], "netskope": ["/opt/netskope"]}


@_lp("work.security_agents", level="L1", family=COMMS, tier="T0", collect="core")
def work_security_agents(h, facts):
    """EDR/ZTNA/RMM agents by install path (no service queries)."""
    hits = sorted(k for k, ps in _EDR.items() if _ex(*ps))
    return {"present": True, "agents": hits, "checked": len(_EDR)}


@_lp("work.mdm", level="L1", family=COMMS, tier="T0", collect="core")
def work_mdm(h, facts):
    """Device management on Linux: Intune for Linux, Canonical Landscape, Ubuntu Pro attach, Puppet/Chef/Salt agents."""
    found = {"intune": _ex("/opt/microsoft/intune", "/usr/bin/intune-portal"),
             "landscape": _ex("/etc/landscape/client.conf"), "ubuntu_pro_attached": _ex("/var/lib/ubuntu-advantage/private/machine-token.json"),
             "puppet": _ex("/etc/puppetlabs"), "chef": _ex("/etc/chef"), "salt_minion": _ex("/etc/salt/minion"),
             "fleetd": _ex("/opt/orbit")}
    real = [k for k, v in found.items() if v and k != "ubuntu_pro_attached"]
    return {"present": True, "real_enrollments": len(real), "enrollments": real, "ubuntu_pro_attached": found["ubuntu_pro_attached"]}


@_lp("work.join_state", level="L1", family=COMMS, tier="T0", collect="core")
def work_join_state(h, facts):
    """Directory join: sssd/realmd/winbind config presence and krb5 default realm set (realm name not emitted)."""
    krb = _open_text("/etc/krb5.conf", 200_000) or ""
    sssd = _ex("/etc/sssd/sssd.conf")
    winbind = "winbind" in (_open_text("/etc/nsswitch.conf", 100_000) or "")
    joined = sssd or winbind
    return {"present": True, "sssd": sssd, "winbind": winbind, "realmd": _ex("/etc/realmd.conf"),
            "krb5_default_realm_set": bool(re.search(r"^\s*default_realm\s*=", krb, re.M)),
            "state": "joined" if joined else "workgroup", "domain_joined": joined, "azure_ad_joined": _ex("/opt/microsoft/intune")}


@_lp("work.policies", level="L1", family=COMMS, tier="T0", collect="core")
def work_policies(h, facts):
    """Managed browser policy files (Chrome, Chromium, Brave, Edge, Firefox): file counts only."""
    dirs = {"chrome": "/etc/opt/chrome/policies/managed", "chromium": "/etc/chromium/policies/managed",
            "brave": "/etc/brave/policies/managed", "edge": "/etc/opt/edge/policies/managed"}
    apps = {k: len([n for n in (_ls(d) or []) if n.endswith(".json")]) for k, d in dirs.items()}
    if _ex("/etc/firefox/policies/policies.json", "/usr/lib/firefox/distribution/policies.json"):
        apps["firefox"] = 1
    configured = {k: {"values": v, "subkeys": 0} for k, v in apps.items() if v}
    return {"present": True, "app_policies": configured, "app_policy_count": len(configured)}
