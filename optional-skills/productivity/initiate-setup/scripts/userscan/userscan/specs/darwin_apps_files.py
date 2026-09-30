"""macOS user-level probes: files, gaming, media.

Registration only at import time. Shared helpers and registration wrappers live in darwin_apps.
"""
from __future__ import annotations

import collections
import ctypes
import ctypes.util
import glob
import os
import plistlib
import re
import time

from . import browser_files as bf
from . import usage_gaming as ug
from . import browser_files_content as bf_content
from . import usage_gaming_games as ug_games
from .darwin_apps import (FILES, GAMING, MAC_EPOCH, MEDIA, _AS, _LIB, _OP_DEEP, _OP_TOP, _U,
    _app, _day, _dir_readable, _ex, _first, _isdir, _ls, _mirror, _mp, _mtime, _newest_mtime, _plist,
    _read_json, _which)

# ================================================================== files

def _known_folders_mac(h):
    U = _U(h)
    out = {n: os.path.join(U, n) for n in ("Desktop", "Documents", "Downloads", "Pictures", "Music")}
    out["Videos"] = os.path.join(U, "Movies")
    sc = _plist(_LIB(h, "Preferences", "com.apple.screencapture.plist")) or {}
    loc = sc.get("location") if isinstance(sc, dict) else None
    out["Screenshots"] = os.path.expanduser(loc) if isinstance(loc, str) and loc else out["Desktop"]
    return out


def _prime_kf(h):
    bf._cached(h, "kf", _known_folders_mac)


_STANDARD = {"desktop", "documents", "downloads", "pictures", "movies", "music", "public", "applications", "library"}


def _home_root(h):
    U = _U(h)
    cls, loose = collections.Counter(), 0
    try:
        ents = list(zip(range(5000), os.scandir(U)))
    except OSError:
        return None
    for _, e in ents:
        try:
            is_dir = e.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if not is_dir:
            loose += 0 if e.name.startswith(".") else 1
            continue
        low = e.name.lower()
        cls["operator" if _OP_TOP.match(e.name) else "dot" if low.startswith(".") else
            "standard" if low in _STANDARD else "custom"] += 1
    return {"dirs_by_class": dict(cls), "loose_visible_files": loose}


@_mp("files.known_folders", level="L1", family=FILES, tier="T0", collect="core")
def files_known_folders(h, facts):
    """Home folders (Desktop, Documents, Downloads, Pictures, Movies, Music) and the screenshot location;
    whether Desktop & Documents sync to iCloud Drive."""
    kf = bf._cached(h, "kf", _known_folders_mac)
    U = _U(h)
    icloud_dd = _isdir(_LIB(h, "Mobile Documents", "com~apple~CloudDocs", "Desktop"))
    rows = {n: {"path": ("~" + p[len(U):]) if p and p.startswith(U) else p, "exists": bool(p and _isdir(p)),
                "under_profile": bool(p and p.startswith(U)), "under_onedrive": False} for n, p in kf.items()}
    return {"present": any(r["exists"] for r in rows.values()), "folders": rows, "icloud_desktop_documents": icloud_dd,
            "screenshot_location_custom": kf["Screenshots"] != kf["Desktop"]}


@_mp("files.composition", level="L2", family=FILES, tier="T1", collect="extended", gate="files.known_folders",
     timeout_ms=9000)
def files_composition(h, facts):
    """Per home folder: files, bytes, extension/kind/age histograms (depth <= 6, 100k entries), plus home-root dir
    classes (dot/standard/custom/operator). No file names."""
    _prime_kf(h)
    v = bf_content.files_composition(h, facts)
    if isinstance(v, dict):
        v["home_root"] = _home_root(h)
    return v


_mirror("files.screenshots", bf_content.files_screenshots, _prime_kf)


@_mp("files.recent_lnk", level="L2", family=FILES, tier="T1", collect="core", gate="files.known_folders")
def files_recent_lnk(h, facts):
    """Recent-document lists (com.apple.sharedfilelist *.sfl3): number of apps with a recent list, list file dates.
    The lists themselves are not parsed. The directory is TCC-protected (Full Disk Access)."""
    d = _AS(h, "com.apple.sharedfilelist", "com.apple.LSSharedFileList.ApplicationRecentDocuments")
    if not _isdir(_AS(h, "com.apple.sharedfilelist")):
        return {"present": False}
    names = _ls(d, 2000)
    if names is None:
        return {"present": False, "needs_fda": True}
    mts = [m for m in (_newest_mtime([os.path.join(d, n)]) for n in names if n.endswith((".sfl2", ".sfl3"))) if m]
    now = time.time()
    return {"present": bool(mts), "count": len(mts), "apps_with_recents": len(mts),
            "oldest": bf._iso(min(mts)) if mts else None, "newest": bf._iso(max(mts)) if mts else None,
            "active_days_30d": len({time.strftime("%Y-%m-%d", time.localtime(t)) for t in mts if now - t < 30 * 86400})}


@_mp("files.clutter", level="L2", family=FILES, tier="T1", collect="extended", gate="files.known_folders", timeout_ms=6000)
def files_clutter(h, facts):
    """Trash (needs FDA), ~/Library/Caches size, this user's files in TMPDIR, Desktop item count (bounded walks)."""
    _prime_kf(h)
    out = {"present": True}
    tr = os.path.join(_U(h), ".Trash")
    if _dir_readable(tr):
        st = bf._walk(tr, max_depth=6, max_entries=100000, budget_s=1.5)
        out["recycle_bin"] = {"items": len(_ls(tr, 100000) or []), "files": st["files"], "bytes": st["bytes"],
                              "truncated": st["truncated"]}
    else:
        out["recycle_bin"] = {"needs_fda": _isdir(tr)}
    st = bf._walk(_LIB(h, "Caches"), max_depth=6, max_entries=150000, budget_s=1.5)
    out["cache_bytes"], out["cache_files"], out["cache_truncated"] = st["bytes"], st["files"], st["truncated"]
    me, n, b = os.getuid(), [0], [0]

    def mine(e, s, depth):
        if s.st_uid == me:
            n[0] += 1
            b[0] += s.st_size
    # no-tmp: ok — measures the user's own temp dir, which falls back to /tmp
    tmp = os.environ.get("TMPDIR") or "/tmp"
    st = bf._walk(tmp, max_depth=3, max_entries=50000, budget_s=1.0, on_file=mine,
                  skip_dir=lambda nm, p: nm == "userscan" or bool(_OP_DEEP.search(p)))
    out["temp_files"], out["temp_bytes"], out["temp_truncated"] = n[0], b[0], st["truncated"]
    desk = bf_content._kf(h, "Desktop")
    names = [x for x in h.list_dir(desk, 5000) if not x.startswith(".")]
    out["desktop_items"] = len(names)
    out["desktop_screenshots"] = sum(1 for x in names if bf_content.SHOT_NAME.match(x))
    return out


def _where_froms(path):
    """kMDItemWhereFroms xattr (binary plist of source URLs) via getxattr(2); None when absent."""
    try:
        lib = _where_froms.lib
    except AttributeError:
        lib = _where_froms.lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        lib.getxattr.restype = ctypes.c_ssize_t
        lib.getxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
    name = b"com.apple.metadata:kMDItemWhereFroms"
    p = os.fsencode(path)
    n = lib.getxattr(p, name, None, 0, 0, 1)  # XATTR_NOFOLLOW
    if n <= 0 or n > 65536:
        return None
    buf = ctypes.create_string_buffer(n)
    n = lib.getxattr(p, name, buf, n, 0, 1)
    if n <= 0:
        return None
    try:
        v = plistlib.loads(buf.raw[:n])
    except Exception:
        return None
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else None


QUARANTINE = ("Library", "Preferences", "com.apple.LaunchServices.QuarantineEventsV2")


@_mp("files.download_sources", level="L2", family=FILES, tier="T1", collect="extended", gate="files.known_folders",
     timeout_ms=6000)
def files_download_sources(h, facts):
    """Where downloads come from, as counts: LaunchServices QuarantineEventsV2 (events by downloading app, by month,
    source-site categories when a URL was recorded) and kMDItemWhereFroms xattrs on files in Downloads (depth <= 2):
    source-site category counts and distinct host count. No hosts, URLs or file names are emitted."""
    _prime_kf(h)
    out = {"present": False}
    q = os.path.join(_U(h), *QUARANTINE)
    if os.path.exists(q):
        rows = h.sqlite(q, "select LSQuarantineTimeStamp, LSQuarantineAgentName, LSQuarantineOriginURLString, "
                           "LSQuarantineDataURLString from LSQuarantineEvent")
        if rows is not None:
            agents, months, cats, hosts = collections.Counter(), collections.Counter(), collections.Counter(), set()
            ts = []
            for t, agent, origin, data in rows:
                if t:
                    ts.append(t + MAC_EPOCH)
                    months[_day(t + MAC_EPOCH)[:7]] += 1
                agents[(agent or "?")[:40]] += 1
                url = origin or data
                if url:
                    host, _ = bf._host_of(url)
                    if host:
                        hosts.add(host)
                        c = bf._categorize(bf_content._reg_domain(host))
                        cats["sensitive" if c in bf.SENSITIVE_CATS else c] += 1
            out.update({"present": bool(rows), "quarantine": {
                "events": len(rows), "first": bf._iso(min(ts)) if ts else None, "last": bf._iso(max(ts)) if ts else None,
                "by_agent": dict(agents.most_common(10)), "by_month": dict(sorted(months.items())[-24:]),
                "with_url": sum(cats.values()), "distinct_hosts": len(hosts), "source_categories": dict(cats.most_common())}})
    root = bf_content._kf(h, "Downloads")
    if _isdir(root):
        cats, ext_cat, hosts = collections.Counter(), collections.Counter(), set()
        n = with_x = 0
        deadline = time.perf_counter() + 3.0
        stack, trunc = [(root, 0)], None
        while stack:
            d, depth = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if depth + 1 < 2 and not _OP_DEEP.search(e.path) and not e.name.endswith((".app", ".photoslibrary")):
                            stack.append((e.path, depth + 1))
                        continue
                except OSError:
                    continue
                n += 1
                if n > 5000 or time.perf_counter() > deadline:
                    trunc = "cap"
                    stack.clear()
                    break
                urls = _where_froms(e.path)
                if not urls:
                    continue
                with_x += 1
                host, _ = bf._host_of(urls[0])
                if host:
                    hosts.add(host)
                c = bf._categorize(bf_content._reg_domain(host)) if host else "other"
                c = "sensitive" if c in bf.SENSITIVE_CATS else c
                cats[c] += 1
                ext_cat[f"{c}:{os.path.splitext(e.name)[1].lower() or '(none)'}"] += 1
        out["downloads_folder"] = {"files_scanned": n, "depth": 2, "truncated": trunc, "with_origin_xattr": with_x,
                                   "distinct_hosts": len(hosts), "source_categories": dict(cats.most_common()),
                                   "category_ext_top": dict(ext_cat.most_common(10))}
        out["present"] = out["present"] or n > 0
    return out


@_mp("files.profile_size_walk", level="L2", family=FILES, tier="T2", collect="deep", gate="files.known_folders",
     timeout_ms=30000)
def files_profile_size_walk(h, facts):
    """Size of each top-level home dir (depth <= 10, 300k entries/dir, 25 s total); operator dirs excluded."""
    U = _U(h)
    deadline = time.perf_counter() + 25.0
    rows, loose = [], 0
    try:
        entries = list(os.scandir(U))
    except OSError:
        return {"present": False}
    for e in entries:
        try:
            if not e.is_dir(follow_symlinks=False):
                loose += 1
                continue
        except OSError:
            continue
        low = e.name.lower()
        cls = ("operator" if _OP_TOP.match(e.name) else "dot" if low.startswith(".") else
               "standard" if low in _STANDARD else "custom")
        if cls == "operator":
            rows.append({"name": e.name, "class": cls, "bytes": 0, "skipped": "operator"})
            continue
        remain = deadline - time.perf_counter()
        if remain <= 0:
            rows.append({"name": e.name, "class": cls, "bytes": 0, "skipped": "budget"})
            continue
        st = bf._walk(e.path, max_depth=10, max_entries=300000, budget_s=min(4.0, remain))
        rows.append({"name": e.name, "class": cls, "bytes": st["bytes"], "files": st["files"], "truncated": st["truncated"]})
    rows.sort(key=lambda r: -r["bytes"])
    shown = 0
    for r in rows:
        if r["class"] == "custom":
            shown += 1
            if shown > 5:
                r["name"] = f"custom#{shown}"
    cls = collections.Counter(r["class"] for r in rows)
    return {"present": True, "top_dirs": [r for r in rows if r["class"] != "operator"][:12], "class_counts": dict(cls),
            "custom_root_dirs": cls.get("custom", 0), "operator_dirs_excluded": cls.get("operator", 0),
            "custom_bytes": sum(r["bytes"] for r in rows if r["class"] == "custom"),
            "total_bytes": sum(r["bytes"] for r in rows), "loose_root_files": loose,
            "truncated_dirs": sum(1 for r in rows if r.get("truncated") or r.get("skipped") == "budget")}


@_mp("sync.other", level="L1", family=FILES, tier="T0", collect="core")
def sync_other(h, facts):
    """Sync clients: iCloud Drive, Dropbox, Google Drive, Box, pCloud, Syncthing, MEGA, Proton Drive, Nextcloud
    (File Provider roots under ~/Library/CloudStorage plus app data dirs). OneDrive is onedrive.accounts."""
    cs = _ls(_LIB(h, "CloudStorage")) or []

    def root(prefix):
        return any(n.startswith(prefix) for n in cs)
    cands = {"icloud_drive": _isdir(_LIB(h, "Mobile Documents", "com~apple~CloudDocs")),
             "dropbox": root("Dropbox") or _isdir(os.path.join(_U(h), ".dropbox")),
             "google_drive": root("GoogleDrive") or _isdir(_AS(h, "Google", "DriveFS")),
             "box": root("Box") or _isdir(_AS(h, "Box")), "pcloud": root("pCloud"),
             "syncthing": _isdir(_AS(h, "Syncthing")) or _isdir(_LIB(h, "Application Support", "Syncthing")),
             "mega": _isdir(_AS(h, "Mega Limited")), "proton_drive": root("ProtonDrive"),
             "nextcloud": _isdir(_AS(h, "Nextcloud")) or root("Nextcloud")}
    found = sorted(k for k, v in cands.items() if v)
    return {"present": bool(found), "clients": found, "cloudstorage_roots": len(cs)}


# ================================================================== gaming

def _steam_root(h):
    p = _AS(h, "Steam")
    if _isdir(os.path.join(p, "steamapps")) or _isdir(os.path.join(p, "userdata")):
        return p
    return None


def _steam_mac(h):
    def load():
        sp = _steam_root(h)
        if not sp:
            return None
        libs = []
        txt = ug._read(os.path.join(sp, "steamapps", "libraryfolders.vdf"))
        if txt:
            for _k, v in (ug_games._ci(ug_games._vdf(txt), "libraryfolders") or {}).items():
                if isinstance(v, dict) and v.get("path"):
                    libs.append(os.path.normpath(v["path"]))
        libs = libs or [sp]
        installed = []
        for lib in libs[:20]:
            sa = os.path.join(lib, "steamapps")
            for n in h.list_dir(sa, 2000):
                if n.startswith("appmanifest_") and n.endswith(".acf"):
                    st = ug_games._ci(ug_games._vdf(ug._read(os.path.join(sa, n)) or ""), "AppState") or {}
                    installed.append({"appid": int(st.get("appid", 0) or 0), "name": st.get("name"),
                                      "size_gb": round(int(st.get("SizeOnDisk", 0) or 0) / 1e9, 1),
                                      "drive": "home" if lib.startswith(_U(h)) else "other",
                                      "last_played": int(st.get("LastPlayed", 0) or 0)})
        ud = os.path.join(sp, "userdata")
        uids = [u for u in h.list_dir(ud, 50) if u.isdigit()]
        users = {}
        for uid in uids:
            apps = {}
            t = ug._read(os.path.join(ud, uid, "config", "localconfig.vdf"))
            if t:
                node = ug_games._ci(ug_games._vdf(t), "UserLocalConfigStore", "Software", "Valve", "Steam", "apps") or {}
                for aid, v in node.items():
                    if isinstance(v, dict) and aid.isdigit() and ("Playtime" in v or "LastPlayed" in v):
                        apps[int(aid)] = {"min": int(v.get("Playtime", 0) or 0), "min2wk": int(v.get("Playtime2wks", 0) or 0),
                                          "last": int(v.get("LastPlayed", 0) or 0)}
            users[uid] = apps
        return {"path": sp, "libs": libs, "installed": installed, "uids": uids, "users": users,
                "appinfo_path": os.path.join(sp, "appcache", "appinfo.vdf")}
    return load


def _prime_steam(h):
    ug._cached(("steam", h.l0.get("run_id")), _steam_mac(h))


@_mp("steam.present", level="L1", family=GAMING, tier="T0", collect="core")
def steam_present(h, facts):
    """Steam client data dir (~/Library/Application Support/Steam) and app bundle; gates the Steam subtree."""
    sp = _steam_root(h)
    if not sp:
        return None
    b = _app(h, "Steam")
    return {"present": True, "via": "native", "app": bool(b), "userdata_accounts": len([u for u in (_ls(os.path.join(sp, "userdata")) or [])
                                                                                        if u.isdigit()])}


for _id, _fn in (("steam.installed", ug_games.steam_installed), ("steam.playtime", ug_games.steam_playtime),
                 ("steam.appinfo_genres", ug_games.steam_appinfo_genres), ("steam.local_sessions", ug_games.steam_local_sessions),
                 ("steam.non_steam_shortcuts", ug_games.steam_non_steam_shortcuts), ("steam.screenshots", ug_games.steam_screenshots),
                 ("steam.login_users", ug_games.steam_login_users), ("steam.remote_clients", ug_games.steam_remote_clients)):
    _mirror(_id, _fn, _prime_steam)


def _epic_manifests(h):
    return _first(_AS(h, "Epic", "EpicGamesLauncher", "Data", "Manifests"))


@_mp("epic.present", level="L1", family=GAMING, tier="T0", collect="core")
def epic_present(h, facts):
    """Epic Games Launcher for Mac (app bundle, ~/Library/Application Support/Epic) or Heroic."""
    b = _app(h, "Epic Games Launcher")
    d = _AS(h, "Epic")
    heroic = _AS(h, "heroic")
    if not (b or _isdir(d) or _isdir(heroic)):
        return None
    return {"present": True, "via": "epic" if (b or _isdir(d)) else "heroic", "mtime": _mtime(d if _isdir(d) else heroic)}


@_mp("epic.installs", level="L2", family=GAMING, tier="T0", collect="core", gate="epic.present")
def epic_installs(h, facts):
    """Epic installs from launcher .item manifests: titles and GB."""
    d = _epic_manifests(h)
    rows = []
    for f in glob.glob(os.path.join(d, "*.item"))[:200] if d else []:
        j = _read_json(f) or {}
        if isinstance(j, dict):
            rows.append([j.get("DisplayName") or os.path.basename(f), round((j.get("InstallSize") or 0) / 1e9, 1)])
    if not rows:
        return {"present": False}
    rows.sort(key=lambda r: -r[1])
    return {"present": True, "count": len(rows), "total_gb": round(sum(r[1] for r in rows), 1), "manifests": rows[:20]}


@_mp("launchers.other", level="L1", family=GAMING, tier="T0", collect="core")
def launchers_other(h, facts):
    """macOS game launchers and compatibility layers: Battle.net, GOG Galaxy, Heroic, CrossOver, Whisky, Game
    Porting Toolkit, Parallels, Prism/Minecraft, itch, Porting Kit, Apple Games."""
    U = _U(h)
    apps = {"battlenet": ["Battle.net"], "gog": ["GOG Galaxy"], "heroic": ["Heroic"], "crossover": ["CrossOver"],
            "whisky": ["Whisky"], "parallels": ["Parallels Desktop"], "prismlauncher": ["PrismLauncher", "Prism Launcher"],
            "itch": ["itch"], "porting_kit": ["Porting Kit"], "apple_games": ["Games"], "minecraft_launcher": ["Minecraft"],
            "ea": ["EA app"], "ubisoft": ["Ubisoft Connect"], "moonlight": ["Moonlight"], "geforce_now": ["GeForceNOW"]}
    hits = {k for k, names in apps.items() if _app(h, *names)}
    for k, p in (("minecraft", _AS(h, "minecraft")), ("prismlauncher", _AS(h, "PrismLauncher")),
                 ("heroic", _AS(h, "heroic")), ("gptk", "/usr/local/opt/game-porting-toolkit"), ("wine", os.path.join(U, ".wine"))):
        if _ex(p):
            hits.add(k)
    return {"present": True, "launchers": sorted(hits)} if hits else None


@_mp("emulators", level="L1", family=GAMING, tier="T0", collect="core")
def emulators(h, facts):
    """Emulators: OpenEmu, RetroArch, Dolphin, PPSSPP, Ryujinx, PCSX2, RPCS3, Cemu, DuckStation, melonDS, UTM."""
    c = {"OpenEmu": ["OpenEmu"], "RetroArch": ["RetroArch"], "Dolphin": ["Dolphin"], "PPSSPP": ["PPSSPPSDL", "PPSSPP"],
         "Ryujinx": ["Ryujinx"], "PCSX2": ["PCSX2"], "RPCS3": ["RPCS3"], "Cemu": ["Cemu"], "DuckStation": ["DuckStation"],
         "melonDS": ["melonDS"], "UTM": ["UTM"], "Provenance": ["Provenance"]}
    hits = sorted(k for k, names in c.items() if _app(h, *names))
    if _isdir(_AS(h, "OpenEmu")) and "OpenEmu" not in hits:
        hits.append("OpenEmu")
    return {"present": True, "found": hits} if hits else None


# ================================================================== media

@_mp("media.libraries", level="L1", family=MEDIA, tier="T1", collect="core")
def media_libraries(h, facts):
    """Music/Movies/Pictures file counts by type and size (bounded walk; Photos/Music library packages not
    entered), Apple Photos and Music library package sizes (stat only), screenshot count."""
    kf = bf._cached(h, "kf", _known_folders_mac)
    out = {}
    for short in ("Music", "Videos", "Pictures"):
        p = kf.get(short) or os.path.join(_U(h), short)
        s = ug_games._walk_media(p, budget_s=0.6)
        s["gb"] = round(s.pop("bytes") / 1e9, 2)
        s["onedrive"] = False
        out[short] = s
    photos = [n for n in (_ls(kf.get("Pictures") or "") or []) if n.endswith(".photoslibrary")]
    out["photos_libraries"] = len(photos)
    out["apple_music_library"] = _ex(os.path.join(kf.get("Music") or "", "Music", "Music Library.musiclibrary"))
    out["screenshots"] = sum(1 for n in (_ls(kf.get("Screenshots") or "", 20000) or []) if bf_content.SHOT_NAME.match(n))
    return {"present": True, **out}


@_mp("media.obs", level="L1", family=MEDIA, tier="T0", collect="core")
def media_obs(h, facts):
    """OBS Studio app bundle and whether a config dir exists (launched at least once); other recorders."""
    b = _app(h, "OBS")
    cfg = _isdir(_AS(h, "obs-studio"))
    others = [n for n in ("Screen Studio", "Loom", "CleanShot X", "Cap", "Kap", "ScreenFlow", "Descript") if _app(h, n)]
    if not (b or cfg):
        return None
    return {"present": True, "obs_installs": ["app"] if b else [], "obs_configs": ["native"] if cfg else [],
            "config_dir": cfg, "version": (b or {}).get("version"), "obs_config_in_agent_container": 0,
            "streamlabs": bool(_app(h, "Streamlabs Desktop")), "streamlabs_config": False, "other_recorders": others}


@_mp("media.players", level="L1", family=MEDIA, tier="T0", collect="core")
def media_players(h, facts):
    """Music/video players and streaming apps: Spotify, Apple Music/TV/Podcasts (library present), IINA, VLC, mpv,
    Infuse, Plex, Jellyfin, TIDAL, ..."""
    kf = bf._cached(h, "kf", _known_folders_mac)
    found = {"spotify": bool(_app(h, "Spotify")) or _isdir(_AS(h, "Spotify")),
             "apple_music": _ex(os.path.join(kf.get("Music") or "", "Music", "Music Library.musiclibrary")),
             "apple_tv": _isdir(_AS(h, "TV")) or _ex(os.path.join(kf.get("Videos") or "", "TV")),
             "podcasts": _isdir(_LIB(h, "Group Containers", "243LU875E5.groups.com.apple.podcasts")),
             "iina": bool(_app(h, "IINA")), "vlc": bool(_app(h, "VLC")), "mpv": bool(_which(h, "mpv")) or bool(_app(h, "mpv")),
             "infuse": bool(_app(h, "Infuse 7", "Infuse")), "plex": bool(_app(h, "Plex", "Plexamp")),
             "jellyfin": bool(_app(h, "Jellyfin Media Player")), "tidal": bool(_app(h, "TIDAL")),
             "youtube_music": bool(_app(h, "YouTube Music")), "audirvana": bool(_app(h, "Audirvana Studio"))}
    hits = sorted(k for k, v in found.items() if v)
    return {"present": True, "players": hits} if hits else None


VLC_PREFS = ("Library", "Preferences", "org.videolan.vlc.plist")


@_mp("media.vlc", level="L1", family=MEDIA, tier="T0", collect="core")
def media_vlc(h, facts):
    """VLC app bundle and whether its preferences exist (launched)."""
    b = _app(h, "VLC")
    cfg = _ex(os.path.join(_U(h), *VLC_PREFS), _AS(h, "org.videolan.vlc"))
    return {"present": True, "installed": bool(b), "version": (b or {}).get("version"), "user_config": cfg} \
        if (b or cfg) else None


@_mp("media.vlc_recents", level="L2", family=MEDIA, tier="T2", collect="extended", gate="media.vlc")
def media_vlc_recents(h, facts):
    """VLC recent-media count, local vs stream split and extension histogram; the list itself is not emitted."""
    d = _plist(os.path.join(_U(h), *VLC_PREFS)) or {}
    items = d.get("recentlyPlayedMedia") or d.get("NSRecentDocuments") or []
    if isinstance(items, dict):
        items = list(items.keys())
    items = [str(x) for x in items if isinstance(x, (str, bytes))][:500] if isinstance(items, list) else []
    if not items:
        return {"present": False}
    kinds = collections.Counter("local" if x.startswith(("file:", "/")) else "stream" for x in items)
    exts = collections.Counter()
    for x in items:
        m2 = re.search(r"\.([A-Za-z0-9]{1,5})$", x.split("?")[0])
        exts["." + m2.group(1).lower() if m2 else ""] += 1
    return {"present": True, "count": len(items), "kinds": dict(kinds), "ext": dict(exts),
            "ini_mtime": _mtime(os.path.join(_U(h), *VLC_PREFS))}
