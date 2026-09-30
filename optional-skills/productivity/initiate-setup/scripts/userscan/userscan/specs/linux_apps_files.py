"""Linux user-level probes: files, gaming, media.

Registration only at import time. Shared helpers and registration wrappers live in linux_apps.
"""
from __future__ import annotations

import collections
import datetime as dt
import os
import re
import time

from . import browser_files as bf
from . import usage_gaming as ug
from . import browser_files_content as bf_content
from . import usage_gaming_games as ug_games
from .linux_apps import (FILES, GAMING, MEDIA, _OP_DEEP, _OP_TOP, _U, _age_bucket, _cachedir,
    _cfg, _data, _ex, _first, _flat, _isdir, _lp, _ls, _mirror, _mtime, _open_text, _read_json, _snapu,
    _state)
from .linux_apps_dev import tempfile_dir

# ================================================================== files

_XDG_DEFAULTS = {"DESKTOP": "Desktop", "DOCUMENTS": "Documents", "DOWNLOAD": "Downloads", "PICTURES": "Pictures",
                 "VIDEOS": "Videos", "MUSIC": "Music", "TEMPLATES": "Templates", "PUBLICSHARE": "Public"}
_KF_NAME = {"DESKTOP": "Desktop", "DOCUMENTS": "Documents", "DOWNLOAD": "Downloads", "PICTURES": "Pictures",
            "VIDEOS": "Videos", "MUSIC": "Music", "TEMPLATES": "Templates", "PUBLICSHARE": "Public"}


def _known_folders_lx(h):
    U = _U(h)
    out = {v: os.path.join(U, d) for k, d in _XDG_DEFAULTS.items() for v in [_KF_NAME[k]]}
    t = _open_text(_cfg(h, "user-dirs.dirs"), 100_000) or ""
    for m in re.finditer(r'^XDG_(\w+)_DIR="([^"]*)"', t, re.M):
        k, v = m.group(1), m.group(2).replace("$HOME", U)
        if k in _KF_NAME:
            out[_KF_NAME[k]] = v if v.rstrip("/") != U.rstrip("/") else None
    pics = out.get("Pictures") or os.path.join(U, "Pictures")
    out["Screenshots"] = os.path.join(pics, "Screenshots")
    return out


def _prime_kf(h):
    bf._cached(h, "kf", _known_folders_lx)


_STANDARD = {"desktop", "documents", "downloads", "pictures", "videos", "music", "templates", "public", "snap"}


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


@_lp("files.known_folders", level="L1", family=FILES, tier="T0", collect="core")
def files_known_folders(h, facts):
    """XDG user dirs (user-dirs.dirs): path under home, exists."""
    kf = bf._cached(h, "kf", _known_folders_lx)
    U = _U(h)
    rows = {n: {"path": ("~" + p[len(U):]) if p and p.startswith(U) else p, "exists": bool(p and _isdir(p)),
                "under_profile": bool(p and p.startswith(U)), "under_onedrive": False} for n, p in kf.items()}
    return {"present": any(r["exists"] for r in rows.values()), "folders": rows,
            "user_dirs_file": os.path.exists(_cfg(h, "user-dirs.dirs"))}


@_lp("files.composition", level="L2", family=FILES, tier="T1", collect="extended", gate="files.known_folders",
     timeout_ms=9000)
def files_composition(h, facts):
    """Per XDG folder: files, bytes, extension/kind/age histograms (depth <= 6, 100k entries), plus home-root dir
    classes (dot/standard/custom/operator). No file names."""
    _prime_kf(h)
    v = bf_content.files_composition(h, facts)
    if isinstance(v, dict):
        v["home_root"] = _home_root(h)
    return v
_mirror("files.screenshots", bf_content.files_screenshots, _prime_kf)


@_lp("files.recent_lnk", level="L2", family=FILES, tier="T1", collect="core", gate="files.known_folders")
def files_recent_lnk(h, facts):
    """Recent files: recently-used.xbel + KDE RecentDocuments (counts, ext histogram, date range) and the KDE
    activity-manager score cache (row count, top agent apps, range). File names and paths are never emitted."""
    now = time.time()
    mts, ext, ages = [], collections.Counter(), collections.Counter()
    op = 0
    xb = _open_text(_data(h, "recently-used.xbel"), 20_000_000)
    if xb:
        for m in re.finditer(r'<bookmark href="([^"]+)"[^>]*?modified="([^"]+)"', xb):
            href, mod = m.group(1), m.group(2)
            if _OP_DEEP.search(href):
                op += 1
                continue
            try:
                t = dt.datetime.fromisoformat(mod.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            mts.append(t)
            ages[_age_bucket(now - t)] += 1
            x = os.path.splitext(href.split("?")[0])[1].lower()
            ext[x if re.fullmatch(r"\.[a-z0-9]{1,5}", x or "") else "(folder_or_none)"] += 1
    rd = _data(h, "RecentDocuments")
    for n in (_ls(rd, 5000) or []):
        try:
            t = os.stat(os.path.join(rd, n)).st_mtime
        except OSError:
            continue
        mts.append(t)
        ages[_age_bucket(now - t)] += 1
        x = os.path.splitext(n[:-8] if n.endswith(".desktop") else n)[1].lower()
        ext[x if re.fullmatch(r"\.[a-z0-9]{1,5}", x or "") else "(folder_or_none)"] += 1
    ka = None
    db = _data(h, "kactivitymanagerd", "resources", "database")
    if os.path.exists(db):
        rows = h.sqlite(db, "select initiatingAgent, count(*), min(firstUpdate), max(lastUpdate) from ResourceScoreCache group by 1")
        if rows:
            agents = collections.Counter({(a or "?"): n for a, n, _f, _l in rows})
            ka = {"rows": sum(agents.values()), "agents": len(agents), "top_agents": dict(agents.most_common(10)),
                  "first": bf._iso(min((r[2] or 0) for r in rows)), "last": bf._iso(max((r[3] or 0) for r in rows))}
    if not mts and not ka:
        return {"present": False}
    folders = ext.get("(folder_or_none)", 0)
    return {"present": True, "count": len(mts), "operator_filtered": op, "oldest": bf._iso(min(mts)) if mts else None,
            "newest": bf._iso(max(mts)) if mts else None, "folder_or_none": folders, "files": len(mts) - folders,
            "target_ext_top": dict(ext.most_common(10)), "age_mtime": {k: ages.get(k, 0) for k in bf.AGE_KEYS},
            "active_days_30d": len({time.strftime("%Y-%m-%d", time.localtime(t)) for t in mts if now - t < 30 * 86400}),
            "kactivity": ka}


@_lp("files.clutter", level="L2", family=FILES, tier="T1", collect="extended", gate="files.known_folders", timeout_ms=6000)
def files_clutter(h, facts):
    """Trash, ~/.cache and thumbnail sizes, this user's files in TMPDIR, Desktop item count (bounded walks)."""
    _prime_kf(h)
    out = {"present": True}
    tr = _data(h, "Trash")
    if _isdir(tr):
        st = bf._walk(os.path.join(tr, "files"), max_depth=6, max_entries=100000, budget_s=1.5)
        out["recycle_bin"] = {"items": len(_ls(os.path.join(tr, "info"), 100000) or []), "files": st["files"],
                              "bytes": st["bytes"], "truncated": st["truncated"]}
    c = _cachedir(h)
    if _isdir(c):
        st = bf._walk(c, max_depth=6, max_entries=150000, budget_s=1.5)
        out["cache_bytes"], out["cache_files"], out["cache_truncated"] = st["bytes"], st["files"], st["truncated"]
    th = bf._walk(_cachedir(h, "thumbnails"), max_depth=3, max_entries=100000, budget_s=0.8)
    out["thumbcache_bytes"] = th["bytes"]
    me, n, b = os.getuid(), [0], [0]

    def mine(e, s, depth):
        if s.st_uid == me:
            n[0] += 1
            b[0] += s.st_size
    tmp = tempfile_dir()
    st = bf._walk(tmp, max_depth=3, max_entries=50000, budget_s=1.0, on_file=mine,
                  skip_dir=lambda nm, p: nm == "userscan" or bool(_OP_DEEP.search(p)))
    out["temp_files"], out["temp_bytes"], out["temp_truncated"] = n[0], b[0], st["truncated"]
    desk = bf_content._kf(h, "Desktop")
    names = [x for x in h.list_dir(desk, 5000) if not x.startswith(".")]
    out["desktop_items"] = len(names)
    out["desktop_shortcuts"] = sum(1 for x in names if x.endswith(".desktop"))
    out["baloo_index_bytes"] = h.meta(_data(h, "baloo", "index")).get("bytes")
    return out


@_lp("files.download_sources", level="L2", family=FILES, tier="T1", collect="deep", gate="files.known_folders",
     timeout_ms=6000)
def files_download_sources(h, facts):
    """user.xdg.origin.url xattrs in Downloads (depth <= 2; written by Chrome/Firefox): source-site category counts
    (no hosts, no file names). The Linux counterpart of Zone.Identifier."""
    _prime_kf(h)
    root = bf_content._kf(h, "Downloads")
    if not _isdir(root):
        return {"present": False}
    cats, ext_cat = collections.Counter(), collections.Counter()
    n = with_x = 0
    deadline = time.perf_counter() + 4.0
    stack, trunc = [(root, 0)], None
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                if depth + 1 < 2 and not _OP_DEEP.search(e.path):
                    stack.append((e.path, depth + 1))
                continue
            n += 1
            if n > 5000 or time.perf_counter() > deadline:
                trunc = "cap"
                stack.clear()
                break
            try:
                url = os.getxattr(e.path, "user.xdg.origin.url", follow_symlinks=False).decode("utf-8", "replace")
            except (OSError, AttributeError):
                continue
            with_x += 1
            host, _ = bf._host_of(url)
            c = bf._categorize(bf_content._reg_domain(host)) if host else "other"
            c = "sensitive" if c in bf.SENSITIVE_CATS else c
            cats[c] += 1
            ext_cat[f"{c}:{os.path.splitext(e.name)[1].lower() or '(none)'}"] += 1
    return {"present": n > 0, "files_scanned": n, "depth": 2, "truncated": trunc, "with_origin_xattr": with_x,
            "source_categories": dict(cats.most_common()), "category_ext_top": dict(ext_cat.most_common(10))}


@_lp("files.profile_size_walk", level="L2", family=FILES, tier="T2", collect="deep", gate="files.known_folders",
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


@_lp("sync.other", level="L1", family=FILES, tier="T0", collect="core")
def sync_other(h, facts):
    """Sync clients: Dropbox, Nextcloud, ownCloud, Syncthing, MEGA, pCloud, Insync, onedrive (abraunegg), rclone, GOA."""
    U = _U(h)
    cands = {"dropbox": [os.path.join(U, ".dropbox")], "nextcloud": [_cfg(h, "Nextcloud")], "owncloud": [_cfg(h, "ownCloud")],
             "syncthing": [_state(h, "syncthing"), _cfg(h, "syncthing")], "mega": [_data(h, "data", "Mega Limited")],
             "pcloud": [os.path.join(U, ".pcloud")], "insync": [_data(h, "Insync"), _cfg(h, "Insync")],
             "onedrive": [_cfg(h, "onedrive")], "rclone": [_cfg(h, "rclone", "rclone.conf")],
             "gnome_online_accounts": [_cfg(h, "goa-1.0", "accounts.conf")], "kde_accounts": [_cfg(h, "libaccounts-glib", "accounts.db")],
             "google_drive_ocamlfuse": [os.path.join(U, ".gdfuse")], "proton_drive": [_cfg(h, "Proton Drive")]}
    found = [k for k, ps in cands.items() if _ex(*ps)]
    if "kde_accounts" in found:
        rows = h.sqlite(_cfg(h, "libaccounts-glib", "accounts.db"), "select count(*) from Accounts")
        if not rows or not rows[0][0]:
            found.remove("kde_accounts")
    return {"present": bool(found), "clients": found}


# ================================================================== gaming

def _steam_root(h):
    for p in (_data(h, "Steam"), os.path.join(_U(h), ".steam", "steam"), os.path.join(_U(h), ".steam", "debian-installation"),
              _flat(h, "com.valvesoftware.Steam", ".local", "share", "Steam"), _snapu(h, "steam", "common", ".local", "share", "Steam")):
        if _isdir(os.path.join(p, "steamapps")) or os.path.isfile(os.path.join(p, "steam.sh")):
            return os.path.realpath(p)
    return None


def _steam_lx(h):
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
    ug._cached(("steam", h.l0.get("run_id")), _steam_lx(h))


@_lp("steam.present", level="L1", family=GAMING, tier="T0", collect="core")
def steam_present(h, facts):
    """Steam client data dir (native ~/.local/share/Steam, ~/.steam, flatpak, snap); gates the Steam subtree."""
    sp = _steam_root(h)
    if not sp:
        return None
    via = "flatpak" if "/.var/app/" in sp else "snap" if "/snap/" in sp else "native"
    return {"present": True, "via": via, "proton_prefixes": len(_ls(os.path.join(sp, "steamapps", "compatdata"), 5000) or [])}


for _id, _fn in (("steam.installed", ug_games.steam_installed), ("steam.playtime", ug_games.steam_playtime),
                 ("steam.appinfo_genres", ug_games.steam_appinfo_genres), ("steam.local_sessions", ug_games.steam_local_sessions),
                 ("steam.non_steam_shortcuts", ug_games.steam_non_steam_shortcuts), ("steam.screenshots", ug_games.steam_screenshots),
                 ("steam.login_users", ug_games.steam_login_users), ("steam.remote_clients", ug_games.steam_remote_clients)):
    _mirror(_id, _fn, _prime_steam)


def _legendary_dir(h):
    return _first(_cfg(h, "heroic", "legendaryConfig", "legendary"), _cfg(h, "legendary"),
                  _flat(h, "com.heroicgameslauncher.hgl", "config", "heroic", "legendaryConfig", "legendary"))


@_lp("epic.present", level="L1", family=GAMING, tier="T0", collect="core")
def epic_present(h, facts):
    """Epic Games library via Heroic/legendary config (the Linux route to Epic)."""
    d = _legendary_dir(h)
    if not d:
        return None
    return {"present": True, "via": "heroic" if "heroic" in d else "legendary", "mtime": _mtime(d)}


@_lp("epic.installs", level="L2", family=GAMING, tier="T0", collect="core", gate="epic.present")
def epic_installs(h, facts):
    """Epic installs from legendary installed.json: titles and GB."""
    d = _legendary_dir(h)
    j = _read_json(os.path.join(d, "installed.json")) if d else None
    if not isinstance(j, dict) or not j:
        return {"present": False}
    rows = [[v.get("title") or k, round((v.get("install_size") or 0) / 1e9, 1)] for k, v in j.items() if isinstance(v, dict)]
    rows.sort(key=lambda r: -r[1])
    return {"present": True, "count": len(rows), "total_gb": round(sum(r[1] for r in rows), 1), "manifests": rows[:20]}


@_lp("launchers.other", level="L1", family=GAMING, tier="T0", collect="core")
def launchers_other(h, facts):
    """Linux game launchers: Heroic, Lutris, Bottles, itch, Minigalaxy, Prism/Minecraft, Wine prefix, ProtonUp-Qt."""
    U = _U(h)
    cands = {"heroic": [_cfg(h, "heroic"), _flat(h, "com.heroicgameslauncher.hgl")],
             "lutris": [_data(h, "lutris"), _cfg(h, "lutris"), _flat(h, "net.lutris.Lutris")],
             "bottles": [_data(h, "bottles"), _flat(h, "com.usebottles.bottles")], "itch": [_cfg(h, "itch")],
             "minigalaxy": [_cfg(h, "minigalaxy")], "gog_via_heroic": [_cfg(h, "heroic", "gog_store")],
             "prismlauncher": [_data(h, "PrismLauncher"), _flat(h, "org.prismlauncher.PrismLauncher")],
             "minecraft": [os.path.join(U, ".minecraft")], "wine": [os.path.join(U, ".wine")],
             "protonup_qt": [_flat(h, "net.davidotek.pupgui2"), _cfg(h, "pupgui")], "gamemode": ["/usr/bin/gamemoderun"],
             "mangohud": [_cfg(h, "MangoHud")]}
    hits = sorted(k for k, ps in cands.items() if _ex(*ps))
    return {"present": True, "launchers": hits} if hits else None


@_lp("emulators", level="L1", family=GAMING, tier="T0", collect="core")
def emulators(h, facts):
    """Emulator config dirs (RetroArch, Ryujinx, Dolphin, PCSX2, RPCS3, Cemu, DuckStation, PPSSPP, ...), native or flatpak."""
    c = {"RetroArch": [_cfg(h, "retroarch"), _flat(h, "org.libretro.RetroArch")], "Ryujinx": [_cfg(h, "Ryujinx"), _flat(h, "org.ryujinx.Ryujinx")],
         "yuzu/suyu": [_data(h, "yuzu"), _data(h, "suyu")], "Dolphin": [_data(h, "dolphin-emu"), _flat(h, "org.DolphinEmu.dolphin-emu")],
         "PCSX2": [_cfg(h, "PCSX2"), _flat(h, "net.pcsx2.PCSX2")], "RPCS3": [_cfg(h, "rpcs3"), _flat(h, "net.rpcs3.RPCS3")],
         "Cemu": [_data(h, "Cemu"), _flat(h, "info.cemu.Cemu")], "DuckStation": [_data(h, "duckstation"), _flat(h, "org.duckstation.DuckStation")],
         "PPSSPP": [_cfg(h, "ppsspp"), _flat(h, "org.ppsspp.PPSSPP")], "xemu": [_data(h, "xemu")], "melonDS": [_cfg(h, "melonDS")]}
    hits = sorted(k for k, ps in c.items() if _ex(*ps))
    return {"present": True, "found": hits} if hits else None


# ================================================================== media

@_lp("media.libraries", level="L1", family=MEDIA, tier="T1", collect="core")
def media_libraries(h, facts):
    """XDG Music/Videos/Pictures file counts by type and size (bounded walk), plus Screenshots count."""
    kf = bf._cached(h, "kf", _known_folders_lx)
    out = {}
    for short in ("Music", "Videos", "Pictures"):
        p = kf.get(short) or os.path.join(_U(h), short)
        s = ug_games._walk_media(p)
        s["gb"] = round(s.pop("bytes") / 1e9, 2)
        s["onedrive"] = False
        out[short] = s
    out["screenshots"] = max(h.count_dir(kf.get("Screenshots") or "", 20000), 0)
    return {"present": True, **out}


@_lp("media.obs", level="L1", family=MEDIA, tier="T0", collect="core")
def media_obs(h, facts):
    """OBS Studio: package/flatpak install and whether a config dir exists (launched at least once)."""
    inst = [k for k, p in (("deb", "/usr/bin/obs"), ("flatpak", "/var/lib/flatpak/app/com.obsproject.Studio"),
                           ("snap", "/snap/obs-studio")) if os.path.exists(p)]
    cfgs = [k for k, p in (("native", _cfg(h, "obs-studio")), ("flatpak", _flat(h, "com.obsproject.Studio", "config", "obs-studio")))
            if _isdir(p)]
    if not (inst or cfgs):
        return None
    return {"present": True, "obs_installs": inst, "obs_configs": cfgs, "config_dir": bool(cfgs),
            "obs_config_in_agent_container": 0, "streamlabs": False, "streamlabs_config": False}


@_lp("media.players", level="L1", family=MEDIA, tier="T0", collect="core")
def media_players(h, facts):
    """Music/video players and streaming apps (Spotify, Elisa, mpv, Strawberry, Kodi, Haruna, ...), native/snap/flatpak."""
    U = _U(h)
    c = {"spotify": [_cfg(h, "spotify"), "/snap/spotify", _flat(h, "com.spotify.Client"), "/usr/share/spotify"],
         "elisa": [_cfg(h, "elisarc"), "/usr/bin/elisa"], "mpv": [_cfg(h, "mpv"), "/usr/bin/mpv"],
         "strawberry": [_cfg(h, "strawberry")], "clementine": [_cfg(h, "Clementine")], "rhythmbox": [_data(h, "rhythmbox")],
         "audacious": [_cfg(h, "audacious")], "kodi": [os.path.join(U, ".kodi")], "plex": [_data(h, "plex"), "/snap/plex-desktop"],
         "jellyfin": [_data(h, "jellyfinmediaplayer"), _flat(h, "com.github.iwalton3.jellyfin-media-player")],
         "haruna": [_cfg(h, "haruna"), "/usr/bin/haruna"], "celluloid": [_cfg(h, "celluloid")], "tidal": [_cfg(h, "tidal-hifi")],
         "youtube_music": [_cfg(h, "YouTube Music")], "smplayer": [_cfg(h, "smplayer")]}
    hits = sorted(k for k, ps in c.items() if _ex(*ps))
    return {"present": True, "players": hits} if hits else None


def _vlc_cfg(h):
    return _first(_cfg(h, "vlc"), _snapu(h, "vlc", "common", ".config", "vlc"), _flat(h, "org.videolan.VLC", "config", "vlc"))


@_lp("media.vlc", level="L1", family=MEDIA, tier="T0", collect="core")
def media_vlc(h, facts):
    """VLC installed (deb/snap/flatpak) and whether a user config exists."""
    inst = _ex("/usr/bin/vlc", "/snap/vlc", "/var/lib/flatpak/app/org.videolan.VLC")
    cfg = _vlc_cfg(h)
    return {"present": True, "installed": inst, "user_config": bool(cfg)} if (inst or cfg) else None


@_lp("media.vlc_recents", level="L2", family=MEDIA, tier="T2", collect="extended", gate="media.vlc")
def media_vlc_recents(h, facts):
    """VLC recent-media count, local vs stream split and extension histogram; the MRL list itself is not emitted."""
    d = _vlc_cfg(h)
    qi = os.path.join(d, "vlc-qt-interface.conf") if d else None
    t = ug._read(qi, 2_000_000) if qi else None
    if not t:
        return {"present": False}
    m = re.search(r"^\[RecentsMRL\][^\[]*?^list=(.*)$", t, re.M | re.S)
    items = [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()] if m else []
    kinds = collections.Counter("local" if x.startswith("file:") else "stream" for x in items)
    exts = collections.Counter()
    for x in items:
        if x.startswith("file:"):
            m2 = re.search(r"\.([A-Za-z0-9]{1,5})$", x.split("?")[0])
            exts["." + m2.group(1).lower() if m2 else ""] += 1
    return {"present": bool(items), "count": len(items), "kinds": dict(kinds), "ext": dict(exts), "ini_mtime": _mtime(qi)}
