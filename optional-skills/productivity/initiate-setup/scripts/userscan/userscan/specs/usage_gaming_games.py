"""Gaming and media probes (Windows).

Registration only at import time. Shared helpers live in usage_gaming.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import re
import struct
import time

from ..registry import probe
from .usage_gaming import (_base, _cached, _ex, _ft, _iso, _mtime, _noise, _paths, _read,
    _reg_keys, _reg_values, _unmatch)

# ================================================================= GAMING

def _steam(h):
    """Shared Steam state: path, libraries, manifests, userdata ids, localconfig apps, appinfo buffer."""
    def load():
        sp = h.reg(r"HKCU\Software\Valve\Steam", "SteamPath")
        if not sp:
            cand = os.path.join(_paths()["PF86"], "Steam")
            sp = cand if _ex(cand) else None
        if not sp:
            return None
        sp = os.path.normpath(sp)
        libs = []
        txt = _read(os.path.join(sp, "steamapps", "libraryfolders.vdf"))
        if txt:
            for _k, v in (_ci(_vdf(txt), "libraryfolders") or {}).items():
                if isinstance(v, dict) and v.get("path"):
                    libs.append(os.path.normpath(v["path"]))
        libs = libs or [sp]
        installed = []
        for lib in libs[:20]:
            sa = os.path.join(lib, "steamapps")
            for n in h.list_dir(sa, 2000):
                if n.startswith("appmanifest_") and n.endswith(".acf"):
                    st = _ci(_vdf(_read(os.path.join(sa, n)) or ""), "AppState") or {}
                    installed.append({"appid": int(st.get("appid", 0) or 0), "name": st.get("name"),
                                      "size_gb": round(int(st.get("SizeOnDisk", 0) or 0) / 1e9, 1), "drive": lib[:2].upper(),
                                      "last_played": int(st.get("LastPlayed", 0) or 0)})
        ud = os.path.join(sp, "userdata")
        uids = [u for u in h.list_dir(ud, 50) if u.isdigit()]
        users = {}
        for uid in uids:
            lc = os.path.join(ud, uid, "config", "localconfig.vdf")
            apps = {}
            t = _read(lc)
            if t:
                d = _vdf(t)
                node = _ci(d, "UserLocalConfigStore", "Software", "Valve", "Steam", "apps") or {}
                for aid, v in node.items():
                    if isinstance(v, dict) and aid.isdigit() and ("Playtime" in v or "LastPlayed" in v):
                        apps[int(aid)] = {"min": int(v.get("Playtime", 0) or 0), "min2wk": int(v.get("Playtime2wks", 0) or 0),
                                          "last": int(v.get("LastPlayed", 0) or 0)}
            users[uid] = apps
        return {"path": sp, "libs": libs, "installed": installed, "uids": uids, "users": users,
                "appinfo_path": os.path.join(sp, "appcache", "appinfo.vdf")}
    return _cached(("steam", h.l0.get("run_id")), load)


_TOK = re.compile(r'"((?:[^"\\]|\\.)*)"|(\{)|(\})', re.S)


def _vdf(text):
    stack, key = [{}], None
    for m in _TOK.finditer(text):
        s, ob, cb = m.groups()
        if ob:
            d = {}
            stack[-1][key] = d
            stack.append(d)
            key = None
        elif cb:
            if len(stack) > 1:
                stack.pop()
        else:
            s = s.replace("\\\\", "\\")
            if key is None:
                key = s
            else:
                stack[-1][key] = s
                key = None
    return stack[0]


def _ci(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        lk = {kk.lower(): kk for kk in d}
        if k.lower() not in lk:
            return None
        d = d[lk[k.lower()]]
    return d


_GENRES = {"1": "Action", "2": "Strategy", "3": "RPG", "4": "Casual", "9": "Racing", "18": "Sports", "23": "Indie",
           "25": "Adventure", "28": "Simulation", "29": "Massively Multiplayer", "37": "Free to Play", "70": "Early Access",
           "51": "Animation & Modeling", "52": "Audio Production", "53": "Design & Illustration", "54": "Education",
           "55": "Photo Editing", "56": "Software Training", "57": "Utilities", "58": "Video Production",
           "59": "Web Publishing", "60": "Game Development"}


def _cstr(buf, i):
    j = buf.index(b"\x00", i)
    return buf[i:j].decode("utf-8", "replace"), j + 1


def _bkv(buf, i, strtab, depth=0):
    d = {}
    while True:
        t = buf[i]
        i += 1
        if t in (8, 11):
            return d, i
        if strtab is not None:
            (idx,) = struct.unpack_from("<I", buf, i)
            i += 4
            key = strtab[idx] if idx < len(strtab) else str(idx)
        else:
            key, i = _cstr(buf, i)
        if t == 0:
            if depth > 30:
                raise ValueError("bkv too deep")
            v, i = _bkv(buf, i, strtab, depth + 1)
        elif t == 1:
            v, i = _cstr(buf, i)
        elif t in (2, 4, 6):
            (v,) = struct.unpack_from("<i", buf, i)
            i += 4
        elif t == 3:
            (v,) = struct.unpack_from("<f", buf, i)
            i += 4
        elif t == 7:
            (v,) = struct.unpack_from("<Q", buf, i)
            i += 8
        elif t == 10:
            (v,) = struct.unpack_from("<q", buf, i)
            i += 8
        elif t == 5:
            j = i
            while buf[j:j + 2] != b"\x00\x00":
                j += 2
            v = buf[i:j].decode("utf-16-le", "replace")
            i = j + 2
        else:
            raise ValueError(f"bkv type {t}")
        d[key] = v


def _appinfo(h, want):
    """{appid: {name, type, genres}} for wanted appids, reading appinfo.vdf once per run."""
    st = _steam(h) or {}
    path = st.get("appinfo_path")

    def load():
        with open(path, "rb") as f:
            return f.read(200_000_000)
    buf = _cached(("appinfo_buf", h.l0.get("run_id")), load) if path and _ex(path) else None
    if not isinstance(buf, (bytes, bytearray)):
        return {}
    magic, _u = struct.unpack_from("<II", buf, 0)
    ver = magic & 0xFF
    off, strtab, end = 8, None, len(buf)
    if ver >= 0x29:
        (st_off,) = struct.unpack_from("<q", buf, 8)
        off, end = 16, st_off
        (n,) = struct.unpack_from("<I", buf, st_off)
        strtab, j = [], st_off + 4
        for _ in range(min(n, 500000)):
            s, j = _cstr(buf, j)
            strtab.append(s)
    out = {}
    while off < end - 8:
        appid, size = struct.unpack_from("<II", buf, off)
        if appid == 0:
            break
        body = off + 8
        if appid in want:
            kv_start = body + 4 + 4 + 8 + 20 + 4 + (20 if ver >= 0x28 else 0)
            try:
                kv, _ = _bkv(buf, kv_start, strtab)
                common = _ci(kv, "appinfo", "common") or {}
                g = common.get("genres") or {}
                out[appid] = {"name": common.get("name"), "type": common.get("type"),
                              "genres": [_GENRES.get(str(x), str(x)) for x in (g.values() if isinstance(g, dict) else [])]}
            except Exception:
                out[appid] = {"name": None, "type": None, "genres": []}
        off = body + size
    return out


@probe(id="steam.present", level="L1", family="gaming", tier="T0", collect="core")
def steam_present(h, facts):
    """Steam client installed (HKCU SteamPath or Program Files); gates the Steam subtree."""
    sp = h.reg(r"HKCU\Software\Valve\Steam", "SteamPath")
    if sp and _ex(sp):
        return {"present": True, "via": "registry"}
    if _ex(os.path.join(_paths()["PF86"], "Steam")):
        return {"present": True, "via": "path"}
    return None


def _lib_volume(p):
    """Volume holding a Steam library: drive letter ('D:') for a Windows path, else the POSIX mount point."""
    if re.match(r"^[A-Za-z]:", p or ""):
        return p[:2].upper()
    q = os.path.realpath(p) if p else "/"
    for _ in range(64):
        if os.path.ismount(q) or os.path.dirname(q) == q:
            return q
        q = os.path.dirname(q)
    return q


@probe(id="steam.installed", level="L2", family="gaming", tier="T1", collect="core", gate="steam.present")
def steam_installed(h, facts):
    """Installed Steam titles and GB from libraryfolders.vdf + appmanifest_*.acf."""
    st = _steam(h)
    if not st or "error" in st:
        return {"present": False, "error": (st or {}).get("error")}
    inst = sorted(st["installed"], key=lambda x: -x["size_gb"])
    games = [x for x in inst if not re.search(r"Redistributable|Dedicated Server|SDK|Proton|Steamworks", x["name"] or "", re.I)]
    return {"present": True, "libraries": len(st["libs"]), "drives": sorted({_lib_volume(p) for p in st["libs"]}),
            "apps": len(inst), "games": len(games), "total_gb": round(sum(x["size_gb"] for x in inst), 1),
            "top": [[x["name"], x["size_gb"], _iso(x["last_played"])] for x in games[:12]]}


@probe(id="steam.playtime", level="L2", family="gaming", tier="T1", collect="core", gate="steam.present")
def steam_playtime(h, facts):
    """Account-wide playtime per Steam userdata account from localconfig.vdf: hours, 2-week hours, top games."""
    st = _steam(h)
    if not st or not st.get("users"):
        return {"present": False}
    want = set()
    for apps in st["users"].values():
        want |= set(sorted(apps, key=lambda a: -apps[a]["min"])[:15])
    info = _appinfo(h, want) if want else {}
    names = {x["appid"]: x["name"] for x in st["installed"]}
    accounts = []
    for uid, apps in st["users"].items():
        rows = []
        for aid, v in apps.items():
            t = (info.get(aid) or {}).get("type")
            if t and t.lower() not in ("game", "demo", "mod", "beta"):
                continue
            rows.append((aid, v))
        rows.sort(key=lambda kv: -kv[1]["min"])
        lasts = [v["last"] for _, v in rows if v["last"]]
        accounts.append({
            "account": f"acct{len(accounts) + 1}", "games_with_record": len(rows),
            "games_played": sum(1 for _, v in rows if v["min"] > 0),
            "total_h": round(sum(v["min"] for _, v in rows) / 60, 1),
            "h_last_2wk": round(sum(v["min2wk"] for _, v in rows) / 60, 1),
            "played_7d": sum(1 for x in lasts if time.time() - x <= 7 * 86400),
            "played_30d": sum(1 for x in lasts if time.time() - x <= 30 * 86400),
            "last_played": _iso(max(lasts)) if lasts else None,
            "top": [[(info.get(a) or {}).get("name") or names.get(a) or f"app{a}", round(v["min"] / 60, 1), _iso(v["last"])]
                    for a, v in rows[:10] if v["min"]]})
    accounts.sort(key=lambda a: -a["total_h"])
    return {"present": True, "accounts": len(accounts), "total_hours": round(sum(a["total_h"] for a in accounts), 1),
            "played_last_30d": max((a["played_30d"] for a in accounts), default=0), "per_account": accounts}


@probe(id="steam.appinfo_genres", level="L2", family="gaming", tier="T0", collect="extended", gate="steam.present")
def steam_appinfo_genres(h, facts):
    """Genre hours per account from binary appinfo.vdf (static genre-id map)."""
    st = _steam(h)
    if not st or not st.get("users"):
        return {"present": False}
    want = set()
    for apps in st["users"].values():
        want |= {a for a, v in apps.items() if v["min"] > 0}
    info = _appinfo(h, want)
    out = []
    for i, apps in enumerate(st["users"].values()):
        gh = collections.Counter()
        for a, v in apps.items():
            for g in (info.get(a) or {}).get("genres", []):
                if g not in ("Free to Play", "Early Access"):
                    gh[g] += v["min"] / 60
        out.append({"account": f"acct{i + 1}", "genre_h": [[g, round(x, 1)] for g, x in gh.most_common(8)]})
    return {"present": True, "resolved": len(info), "unresolved": len(want - set(info)), "per_account": out}


@probe(id="steam.local_sessions", level="L2", family="gaming", tier="T1", collect="core", gate="steam.present")
def steam_local_sessions(h, facts):
    """Per-machine play sessions from gameprocess_log: hours, last-30-day hours, start hour and weekday histograms."""
    st = _steam(h)
    if not st:
        return {"present": False}
    rx_add = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] AppID (\d+) adding PID")
    rx_end = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Remove (\d+) from running list")
    active, sessions, first = {}, [], None
    for fn in ("gameprocess_log.previous.txt", "gameprocess_log.txt"):
        txt = _read(os.path.join(st["path"], "logs", fn), 20_000_000)
        if not txt:
            continue
        last_ts = None
        for line in txt.splitlines():
            if not line.startswith("[20"):
                continue
            if first is None:
                first = line[1:11]
            if "] Client version:" in line and active and last_ts:
                sessions += [(a, s0, last_ts) for a, s0 in active.items()]
                active = {}
            try:
                last_ts = dt.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            m = rx_add.match(line)
            if m:
                active.setdefault(int(m.group(2)), last_ts)
                continue
            m = rx_end.match(line)
            if m and int(m.group(2)) in active:
                sessions.append((int(m.group(2)), active.pop(int(m.group(2))), last_ts))
    dropped = sum(1 for _, s0, e0 in sessions if (e0 - s0).total_seconds() > 16 * 3600)
    sessions = [x for x in sessions if (x[2] - x[1]).total_seconds() <= 16 * 3600]
    if not sessions:
        return {"present": False, "log_starts": first}
    per = {}
    hh, dh = [0] * 24, [0] * 7
    now = dt.datetime.now()
    for a, s0, e0 in sessions:
        p = per.setdefault(a, [0, 0.0, ""])
        p[0] += 1
        p[1] += max(0, (e0 - s0).total_seconds()) / 3600
        p[2] = max(p[2], e0.strftime("%Y-%m-%d"))
        hh[s0.hour] += 1
        dh[s0.weekday()] += 1
    names = {x["appid"]: x["name"] for x in st["installed"]}
    top_ids = sorted(per, key=lambda a: -per[a][1])[:10]
    info = _appinfo(h, set(top_ids) - set(names))
    return {"present": True, "log_starts": first, "sessions": len(sessions), "dropped_over_16h": dropped, "apps": len(per),
            "total_hours": round(sum(p[1] for p in per.values()), 1),
            "hours_30d": round(sum(max(0, (e - s).total_seconds()) / 3600 for _, s, e in sessions if (now - e).days <= 30), 1),
            "start_hours": hh, "start_dow_hist_mon0": dh,
            "top": [[names.get(a) or (info.get(a) or {}).get("name") or f"app{a}", per[a][0], round(per[a][1], 1), per[a][2]]
                    for a in top_ids]}


@probe(id="steam.non_steam_shortcuts", level="L2", family="gaming", tier="T0", collect="core", gate="steam.present")
def steam_non_steam_shortcuts(h, facts):
    """Count of non-Steam game shortcuts (shortcuts.vdf entries) per account; count only."""
    st = _steam(h)
    if not st:
        return {"present": False}
    total, files = 0, 0
    for uid in st["uids"]:
        p = os.path.join(st["path"], "userdata", uid, "config", "shortcuts.vdf")
        try:
            with open(p, "rb") as f:
                data = f.read(5_000_000)
        except OSError:
            continue
        files += 1
        total += len(re.findall(rb"\x01(?i:appname)\x00", data))
    return {"present": files > 0, "files": files, "shortcuts": total}


@probe(id="steam.screenshots", level="L2", family="gaming", tier="T0", collect="core", gate="steam.present")
def steam_screenshots(h, facts):
    """Steam screenshot counts per account (userdata\\*\\760\\remote, depth <= 3)."""
    st = _steam(h)
    if not st:
        return {"present": False}
    counts = []
    for uid in st["uids"]:
        rem = os.path.join(st["path"], "userdata", uid, "760", "remote")
        n, seen = 0, 0
        for root, dirs, files in os.walk(rem):
            seen += 1
            if seen > 2000 or root.count(os.sep) - rem.count(os.sep) >= 3:
                dirs[:] = []
            n += sum(1 for f in files if f.lower().endswith((".jpg", ".png")))
        counts.append(n)
    return {"present": True, "accounts": len(counts), "screenshots": sum(counts), "per_account": counts}


@probe(id="steam.login_users", level="L2", family="gaming", tier="T2", collect="extended", gate="steam.present")
def steam_login_users(h, facts):
    """loginusers.vdf: account count, last-login dates, autologin; persona names (T2). Login AccountName never emitted."""
    st = _steam(h)
    if not st:
        return {"present": False}
    txt = _read(os.path.join(st["path"], "config", "loginusers.vdf"))
    if not txt:
        return {"present": False}
    users = []
    for _sid, v in (_ci(_vdf(txt), "users") or {}).items():
        if isinstance(v, dict):
            users.append({"persona_name": v.get("PersonaName"), "most_recent": v.get("MostRecent") == "1",
                          "auto_login": v.get("AutoLogin") == "1", "last_login": _iso(int(v.get("Timestamp", 0) or 0))})
    return {"present": bool(users), "count": len(users), "users": users}


@probe(id="steam.remote_clients", level="L2", family="gaming", tier="T2", collect="deep", gate="steam.present")
def steam_remote_clients(h, facts):
    """Steam Remote Play peers: peer count and hostnames of the user's other machines (T2)."""
    st = _steam(h)
    if not st:
        return {"present": False}
    txt = _read(os.path.join(st["path"], "config", "remoteclients.vdf"))
    if not txt:
        return {"present": False}
    hn = re.findall(r'"hostname"\s+"([^"]*)"', txt)
    return {"present": bool(hn), "count": len(hn), "hostnames": hn[:10]}


@probe(id="epic.present", level="L1", family="gaming", tier="T0", collect="core")
def epic_present(h, facts):
    """Epic Games Launcher data dir present; gates Epic subtree."""
    return {"present": True} if _ex(os.path.join(_paths()["PD"], "Epic", "EpicGamesLauncher")) else None


@probe(id="epic.installs", level="L2", family="gaming", tier="T0", collect="core", gate="epic.present")
def epic_installs(h, facts):
    """Epic installs: LauncherInstalled.dat, manifests (incl. Pending), game dirs with .egstore."""
    P = _paths()
    items = []
    t = _read(os.path.join(P["PD"], "Epic", "UnrealEngineLauncher", "LauncherInstalled.dat"))
    if t:
        try:
            items = [e.get("AppName") for e in json.loads(t).get("InstallationList", [])]
        except Exception:
            pass
    man = os.path.join(P["PD"], "Epic", "EpicGamesLauncher", "Data", "Manifests")
    manifests = []
    for sub in ("", "Pending"):
        d = os.path.join(man, sub)
        for n in h.list_dir(d, 500):
            if n.endswith(".item"):
                try:
                    j = json.loads(_read(os.path.join(d, n)) or "{}")
                    manifests.append([j.get("DisplayName"), round((j.get("InstallSize") or 0) / 1e9, 1), bool(sub)])
                except Exception:
                    pass
    dirs = []
    for base in (os.path.join(P["PF"], "Epic Games"), os.path.join(P["PF86"], "Epic Games")):
        for n in h.list_dir(base, 200):
            if n not in ("Launcher", "DirectXRedist", "Epic Online Services") and _ex(os.path.join(base, n, ".egstore")):
                dirs.append(n)
    return {"present": True, "launcher": _unmatch(h, r"Epic Games Launcher")[:1], "installed_list": items[:30],
            "manifests": manifests[:30], "pending": sum(1 for m in manifests if m[2]), "egstore_dirs": dirs[:30]}


@probe(id="xbox.present", level="L1", family="gaming", tier="T0", collect="core")
def xbox_present(h, facts):
    """Xbox app package dir present (inbox app, weak alone)."""
    pk = os.path.join(_paths()["LAD"], "Packages")
    app = _ex(os.path.join(pk, "Microsoft.GamingApp_8wekyb3d8bbwe"))
    bar = _ex(os.path.join(pk, "Microsoft.XboxGamingOverlay_8wekyb3d8bbwe"))
    return {"present": True, "xbox_app": app, "game_bar": bar} if (app or bar) else None


@probe(id="xbox.gamepass", level="L2", family="gaming", tier="T0", collect="core", gate="xbox.present")
def xbox_gamepass(h, facts):
    """Game Pass / Gaming Services installs: package repository, XboxGames dirs, ModifiableWindowsApps."""
    repo = _reg_keys(h, r"HKLM\SOFTWARE\Microsoft\GamingServices\PackageRepository\Root", 500)
    dirs = []
    for d in (r"C:\XboxGames", r"D:\XboxGames", r"E:\XboxGames"):
        dirs += [n for n in h.list_dir(d, 200) if n != "GameSave"]
    mwa = h.count_dir(os.path.join(_paths()["PF"], "ModifiableWindowsApps"), 500)
    return {"present": True, "gaming_services_repo": len(repo), "xboxgames_dirs": len(dirs),
            "modifiable_windows_apps": max(mwa, 0),
            "game_mode_auto": h.reg(r"HKCU\Software\Microsoft\GameBar", "AutoGameModeEnabled"),
            "gamedvr_capture": h.reg(r"HKCU\Software\Microsoft\Windows\CurrentVersion\GameDVR", "AppCaptureEnabled")}


@probe(id="gcs.present", level="L1", family="gaming", tier="T0", collect="core")
def gcs_present(h, facts):
    """GameConfigStore Children key present (Game Bar game classification); gates gcs.game_history."""
    n = len(_reg_keys(h, r"HKCU\System\GameConfigStore\Children", 5000))
    return {"present": True, "children": n} if n else None


@probe(id="gcs.game_history", level="L2", family="gaming", tier="T2", collect="core", gate="gcs.present")
def gcs_game_history(h, facts):
    """Game exes ever run (incl. since uninstalled) from GameConfigStore; game names and launcher, paths stripped."""
    base = r"HKCU\System\GameConfigStore\Children"
    by = {}
    for sk in _reg_keys(h, base, 2000):
        v = _reg_values(base + "\\" + sk, 100)
        exe = v.get("MatchedExeFullPath")
        if not exe or _noise(exe):
            continue
        low = exe.lower()
        launcher = ("steam" if "\\steamapps\\" in low else "epic" if "\\epic games\\" in low else
                    "riot" if "riot games" in low else "xbox" if ("xboxgames" in low or "windowsapps" in low) else
                    "ea" if ("\\ea games\\" in low or "electronic arts" in low) else "ubisoft" if "ubisoft" in low else
                    "rockstar" if "rockstar" in low else "gog" if "gog" in low else "other")
        m = re.search(r"\\(?:steamapps\\common|Epic Games|Riot Games|Rockstar Games|Games|XboxGames)\\([^\\]+)", exe, re.I)
        game = m.group(1) if m else (exe.split("\\")[1] if re.match(r"^[A-Z]:\\[^\\]+\\", exe) else _base(exe))
        la = v.get("LastAccessed")
        t = _ft(la) if isinstance(la, int) else None
        ts = t.timestamp() if t else None
        k = game.lower()
        cur = by.get(k)
        present = _ex(exe)
        if cur is None or (ts or 0) > (cur["ts"] or 0):
            by[k] = {"game": game, "launcher": launcher, "ts": ts, "present": present or (cur or {}).get("present", False)}
        elif present:
            cur["present"] = True
    if not by:
        return {"present": False}
    games = sorted(by.values(), key=lambda g: -(g["ts"] or 0))
    return {"present": True, "distinct_games": len(games),
            "uninstalled_since": sum(1 for g in games if not g["present"]),
            "last_30d": sum(1 for g in games if g["ts"] and time.time() - g["ts"] <= 30 * 86400),
            "by_launcher": dict(collections.Counter(g["launcher"] for g in games)),
            "games": [[g["game"], g["launcher"], _iso(g["ts"]), g["present"]] for g in games[:30]]}


@probe(id="launchers.other", level="L1", family="gaming", tier="T0", collect="core")
def launchers_other(h, facts):
    """Presence of Battle.net, Riot, EA, Ubisoft, GOG, Rockstar, Amazon, Playnite, Heroic launchers."""
    P = _paths()
    cands = {
        "battlenet": [os.path.join(P["PF86"], "Battle.net"), os.path.join(P["PD"], "Battle.net")],
        "riot": [r"C:\Riot Games", os.path.join(P["LAD"], "Riot Games")],
        "ea": [os.path.join(P["PD"], "EA Desktop"), os.path.join(P["RAD"], "Electronic Arts"), os.path.join(P["LAD"], "Electronic Arts")],
        "ubisoft": [os.path.join(P["PF86"], "Ubisoft"), os.path.join(P["LAD"], "Ubisoft Game Launcher")],
        "gog": [os.path.join(P["PF86"], "GOG Galaxy"), os.path.join(P["PD"], "GOG.com")],
        "rockstar": [os.path.join(P["PF"], "Rockstar Games"), os.path.join(P["LAD"], "Rockstar Games")],
        "amazon": [os.path.join(P["LAD"], "Amazon Games")],
        "playnite": [os.path.join(P["RAD"], "Playnite"), os.path.join(P["LAD"], "Playnite")],
        "heroic": [os.path.join(P["RAD"], "heroic")],
    }
    hits = sorted(k for k, ps in cands.items() if any(_ex(p) for p in ps))
    return {"present": True, "launchers": hits} if hits else None


@probe(id="emulators", level="L1", family="gaming", tier="T0", collect="core")
def emulators(h, facts):
    """Emulator config dirs (RetroArch, Ryujinx, yuzu, Dolphin, PCSX2, RPCS3, Cemu, ...)."""
    P = _paths()
    docs = os.path.join(P["HOME"], "Documents")
    cands = {"RetroArch": [os.path.join(P["RAD"], "RetroArch"), r"C:\RetroArch-Win64"], "Ryujinx": [os.path.join(P["RAD"], "Ryujinx")],
             "yuzu/suyu": [os.path.join(P["RAD"], "yuzu"), os.path.join(P["RAD"], "suyu")],
             "Dolphin": [os.path.join(P["RAD"], "Dolphin Emulator"), os.path.join(docs, "Dolphin Emulator")],
             "PCSX2": [os.path.join(P["RAD"], "PCSX2"), os.path.join(docs, "PCSX2")], "RPCS3": [os.path.join(P["RAD"], "rpcs3")],
             "Cemu": [os.path.join(P["RAD"], "Cemu")], "DuckStation": [os.path.join(docs, "DuckStation")],
             "PPSSPP": [os.path.join(docs, "PPSSPP")], "xemu": [os.path.join(P["RAD"], "xemu")],
             "Xenia": [os.path.join(docs, "Xenia")], "melonDS": [os.path.join(P["RAD"], "melonDS")]}
    hits = sorted(k for k, ps in cands.items() if any(_ex(p) for p in ps))
    return {"present": True, "found": hits} if hits else None


@probe(id="gaming.anticheat", level="L1", family="gaming", tier="T0", collect="core")
def gaming_anticheat(h, facts):
    """Anti-cheat drivers/dirs (BattlEye, EasyAntiCheat, Vanguard, FACEIT): competitive-shooter hint."""
    P = _paths()
    cands = {"BattlEye": [os.path.join(P["LAD"], "BattlEye"), os.path.join(P["PF86"], "Common Files", "BattlEye")],
             "EasyAntiCheat": [os.path.join(P["PF86"], "EasyAntiCheat_EOS"), os.path.join(P["PF86"], "EasyAntiCheat")],
             "Vanguard": [os.path.join(P["PF"], "Riot Vanguard")], "FACEIT": [os.path.join(P["PF"], "FACEIT AC")]}
    hits = sorted(k for k, ps in cands.items() if any(_ex(p) for p in ps))
    return {"present": True, "found": hits} if hits else None


@probe(id="gaming.secondary_drive", level="L1", family="gaming", tier="T0", collect="core")
def gaming_secondary_drive(h, facts):
    """Game folders on D:/E: (Games, SteamLibrary, XboxGames, Epic Games, GOG Games) with entry counts."""
    out = []
    for drv in ("D:\\", "E:\\", "F:\\"):
        if not _ex(drv):
            continue
        for n in ("Games", "SteamLibrary", "XboxGames", "Epic Games", "GOG Games"):
            p = os.path.join(drv, n)
            if _ex(p):
                out.append([p, max(h.count_dir(p, 1000), 0)])
    return {"present": True, "dirs": out} if out else None


@probe(id="gaming.save_roots", level="L1", family="gaming", tier="T0", collect="core")
def gaming_save_roots(h, facts):
    """My Games / Saved Games / LocalLow roots exist; gates gaming.save_dirs."""
    home = os.path.expanduser("~")
    roots = [p for p in (os.path.join(home, "Documents", "My Games"), os.path.join(home, "Saved Games"),
                         os.path.join(home, "AppData", "LocalLow")) if _ex(p)]
    return {"present": True, "roots": len(roots)} if roots else None


@probe(id="gaming.save_dirs", level="L2", family="gaming", tier="T2", collect="extended", gate="gaming.save_roots")
def gaming_save_dirs(h, facts):
    """Game titles/publishers from My Games, Saved Games and LocalLow dir names (bounded listdir)."""
    home = os.path.expanduser("~")
    skip = {"microsoft", "nvidia", "intel", "adobe", "com.adobe.crashreporter", "temp", "desktop.ini", "sun", "oracle",
            "unity", "google", "mozilla", "igdump"}
    out = {}
    for key, p in (("my_games", os.path.join(home, "Documents", "My Games")), ("saved_games", os.path.join(home, "Saved Games")),
                   ("locallow", os.path.join(home, "AppData", "LocalLow"))):
        names = [n for n in h.list_dir(p, 300) if n.lower() not in skip and not n.endswith(".ini")
                 and not re.fullmatch(r"[0-9a-fA-F-]{32,}", n) and not _noise("\\" + n + "\\")]
        out[key] = sorted(names)[:40]
    return {"present": any(out.values()), **out}


# ================================================================= MEDIA

@probe(id="nvidia.app", level="L1", family="hardware", tier="T0", collect="core")
def nvidia_app(h, facts):
    """NVIDIA App backend dir present; gates the NVIDIA subtree."""
    return {"present": True} if _ex(os.path.join(_paths()["LAD"], "NVIDIA Corporation", "NVIDIA App", "NvBackend")) else None


@probe(id="nvidia.library", level="L2", family="gaming", tier="T0", collect="core", gate="nvidia.app")
def nvidia_library(h, facts):
    """Apps and games detected by NVIDIA App (ApplicationStorage.json)."""
    p = os.path.join(_paths()["LAD"], "NVIDIA Corporation", "NVIDIA App", "NvBackend", "ApplicationStorage.json")
    t = _read(p)
    if not t:
        return {"present": False}
    try:
        j = json.loads(t)
    except Exception:
        return {"present": False, "error": "json"}
    names = []
    for a in j.get("Applications", [])[:500]:
        ap = a.get("Application", a) if isinstance(a, dict) else {}
        if ap.get("DisplayName"):
            names.append(ap["DisplayName"])
    return {"present": bool(names), "count": len(names), "apps": names[:40], "mtime": _mtime(p),
            "app_version": (_unmatch(h, r"^NVIDIA App \d") or [{}])[0].get("version")}


@probe(id="nvidia.recommendations", level="L2", family="gaming", tier="T2", collect="core", gate="nvidia.app")
def nvidia_recommendations(h, facts):
    """Per-game optimisation dirs in NvBackend\\Recommendations; persist after uninstall (game history)."""
    d = os.path.join(_paths()["LAD"], "NVIDIA Corporation", "NVIDIA App", "NvBackend", "Recommendations")
    names = sorted(n for n in h.list_dir(d, 500) if os.path.isdir(os.path.join(d, n)))
    return {"present": bool(names), "count": len(names), "games": names[:60]}


@probe(id="captures.nvidia", level="L2", family="media", tier="T2", collect="core", gate="nvidia.app")
def captures_nvidia(h, facts):
    """NVIDIA overlay captures: file count and size; per-game folder counts (folder names are game titles)."""
    P = _paths()
    capdir = None
    t = _read(os.path.join(P["LAD"], "NVIDIA Corporation", "NVIDIA Overlay", "GallerySettings.json"))
    if t:
        try:
            capdir = json.loads(t).get("settings", {}).get("currentDirectoryV2")
        except Exception:
            pass
    capdir = capdir or os.path.join(P["HOME"], "Videos", "NVIDIA")
    if not _ex(capdir):
        return {"present": False}
    n = b = seen = 0
    per = {}
    for root, dirs, files in os.walk(capdir):
        seen += 1
        if seen > 500 or root.count(os.sep) - capdir.count(os.sep) >= 2:
            dirs[:] = []
        for f in files[:5000]:
            n += 1
            try:
                b += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
        if root != capdir and files:
            per[os.path.basename(root)] = len(files)
    return {"present": True, "dir_is_default": capdir.lower().endswith("videos\\nvidia"), "files": n,
            "mb": round(b / 1e6), "game_folders": dict(sorted(per.items(), key=lambda kv: -kv[1])[:20])}


@probe(id="captures.gamebar", level="L2", family="media", tier="T0", collect="core", gate="xbox.present")
def captures_gamebar(h, facts):
    """Game Bar captures dir (Videos\\Captures): file count and size."""
    cap = os.path.join(os.path.expanduser("~"), "Videos", "Captures")
    n = b = 0
    try:
        with os.scandir(cap) as it:
            for e in it:
                if n >= 5000:
                    break
                if e.is_file():
                    n += 1
                    b += e.stat().st_size
    except OSError:
        return {"present": False}
    return {"present": True, "files": n, "mb": round(b / 1e6)}


@probe(id="nvidia.shadowplay", level="L1", family="media", tier="T0", collect="core")
def nvidia_shadowplay(h, facts):
    """NVIDIA overlay/instant replay/highlights/mic flags (NVSPCAPS REG_BINARY dwords)."""
    sp = r"HKCU\Software\NVIDIA Corporation\Global\ShadowPlay\NVSPCAPS"
    out = {}
    for k, name in (("overlay_enabled", "IsShadowPlayEnabledUser"), ("instant_replay_or_rec", "RecEnabled"),
                    ("highlights", "HLEnabled"), ("mic", "EnableMicrophone")):
        v = h.reg(sp, name)
        if isinstance(v, (bytes, bytearray)) and len(v) >= 4:
            v = int.from_bytes(v[:4], "little")
        if v is not None:
            out[k] = v
    return {"present": True, **out} if out else None


@probe(id="nvidia.broadcast", level="L1", family="media", tier="T0", collect="core")
def nvidia_broadcast(h, facts):
    """NVIDIA Broadcast / G-Assist presence."""
    P = _paths()
    b = any(_ex(p) for p in (os.path.join(P["PD"], "NVIDIA Corporation", "NVIDIA Broadcast"),
                             os.path.join(P["PF"], "NVIDIA Corporation", "NVIDIA Broadcast")))
    g = _ex(os.path.join(P["PD"], "NVIDIA Corporation", "nvtopps", "rise")) or _ex(os.path.join(P["PF"], "NVIDIA Corporation", "NVIDIA G-Assist"))
    return {"present": True, "broadcast": b, "g_assist": g} if (b or g) else None


@probe(id="media.obs", level="L1", family="media", tier="T0", collect="core")
def media_obs(h, facts):
    """OBS/Streamlabs presence and whether a config dir exists (launched at least once); agent-container copies flagged."""
    P = _paths()
    installs, configs = [], []
    if _ex(os.path.join(P["PF"], "obs-studio")):
        installs.append("program_files")
    wg = os.path.join(P["LAD"], "Microsoft", "WinGet", "Packages")
    for n in h.list_dir(wg, 1000):
        if n.lower().startswith("obsproject.obsstudio"):
            installs.append("winget_portable")
            if _ex(os.path.join(wg, n, "config", "obs-studio")):
                configs.append("winget_portable")
    if _ex(os.path.join(P["RAD"], "obs-studio")):
        configs.append("roaming")
    agent_cfg = 0
    for n in h.list_dir(os.path.join(P["LAD"], "Packages"), 2000):
        if n.lower().startswith(("openai.codex", "anthropic.claude")):
            if _ex(os.path.join(P["LAD"], "Packages", n, "LocalCache", "Roaming", "obs-studio")):
                agent_cfg += 1
    sl = _ex(os.path.join(P["PF"], "Streamlabs OBS"))
    sl_cfg = _ex(os.path.join(P["RAD"], "slobs-client"))
    if not (installs or configs or sl or agent_cfg):
        return None
    return {"present": True, "obs_installs": installs, "obs_configs": configs, "config_dir": bool(configs), "obs_config_in_agent_container": agent_cfg,
            "streamlabs": sl, "streamlabs_config": sl_cfg}


@probe(id="media.players", level="L1", family="media", tier="T0", collect="core")
def media_players(h, facts):
    """Music/video players and streaming apps (Spotify, Apple Music, Store players, mpv, Plex, ...)."""
    P = _paths()
    pkgs = [n.lower() for n in h.list_dir(os.path.join(P["LAD"], "Packages"), 3000)]

    def pkg(prefix):
        return any(n.startswith(prefix.lower()) for n in pkgs)
    found = {"spotify": _ex(os.path.join(P["RAD"], "Spotify")) or pkg("SpotifyAB.SpotifyMusic"),
             "apple_music": pkg("AppleInc.AppleMusic"), "media_player_uwp": pkg("Microsoft.ZuneMusic"),
             "netflix": pkg("4DF9E0F8.Netflix"), "prime_video": pkg("AmazonVideo.PrimeVideo"), "disney": pkg("Disney."),
             "mpv": _ex(os.path.join(P["RAD"], "mpv")), "mpc_hc": _ex(os.path.join(P["PF"], "MPC-HC")),
             "potplayer": _ex(os.path.join(P["PF"], "DAUM", "PotPlayer")), "foobar2000": _ex(os.path.join(P["RAD"], "foobar2000")),
             "musicbee": _ex(os.path.join(P["RAD"], "MusicBee")), "plex": _ex(os.path.join(P["LAD"], "Plex")),
             "kodi": _ex(os.path.join(P["RAD"], "Kodi")), "tidal": _ex(os.path.join(P["LAD"], "TIDAL"))}
    hits = sorted(k for k, v in found.items() if v)
    return {"present": True, "players": hits} if hits else None


@probe(id="media.vlc", level="L1", family="media", tier="T0", collect="core")
def media_vlc(h, facts):
    """VLC installed and whether a user config exists."""
    P = _paths()
    inst = _ex(os.path.join(P["PF"], "VideoLAN", "VLC")) or _ex(os.path.join(P["PF86"], "VideoLAN", "VLC"))
    cfg = _ex(os.path.join(P["RAD"], "vlc"))
    return {"present": True, "installed": inst, "user_config": cfg} if (inst or cfg) else None


@probe(id="media.vlc_recents", level="L2", family="media", tier="T2", collect="extended", gate="media.vlc")
def media_vlc_recents(h, facts):
    """VLC recent-media count, local vs stream split and extension histogram; the MRL list itself is not emitted."""
    qi = os.path.join(_paths()["RAD"], "vlc", "vlc-qt-interface.ini")
    t = _read(qi, 2_000_000)
    if not t:
        return {"present": False}
    m = re.search(r"^\[RecentsMRL\][^\[]*?^list=(.*)$", t, re.M | re.S)
    items = [x.strip() for x in m.group(1).split(",") if x.strip()] if m else []
    kinds = collections.Counter("local" if x.startswith("file:") else "stream" for x in items)
    exts = collections.Counter()
    for x in items:
        if x.startswith("file:"):
            m2 = re.search(r"\.([A-Za-z0-9]{1,5})$", x.split("?")[0])
            exts["." + m2.group(1).lower() if m2 else ""] += 1
    own = sum(1 for x in items if "/Videos/Captures/" in x or "/Videos/NVIDIA/" in x)
    return {"present": bool(items), "count": len(items), "kinds": dict(kinds), "ext": dict(exts),
            "from_own_capture_dirs": own, "ini_mtime": _mtime(qi)}


_MEDIA_EXT = {"video": {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".m4v", ".flv", ".ts"},
              "audio": {".mp3", ".flac", ".m4a", ".wav", ".ogg", ".opus", ".aac", ".wma"},
              "image": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp", ".tif", ".tiff", ".raw", ".dng", ".jxr"}}


def _walk_media(root, max_depth=4, max_files=20000, budget_s=1.5):
    t0 = time.perf_counter()
    st = {"files": 0, "bytes": 0, "video": 0, "audio": 0, "image": 0, "cloud_only": 0, "truncated": False}
    stack = [(root, 0)]
    while stack:
        p, dep = stack.pop()
        if _noise(p + "\\"):
            continue
        try:
            it = os.scandir(p)
        except OSError:
            continue
        with it:
            for e in it:
                if time.perf_counter() - t0 > budget_s or st["files"] >= max_files:
                    st["truncated"] = True
                    return st
                try:
                    if e.is_dir(follow_symlinks=False):
                        if dep < max_depth:
                            stack.append((e.path, dep + 1))
                        continue
                    s = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                st["files"] += 1
                if getattr(s, "st_file_attributes", 0) & (0x400000 | 0x1000):
                    st["cloud_only"] += 1
                else:
                    st["bytes"] += s.st_size
                ext = os.path.splitext(e.name)[1].lower()
                for k, exts in _MEDIA_EXT.items():
                    if ext in exts:
                        st[k] += 1
                        break
    return st


@probe(id="media.libraries", level="L1", family="media", tier="T1", collect="core")
def media_libraries(h, facts):
    """Music/Videos/Pictures known-folder file counts by type and size (bounded walk), plus Screenshots count."""
    home = os.path.expanduser("~")
    usf = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
    out = {}
    for short, regname in (("Music", "My Music"), ("Videos", "My Video"), ("Pictures", "My Pictures")):
        p = os.path.expandvars(h.reg(usf, regname) or os.path.join(home, short))
        s = _walk_media(p)
        s["gb"] = round(s.pop("bytes") / 1e9, 2)
        s["onedrive"] = "onedrive" in p.lower()
        out[short] = s
    out["screenshots"] = max(h.count_dir(os.path.join(home, "Pictures", "Screenshots"), 20000), 0)
    return {"present": True, **out}
