"""Files and content probes (files family, Windows): known folders, composition, screenshots.

Registration only at import time. Shared helpers and caches live in browser_files.
"""
from __future__ import annotations

import collections
import os
import re
import statistics
import time

from .browser_files import (AGE_KEYS, FILE_ATTRIBUTE_REPARSE_POINT, HKCU_EXPLORER,
    OPERATOR_DEEP_RX, OPERATOR_RX, SENSITIVE_CATS, _age_bucket, _cache_ms, _cached, _categorize, _env, _home,
    _host_of, _isdir, _iso, _probe, _redact, _reg_key, _reg_lastwrite, _reg_values, _walk, _winreg)

# ================================================================ family: files

KF_GUIDS = {
    "Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}", "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "Downloads": "{374DE290-123F-4565-9164-39C4925E467B}", "Pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "Videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}", "Music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "Screenshots": "{B7BEDE81-DF94-4682-A7D8-57A52620B86F}", "Captures": "{EDC0FE71-98D8-4F4A-B920-C8DC133CB165}",
    "CameraRoll": "{AB5FB87B-7CE2-4F83-915D-550846C9537B}",
}


def _known_folders(h):
    import ctypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort),
                    ("Data4", ctypes.c_ubyte * 8)]
    out = {}
    for name, g in KF_GUIDS.items():
        guid = GUID()
        ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(g), ctypes.byref(guid))
        p = ctypes.c_wchar_p()
        try:
            ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0x4000, None, ctypes.byref(p))  # DONT_VERIFY
            out[name] = p.value
        except OSError:
            out[name] = None
        finally:
            if p:
                ctypes.windll.ole32.CoTaskMemFree(p)
    return out


def _kf(h, name):
    try:
        p = _cached(h, "kf", _known_folders).get(name)
    except RuntimeError:
        p = None
    return p or os.path.join(_home(), name)


@_probe(id="files.known_folders", level="L1", family="files", tier="T0", collect="core")
def files_known_folders(h, facts):
    """Known-folder paths (SHGetKnownFolderPath, DONT_VERIFY): exists, under profile, under OneDrive."""
    kf = _cached(h, "kf", _known_folders)
    home = _home().lower()
    rows = {n: {"path": _redact(p), "exists": bool(p and _isdir(p)),
                "under_profile": bool(p and p.lower().startswith(home)),
                "under_onedrive": bool(p and "onedrive" in p.lower())} for n, p in kf.items()}
    return {"present": any(r["exists"] for r in rows.values()), "folders": rows}


@_probe(id="sync.other", level="L1", family="files", tier="T0", collect="core")
def sync_other(h, facts):
    """Other sync clients: Dropbox, Google Drive, iCloud, Box, MEGA, Proton Drive, pCloud, Syncthing."""
    cands = {"dropbox": r"%LOCALAPPDATA%\Dropbox\info.json", "google_drive": r"%LOCALAPPDATA%\Google\DriveFS",
             "icloud": r"%LOCALAPPDATA%\Apple Inc\iCloud", "box": r"%LOCALAPPDATA%\Box",
             "mega": r"%LOCALAPPDATA%\MEGAsync", "proton_drive": r"%LOCALAPPDATA%\Proton\Proton Drive",
             "pcloud": r"%APPDATA%\pCloud", "syncthing": r"%LOCALAPPDATA%\Syncthing"}
    found = [k for k, p in cands.items() if h.exists(p)]
    roots = h.reg(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager") or []
    providers = sorted({r.split("!")[0] for r in roots if not r.lower().startswith("onedrive")})
    return {"present": bool(found), "clients": found, "sync_root_providers_non_onedrive": providers[:10]}


@_probe(id="jumplists.counts", level="L1", family="files", tier="T0", collect="core")
def jumplists_counts(h, facts):
    """Jump-list file counts and sizes (Automatic/CustomDestinations); entries never parsed."""
    base = h.expand(r"%APPDATA%\Microsoft\Windows\Recent")
    out = {}
    for sub in ("AutomaticDestinations", "CustomDestinations"):
        n = b = 0
        try:
            with os.scandir(os.path.join(base, sub)) as it:
                for i, e in enumerate(it):
                    if i >= 5000:
                        break
                    if e.is_file():
                        n += 1
                        b += e.stat().st_size
        except OSError:
            continue
        out[sub] = {"files": n, "bytes": b}
    return {"present": bool(out), **out}


@_probe(id="onedrive.kfm", level="L1", family="files", tier="T0", collect="core", gate="onedrive.accounts")
def onedrive_kfm(h, facts):
    """Known-folder move: Desktop/Documents/Pictures redirected into OneDrive (User Shell Folders)."""
    usf = _reg_values(HKCU_EXPLORER + r"\User Shell Folders") or {}
    red = [k for k in ("Desktop", "Personal", "My Pictures", "My Video", "My Music") if "onedrive" in str(usf.get(k, "")).lower()]
    return {"present": True, "redirected": bool(red), "folders": red}


@_probe(id="onedrive.root_size", level="L2", family="files", tier="T0", collect="core", gate="onedrive.accounts")
def onedrive_root_size(h, facts):
    """Bounded size of %USERPROFILE%\\OneDrive* without hydrating placeholders."""
    home = _home()
    roots = {}
    for name in h.list_dir(home, 500):
        if name.lower().startswith("onedrive") and _isdir(os.path.join(home, name)):
            st = _walk(os.path.join(home, name), max_depth=6, max_entries=50000, budget_s=2.0)
            roots[name] = {k: st[k] for k in ("files", "bytes", "cloud_placeholders", "truncated")}
    if not roots:
        return {"present": False}
    return {"present": True, "roots": roots, "files": sum(r["files"] for r in roots.values()),
            "bytes": sum(r["bytes"] for r in roots.values())}


def _recent(h):
    rec = h.expand(r"%APPDATA%\Microsoft\Windows\Recent")
    now = time.time()
    ext, ages, mts = collections.Counter(), collections.Counter(), []
    operator = 0
    try:
        it = os.scandir(rec)
    except OSError:
        return None
    with it:
        for i, e in enumerate(it):
            if i >= 20000:
                break
            if not e.name.lower().endswith(".lnk") or not e.is_file():
                continue
            if OPERATOR_DEEP_RX.search(e.name) or OPERATOR_RX.match(e.name[:-4]):
                operator += 1
                continue
            try:
                s = e.stat()
            except OSError:
                continue
            mts.append(s.st_mtime)
            ages[_age_bucket(now - s.st_mtime)] += 1
            x = os.path.splitext(e.name[:-4])[1].lower()
            ext[x if re.fullmatch(r"\.[a-z0-9]{1,5}", x or "") else "(folder_or_none)"] += 1
    return {"count": len(mts), "operator_filtered": operator, "ext": ext, "ages": ages, "mts": mts}


@_probe(id="files.recent_lnk", level="L2", family="files", tier="T1", collect="core", gate="files.known_folders")
def files_recent_lnk(h, facts):
    """Recent .lnk count, date range, folder vs file split, target extension counts (no names)."""
    r = _recent(h)
    if not r or not r["count"]:
        return {"present": False}
    now = time.time()
    mts = r["mts"]
    folders = r["ext"].get("(folder_or_none)", 0)
    return {"present": True, "count": r["count"], "operator_filtered": r["operator_filtered"],
            "oldest": _iso(min(mts)), "newest": _iso(max(mts)), "folder_or_none": folders,
            "files": r["count"] - folders, "target_ext_top": dict(r["ext"].most_common(10)),
            "age_mtime": {k: r["ages"].get(k, 0) for k in AGE_KEYS},
            "active_days_30d": len({time.strftime("%Y-%m-%d", time.localtime(t)) for t in mts if now - t < 30 * 86400})}


SHOT_NAME = re.compile(r"^(screenshot|screen shot|capture|snip|screen ?recording|scr_|shot_)", re.I)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".jxr", ".heic", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".m4v"}
AUDIO_EXT = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus"}
DOC_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".md", ".csv", ".odt", ".rtf"}
CODE_EXT = {".py", ".js", ".ts", ".json", ".toml", ".yaml", ".yml", ".sh", ".ps1", ".bat", ".rs", ".go", ".c", ".cpp",
            ".h", ".java", ".ipynb", ".log", ".rst", ".html", ".css"}
INSTALLER_EXT = {".exe", ".msi", ".msix", ".appx", ".msixbundle", ".appxbundle"}
ARCHIVE_EXT = {".zip", ".7z", ".rar", ".tar", ".gz", ".iso", ".xz", ".bz2", ".zst"}


def _kind(ext):
    for k, s in (("image", IMG_EXT), ("video", VIDEO_EXT), ("audio", AUDIO_EXT), ("document", DOC_EXT),
                 ("code", CODE_EXT), ("installer", INSTALLER_EXT), ("archive", ARCHIVE_EXT)):
        if ext in s:
            return k
    return "other"


def _composition(h):
    now = time.time()
    deadline = time.perf_counter() + 8.0
    shots = []
    seen = set()
    res = {}
    for name in ("Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music"):
        p = _kf(h, name)
        ext_n, ext_b, kind_b = collections.Counter(), collections.Counter(), collections.Counter()
        ages, years = collections.Counter(), collections.Counter()
        dup = [0]

        def on_file(e, s, depth, label=name):
            ext = os.path.splitext(e.name)[1].lower() or "(none)"
            if len(ext) > 12:
                ext = "(long)"
            ext_n[ext] += 1
            ext_b[ext] += s.st_size
            kind_b[_kind(ext)] += s.st_size
            ages[_age_bucket(now - s.st_mtime)] += 1
            years[time.localtime(s.st_mtime).tm_year] += 1
            if re.search(r"\(\d+\)(\.[^.]+)?$", e.name):
                dup[0] += 1
            if ext in IMG_EXT or ext in (".mp4", ".mkv"):
                low = e.path.lower()
                if (SHOT_NAME.match(e.name) or any(s_ in low for s_ in ("\\screenshots\\", "\\captures\\", "\\sharex\\",
                                                                        "\\ansel\\"))) and low not in seen:
                    seen.add(low)
                    shots.append((s.st_mtime, label, ext in IMG_EXT))
        budget = max(0.2, min(3.0, deadline - time.perf_counter()))
        st = _walk(p, max_depth=6, max_entries=100000, budget_s=budget, on_file=on_file,
                   skip_dir=lambda n, path: bool(OPERATOR_DEEP_RX.search(n)))
        top = {"files": 0, "dirs": 0, "shortcuts": 0}
        for e in h.list_dir(p, 5000):
            full = os.path.join(p, e)
            if e.lower() == "desktop.ini":
                continue
            if _isdir(full):
                top["dirs"] += 1
            else:
                top["files"] += 1
                top["shortcuts"] += e.lower().endswith((".lnk", ".url"))
        res[name] = {"files": st["files"], "dirs": st["dirs"], "bytes": st["bytes"], "truncated": st["truncated"],
                     "depth": 6, "top_level": top, "ext_top_by_count": dict(ext_n.most_common(10)),
                     "ext_top_by_bytes": dict(ext_b.most_common(6)), "kind_bytes": dict(kind_b),
                     "age_mtime": {k: ages.get(k, 0) for k in AGE_KEYS},
                     "oldest_year": min(years) if years else None, "newest_year": max(years) if years else None,
                     "dup_suffix_files": dup[0], "ms": st.get("ms")}
    return {"folders": res, "shots": shots}


@_probe(id="files.composition", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders",
        timeout_ms=9000)
def files_composition(h, facts):
    """Per known folder: files, bytes, extension/kind/age histograms, depth <= 6 (no file names)."""
    c = _cached(h, "composition", _composition)
    folders = c["folders"]
    kind = collections.Counter()
    for f in folders.values():
        kind.update(f["kind_bytes"])
    tot = sum(kind.values()) or 1
    dl = folders.get("Downloads", {})
    old = sum(dl.get("age_mtime", {}).get(k, 0) for k in ("lt_1y", "lt_2y", "ge_2y"))
    return {"present": True, "folders": folders, "total_bytes": sum(f["bytes"] for f in folders.values()),
            "byte_share": {k: round(v / tot, 3) for k, v in kind.most_common()},
            "downloads_older_90d_share": round(old / dl["files"], 3) if dl.get("files") else None,
            "shared_block_ms": _cache_ms(h, "composition")}


@_probe(id="files.screenshots", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders",
        timeout_ms=9000)
def files_screenshots(h, facts):
    """Screenshot/capture count, active days, cadence and local hour/weekday histograms (from the composition walk)."""
    c = _cached(h, "composition", _composition)
    imgs = sorted(t for t, _l, is_img in c["shots"] if is_img)
    vids = sum(1 for _t, _l, is_img in c["shots"] if not is_img)
    shots_kf = _kf(h, "Screenshots")
    if not imgs:
        return {"present": False}
    now = time.time()
    days = {time.strftime("%Y-%m-%d", time.localtime(t)) for t in imgs}
    gaps = [(b - a) / 3600 for a, b in zip(imgs, imgs[1:])]
    return {"present": True, "count": len(imgs), "video_captures": vids,
            "by_folder": dict(collections.Counter(l for _t, l, i in c["shots"] if i)),
            "screenshots_kf_exists": _isdir(shots_kf), "first": _iso(imgs[0]), "last": _iso(imgs[-1]),
            "last_30d": sum(1 for t in imgs if now - t < 30 * 86400), "active_days": len(days),
            "per_active_day": round(len(imgs) / len(days), 2),
            "median_gap_h": round(statistics.median(gaps), 2) if gaps else None,
            "hours": [sum(1 for t in imgs if time.localtime(t).tm_hour == hr) for hr in range(24)],
            "weekdays_mon0": [sum(1 for t in imgs if time.localtime(t).tm_wday == d) for d in range(7)]}


def _user_sid(h):
    home = _home().lower()
    base = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList"
    for sid in h.reg(base) or []:
        if sid.startswith("S-1-5-21-") and (h.reg(base + "\\" + sid, "ProfileImagePath") or "").lower() == home:
            return sid
    return None


@_probe(id="files.clutter", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders",
        timeout_ms=6000)
def files_clutter(h, facts):
    """Recycle Bin, TEMP and thumbnail-cache sizes; Desktop item count; Windows Search index size."""
    out = {"present": True}
    sid = _user_sid(h)
    rb = rf"C:\$Recycle.Bin\{sid}" if sid else None
    if rb and _isdir(rb):
        items = [0]

        def cb(e, s, depth):
            if depth == 0 and e.name.startswith("$R"):
                items[0] += 1
        st = _walk(rb, max_depth=6, max_entries=100000, budget_s=2.0, on_file=cb)
        out["recycle_bin"] = {"items": items[0], "files": st["files"], "bytes": st["bytes"], "truncated": st["truncated"]}
    tmp = _env("TEMP")
    if tmp and _isdir(tmp):
        st = _walk(tmp, max_depth=5, max_entries=100000, budget_s=2.0,
                   skip_dir=lambda n, p: n == "userscan" or bool(OPERATOR_DEEP_RX.search(n)))
        out["temp_files"], out["temp_bytes"], out["temp_truncated"] = st["files"], st["bytes"], st["truncated"]
    ex = h.expand(r"%LOCALAPPDATA%\Microsoft\Windows\Explorer")
    tb = sum((h.meta(os.path.join(ex, n)).get("bytes") or 0) for n in h.list_dir(ex, 500)
             if n.lower().startswith(("thumbcache_", "iconcache_")))
    out["thumbcache_bytes"] = tb
    idx = h.meta(r"C:\ProgramData\Microsoft\Search\Data\Applications\Windows\Windows.db")
    out["search_index_bytes"] = idx.get("bytes")
    desk = _kf(h, "Desktop")
    names = [n for n in h.list_dir(desk, 5000) if n.lower() != "desktop.ini"]
    out["desktop_items"] = len(names)
    out["desktop_shortcuts"] = sum(1 for n in names if n.lower().endswith((".lnk", ".url")))
    return out


def _mru_counts(path):
    try:
        subs = []
        with _reg_key(path) as k:
            wr = _winreg()
            i = 0
            while i < 2000:
                try:
                    subs.append(wr.EnumKey(k, i))
                except OSError:
                    break
                i += 1
    except OSError:
        return None
    per = {}
    for sk in subs:
        v = _reg_values(path + "\\" + sk) or {}
        per[sk.lower()] = sum(1 for n in v if n.isdigit())
    return per


EXT_OK = re.compile(r"^\.?[a-z0-9]{1,6}$|^folder$|^\*$")


def _bucket(per):
    clean, odd = collections.Counter(), 0
    for e, c in per.items():
        if EXT_OK.match(e):
            clean[e] += c
        else:
            odd += 1
            clean["(url_or_odd)"] += c
    return clean, odd


def _utf16_first(b):
    try:
        return b.decode("utf-16le", errors="ignore").split("\x00", 1)[0]
    except Exception:
        return None


@_probe(id="files.opensave_mru", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders")
def files_opensave_mru(h, facts):
    """File-dialog history: per-extension item counts and the app exe names that used the dialog (no paths)."""
    cd = HKCU_EXPLORER + r"\ComDlg32"
    per = _mru_counts(cd + r"\OpenSavePidlMRU")
    apps = collections.Counter()
    for sub in ("LastVisitedPidlMRU", "CIDSizeMRU"):
        for n, v in (_reg_values(cd + "\\" + sub) or {}).items():
            if n.isdigit() and isinstance(v, bytes):
                a = _utf16_first(v)
                if a and re.fullmatch(r"[\w .\-]{1,64}\.exe", a, re.I):
                    apps[a.lower()] += 1
    if per is None and not apps:
        return {"present": False}
    clean, odd = _bucket({k: v for k, v in (per or {}).items() if k != "*"})
    return {"present": True, "all_items": (per or {}).get("*"), "ext_top": dict(clean.most_common(12)),
            "odd_keys_redacted": odd, "apps": dict(apps.most_common(12)),
            "last_write": _reg_lastwrite(cd + r"\OpenSavePidlMRU")}


@_probe(id="files.recentdocs", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders")
def files_recentdocs(h, facts):
    """Explorer RecentDocs: total items and per-extension counts; URL/search-built subkeys bucketed."""
    path = HKCU_EXPLORER + r"\RecentDocs"
    top = _reg_values(path)
    if top is None:
        return {"present": False}
    per = _mru_counts(path) or {}
    clean, odd = _bucket(per)
    return {"present": True, "total_items": sum(1 for n in top if n.isdigit()), "ext_subkeys": len(per),
            "odd_keys_redacted": odd, "ext_top": dict(clean.most_common(12)), "last_write": _reg_lastwrite(path)}


def _reg_domain(host):
    if not host or re.fullmatch(r"[\d.]+", host):
        return host
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "ac", "gov") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


@_probe(id="files.download_sources", level="L2", family="files", tier="T1", collect="deep", gate="files.known_folders",
        timeout_ms=6000)
def files_download_sources(h, facts):
    """Zone.Identifier ADS in Downloads (depth <= 2): zone counts and source-site category counts (no hosts)."""
    root = _kf(h, "Downloads")
    if not _isdir(root):
        return {"present": False}
    zone, cats, ext_cat = collections.Counter(), collections.Counter(), collections.Counter()
    n = with_ads = no_host = 0
    deadline = time.perf_counter() + 5.0
    stack, trunc = [(root, 0)], None
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                if depth + 1 < 2 and not OPERATOR_DEEP_RX.search(e.name):
                    stack.append((e.path, depth + 1))
                continue
            n += 1
            if n > 5000 or time.perf_counter() > deadline:
                trunc = "cap"
                stack.clear()
                break
            try:
                with open(e.path + ":Zone.Identifier", "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read(4096)
            except OSError:
                continue
            with_ads += 1
            kv = dict(line.split("=", 1) for line in txt.splitlines() if "=" in line)
            zone[kv.get("ZoneId", "?").strip()] += 1
            url = (kv.get("HostUrl") or "").strip()
            if not url or url.startswith("about:"):
                no_host += 1
                continue
            host, _ = _host_of(url)
            c = _categorize(_reg_domain(host)) if host else "other"
            c = "sensitive" if c in SENSITIVE_CATS else c
            cats[c] += 1
            ext_cat[f"{c}:{os.path.splitext(e.name)[1].lower() or '(none)'}"] += 1
    return {"present": True, "files_scanned": n, "depth": 2, "truncated": trunc, "with_zone_identifier": with_ads,
            "no_host_url": no_host, "zones": dict(zone), "source_categories": dict(cats.most_common()),
            "category_ext_top": dict(ext_cat.most_common(10))}


STANDARD_DIRS = {"appdata", "desktop", "documents", "downloads", "pictures", "videos", "music", "onedrive", "contacts",
                 "favorites", "links", "saved games", "searches", "3d objects"}


@_probe(id="files.profile_size_walk", level="L2", family="files", tier="T2", collect="deep", gate="files.known_folders",
        timeout_ms=30000)
def files_profile_size_walk(h, facts):
    """Size of each top-level profile dir (depth <= 10, 300k entries/dir, 25 s total); operator dirs excluded."""
    home = _home()
    deadline = time.perf_counter() + 25.0
    rows = []
    loose = 0
    try:
        entries = list(os.scandir(home))
    except OSError:
        return {"present": False}
    for e in entries:
        try:
            s = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if not e.is_dir(follow_symlinks=False):
            loose += 1
            continue
        if getattr(s, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        low = e.name.lower()
        cls = ("operator" if OPERATOR_RX.match(e.name) else "dot" if low.startswith(".") else
               "standard" if low in STANDARD_DIRS else "custom")
        if cls == "operator":
            rows.append({"name": e.name, "class": cls, "bytes": 0, "skipped": "operator"})
            continue
        remain = deadline - time.perf_counter()
        if remain <= 0:
            rows.append({"name": e.name, "class": cls, "bytes": 0, "skipped": "budget"})
            continue
        st = _walk(e.path, max_depth=10, max_entries=300000,
                   budget_s=min(12.0 if low == "appdata" else 4.0, remain))
        rows.append({"name": e.name, "class": cls, "bytes": st["bytes"], "files": st["files"],
                     "truncated": st["truncated"], "ms": st.get("ms")})
    rows.sort(key=lambda r: -r["bytes"])
    shown = 0
    for r in rows:
        if r["class"] == "custom":
            shown += 1
            if shown > 5:
                r["name"] = f"custom#{shown}"
    cls = collections.Counter(r["class"] for r in rows)
    return {"present": True, "top_dirs": [r for r in rows if r["class"] != "operator"][:12],
            "class_counts": dict(cls), "custom_root_dirs": cls.get("custom", 0),
            "operator_dirs_excluded": cls.get("operator", 0),
            "custom_bytes": sum(r["bytes"] for r in rows if r["class"] == "custom"),
            "total_bytes": sum(r["bytes"] for r in rows), "loose_root_files": loose,
            "truncated_dirs": sum(1 for r in rows if r.get("truncated") or r.get("skipped") == "budget")}
