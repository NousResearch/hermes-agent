"""macOS user-level probes: apps, ai_agents, dev, browser, comms_work, files, gaming, media.

Registration only at import time. Signal ids match the Windows modules wherever macOS has a source, so derive.py
rules fire unchanged. Where a Windows extractor is path-agnostic (Claude/Codex JSONL scans, Chromium History SQL,
Steam VDF parsing, known-folder composition) this module primes the shared cache of that extractor with macOS
paths and calls it, the same way linux_apps.py does.

Scope is the invoking user's home plus world-readable system inventories (/Applications, Homebrew prefixes,
/Library/LaunchAgents). Stores behind Transparency Consent and Control (Safari History.db, Messages chat.db, Notes
NoteStore.sqlite, ~/.Trash, recent-document lists) need Full Disk Access for the process running the collector;
without it the probe returns {"present": false, "needs_fda": true}. Credential files (cookies, Login Data,
auth.json, .env, ~/.ssh keys) are stat'ed only.
"""
from __future__ import annotations

import collections
import ctypes
import ctypes.util
import datetime as dt
import glob
import json
import os
import plistlib
import re
import shutil
import sqlite3
import subprocess
import time

from userscan.registry import REGISTRY, probe
from userscan.specs import apps_dev as ad
from userscan.specs import browser_files as bf
from userscan.specs import usage_gaming as ug

MAC = "darwin"
APPS, AI, DEV, BROWSER, COMMS, FILES, GAMING, MEDIA = (
    "apps", "ai_agents", "dev", "browser", "comms_work", "files", "gaming", "media")
MAC_EPOCH = 978307200  # 2001-01-01 UTC (Cocoa absolute time)


# ------------------------------------------------------------------ registration helpers

def _taken(id):
    return any(q.id == id and q.os in (MAC, "any") for q in REGISTRY.values())


def _mp(id, **kw):
    """Register a macOS probe unless another module already registered this id for macOS."""
    kw.setdefault("os", MAC)
    if _taken(id):
        return lambda fn: fn
    return probe(id, **kw)


def _win_meta(id):
    return next((q for q in REGISTRY.values() if q.id == id and q.os == "windows"), None)


def _mirror(id, fn, prime=None, **override):
    """Register a macOS probe that primes a shared cache with macOS paths, then runs the Windows extractor fn."""
    w = _win_meta(id)
    meta = {"level": w.level, "family": w.family, "tier": w.tier, "collect": w.collect, "gate": w.gate,
            "timeout_ms": w.timeout_ms} if w else {}
    meta.update(override)

    def run(h, facts):
        if prime:
            prime(h)
        return fn(h, facts)
    run.__name__ = getattr(fn, "__name__", id.replace(".", "_"))
    run.__doc__ = (fn.__doc__ or "").strip() + " [macOS: shared extractor over macOS paths]"
    _mp(id, **meta)(run)


# ------------------------------------------------------------------ paths and small helpers

def _U(h):
    return ad._U(h) if h is not None else os.path.expanduser("~")


def _AS(h, *p):
    return os.path.join(_U(h), "Library", "Application Support", *p)


def _LIB(h, *p):
    return os.path.join(_U(h), "Library", *p)


def _cfg(h, *p):
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(_U(h), ".config"), *p)


def _data(h, *p):
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(_U(h), ".local", "share"), *p)


def _cachedir(h, *p):
    return os.path.join(_U(h), ".cache", *p)


_isdir, _ls, _mtime, _iso, _read_json = ad._isdir, ad._ls, ad._mtime, ad._iso, ad._read_json


def _ex(*paths):
    return any(p and os.path.exists(p) for p in paths)


def _first(*paths):
    return next((p for p in paths if p and os.path.exists(p)), None)


def _newest_mtime(paths):
    best = 0
    for p in paths:
        try:
            best = max(best, os.stat(p).st_mtime)
        except OSError:
            continue
    return best or None


def _day(ts):
    try:
        return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d") if ts else None
    except (OSError, ValueError, OverflowError, TypeError):
        return None


def _birth(p):
    try:
        st = os.stat(p)
        return getattr(st, "st_birthtime", None) or st.st_ctime
    except OSError:
        return None


def _memo(h, name, fn):
    return ad._memo(h, "mac:" + name, fn)


def _fact(facts, id):
    f = facts.get(id)
    if isinstance(f, dict) and "status" in f and "value" in f:
        return f["value"] if f.get("status") == "ok" else None
    return f


def _plist(p, maxb=20_000_000):
    try:
        if os.path.getsize(p) > maxb:
            return None
        with open(p, "rb") as f:
            return plistlib.load(f)
    except Exception:
        return None


def _readable(p):
    try:
        fd = os.open(p, os.O_RDONLY)
        os.close(fd)
        return True
    except OSError:
        return False


def _fda_meta(p):
    """presence + size + readable, contents never read. readable=false on an existing file means TCC (FDA)."""
    try:
        st = os.stat(p)
    except OSError:
        return {"present": False}
    ok = _readable(p)
    return {"present": True, "bytes": st.st_size, "mtime": _day(st.st_mtime), "readable": ok, "needs_fda": not ok}


def _dir_readable(p):
    try:
        with os.scandir(p) as it:
            next(it, None)
        return True
    except OSError:
        return False


def _extra_bins(h):
    U = _U(h) if h is not None else os.path.expanduser("~")
    return ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin", os.path.join(U, ".local", "bin"),
            os.path.join(U, "bin"), os.path.join(U, ".bun", "bin"), os.path.join(U, ".cargo", "bin"),
            os.path.join(U, "go", "bin"), os.path.join(U, ".deno", "bin"), os.path.join(U, ".npm-global", "bin"),
            os.path.join(U, "Library", "pnpm"), os.path.join(U, ".volta", "bin"), os.path.join(U, ".hermes", "bin"),
            os.path.join(U, ".nix-profile", "bin"), "/nix/var/nix/profiles/default/bin", os.path.join(U, ".lmstudio", "bin"),
            os.path.join(U, ".rye", "shims"), os.path.join(U, ".pyenv", "shims"),
            "/Applications/Docker.app/Contents/Resources/bin", os.path.join(U, ".orbstack", "bin")]


def _which(h, name):
    p = shutil.which(name)
    if p:
        return p
    for d in _extra_bins(h):
        c = os.path.join(d, name)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _spawn(args, timeout=5.0):
    """(rc, stdout, ms) with stdin closed; stderr dropped. PATH gains Homebrew and the user's tool dirs."""
    t = time.perf_counter()
    try:
        path = os.environ.get("PATH", "") + ":" + ":".join(d for d in _extra_bins(None) if os.path.isdir(d))
        r = subprocess.run(args, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           env=dict(os.environ, PATH=path, NO_COLOR="1", HERMES_NO_UPDATE_CHECK="1",
                                    HOMEBREW_NO_AUTO_UPDATE="1", LC_ALL="C"))
        return r.returncode, (r.stdout or b"").decode("utf-8", "replace").strip(), round((time.perf_counter() - t) * 1000, 1)
    except Exception:
        return None, "", round((time.perf_counter() - t) * 1000, 1)


def _open_text(p, maxb=30_000_000):
    try:
        if os.path.getsize(p) > maxb:
            return None
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _ro(path, sql, args=(), timeout=2.0):
    """Read-only query on a live SQLite file (mode=ro). Used for large WAL databases where a copy costs seconds."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


_OP_TOP = re.compile(r"^(hn-e2e|ns960|ns923.*|lhm|shots|uil|user-insights-lab|userscan.*|\.?hermes-(?!agent$).+)$", re.I)
_OP_DEEP = re.compile(r"(^|/)(user-insights-lab|userscan[^/]*|hn-e2e|ns960|ns923[^/]*)(/|$)", re.I)


def _is_op(h, path):
    """Operator (lab) noise: side homes and lab dirs directly under home, or lab dir names at any depth."""
    U = _U(h).rstrip("/") + "/"
    p = os.path.abspath(path)
    if _OP_DEEP.search(p):
        return True
    if p.startswith(U):
        first = p[len(U):].split("/", 1)[0]
        return bool(first and _OP_TOP.match(first))
    return False


# ------------------------------------------------------------------ own-user process table

def _procs(h):
    """Counter of executable basenames for the invoking uid (one `ps` spawn, comm only, no argv),
    plus the total process count across all users (a count, no names)."""
    def build():
        rc, txt, _ms = _spawn(["ps", "-axo", "uid=,comm="], timeout=4)
        me = os.getuid()
        names, total = collections.Counter(), 0
        for line in txt.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            total += 1
            if int(parts[0]) != me:
                continue
            comm = parts[1]
            if comm.startswith("<defunct"):
                continue
            app = re.search(r"/([^/]+)\.app/Contents/MacOS/", comm)
            n = app.group(1) if app else os.path.basename(comm).lstrip("-")
            names[n[:40]] += 1
        return {"names": names, "total_all_users": total}
    return _memo(h, "procs", build)


def _running(h, pattern):
    rx = re.compile(pattern, re.I)
    return {k: v for k, v in _procs(h)["names"].items() if rx.search(k)}


# ================================================================== apps

_LS_CAT = [("gaming", r"games"), ("dev", r"developer-tools"), ("creative", r"graphics-design|photography|video|music"),
           ("comms", r"social-networking"), ("productivity", r"productivity|business|education|reference|finance"),
           ("utilities", r"utilities")]


def _cls(name, bundle_id="", ls_cat=""):
    c = ad._classify(name, bundle_id, "")
    if c != "other":
        return c
    for cat, rx in _LS_CAT:
        if re.search(rx, ls_cat or ""):
            return cat
    return "other"


def _app_roots(h):
    return [("applications", "/Applications", 2), ("user_applications", os.path.join(_U(h), "Applications"), 2),
            ("system", "/System/Applications", 2)]


def _bundles(h):
    """Every .app bundle under /Applications (depth 2), ~/Applications (depth 2) and /System/Applications:
    Info.plist name, bundle id, version, category, Mac App Store receipt, and bundle birth time (install date)."""
    def build():
        out, seen = [], set()
        for loc, root, depth in _app_roots(h):
            stack = [(root, 0)]
            while stack:
                d, dep = stack.pop()
                for n in _ls(d, 3000) or []:
                    p = os.path.join(d, n)
                    if n.endswith(".app"):
                        rp = os.path.realpath(p)
                        if rp in seen:
                            continue
                        seen.add(rp)
                        info = _plist(os.path.join(p, "Contents", "Info.plist"), 5_000_000) or {}
                        bid = str(info.get("CFBundleIdentifier") or "")
                        name = str(info.get("CFBundleDisplayName") or info.get("CFBundleName") or n[:-4])
                        b = _birth(p)
                        out.append({"name": n[:-4], "display": name, "bundle_id": bid,
                                    "version": str(info.get("CFBundleShortVersionString") or info.get("CFBundleVersion") or "")[:30] or None,
                                    "ls_category": str(info.get("LSApplicationCategoryType") or ""),
                                    "mas": os.path.exists(os.path.join(p, "Contents", "_MASReceipt")),
                                    "location": loc, "install_ts": b, "install_date": _day(b),
                                    "apple": bid.startswith("com.apple."),
                                    "arch_arm64": None, "path": p})
                    elif dep + 1 < depth and not n.startswith(".") and _isdir(p) and loc != "system":
                        stack.append((p, dep + 1))
        for a in out:
            a["category"] = _cls(a["name"], a["bundle_id"], a["ls_category"])
        return out
    return _memo(h, "bundles", build)


def _brew_prefix():
    for p in ("/opt/homebrew", "/usr/local"):
        if os.path.isdir(os.path.join(p, "Cellar")) or os.path.isdir(os.path.join(p, "Caskroom")):
            return p
    return None


def _brew(h):
    """Homebrew formulae (Cellar + INSTALL_RECEIPT.json: install time, installed_on_request) and casks (Caskroom
    dir birth time). Equals `brew list --formula` / `brew list --cask` without the ~1 s Ruby spawn."""
    def build():
        pre = _brew_prefix()
        if not pre:
            return None
        formulae = []
        cellar = os.path.join(pre, "Cellar")
        for n in _ls(cellar, 3000) or []:
            vers = sorted(_ls(os.path.join(cellar, n), 50) or [])
            if not vers:
                continue
            rec = _read_json(os.path.join(cellar, n, vers[-1], "INSTALL_RECEIPT.json"), maxb=1_000_000) or {}
            t = rec.get("time") if isinstance(rec, dict) else None
            formulae.append({"name": n, "version": vers[-1], "on_request": bool(rec.get("installed_on_request")) if isinstance(rec, dict) else None,
                             "time": t if isinstance(t, (int, float)) else _birth(os.path.join(cellar, n))})
        casks = []
        room = os.path.join(pre, "Caskroom")
        for n in _ls(room, 2000) or []:
            if n.startswith("."):
                continue
            vers = [v for v in (_ls(os.path.join(room, n), 50) or []) if not v.startswith(".")]
            casks.append({"name": n, "version": sorted(vers)[-1] if vers else None, "time": _birth(os.path.join(room, n))})
        taps = glob.glob(os.path.join(pre, "Library", "Taps", "*", "*"))
        return {"prefix": pre, "formulae": formulae, "casks": casks, "taps": len(taps)}
    return _memo(h, "brew", build)


def _taxonomy(h):
    """User-facing inventory: app bundles (Apple and /System bundles = preinstalled), brew casks and on-request
    formulae, ~/.local/bin AI CLIs."""
    def build():
        apps = {}

        def add(name, cat, date, src, pre):
            n = ad._norm(name)
            if not n:
                return
            a = apps.setdefault(n, {"name": name, "category": cat, "date": None, "sources": set(), "pre": pre})
            a["sources"].add(src)
            a["pre"] = a["pre"] and pre
            if cat != "other" and a["category"] == "other":
                a["category"] = cat
            if date and (not a["date"] or date < a["date"]):
                a["date"] = date
        for b in _bundles(h):
            pre = b["location"] == "system" or (b["apple"] and not b["mas"])
            add(b["name"], b["category"], None if pre else b["install_date"], "bundle_" + b["location"], pre)
        br = _brew(h) or {"formulae": [], "casks": []}
        for c in br["casks"]:
            add(c["name"], _cls(c["name"]), _day(c["time"]), "brew_cask", False)
        for f in br["formulae"]:
            if f["on_request"]:
                add(f["name"], _cls(f["name"]), _day(f["time"]), "brew_formula", False)
        for nm in _ls(os.path.join(_U(h), ".local", "bin")) or []:
            if nm.lower() in ("claude", "codex", "hermes", "ollama", "gemini", "aider", "goose", "opencode", "cua-driver",
                              "agent-browser", "copilot"):
                add(nm, "ai", _day(_newest_mtime([os.path.join(_U(h), ".local", "bin", nm)])), "local_bin", False)
        allv = list(apps.values())
        user = [v for v in allv if v["category"] in ad.USER_CATS or v["category"] == "other"]
        inst = _birth("/var/db/.AppleSetupDone")
        return {"os_install": _day(inst), "all": allv, "user": user}
    return _memo(h, "taxonomy", build)


@_mp("apps.uninstall", level="L1", family=APPS, tier="T0", collect="core")
def apps_uninstall(h, facts):
    """App inventory: .app bundles in /Applications, ~/Applications, /System/Applications with bundle version,
    install date (bundle birth time) and Mac App Store receipt. The macOS counterpart of the Uninstall keys."""
    b = _bundles(h)
    if not b:
        return None
    by_loc = collections.Counter(x["location"] for x in b)
    return {"present": True, "listed": len(b), "by_location": dict(by_loc),
            "with_install_date": sum(1 for x in b if x["install_date"]), "mas": sum(1 for x in b if x["mas"]),
            "apple": sum(1 for x in b if x["apple"]),
            "items": [{"name": x["name"], "publisher": x["bundle_id"], "version": x["version"],
                       "install_date": x["install_date"], "category": x["category"]}
                      for x in sorted(b, key=lambda x: x["name"].lower()) if x["location"] != "system"]}


@_mp("apps.packages", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_packages(h, facts):
    """Homebrew: formula/cask counts, formulae installed on request, cask names, install events by month."""
    br = _brew(h)
    if not br:
        return {"present": False}
    req = sorted(f["name"] for f in br["formulae"] if f["on_request"])
    months = collections.Counter(_day(x["time"])[:7] for x in br["formulae"] + br["casks"] if _day(x["time"]))
    return {"present": True, "manager": "homebrew", "prefix": br["prefix"], "count": len(br["formulae"]),
            "user_requested": len(req), "user_requested_names": req[:80], "casks": len(br["casks"]),
            "cask_names": sorted(c["name"] for c in br["casks"])[:80], "taps": br["taps"],
            "install_events_by_month": dict(sorted(months.items()))}


@_mp("apps.taxonomy", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_taxonomy(h, facts):
    """Merged user-facing inventory (bundles + brew casks + on-request formulae + ~/.local/bin AI CLIs) by category."""
    t = _taxonomy(h)
    cats = ad.USER_CATS + ["other"]
    return {"present": True, "unique_all": len(t["all"]), "unique_user_facing": len(t["user"]),
            "counts_all": dict(collections.Counter(v["category"] for v in t["all"]).most_common()),
            "counts_user_facing": {c: sum(1 for v in t["user"] if v["category"] == c) for c in cats},
            "names": {c: sorted(v["name"] for v in t["user"] if v["category"] == c)[:60] for c in cats}}


@_mp("apps.taxonomy_user_added", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_taxonomy_user_added(h, facts):
    """Category counts of apps the user added (not Apple-bundled, not in /System)."""
    t = _taxonomy(h)
    cats = ad.USER_CATS + ["other"]
    added = [v for v in t["user"] if not v["pre"]]
    months = collections.Counter(v["date"][:7] for v in added if v["date"])
    return {"present": True, "profile_created": _day(_birth(_U(h))), "image_date": t["os_install"],
            "user_added_total": len(added), "preinstalled_total": len(t["user"]) - len(added),
            "counts": {c: sum(1 for v in added if v["category"] == c) for c in cats},
            "by_month": dict(sorted(months.items())), "undated": sum(1 for v in added if not v["date"])}


@_mp("apps.install_timeline", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_install_timeline(h, facts):
    """Month spread of user installs (bundle birth times, brew casks and on-request formulae)."""
    t = _taxonomy(h)
    months = collections.Counter(v["date"][:7] for v in t["user"] if not v["pre"] and v["date"])
    before = sum(1 for v in t["user"] if not v["pre"] and v["date"] and t["os_install"] and v["date"] < t["os_install"])
    return {"present": bool(months), "distinct_months": len(months), "by_month": dict(sorted(months.items())),
            "first": min(months) if months else None, "last": max(months) if months else None,
            "dated_before_os_install": before, "os_install": t["os_install"]}


def _launchd_dir(p):
    rows = []
    for n in _ls(p, 1000) or []:
        if n.endswith(".plist"):
            rows.append(n[:-6])
    return rows


@_mp("apps.autostart", level="L1", family=APPS, tier="T0", collect="core")
def apps_autostart(h, facts):
    """launchd jobs: ~/Library/LaunchAgents, /Library/LaunchAgents, /Library/LaunchDaemons labels (file names only).
    Login Items (BackgroundItems-v*.btm) are root-only and not read."""
    user = _launchd_dir(_LIB(h, "LaunchAgents"))
    op_rx = re.compile(r"hermes-|user-insights|userscan", re.I)
    la = _launchd_dir("/Library/LaunchAgents")
    ld = _launchd_dir("/Library/LaunchDaemons")
    vendors = collections.Counter(".".join(x.split(".")[:2]) for x in la + ld + user if not op_rx.search(x))
    return {"present": True, "user_agents": sorted(x for x in user if not op_rx.search(x))[:40],
            "user_agents_operator": sum(1 for x in user if op_rx.search(x)),
            "system_agents": len(la), "system_daemons": len(ld), "vendors_top": dict(vendors.most_common(12))}


_PROC_INTEREST = re.compile(r"claude|codex|hermes|ollama|lm ?studio|cursor|windsurf|copilot|gemini|opencode|aider|cua|"
                            r"agent-browser|docker|orbstack|discord|steam|slack|teams|^code$|electron|brave|chrome|firefox|"
                            r"safari|dia|arc|tailscale|obs|spotify|tmux|zellij|node|bun|python|signal|telegram|zoom|"
                            r"whatsapp|zed|ghostty|cmux|warp|iterm|raycast", re.I)


@_mp("tasks.nonms", level="L2", family=APPS, tier="T2", collect="extended", gate="apps.uninstall", timeout_ms=3000)
def tasks_nonms(h, facts):
    """This user's scheduled jobs: `crontab -l` entry count and command names, user LaunchAgents with a
    StartInterval/StartCalendarInterval; lab jobs counted separately."""
    rc, txt, _ms = _spawn(["crontab", "-l"], timeout=2) if shutil.which("crontab") else (None, "", 0)
    jobs = [l for l in txt.splitlines() if l.strip() and not l.lstrip().startswith("#") and not re.match(r"^\w+=", l.strip())]
    names, op = collections.Counter(), 0
    for j in jobs:
        parts = j.split()
        cmd = parts[5] if not j.startswith("@") and len(parts) > 5 else (parts[1] if len(parts) > 1 else "")
        if _OP_DEEP.search(j) or re.search(r"hermes-|user-insights|uil/", j):
            op += 1
            continue
        names[os.path.basename(cmd)[:40]] += 1
    timed = []
    for n in _launchd_dir(_LIB(h, "LaunchAgents")):
        d = _plist(_LIB(h, "LaunchAgents", n + ".plist"), 200_000) or {}
        if isinstance(d, dict) and ("StartInterval" in d or "StartCalendarInterval" in d):
            if re.search(r"hermes-|user-insights|userscan", n):
                op += 1
            else:
                timed.append(n)
    if not jobs and not timed:
        return {"present": False}
    return {"present": True, "count": len(jobs) - op + len(timed), "cron_jobs": len(jobs),
            "cron_commands": dict(names.most_common(10)), "timed_launch_agents": sorted(timed)[:20], "operator_lab_jobs": op}


@_mp("proc.snapshot", level="L1", family=APPS, tier="T1", collect="core")
def proc_snapshot(h, facts):
    """`ps -axo uid=,comm=` for the invoking uid only: count, top executable names, running agents/apps."""
    p = _procs(h)
    n = p["names"]
    if not n:
        return None
    return {"present": True, "total": sum(n.values()), "distinct": len(n), "top": dict(n.most_common(20)),
            "interesting": dict(sorted((k, v) for k, v in n.items() if _PROC_INTEREST.search(k))),
            "all_users_total": p["total_all_users"]}


# ================================================================== ai_agents

def _app(h, *names):
    """First /Applications or ~/Applications bundle whose name matches one of names (case-insensitive)."""
    want = [n.lower() for n in names]
    for b in _bundles(h):
        if b["name"].lower() in want:
            return b
    return None


@_mp("ai.catalog_absent_checks", level="L1", family=AI, tier="T0", collect="core")
def ai_catalog_absent_checks(h, facts):
    """Path stats for AI tools the user may have or lack (Cursor, Windsurf, Gemini CLI, opencode, LM Studio, ...)."""
    U = _U(h)
    cat = {
        "cursor": [os.path.join(U, ".cursor"), _AS(h, "Cursor"), "/Applications/Cursor.app"],
        "windsurf": [os.path.join(U, ".codeium"), os.path.join(U, ".windsurf"), _AS(h, "Windsurf"), "/Applications/Windsurf.app"],
        "gemini_cli": [os.path.join(U, ".gemini")],
        "opencode": [_data(h, "opencode"), _cfg(h, "opencode"), _AS(h, "ai.opencode.desktop")],
        "aider": [os.path.join(U, ".aider.conf.yml"), os.path.join(U, ".aider"), os.path.join(U, ".aider.chat.history.md")],
        "continue": [os.path.join(U, ".continue")], "cline": [os.path.join(U, ".cline")],
        "goose": [_cfg(h, "goose"), _AS(h, "Goose")], "amp": [_cfg(h, "amp")],
        "chatgpt_desktop": ["/Applications/ChatGPT.app", _AS(h, "com.openai.chat")],
        "chatgpt_atlas": ["/Applications/ChatGPT Atlas.app", _AS(h, "com.openai.atlas")],
        "perplexity": ["/Applications/Perplexity.app", "/Applications/Comet.app"],
        "lm_studio": [os.path.join(U, ".lmstudio"), _AS(h, "LM Studio"), "/Applications/LM Studio.app"],
        "jan": [_AS(h, "Jan"), os.path.join(U, "jan")], "gpt4all": [_AS(h, "nomic.ai")],
        "msty": [_AS(h, "Msty")], "anythingllm": [_AS(h, "anythingllm-desktop")],
        "hf_hub": [_cachedir(h, "huggingface", "hub")], "llama_cpp": [_cachedir(h, "llama.cpp")],
        "comfyui": [os.path.join(U, "ComfyUI"), _AS(h, "ComfyUI")], "llm_cli": [_AS(h, "io.datasette.llm")],
        "repoprompt": [_AS(h, "RepoPrompt"), _AS(h, "com.pvncher.repoprompt")], "conductor": [_AS(h, "com.conductor.app")],
        "wispr_flow": [_AS(h, "Wispr Flow")], "granola": [_AS(h, "Granola")], "grok": [_AS(h, "Grok Bot")],
        "mini_swe_agent": [_AS(h, "mini-swe-agent")], "t3code": [_AS(h, "t3code")],
        "apple_intelligence_models": ["/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels"],
    }
    found = {k: _ex(*v) for k, v in cat.items()}
    hf_models = sum(1 for n in (_ls(_cachedir(h, "huggingface", "hub")) or []) if n.startswith("models--"))
    return {"present": True, "found": sorted(k for k, v in found.items() if v),
            "absent": sorted(k for k, v in found.items() if not v), "hf_hub_models": hf_models}


def _npm_roots(h):
    U = _U(h)
    roots = [os.path.join(U, ".npm-global", "lib", "node_modules"), os.path.join(U, ".local", "lib", "node_modules"),
             "/opt/homebrew/lib/node_modules", "/usr/local/lib/node_modules"]
    roots += [os.path.join(d, "aliases", "default", "lib", "node_modules")
              for d in (_data(h, "fnm"), _AS(h, "fnm"), os.path.join(U, ".fnm"))]
    roots += glob.glob(_AS(h, "fnm", "node-versions", "*", "installation", "lib", "node_modules"))[:10]
    roots += glob.glob(_data(h, "fnm", "node-versions", "*", "installation", "lib", "node_modules"))[:10]
    roots += glob.glob(os.path.join(U, ".nvm", "versions", "node", "*", "lib", "node_modules"))[:10]
    roots += glob.glob(os.path.join(U, ".volta", "tools", "image", "packages", "*", "lib", "node_modules"))[:20]
    return [r for r in roots if _isdir(r)]


def _npm_pkg(h, name):
    for r in _npm_roots(h):
        pj = _read_json(os.path.join(r, name, "package.json"))
        if isinstance(pj, dict):
            return {"root": "user" if r.startswith(_U(h)) else "system", "version": pj.get("version")}
    return None


@_mp("browser.automation", level="L1", family=AI, tier="T0", collect="core")
def browser_automation(h, facts):
    """Browser-automation runtimes on disk: Playwright, puppeteer, camoufox, agent-browser (npm or binary)."""
    U = _U(h)
    out = {}
    for label, p in (("playwright_cache", _LIB(h, "Caches", "ms-playwright")), ("puppeteer", _cachedir(h, "puppeteer")),
                     ("camoufox", _LIB(h, "Caches", "camoufox")), ("agent_browser_browsers", os.path.join(U, ".agent-browser", "browsers"))):
        names = _ls(p, 50)
        if names is not None:
            out[label] = {"entries": sorted(names)[:20], "mtime": _mtime(p)}
    ab = []
    npm = _npm_pkg(h, "agent-browser")
    if npm:
        ab.append({"where": "npm_" + npm["root"], "version": npm["version"]})
    exe = _which(h, "agent-browser")
    if exe:
        ab.append({"where": "binary", "path_kind": "user" if exe.startswith(U) else "system"})
    if _isdir(os.path.join(U, ".agent-browser")):
        out["agent_browser_home"] = {"mtime": _mtime(os.path.join(U, ".agent-browser"))}
    if ab:
        out["agent_browser"] = ab
    return {"present": bool(out), **out}


_mirror("browser_harness.present", ad.browser_harness_present)


@_mp("claude_code.present", level="L1", family=AI, tier="T0", collect="core")
def claude_code_present(h, facts):
    """Claude Code CLI: ~/.claude home, native install (~/.local/bin/claude -> ~/.local/share/claude/versions),
    npm global, Homebrew cask."""
    U = _U(h)
    home = os.path.join(U, ".claude")
    link = os.path.join(U, ".local", "bin", "claude")
    npm = _npm_pkg(h, "@anthropic-ai/claude-code")
    cask = _isdir(os.path.join(_brew_prefix() or "/opt/homebrew", "Caskroom", "claude-code"))
    if not (_isdir(home) or os.path.lexists(link) or npm or cask):
        return None
    ver = None
    try:
        ver = os.path.basename(os.readlink(link))
    except OSError:
        pass
    vers = _ls(_data(h, "claude", "versions")) or []
    return {"present": True, "home": _isdir(home), "native_exe": os.path.exists(link), "exe_version": ver,
            "npm_install": bool(npm), "npm_version": (npm or {}).get("version"), "brew_cask": cask,
            "installed_versions": len(vers), "home_mtime": _mtime(home),
            "running": sum(_running(h, r"^claude$|claude-code").values())}


_mirror("claude_code.config", ad.claude_code_config)
_mirror("claude_code.claude_json", ad.claude_code_claude_json)
_mirror("claude_code.sessions", ad.claude_code_sessions)
_mirror("claude_code.titles", ad.claude_code_titles)


@_mp("claude_desktop.present", level="L1", family=AI, tier="T0", collect="core")
def claude_desktop_present(h, facts):
    """Claude desktop app: bundle version, ~/Library/Application Support/Claude, MCP server names from its config."""
    d = _AS(h, "Claude")
    b = _app(h, "Claude")
    if not (b or _isdir(d)):
        return None
    cfg = _read_json(os.path.join(d, "claude_desktop_config.json"))
    return {"present": True, "installed": bool(b), "version": (b or {}).get("version"),
            "install_date": (b or {}).get("install_date"), "data_mtime": _mtime(d),
            "mcp_servers": sorted((cfg.get("mcpServers") or {}).keys()) if isinstance(cfg, dict) else [],
            "claude_3p_dir": _isdir(_AS(h, "Claude-3p")), "running": sum(_running(h, r"^Claude$").values())}


@_mp("codex.present", level="L1", family=AI, tier="T0", collect="core")
def codex_present(h, facts):
    """OpenAI Codex: ~/.codex home (or CODEX_HOME), Codex desktop app bundle, npm global, binary on PATH."""
    home = ad._codex_home(h)
    npm = _npm_pkg(h, "@openai/codex")
    exe = _which(h, "codex")
    b = _app(h, "Codex")
    if not (_isdir(home) or npm or exe or b):
        return None
    return {"present": True, "home": _isdir(home), "desktop_app": bool(b), "desktop_version": (b or {}).get("version"),
            "npm_cli": bool(npm), "npm_version": (npm or {}).get("version"), "binary": bool(exe),
            "home_mtime": _mtime(home)}


@_mp("codex.auth_presence", level="L1", family=AI, tier="T3", collect="core")
def codex_auth_presence(h, facts):
    """Codex auth.json presence/size/mtime (never opened)."""
    return h.meta(os.path.join(ad._codex_home(h), "auth.json"))


_mirror("codex.config", ad.codex_config)
_TURN_MODEL = re.compile(rb'"model":"([^"]{1,60})"')
_TURN_EFFORT = re.compile(rb'"(?:reasoning_)?effort":"([^"]{1,20})"')


@_mp("codex.usage", level="L2", family=AI, tier="T1", collect="extended", gate="codex.present", timeout_ms=5000)
def codex_usage(h, facts):
    """Codex rollouts (count/bytes/date range from file names; turn models from a byte-level scan, newest files
    first, 2.5 s budget) + state_*.sqlite thread counts and thread_history turn durations opened read-only in place.
    Same keys as the Windows extractor; `content_scan` says how much of the rollout bytes were scanned."""
    home = ad._codex_home(h)
    if not _isdir(home):
        return None
    files = sorted(glob.glob(os.path.join(home, "sessions", "*", "*", "*", "rollout-*.jsonl")) +
                   glob.glob(os.path.join(home, "archived_sessions", "*.jsonl")))[:3000]
    models, efforts, orig, clis = (collections.Counter() for _ in range(4))
    dates, b, turns, scanned, scanned_b = [], 0, 0, 0, 0
    sized = []
    for f in files:
        try:
            s = os.path.getsize(f)
        except OSError:
            continue
        b += s
        m = re.search(r"rollout-(\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d)", f)
        if m:
            dates.append(m.group(1))
        sized.append((m.group(1) if m else "", f, s))
    deadline = time.perf_counter() + 2.5
    for _d, f, s in sorted(sized, reverse=True):
        if time.perf_counter() > deadline:
            break
        if s > 80_000_000:
            continue
        try:
            with open(f, "rb") as fh:
                for i, line in enumerate(fh):
                    head = line[:120]
                    if i == 0 and b'"session_meta"' in head:
                        try:
                            p = json.loads(line)["payload"]
                            orig[p.get("originator")] += 1
                            clis[p.get("cli_version")] += 1
                        except Exception:
                            pass
                    elif b'"turn_context"' in head:
                        turns += 1
                        mm = _TURN_MODEL.search(line, 0, 4000)
                        me = _TURN_EFFORT.search(line, 0, 4000)
                        models[mm.group(1).decode() if mm else None] += 1
                        efforts[me.group(1).decode() if me else None] += 1
        except OSError:
            continue
        scanned += 1
        scanned_b += s
    dates.sort()
    cutoff = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")
    out = {"present": True, "sessions_30d": sum(1 for d in dates if d[:10] >= cutoff),
           "rollouts": {"files": len(files), "bytes": b, "first": dates[0] if dates else None,
                        "last": dates[-1] if dates else None, "turn_contexts": turns, "models": ad._top(models),
                        "efforts": ad._top(efforts), "originators": ad._top(orig), "cli_versions": ad._top(clis, 5)},
           "content_scan": {"files": scanned, "bytes": scanned_b, "complete": scanned == len(sized)}}
    sdb = sorted(glob.glob(os.path.join(home, "state_*.sqlite")), key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if sdb:
        q = lambda sql: _ro(sdb[-1], sql)
        try:
            cs = {r[1] for r in q("pragma table_info(threads)")}
            by_src = collections.Counter()
            for (s,) in q("select source from threads"):
                if isinstance(s, str) and s.startswith("{"):
                    try:
                        s = "subagent" if "subagent" in s.lower() else next(iter(json.loads(s)))
                    except Exception:
                        s = "object"
                by_src[s] += 1
            a, z = q("select min(created_at), max(updated_at) from threads")[0]
            out["threads"] = {"count": sum(by_src.values()), "first": _iso(a), "last": _iso(z), "by_source": dict(by_src),
                              "by_provider": dict(q("select model_provider, count(*) from threads group by 1")) if "model_provider" in cs else None,
                              "by_model": dict(q("select model, count(*) from threads group by 1")) if "model" in cs else None,
                              "tokens_used_total": q("select sum(tokens_used) from threads")[0][0] if "tokens_used" in cs else None}
            try:
                out["threads"]["subagent_edges"] = q("select count(*) from thread_spawn_edges")[0][0]
            except sqlite3.Error:
                pass
        except sqlite3.Error as e:
            out["threads"] = {"error": type(e).__name__}
    th = sorted(glob.glob(os.path.join(home, "thread_history_*.sqlite")))
    if th:
        try:
            out["turns"] = {"count": _ro(th[-1], "select count(*) from thread_turns")[0][0],
                            "agent_minutes": round((_ro(th[-1], "select sum(duration_ms) from thread_turns")[0][0] or 0) / 60000, 1),
                            "items": _ro(th[-1], "select count(*) from thread_items")[0][0]}
        except sqlite3.Error as e:
            out["turns"] = {"error": type(e).__name__}
    return out

_mirror("codex.extras", ad.codex_extras)
_mirror("codex.chatgpt_catalog", ad.codex_chatgpt_catalog)
_mirror("copilot_cli.present", ad.copilot_cli_present)
_mirror("docker.model_runner", ad.docker_model_runner)


@_mp("cua_driver.present", level="L1", family=AI, tier="T0", collect="core")
def cua_driver_present(h, facts):
    """cua-driver (computer-use): ~/.cua-driver release dirs, binary on PATH, running."""
    home = os.path.join(_U(h), ".cua-driver")
    exe = _which(h, "cua-driver")
    if not (_isdir(home) or exe):
        return None
    rel = sorted(_ls(os.path.join(home, "packages", "releases")) or [])
    vc = _read_json(os.path.join(home, "version_check.json")) or {}
    return {"present": True, "binary": bool(exe), "releases": rel[-5:],
            "first_install": _mtime(os.path.join(home, ".installation_recorded")),
            "latest_available": vc.get("latest_version") if isinstance(vc, dict) else None,
            "running": bool(_running(h, r"^cua-driver"))}


def _hermes_homes_mac(h):
    env = h.l0.get("hermes_home") or os.environ.get("HERMES_HOME") or ""
    U = _U(h)
    cands = []
    for role, p in (("HERMES_HOME", env), ("dot_hermes", os.path.join(U, ".hermes"))):
        if p and _isdir(p) and all(os.path.realpath(p) != os.path.realpath(c["path"]) for c in cands):
            cands.append({"role": role, "path": p, "has_state": os.path.exists(os.path.join(p, "state.db"))})
    primary = next((c for c in cands if c["role"] == "HERMES_HOME"), None) or next((c for c in cands if c["has_state"]), None) \
        or (cands[0] if cands else None)
    side = [p for p in glob.glob(os.path.join(U, ".hermes-*")) + glob.glob(os.path.join(U, "hermes-*")) if _isdir(p)][:50]
    return {"env_set": bool(env), "cands": cands, "primary": primary, "side": side}


def _prime_hermes(h):
    ad._memo(h, "hermes_homes", lambda: _hermes_homes_mac(h))


@_mp("hermes.present", level="L1", family=AI, tier="T0", collect="core")
def hermes_present(h, facts):
    """Hermes Agent: HERMES_HOME / ~/.hermes, install method, agent version, desktop app, launchd gateway, running."""
    _prime_hermes(h)
    hh = ad._hermes_homes(h)
    homes = []
    for c in hh["cands"]:
        p = c["path"]
        im = None
        try:
            with open(os.path.join(p, ".install_method"), encoding="utf-8", errors="replace") as f:
                im = f.read(40).strip()
        except OSError:
            pass
        ver = None
        pp = os.path.join(p, "hermes-agent", "pyproject.toml")
        if ad.tomllib and os.path.exists(pp):
            try:
                with open(pp, "rb") as f:
                    ver = ad.tomllib.load(f).get("project", {}).get("version")
            except Exception:
                pass
        homes.append({"role": c["role"], "primary": hh["primary"] is c,
                      "state_db_bytes": h.meta(os.path.join(p, "state.db")).get("bytes"), "mtime": _mtime(p),
                      "install_method": im, "agent_version": ver, "git_checkout": _isdir(os.path.join(p, "hermes-agent", ".git")),
                      "profiles": len(_ls(os.path.join(p, "profiles")) or []), "cron_jobs": len(_ls(os.path.join(p, "cron")) or [])})
    agents = sorted(n for n in _launchd_dir(_LIB(h, "LaunchAgents")) if "hermes" in n.lower() and not re.search(r"hermes-", n))
    desktop = _app(h, "Hermes")
    if not (homes or hh["side"] or agents or desktop or _which(h, "hermes")):
        return None
    return {"present": True, "HERMES_HOME_set": hh["env_set"], "user_home": hh["primary"] is not None, "homes": homes,
            "side_homes_operator": len(hh["side"]), "cli_on_path": bool(_which(h, "hermes")),
            "launch_agents": agents, "desktop_app": bool(desktop), "desktop_version": (desktop or {}).get("version"),
            "desktop_userdata_dirs": len(glob.glob(_AS(h, "Hermes*"))), "running": sum(_running(h, r"hermes").values())}


_mirror("hermes.auth_presence", ad.hermes_auth_presence, _prime_hermes)
_mirror("hermes.config", ad.hermes_config, _prime_hermes)
_mirror("hermes.skills", ad.hermes_skills, _prime_hermes)


def _hermes_summary(path, titles=False):
    """Aggregates from state.db opened read-only in place (mode=ro, no copy: the file can be several GB)."""
    st = os.path.join(path, "state.db")
    if not os.path.exists(st):
        return None
    q = lambda sql, args=(): _ro(st, sql, args, timeout=3.0)
    cs = {r[1] for r in q("pragma table_info(sessions)")}
    if titles:
        if "title" not in cs:
            return {"titles": []}
        rows = q("select title from sessions where title is not null and title != '' "
                 + ("and parent_session_id is null " if "parent_session_id" in cs else "") + "order by started_at desc limit 5")
        return {"titles": [ad._redact(t) for (t,) in rows]}
    a, z = q("select min(started_at), max(coalesce(ended_at, started_at)) from sessions")[0]
    r = {"sessions": q("select count(*) from sessions")[0][0], "messages": q("select count(*) from messages")[0][0],
         "first": _iso(a), "last": _iso(z),
         "by_source": dict(q("select coalesce(source,'?'), count(*) from sessions group by 1")),
         "by_month": dict(q("select strftime('%Y-%m', started_at, 'unixepoch'), count(*) from sessions group by 1")),
         "by_model": dict(q("select coalesce(model,'?'), count(*) from sessions group by 1 order by 2 desc limit 10"))}
    r["sessions_30d"] = q("select count(*) from sessions where started_at >= ?", (time.time() - 30 * 86400,))[0][0]
    if "parent_session_id" in cs:
        r["subagent_sessions"] = q("select count(*) from sessions where parent_session_id is not null")[0][0]
    if "tool_call_count" in cs:
        r["tool_calls"] = q("select sum(tool_call_count) from sessions")[0][0]
    if "estimated_cost_usd" in cs:
        r["est_cost_usd"] = q("select round(sum(estimated_cost_usd), 2) from sessions")[0][0]
    try:
        r["model_calls"] = [{"model": m, "provider": p, "calls": c} for m, p, c in
                            q("select model, billing_provider, sum(api_call_count) from session_model_usage "
                              "group by 1,2 order by 3 desc limit 10")]
    except sqlite3.Error:
        pass
    return r


@_mp("hermes.usage", level="L2", family=AI, tier="T1", collect="extended", gate="hermes.present", timeout_ms=6000)
def hermes_usage(h, facts):
    """Primary Hermes state.db opened read-only in place: session/message/tool-call counts, date range, sources, models."""
    _prime_hermes(h)
    hh = ad._hermes_homes(h)
    p = hh["primary"]
    if not p:
        return None
    try:
        r = _hermes_summary(p["path"])
    except sqlite3.Error as e:
        return {"present": True, "error": f"sqlite: {e}"}
    if r is None:
        return None
    prof = []
    for db in glob.glob(os.path.join(p["path"], "profiles", "*", "state.db"))[:20]:
        try:
            s = _hermes_summary(os.path.dirname(db)) or {}
        except sqlite3.Error:
            s = {}
        prof.append({"sessions": s.get("sessions"), "first": s.get("first"), "last": s.get("last")})
    return {"present": True, "home_role": p["role"], "via": "live_ro", **r, "profiles": prof,
            "side_homes_excluded": len(hh["side"])}


@_mp("hermes.titles", level="L2", family=AI, tier="T2", collect="deep", gate="hermes.present")
def hermes_titles(h, facts):
    """Most recent top-level Hermes session titles (redacted). T2 content, opt-in."""
    _prime_hermes(h)
    p = ad._hermes_homes(h)["primary"]
    if not p:
        return None
    try:
        r = _hermes_summary(p["path"], titles=True)
    except sqlite3.Error:
        return None
    if not r or not r.get("titles"):
        return None
    return {"present": True, "recent": r["titles"]}


@_mp("l3.mcp_inventory", level="L2", family=AI, tier="T0", collect="core", gate="ai.catalog_absent_checks")
def l3_mcp_inventory(h, facts):
    """Union of MCP server names across agents (Claude Code, Claude desktop, Codex, Hermes, VS Code, Cursor, ...)."""
    _prime_hermes(h)
    U = _U(h)
    by = {}
    cj = ad._claude_json(h)
    if cj:
        by["claude_code"] = sorted(set(cj["global_mcp_servers"]) | set(cj["project_mcp_servers"]))
    cp = os.path.join(ad._codex_home(h), "config.toml")
    if ad.tomllib and os.path.exists(cp):
        try:
            with open(cp, "rb") as f:
                by["codex"] = sorted((ad.tomllib.load(f).get("mcp_servers") or {}).keys())
        except Exception:
            pass
    for label, path, keys in [("vscode", _AS(h, "Code", "User", "mcp.json"), ("servers", "mcpServers")),
                              ("claude_desktop", _AS(h, "Claude", "claude_desktop_config.json"), ("mcpServers",)),
                              ("cursor", os.path.join(U, ".cursor", "mcp.json"), ("mcpServers",)),
                              ("copilot_cli", os.path.join(U, ".copilot", "mcp-config.json"), ("mcpServers",)),
                              ("gemini_cli", os.path.join(U, ".gemini", "settings.json"), ("mcpServers",)),
                              ("windsurf", os.path.join(U, ".codeium", "windsurf", "mcp_config.json"), ("mcpServers",)),
                              ("zed", _cfg(h, "zed", "settings.json"), ("context_servers",))]:
        j = _read_json(path, jsonc=True)
        if isinstance(j, dict):
            for k in keys:
                if isinstance(j.get(k), dict):
                    by[label] = sorted(j[k].keys())
                    break
    p = ad._hermes_homes(h)["primary"]
    if p and os.path.exists(os.path.join(p["path"], "config.yaml")):
        try:
            by["hermes"] = ad._parse_hermes_yaml(os.path.join(p["path"], "config.yaml"))["mcp_servers"]
        except OSError:
            pass
    user = sorted({n for v in by.values() for n in v if n not in ad.BUNDLED_MCP})
    return {"present": True, "by_agent": {k: v for k, v in by.items() if v}, "user_configured": user,
            "user_configured_total": len(user), "vendor_bundled": sorted({n for v in by.values() for n in v if n in ad.BUNDLED_MCP})}


@_mp("ollama.present", level="L1", family=AI, tier="T0", collect="core")
def ollama_present(h, facts):
    """Ollama: app bundle (version), binary, ~/.ollama models dir, running."""
    exe = _which(h, "ollama")
    b = _app(h, "Ollama")
    mroot = ad._ollama_models_root(h)
    if not (exe or b or _isdir(os.path.join(_U(h), ".ollama"))):
        return None
    return {"present": True, "app": bool(b), "version": (b or {}).get("version"), "install_date": (b or {}).get("install_date"),
            "binary": bool(exe), "models_dir": _isdir(mroot), "OLLAMA_MODELS_set": bool(os.environ.get("OLLAMA_MODELS")),
            "running": _running(h, r"^ollama")}


_mirror("ollama.models", ad.ollama_models)


# ================================================================== dev

@_mp("apps.pkg_managers", level="L1", family=DEV, tier="T0", collect="core")
def apps_pkg_managers(h, facts):
    """Package managers: Homebrew, MacPorts, nix, npm/pnpm/bun/yarn, cargo, go, uv, pipx, mas."""
    U = _U(h)
    b = {"brew": bool(_brew_prefix()), "macports": _ex("/opt/local/bin/port"),
         "nix": _ex("/nix", os.path.join(U, ".nix-profile")), "npm": bool(_which(h, "npm")), "pnpm": bool(_which(h, "pnpm")),
         "bun": bool(_which(h, "bun")), "yarn": bool(_which(h, "yarn")), "cargo": bool(_which(h, "cargo")),
         "go": bool(_which(h, "go")), "uv": bool(_which(h, "uv")), "pipx": bool(_which(h, "pipx")),
         "mas": bool(_which(h, "mas")), "fnm": bool(_which(h, "fnm")), "nvm": _isdir(os.path.join(U, ".nvm")),
         "pkgx": bool(_which(h, "pkgx")), "mise": bool(_which(h, "mise")), "asdf": _isdir(os.path.join(U, ".asdf"))}
    return {"present": True, **b, "managers": sorted(k for k, v in b.items() if v)}


def _ls_pkgs(root, cap=500):
    names = _ls(root, cap)
    if names is None:
        return None
    pk = []
    for n in names:
        if n.startswith("@"):
            pk += [f"{n}/{s}" for s in (_ls(os.path.join(root, n), 100) or []) if not s.startswith(".")]
        elif not n.startswith("."):
            pk.append(n)
    return sorted(pk)


@_mp("dev.npm_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
def dev_npm_globals(h, facts):
    """Global JS packages by manager (npm roots incl. Homebrew/fnm/nvm, pnpm, bun, yarn); directory listings, no spawn."""
    U = _U(h)
    out, allp = {}, set()
    for r in _npm_roots(h):
        pk = [p for p in (_ls_pkgs(r) or []) if p not in ("npm", "corepack")]
        if pk:
            key = "npm_user" if r.startswith(U) else "npm_system"
            out.setdefault(key, []).extend(pk)
            allp.update(pk)
    for key, r in (("pnpm", os.path.join(U, "Library", "pnpm", "global", "5", "node_modules")),
                   ("bun", os.path.join(U, ".bun", "install", "global", "node_modules")),
                   ("yarn", _cfg(h, "yarn", "global", "node_modules"))):
        pk = _ls_pkgs(r)
        if pk:
            out[key] = pk
            allp.update(pk)
    if not out:
        return {"present": False}
    return {"present": True, "count": len(allp), "packages": sorted(allp)[:80],
            "by_manager": {k: len(set(v)) for k, v in out.items()}}


@_mp("dev.lang_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
def dev_lang_globals(h, facts):
    """cargo-installed crates (~/.cargo/.crates2.json) and ~/go/bin binaries; names only."""
    U = _U(h)
    out = {}
    cj = _read_json(os.path.join(U, ".cargo", ".crates2.json"))
    if isinstance(cj, dict) and isinstance(cj.get("installs"), dict):
        out["cargo"] = sorted({k.split(" ")[0] for k in cj["installs"]})[:60]
    elif _isdir(os.path.join(U, ".cargo", "bin")):
        out["cargo_bin"] = sorted(n for n in _ls(os.path.join(U, ".cargo", "bin")) or [] if not n.startswith("."))[:60]
    gb = _ls(os.path.join(os.environ.get("GOPATH") or os.path.join(U, "go"), "bin"))
    if gb:
        out["go_bin"] = sorted(gb)[:60]
    if not out:
        return {"present": False}
    return {"present": True, "count": sum(len(v) for v in out.values()), **out}


@_mp("dev.uv_tools", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_uv_tools(h, facts):
    """uv tools and uv-managed Pythons (~/.local/share/uv listing, equals `uv tool list`), ~/.local/bin entries."""
    tools = _ls(_data(h, "uv", "tools"), 200)
    py = _ls(_data(h, "uv", "python"), 200)
    lb = _ls(os.path.join(_U(h), ".local", "bin"), 300)
    if tools is None and py is None and lb is None:
        return {"present": False}
    return {"present": True, "tools": sorted(t for t in (tools or []) if not t.startswith("."))[:60],
            "pythons": sorted(p for p in (py or []) if p.startswith(("cpython", "pypy")))[:30],
            "local_bin_count": len(lb or []), "local_bin": sorted(lb or [])[:40]}


@_mp("dev.git", level="L1", family=DEV, tier="T0", collect="core")
def dev_git(h, facts):
    """git binary (no spawn): Xcode Command Line Tools shim, Homebrew git or other."""
    exe = _which(h, "git")
    if not exe:
        return None
    rp = os.path.realpath(exe)
    kind = "homebrew" if "/homebrew/" in rp or "/Cellar/" in rp else "apple_clt" if exe == "/usr/bin/git" else \
        "user" if exe.startswith(_U(h)) else "other"
    return {"present": True, "on_path": bool(shutil.which("git")), "path_kind": kind,
            "clt_installed": _isdir("/Library/Developer/CommandLineTools"), "xdg_config": os.path.exists(_cfg(h, "git", "config"))}


_mirror("dev.git_global_config", ad.dev_git_global_config)


@_mp("dev.gh_auth_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_gh_auth_presence(h, facts):
    """GitHub CLI hosts.yml presence/size (means gh logged in; never opened)."""
    return h.meta(_cfg(h, "gh", "hosts.yml"))


@_mp("dev.docker_config_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_docker_config_presence(h, facts):
    """~/.docker/config.json presence/size (may hold registry auths; never opened)."""
    return h.meta(os.path.join(_U(h), ".docker", "config.json"))


_mirror("dev.npmrc_presence", ad.dev_npmrc_presence)

_PATH_TOOLS = [("homebrew", r"^/opt/homebrew/|^/usr/local/(s?bin)$"), ("local_bin", r"/\.local/bin$"), ("home_bin", r"^/Users/[^/]+/bin$"),
               ("bun", r"/\.bun/bin"), ("cargo", r"/\.cargo/bin"), ("go", r"/go/bin"), ("fnm", r"fnm"), ("nvm", r"\.nvm"),
               ("volta", r"\.volta"), ("pnpm", r"pnpm"), ("deno", r"\.deno"), ("nix", r"\.nix-profile|/nix/"),
               ("conda", r"conda|miniforge"), ("pyenv", r"\.pyenv"), ("rye", r"\.rye"), ("hermes", r"hermes"),
               ("lmstudio", r"\.lmstudio"), ("orbstack", r"orbstack"), ("docker", r"Docker\.app"), ("macports", r"^/opt/local/"),
               ("postgres_app", r"Postgres\.app"), ("vscode", r"Visual Studio Code\.app")]


@_mp("dev.path_entries", level="L1", family=DEV, tier="T0", collect="core")
def dev_path_entries(h, facts):
    """PATH entry count and known-tool dir matches for this process (paths not emitted); user bin dirs off PATH."""
    ents = [e for e in os.environ.get("PATH", "").split(":") if e]
    tools = sorted({t for t, rx in _PATH_TOOLS for e in ents if re.search(rx, e)})
    U = _U(h)
    off = [os.path.relpath(d, U) for d in _extra_bins(h) if d.startswith(U) and _isdir(d) and d not in ents]
    helper = len(_ls("/etc/paths.d") or [])
    return {"present": True, "entries": len(ents), "user": sum(1 for e in ents if e.startswith(U)),
            "machine": sum(1 for e in ents if not e.startswith(U)), "tools": tools, "user_bin_dirs_off_path": off,
            "paths_d_entries": helper}


_mirror("dev.ssh_config", ad.dev_ssh_config)

_TOOLCHAIN = ["node", "npm", "pnpm", "yarn", "bun", "deno", "fnm", "nvm", "volta", "uv", "pipx", "conda", "poetry",
              "rustc", "cargo", "rustup", "go", "dotnet", "java", "javac", "mvn", "gradle", "cmake", "ninja", "make", "clang",
              "gcc", "zig", "swift", "xcodebuild", "xcrun", "pod", "gh", "docker", "podman", "colima", "orb", "kubectl",
              "terraform", "tailscale", "claude", "codex", "hermes", "ollama", "code", "cursor", "windsurf", "zed", "nvim",
              "vim", "emacs", "python3", "python", "tmux", "zellij", "lazygit", "rg", "fd", "fzf", "zoxide", "mcfly",
              "atuin", "btop", "htop", "jq", "ssh", "mosh", "brew", "mas", "gcloud", "aws", "vercel", "wrangler", "flyctl"]


@_mp("dev.toolchain_presence", level="L1", family=DEV, tier="T0", collect="core")
def dev_toolchain_presence(h, facts):
    """which() for language/infra/AI/editor CLIs over PATH plus Homebrew and user tool dirs; off-PATH install dirs."""
    U = _U(h)
    found, user_found = [], []
    for t in _TOOLCHAIN:
        p = _which(h, t)
        if p:
            found.append(t)
            if p.startswith(U):
                user_found.append(t)
    off = {k: _isdir(p) for k, p in [
        ("cargo_home", os.path.join(U, ".cargo")), ("rustup", os.path.join(U, ".rustup")), ("go_root", "/usr/local/go"),
        ("gopath", os.path.join(U, "go")), ("uv_pythons", _data(h, "uv", "python")), ("pyenv", os.path.join(U, ".pyenv")),
        ("miniconda", os.path.join(U, "miniconda3")), ("anaconda", os.path.join(U, "anaconda3")),
        ("miniforge", os.path.join(U, "miniforge3")), ("nvm", os.path.join(U, ".nvm")), ("sdkman", os.path.join(U, ".sdkman")),
        ("android_sdk", _LIB(h, "Android", "sdk")), ("xcode", "/Applications/Xcode.app"),
        ("xcode_derived_data", _LIB(h, "Developer", "Xcode", "DerivedData")), ("clt", "/Library/Developer/CommandLineTools"),
        ("dotnet", os.path.join(U, ".dotnet")), ("rye", os.path.join(U, ".rye"))]}
    return {"present": True, "on_path": found, "user_installed": user_found, "off_path": sorted(k for k, v in off.items() if v)}


@_mp("dev.toolchain_versions", level="L2", family=DEV, tier="T0", collect="extended", gate="dev.toolchain_presence",
     timeout_ms=12000)
def dev_toolchain_versions(h, facts):
    """--version spawns (parallel, gated on which) for runtimes, package managers, AI CLIs and infra CLIs.
    `hermes --version` and `brew --version` are excluded (both start slow interpreters); versions come from files."""
    from concurrent.futures import ThreadPoolExecutor
    specs = [("node", ["--version"]), ("npm", ["--version"]), ("uv", ["--version"]), ("gh", ["--version"]),
             ("git", ["--version"]), ("docker", ["--version"]), ("bun", ["--version"]), ("pnpm", ["--version"]),
             ("deno", ["--version"]), ("rustc", ["--version"]), ("cargo", ["--version"]), ("go", ["version"]),
             ("python3", ["--version"]), ("claude", ["--version"]), ("codex", ["--version"]), ("ollama", ["--version"]),
             ("tmux", ["-V"]), ("nvim", ["--version"]), ("swift", ["--version"]), ("tailscale", ["version"])]
    todo = [(n, [_which(h, n)] + a) for n, a in specs if _which(h, n)]
    if "git" in dict(todo) and not _isdir("/Library/Developer/CommandLineTools") and not _isdir("/Applications/Xcode.app"):
        todo = [t for t in todo if t[0] != "git"]

    def one(item):
        name, args = item
        rc, txt, ms = _spawn(args, timeout=3)
        line = next((l.strip() for l in txt.splitlines() if l.strip()), None)
        return name, {"version": ad._redact(line, 90) if line else None, "rc": rc, "ms": ms}
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, r in ex.map(one, todo):
            out[name] = r
    return {"present": bool(out), "tools": out}


_EDITORS = {
    # name: (app bundle names, extension dir(s), Application Support user dir)
    "vscode": (["Visual Studio Code"], [".vscode/extensions"], "Code"),
    "vscode_insiders": (["Visual Studio Code - Insiders"], [".vscode-insiders/extensions"], "Code - Insiders"),
    "cursor": (["Cursor"], [".cursor/extensions"], "Cursor"),
    "windsurf": (["Windsurf"], [".windsurf/extensions"], "Windsurf"),
    "vscodium": (["VSCodium"], [".vscode-oss/extensions"], "VSCodium"),
    "kiro": (["Kiro"], [".kiro/extensions"], "Kiro"),
    "trae": (["Trae"], [".trae/extensions"], "Trae"),
}


@_mp("editor.vscode", level="L2", family=DEV, tier="T2", collect="core", gate="dev.toolchain_presence")
def editor_vscode(h, facts):
    """VS Code family: extension ids (AI flagged), workspaces opened, MCP/chat; other editors (JetBrains, Zed, Xcode...)."""
    U = _U(h)
    out = {}
    for name, (apps, extds, userd) in _EDITORS.items():
        b = _app(h, *apps)
        extd = next((os.path.join(U, e) for e in extds if _isdir(os.path.join(U, e))), None)
        user = _AS(h, userd, "User") if userd else None
        if not (b or extd or (user and _isdir(user))):
            continue
        ids = ad._ext_ids(extd) if extd else []
        e = {"installed": bool(b), "version": (b or {}).get("version"), "extensions": len(ids), "extension_ids": ids[:80],
             "ai_extensions": [i for i in ids if ad._AI_EXT.search(i)]}
        if user and _isdir(user):
            ws = _ls(os.path.join(user, "workspaceStorage"), 5000)
            e["workspaces_opened"] = len(ws) if ws is not None else 0
            e["settings_json"] = os.path.exists(os.path.join(user, "settings.json"))
            e["keybindings_json"] = os.path.exists(os.path.join(user, "keybindings.json"))
            e["profiles"] = len(_ls(os.path.join(user, "profiles")) or [])
            mj = _read_json(os.path.join(user, "mcp.json"), jsonc=True)
            e["mcp_servers"] = len((mj.get("servers") or mj.get("mcpServers") or {})) if isinstance(mj, dict) else 0
            e["chat_sessions"] = len(glob.glob(os.path.join(user, "workspaceStorage", "*", "chatSessions", "*"))[:5000])
            e["user_mtime"] = _mtime(user)
        out[name] = e
    other = {"jetbrains": [x for x in (_ls(_AS(h, "JetBrains")) or []) if re.match(r"^[A-Za-z]+\d{4}\.\d", x)],
             "zed": bool(_app(h, "Zed")) or _isdir(_cfg(h, "zed")), "xcode": bool(_app(h, "Xcode")),
             "neovim_config": _isdir(_cfg(h, "nvim")), "vimrc": _ex(os.path.join(U, ".vimrc"), os.path.join(U, ".vim")),
             "emacs": _ex(os.path.join(U, ".emacs.d"), os.path.join(U, ".emacs"), _cfg(h, "emacs")),
             "helix": _isdir(_cfg(h, "helix")), "sublime": _isdir(_AS(h, "Sublime Text")) or bool(_app(h, "Sublime Text")),
             "nova": bool(_app(h, "Nova")), "bbedit": bool(_app(h, "BBEdit"))}
    other = {k: v for k, v in other.items() if v}
    customised = any(e["extensions"] or e.get("settings_json") or e.get("keybindings_json") for e in out.values()) or \
        bool(other.get("neovim_config") or other.get("emacs"))
    return {"present": bool(out or other), "extensions": sum(e["extensions"] for e in out.values()),
            "customised": customised, "editors": out, "other": other}


# ---------------- repos (deep)

_REPO_SKIP = {"node_modules", ".cache", "__pycache__", ".venv", "venv", "site-packages", ".npm", ".rustup", ".cargo",
              ".bun", ".vscode", ".cursor", ".trash", "target", "dist", "build", ".next", ".git", "go", ".gradle", ".m2",
              ".nvm", ".pyenv", "miniconda3", "anaconda3", ".docker", ".ollama", ".orbstack", "library", "applications",
              "movies", "music", "pictures", ".rye", ".lmstudio", ".colima"}


def _repos_mac(h):
    """Bounded repo walk: home depth 5, 60k dirs, 8 s; ~/Library and media folders skipped;
    classes user_area / tool_managed (dot dirs) / operator."""
    def build():
        deadline = time.perf_counter() + 8.0
        U = _U(h)
        hits, seen, visited = [], set(), [0]
        stack = [(U, 0)]
        while stack and time.perf_counter() < deadline and visited[0] < 60000:
            d, depth = stack.pop()
            visited[0] += 1
            try:
                with os.scandir(d) as it:
                    ents = list(zip(range(3000), it))
            except OSError:
                continue
            names = {e.name for _, e in ents}
            if ".git" in names and (os.path.exists(os.path.join(d, ".git", "HEAD")) or os.path.isfile(os.path.join(d, ".git"))):
                rp = os.path.realpath(d)
                if rp not in seen:
                    seen.add(rp)
                    hits.append(d)
                continue
            if depth >= 5:
                continue
            for _, e in ents:
                if e.name.lower() in _REPO_SKIP or e.name.startswith(".Trash"):
                    continue
                try:
                    if not e.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                stack.append((e.path, depth + 1))
        repos = []
        for d in hits[:300]:
            rel = os.path.relpath(d, U)
            cls = "operator" if _is_op(h, d) else "tool_managed" if rel.startswith(".") else "user_area"
            gd = os.path.join(d, ".git")
            remotes, branch, head_mtime, local_email = [], None, None, None
            if os.path.isdir(gd):
                try:
                    with open(os.path.join(gd, "HEAD"), encoding="utf-8", errors="replace") as f:
                        hd = f.read(200).strip()
                    branch = hd[16:] if hd.startswith("ref: refs/heads/") else "detached"
                except OSError:
                    pass
                head_mtime = _mtime(os.path.join(gd, "logs", "HEAD")) or _mtime(os.path.join(gd, "index"))
                try:
                    with open(os.path.join(gd, "config"), encoding="utf-8", errors="replace") as f:
                        cfg = f.read(100_000)
                    remotes = re.findall(r"^\s*url\s*=\s*(\S+)", cfg, re.M)
                    m = re.search(r"^\s*email\s*=\s*(\S+)", cfg, re.M)
                    local_email = m.group(1).lower() if m else None
                except OSError:
                    pass
            markers = [m for m in ("package.json", "pyproject.toml", "uv.lock", "Cargo.toml", "go.mod", "pom.xml",
                                   "build.gradle", "Dockerfile", "flake.nix", "Package.swift", "AGENTS.md", "CLAUDE.md", ".mcp.json")
                       if os.path.exists(os.path.join(d, m))]
            if any(n.endswith(".xcodeproj") for n in (_ls(d, 300) or [])):
                markers.append("xcodeproj")
            repos.append({"path": d, "class": cls, "branch": branch, "head_mtime": head_mtime, "remotes": remotes,
                          "local_email": local_email, "markers": markers})
        return {"repos": repos, "visited_dirs": visited[0], "timed_out": time.perf_counter() >= deadline}
    return build


def _commits_mac(h):
    def build():
        git = _which(h, "git")
        if not git:
            return None
        env_ids = {x.strip().lower() for x in os.environ.get("UIL_GIT_IDENTITY", "").split(",") if x.strip()}
        ids = set(env_ids)
        for gc in (os.path.join(_U(h), ".gitconfig"), _cfg(h, "git", "config")):
            try:
                with open(gc, encoding="utf-8", errors="replace") as f:
                    ids |= {m.lower() for m in re.findall(r"^\s*email\s*=\s*(\S+)", f.read(100_000), re.M)}
            except OSError:
                pass
        res = {"identity_sources": {"env": bool(env_ids), "global": bool(ids - env_ids), "repo_local": 0}, "by_class": {},
               "hours_local": collections.Counter(), "tz_offsets": collections.Counter(), "weekday": collections.Counter(),
               "month": collections.Counter(), "repos_with_own": {}}
        # Newest HEAD first, so when the deadline cuts the scan it drops the stalest repos, not a random set.
        deadline = time.perf_counter() + 10.0
        res["repos_scanned"], res["timed_out"] = 0, False
        for x in sorted(ad._repos(h)["repos"], key=lambda r: str(r.get("head_mtime") or ""), reverse=True):
            if x["class"] == "operator":
                continue
            if time.perf_counter() > deadline:
                res["timed_out"] = True
                break
            rid = set(ids)
            if x["local_email"]:
                rid.add(x["local_email"])
                res["identity_sources"]["repo_local"] += 1
            rc, txt, _ms = _spawn([git, "-C", x["path"], "log", "--all", "--no-merges",
                                   "--format=%ae%x09%an%x09%at%x09%ai", "-n", "100000"], timeout=max(0.5, deadline - time.perf_counter()))
            if rc != 0:
                continue
            res["repos_scanned"] += 1
            mine = total = 0
            for line in txt.splitlines():
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                total += 1
                if parts[0].lower() in rid or parts[1].lower() in rid:
                    mine += 1
                    try:
                        t = dt.datetime.fromtimestamp(int(parts[2]))
                        res["hours_local"][t.hour] += 1
                        res["weekday"][t.strftime("%a")] += 1
                        res["month"][t.strftime("%Y-%m")] += 1
                    except (ValueError, OSError):
                        pass
                    res["tz_offsets"][parts[3].strip()[-5:]] += 1
            c = res["by_class"].setdefault(x["class"], [0, 0])
            c[0] += mine
            c[1] += total
            if mine:
                res["repos_with_own"][x["class"]] = res["repos_with_own"].get(x["class"], 0) + 1
        return res
    return build


def _prime_repos(h):
    ad._memo(h, "repos", _repos_mac(h))


def _prime_commits(h):
    _prime_repos(h)
    ad._memo(h, "commits", _commits_mac(h))


# dev.repos and dev.repos.remotes need only the walk; priming commits there made them wait for git log on every repo.
_mirror("dev.repos", ad.dev_repos, _prime_repos)
_mirror("dev.repos.remotes", ad.dev_repos_remotes, _prime_repos)
_mirror("dev.repos.my_commits", ad.dev_repos_my_commits, _prime_commits)
_mirror("dev.repos.commit_hours", ad.dev_repos_commit_hours, _prime_commits)

_BUILTINS = {"cd", "export", "source", ".", "alias", "unalias", "echo", "exit", "history", "set", "unset", "eval", "exec",
             "type", "which", "for", "if", "while", "sudo", "time", "nohup", "env", "clear", "pwd", "ls", "ll", "la", "fg",
             "bg", "jobs", "kill", "z", "zi", "man", "help", "open", "defaults", "killall"}


def _cmd_name(line, known):
    t = line.strip().split()
    while t and (t[0] in ("sudo", "time", "nohup", "env", "exec") or "=" in t[0]):
        t = t[1:]
    if not t:
        return None
    c = os.path.basename(t[0])
    if not re.fullmatch(r"[A-Za-z0-9._+\-]{1,32}", c):
        return None
    return c if c in _BUILTINS or c in known else None


@_mp("dev.shell_history", level="L2", family=DEV, tier="T1", collect="extended", gate="dev.path_entries")
def dev_shell_history(h, facts):
    """zsh/bash/fish/python/node history: line counts, last write, top command names (resolved executables or
    builtins only), Apple Terminal ~/.zsh_sessions count. Command lines, arguments and paths are never emitted."""
    U = _U(h)
    files = [("zsh", os.environ.get("HISTFILE") or os.path.join(U, ".zsh_history")), ("bash", os.path.join(U, ".bash_history")),
             ("fish", _data(h, "fish", "fish_history")), ("python", os.path.join(U, ".python_history")),
             ("node", os.path.join(U, ".node_repl_history")), ("mcfly_db", _AS(h, "McFly", "history.db")),
             ("atuin_db", _data(h, "atuin", "history.db"))]
    known = set()
    for d in os.environ.get("PATH", "").split(":") + _extra_bins(h):
        known.update(_ls(d, 20000) or [])
    out, cmds, total = {}, collections.Counter(), 0
    seen = set()
    for label, p in files:
        try:
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            st = os.stat(p)
        except OSError:
            continue
        seen.add(rp)
        if label.endswith("_db"):
            out[label] = {"bytes": st.st_size, "mtime": _day(st.st_mtime)}
            continue
        if st.st_size > 50_000_000:
            out[label] = {"bytes": st.st_size, "lines": None, "mtime": _day(st.st_mtime)}
            continue
        n = 0
        with open(p, "rb") as f:
            for raw in f:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if label == "fish":
                    if not line.startswith("- cmd: "):
                        continue
                    line = line[7:]
                elif label == "zsh":
                    line = re.sub(r"^: \d+:\d+;", "", line)
                n += 1
                if label in ("bash", "zsh", "fish"):
                    c = _cmd_name(line, known)
                    if c:
                        cmds[c] += 1
        out[label] = {"lines": n, "bytes": st.st_size, "mtime": _day(st.st_mtime)}
        if label in ("bash", "zsh", "fish"):
            total += n
    zs = _ls(os.path.join(U, ".zsh_sessions"), 20000)
    if zs:
        out["zsh_sessions"] = {"files": sum(1 for x in zs if x.endswith(".history")), "mtime": _day(_newest_mtime([os.path.join(U, ".zsh_sessions")]))}
    if not out:
        return {"present": False}
    return {"present": True, "lines": total, "files": out, "top_commands": dict(cmds.most_common(15)),
            "distinct_commands": len(cmds), "login_shell": os.path.basename(os.environ.get("SHELL", ""))}


def _docker_sock(h):
    for p in (os.path.join(_U(h), ".docker", "run", "docker.sock"), "/var/run/docker.sock",
              os.path.join(_U(h), ".orbstack", "run", "docker.sock"), os.path.join(_U(h), ".colima", "default", "docker.sock")):
        try:
            import stat as _st
            if _st.S_ISSOCK(os.stat(p).st_mode):
                return p
        except OSError:
            continue
    return None


@_mp("dev.docker", level="L1", family=DEV, tier="T0", collect="core")
def dev_docker(h, facts):
    """Docker on macOS: CLI, Docker Desktop / OrbStack / Colima installs, live engine socket (daemon running)."""
    cli = _which(h, "docker")
    desktop = bool(_app(h, "Docker")) or _isdir(_LIB(h, "Group Containers", "group.com.docker"))
    orb = bool(_app(h, "OrbStack")) or _isdir(os.path.join(_U(h), ".orbstack"))
    colima = _isdir(os.path.join(_U(h), ".colima"))
    if not (cli or desktop or orb or colima):
        return None
    sock = _docker_sock(h)
    return {"present": True, "cli": bool(cli), "docker_desktop": desktop, "orbstack": orb, "colima": colima,
            "engine_socket": bool(sock), "socket_access": bool(sock and os.access(sock, os.R_OK | os.W_OK)),
            "podman": bool(_which(h, "podman"))}


def _gb(s):
    m = re.match(r"([\d.]+)\s*([kKMGT]?B)", s or "")
    if not m:
        return 0.0
    return float(m.group(1)) * {"B": 1e-9, "kB": 1e-6, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}.get(m.group(2), 0)


@_mp("dev.docker_runtime", level="L2", family=DEV, tier="T2", collect="extended", gate="dev.docker", timeout_ms=8000)
def dev_docker_runtime(h, facts):
    """`docker ps` names/images and `docker images` repos/sizes only, only when an engine socket is live."""
    d = _fact(facts, "dev.docker") or {}
    cli = _which(h, "docker")
    if not cli or not d.get("socket_access"):
        return {"present": True, "accessible": False, "count": 0}
    rc, ps, ms1 = _spawn([cli, "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.RunningFor}}"], timeout=5)
    rc2, im, ms2 = _spawn([cli, "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}"], timeout=5)
    if rc != 0 and rc2 != 0:
        return {"present": True, "accessible": False, "count": 0}
    running = [dict(zip(("name", "image", "up"), l.split("\t"))) for l in ps.splitlines() if l.strip()]
    images = [l.split("\t") for l in im.splitlines() if l.strip()]
    return {"present": True, "accessible": True, "count": len(running),
            "running": [{"name": r.get("name"), "image": (r.get("image") or "").split("@")[0][:80], "up": r.get("up")}
                        for r in running[:40]],
            "images": len(images), "images_gb": round(sum(_gb(x[1]) for x in images if len(x) > 1), 2),
            "image_repos": sorted({x[0].rsplit(":", 1)[0] for x in images})[:40], "ms": round(ms1 + ms2, 1)}


@_mp("dev.multiplexers", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_multiplexers(h, facts):
    """tmux / zellij / screen session counts for this user (session names not emitted); terminal apps installed."""
    uid = os.getuid()
    out = {}
    tdir = os.path.join(os.environ.get("TMUX_TMPDIR", "/private/tmp"), f"tmux-{uid}")
    socks = [os.path.join(tdir, n) for n in (_ls(tdir) or [])]
    tm = _which(h, "tmux")
    if socks or tm:
        n = 0
        for s in socks[:10]:
            rc, txt, _ = _spawn([tm, "-S", s, "ls"], timeout=2) if tm else (None, "", 0)
            if rc == 0:
                n += len([l for l in txt.splitlines() if l.strip()])
        out["tmux"] = {"installed": bool(tm), "sockets": len(socks), "sessions": n,
                       "config": _ex(os.path.join(_U(h), ".tmux.conf"), _cfg(h, "tmux", "tmux.conf"))}
    zj = _which(h, "zellij")
    # no-tmp: ok — zellij puts its sockets under TMPDIR, which falls back to /tmp
    tmp = os.environ.get("TMPDIR") or "/tmp"
    zsocks = [p for p in glob.glob(os.path.join(tmp, "zellij-*", "*", "*")) if not os.path.isdir(p)]
    zres = glob.glob(_LIB(h, "Caches", "org.Zellij-Contributors.Zellij", "*", "session_info", "*"))
    if zj or zsocks or zres:
        out["zellij"] = {"installed": bool(zj), "live_sessions": len(zsocks), "resurrectable": len(zres),
                         "config": _isdir(_cfg(h, "zellij"))}
    terms = [n for n in ("iTerm", "Ghostty", "WezTerm", "Alacritty", "kitty", "Warp", "cmux", "Hyper", "Tabby")
             if _app(h, n)]
    if terms:
        out["terminal_apps"] = terms
    if not out:
        return {"present": False}
    return {"present": True, "count": sum(v.get("sessions", 0) + v.get("live_sessions", 0) for v in out.values()
                                          if isinstance(v, dict)), **out}


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
                 ("browser.history.workspaces", bf.history_workspaces), ("browser.history.work_hosts", bf.history_work_hosts)):
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
        c = bf._categorize(bf._reg_domain(host)) if host else "other"
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
    v = bf.files_composition(h, facts)
    if isinstance(v, dict):
        v["home_root"] = _home_root(h)
    return v


_mirror("files.screenshots", bf.files_screenshots, _prime_kf)


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
    desk = bf._kf(h, "Desktop")
    names = [x for x in h.list_dir(desk, 5000) if not x.startswith(".")]
    out["desktop_items"] = len(names)
    out["desktop_screenshots"] = sum(1 for x in names if bf.SHOT_NAME.match(x))
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
                        c = bf._categorize(bf._reg_domain(host))
                        cats["sensitive" if c in bf.SENSITIVE_CATS else c] += 1
            out.update({"present": bool(rows), "quarantine": {
                "events": len(rows), "first": bf._iso(min(ts)) if ts else None, "last": bf._iso(max(ts)) if ts else None,
                "by_agent": dict(agents.most_common(10)), "by_month": dict(sorted(months.items())[-24:]),
                "with_url": sum(cats.values()), "distinct_hosts": len(hosts), "source_categories": dict(cats.most_common())}})
    root = bf._kf(h, "Downloads")
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
                c = bf._categorize(bf._reg_domain(host)) if host else "other"
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
            for _k, v in (ug._ci(ug._vdf(txt), "libraryfolders") or {}).items():
                if isinstance(v, dict) and v.get("path"):
                    libs.append(os.path.normpath(v["path"]))
        libs = libs or [sp]
        installed = []
        for lib in libs[:20]:
            sa = os.path.join(lib, "steamapps")
            for n in h.list_dir(sa, 2000):
                if n.startswith("appmanifest_") and n.endswith(".acf"):
                    st = ug._ci(ug._vdf(ug._read(os.path.join(sa, n)) or ""), "AppState") or {}
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
                node = ug._ci(ug._vdf(t), "UserLocalConfigStore", "Software", "Valve", "Steam", "apps") or {}
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


for _id, _fn in (("steam.installed", ug.steam_installed), ("steam.playtime", ug.steam_playtime),
                 ("steam.appinfo_genres", ug.steam_appinfo_genres), ("steam.local_sessions", ug.steam_local_sessions),
                 ("steam.non_steam_shortcuts", ug.steam_non_steam_shortcuts), ("steam.screenshots", ug.steam_screenshots),
                 ("steam.login_users", ug.steam_login_users), ("steam.remote_clients", ug.steam_remote_clients)):
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
        s = ug._walk_media(p, budget_s=0.6)
        s["gb"] = round(s.pop("bytes") / 1e9, 2)
        s["onedrive"] = False
        out[short] = s
    photos = [n for n in (_ls(kf.get("Pictures") or "") or []) if n.endswith(".photoslibrary")]
    out["photos_libraries"] = len(photos)
    out["apple_music_library"] = _ex(os.path.join(kf.get("Music") or "", "Music", "Music Library.musiclibrary"))
    out["screenshots"] = sum(1 for n in (_ls(kf.get("Screenshots") or "", 20000) or []) if bf.SHOT_NAME.match(n))
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
