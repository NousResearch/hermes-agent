"""macOS user-level probes: apps, ai_agents. The dev, browser/comms_work and files/gaming/media families live
in darwin_apps_dev, darwin_apps_browser and darwin_apps_files and share this module's helpers.

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
from userscan.specs import apps_dev_agents as ad_agents

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


def _spawn(h, args, timeout=5.0):
    """(rc, stdout, ms) with stdin closed; stderr dropped. PATH gains Homebrew and the user's tool dirs."""
    t = time.perf_counter()
    try:
        path = h.child_env().get("PATH", "") + ":" + ":".join(d for d in _extra_bins(None) if os.path.isdir(d))
        r = subprocess.run(args, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           env=h.child_env(PATH=path, NO_COLOR="1", HERMES_NO_UPDATE_CHECK="1",
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
        rc, txt, _ms = _spawn(h, ["ps", "-axo", "uid=,comm="], timeout=4)
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
    rc, txt, _ms = _spawn(h, ["crontab", "-l"], timeout=2) if shutil.which("crontab") else (None, "", 0)
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


_mirror("browser_harness.present", ad_agents.browser_harness_present)


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


_mirror("claude_code.config", ad_agents.claude_code_config)
_mirror("claude_code.claude_json", ad_agents.claude_code_claude_json)
_mirror("claude_code.sessions", ad_agents.claude_code_sessions)
_mirror("claude_code.titles", ad_agents.claude_code_titles)


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
    home = ad_agents._codex_home(h)
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
    return h.meta(os.path.join(ad_agents._codex_home(h), "auth.json"))


_mirror("codex.config", ad_agents.codex_config)
_TURN_MODEL = re.compile(rb'"model":"([^"]{1,60})"')
_TURN_EFFORT = re.compile(rb'"(?:reasoning_)?effort":"([^"]{1,20})"')


@_mp("codex.usage", level="L2", family=AI, tier="T1", collect="extended", gate="codex.present", timeout_ms=5000)
def codex_usage(h, facts):
    """Codex rollouts (count/bytes/date range from file names; turn models from a byte-level scan, newest files
    first, 2.5 s budget) + state_*.sqlite thread counts and thread_history turn durations opened read-only in place.
    Same keys as the Windows extractor; `content_scan` says how much of the rollout bytes were scanned."""
    home = ad_agents._codex_home(h)
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

_mirror("codex.extras", ad_agents.codex_extras)
_mirror("codex.chatgpt_catalog", ad_agents.codex_chatgpt_catalog)
_mirror("copilot_cli.present", ad_agents.copilot_cli_present)
_mirror("docker.model_runner", ad_agents.docker_model_runner)


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
    hh = ad_agents._hermes_homes(h)
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


_mirror("hermes.auth_presence", ad_agents.hermes_auth_presence, _prime_hermes)
_mirror("hermes.config", ad_agents.hermes_config, _prime_hermes)
_mirror("hermes.skills", ad_agents.hermes_skills, _prime_hermes)


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
    hh = ad_agents._hermes_homes(h)
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
    p = ad_agents._hermes_homes(h)["primary"]
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
    cj = ad_agents._claude_json(h)
    if cj:
        by["claude_code"] = sorted(set(cj["global_mcp_servers"]) | set(cj["project_mcp_servers"]))
    cp = os.path.join(ad_agents._codex_home(h), "config.toml")
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
    p = ad_agents._hermes_homes(h)["primary"]
    if p and os.path.exists(os.path.join(p["path"], "config.yaml")):
        try:
            by["hermes"] = ad_agents._parse_hermes_yaml(os.path.join(p["path"], "config.yaml"))["mcp_servers"]
        except OSError:
            pass
    user = sorted({n for v in by.values() for n in v if n not in ad_agents.BUNDLED_MCP})
    return {"present": True, "by_agent": {k: v for k, v in by.items() if v}, "user_configured": user,
            "user_configured_total": len(user), "vendor_bundled": sorted({n for v in by.values() for n in v if n in ad_agents.BUNDLED_MCP})}


@_mp("ollama.present", level="L1", family=AI, tier="T0", collect="core")
def ollama_present(h, facts):
    """Ollama: app bundle (version), binary, ~/.ollama models dir, running."""
    exe = _which(h, "ollama")
    b = _app(h, "Ollama")
    mroot = ad_agents._ollama_models_root(h)
    if not (exe or b or _isdir(os.path.join(_U(h), ".ollama"))):
        return None
    return {"present": True, "app": bool(b), "version": (b or {}).get("version"), "install_date": (b or {}).get("install_date"),
            "binary": bool(exe), "models_dir": _isdir(mroot), "OLLAMA_MODELS_set": bool(os.environ.get("OLLAMA_MODELS")),
            "running": _running(h, r"^ollama")}


_mirror("ollama.models", ad_agents.ollama_models)
