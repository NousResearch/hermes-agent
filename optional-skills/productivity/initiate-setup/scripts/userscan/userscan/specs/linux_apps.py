"""Linux user-level probes: apps, ai_agents. The dev, browser/comms_work and files/gaming/media families live
in linux_apps_dev, linux_apps_browser and linux_apps_files and share this module's helpers.

Registration only at import time. Signal ids match the Windows modules wherever Linux has a source, so
derive.py rules fire unchanged. Where a Windows extractor is path-agnostic (Claude/Codex JSONL scans,
Chromium History SQL, Steam VDF parsing, known-folder composition) this module primes the shared cache
of that extractor with Linux paths and calls it, instead of copying it.

Scope is the invoking user's home plus world-readable system inventories (dpkg status, apt/dpkg logs,
.desktop files, /opt). Another account's home is never read. Docker is queried through the CLI with
`ps` and `images` only.
"""
from __future__ import annotations

import collections
import configparser
import datetime as dt
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import time

from userscan.registry import REGISTRY, probe
from userscan.specs import apps_dev as ad
from userscan.specs import browser_files as bf
from userscan.specs import apps_dev_agents as ad_agents

LX = "linux"
APPS, AI, DEV, BROWSER, COMMS, FILES, GAMING, MEDIA = (
    "apps", "ai_agents", "dev", "browser", "comms_work", "files", "gaming", "media")


# ------------------------------------------------------------------ registration helpers

def _taken(id):
    return any(q.id == id and q.os in (LX, "any") for q in REGISTRY.values())


def _lp(id, **kw):
    """Register a Linux probe unless another module already registered this id for Linux."""
    kw.setdefault("os", LX)
    if _taken(id):
        return lambda fn: fn
    return probe(id, **kw)


def _win_meta(id):
    return next((q for q in REGISTRY.values() if q.id == id and q.os == "windows"), None)


def _mirror(id, fn, prime=None, **override):
    """Register a Linux probe that primes a shared cache with Linux paths, then runs the Windows extractor fn.
    Level/family/tier/collect/gate/timeout default to the Windows registration of the same id."""
    w = _win_meta(id)
    meta = {"level": w.level, "family": w.family, "tier": w.tier, "collect": w.collect, "gate": w.gate,
            "timeout_ms": w.timeout_ms} if w else {}
    meta.update(override)

    def run(h, facts):
        if prime:
            prime(h)
        return fn(h, facts)
    run.__name__ = getattr(fn, "__name__", id.replace(".", "_"))
    run.__doc__ = (fn.__doc__ or "").strip() + " [Linux: shared extractor over Linux paths]"
    _lp(id, **meta)(run)


# ------------------------------------------------------------------ paths and small helpers

def _U(h):
    return ad._U(h) if h is not None else os.path.expanduser("~")


def _cfg(h, *p):
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(_U(h), ".config"), *p)


def _data(h, *p):
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(_U(h), ".local", "share"), *p)


def _cachedir(h, *p):
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(_U(h), ".cache"), *p)


def _state(h, *p):
    return os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.join(_U(h), ".local", "state"), *p)


def _flat(h, appid, *p):
    return os.path.join(_U(h), ".var", "app", appid, *p)


def _snapu(h, name, *p):
    return os.path.join(_U(h), "snap", name, *p)


_isdir, _ls, _mtime, _iso, _top, _read_json = ad._isdir, ad._ls, ad._mtime, ad._iso, ad._top, ad._read_json


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
        return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else None
    except (OSError, ValueError, OverflowError):
        return None


def _memo(h, name, fn):
    return ad._memo(h, "lx:" + name, fn)


def _extra_bins_static():
    return [d for d in _extra_bins(None) if os.path.isdir(d)]


def _extra_bins(h):
    U = _U(h) if h is not None else os.path.expanduser("~")
    return [os.path.join(U, ".local", "bin"), os.path.join(U, "bin"), os.path.join(U, ".bun", "bin"),
            os.path.join(U, ".cargo", "bin"), os.path.join(U, "go", "bin"), os.path.join(U, ".deno", "bin"),
            os.path.join(U, ".npm-global", "bin"), _data(h, "pnpm"), os.path.join(U, ".volta", "bin"),
            _data(h, "fnm", "aliases", "default", "bin"), os.path.join(U, ".hermes", "bin"),
            os.path.join(U, ".nix-profile", "bin"), "/usr/local/go/bin", "/snap/bin", "/var/lib/flatpak/exports/bin",
            _data(h, "flatpak", "exports", "bin"), "/home/linuxbrew/.linuxbrew/bin"]


def _fact(facts, id):
    """Value of an earlier probe. The runner passes its record dicts ({'value': ...}); accept bare values too."""
    f = facts.get(id)
    if isinstance(f, dict) and "status" in f and "value" in f:
        return f["value"] if f.get("status") == "ok" else None
    return f


def _which(h, name):
    """shutil.which over PATH, then the user's tool dirs (a non-login ssh shell often lacks ~/.local/bin)."""
    p = shutil.which(name)
    if p:
        return p
    for d in _extra_bins(h):
        c = os.path.join(d, name)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


_SPAWN_PATH = None


def _spawn(h, args, timeout=5.0):
    """(rc, stdout, ms) with stdin closed; stderr dropped. PATH gains the user's tool dirs so #!/usr/bin/env node
    shims (npm, pnpm) resolve their interpreter."""
    t = time.perf_counter()
    try:
        path = h.child_env().get("PATH", "") + ":" + ":".join(_extra_bins_static())
        r = subprocess.run(args, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           env=h.child_env(PATH=path, NO_COLOR="1", HERMES_NO_UPDATE_CHECK="1", LC_ALL="C"))
        return r.returncode, (r.stdout or b"").decode("utf-8", "replace").strip(), round((time.perf_counter() - t) * 1000, 1)
    except Exception:
        return None, "", round((time.perf_counter() - t) * 1000, 1)


def _open_text(p, maxb=30_000_000):
    """Read a text file (plain or .gz). None on failure or when larger than maxb."""
    try:
        if os.path.getsize(p) > maxb:
            return None
        if p.endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                return f.read(maxb * 4)
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


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


def _age_bucket(age_s):
    return bf._age_bucket(age_s)


# ------------------------------------------------------------------ own-user process table

_SAFE_NAME = re.compile(r"^[\w.@+\-]{1,40}$")


def _procs(h):
    """Counter of process names for the invoking uid (argv0 basename; for interpreters also argv1 basename),
    plus the total process count across all users (a count, no names)."""
    def build():
        me = os.getuid()
        names, total = collections.Counter(), 0
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            total += 1
            try:
                if os.stat(f"/proc/{pid}").st_uid != me:
                    continue
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = f.read(4096).split(b"\0")
                with open(f"/proc/{pid}/comm", encoding="utf-8", errors="replace") as f:
                    comm = f.read().strip()
            except OSError:
                continue
            a0 = os.path.basename(argv[0].decode("utf-8", "replace")) if argv and argv[0] else comm
            a0 = a0.split(" ")[0] or comm
            n = a0 if _SAFE_NAME.match(a0) else comm
            if re.match(r"^(python[\d.]*|node|bun|deno|uv|npx|bunx)$", n) and len(argv) > 1 and argv[1]:
                a1 = os.path.basename(argv[1].decode("utf-8", "replace"))
                if _SAFE_NAME.match(a1) and not a1.startswith("-"):
                    n = f"{n}:{a1}"
            names[n] += 1
        return {"names": names, "total_all_users": total}
    return _memo(h, "procs", build)


def _running(h, pattern):
    rx = re.compile(pattern, re.I)
    return {k: v for k, v in _procs(h)["names"].items() if rx.search(k)}


# ================================================================== apps

def _parse_deb822(text):
    for block in text.split("\n\n"):
        d, last = {}, None
        for line in block.splitlines():
            if not line:
                continue
            if line[0] in " \t":
                continue
            k, _, v = line.partition(":")
            d[k] = v.strip()
            last = k
        if d:
            yield d


def _os_install_date():
    for p in ("/var/log/installer", "/lost+found", "/etc/machine-id"):
        try:
            return _day(os.stat(p).st_mtime)
        except OSError:
            continue
    return None


def _dpkg(h):
    """dpkg status + apt extended_states + dpkg.log* + apt history.log*. No spawn (replaces dpkg -l and
    apt list --installed / apt-mark showmanual, which cost 0.3-1.5 s)."""
    def build():
        txt = _open_text("/var/lib/dpkg/status", 60_000_000)
        if txt is None:
            return None
        pkgs = {}
        for d in _parse_deb822(txt):
            if "install ok installed" not in d.get("Status", ""):
                continue
            pkgs[d.get("Package")] = {"section": d.get("Section", ""), "arch": d.get("Architecture", ""),
                                      "kb": int(d.get("Installed-Size", "0") or 0), "priority": d.get("Priority", "")}
        auto = set()
        es = _open_text("/var/lib/apt/extended_states")
        for d in _parse_deb822(es or ""):
            if d.get("Auto-Installed") == "1":
                auto.add(d.get("Package"))
        first_inst, months = {}, collections.Counter()
        oldest_log = None
        for f in sorted(glob.glob("/var/log/dpkg.log*"))[:30]:
            t = _open_text(f)
            if not t:
                continue
            for m in re.finditer(r"^(\d{4}-\d\d-\d\d) [\d:]+ install (\S+?)(?::\S+)? ", t, re.M):
                d, p = m.group(1), m.group(2)
                oldest_log = d if oldest_log is None or d < oldest_log else oldest_log
                months[d[:7]] += 1
                if p not in first_inst or d < first_inst[p]:
                    first_inst[p] = d
        requested = []
        me = os.getuid()
        os_inst = _os_install_date()
        for f in sorted(glob.glob("/var/log/apt/history.log*"))[:30]:
            t = _open_text(f)
            if not t:
                continue
            for block in t.split("\n\n"):
                sd = re.search(r"^Start-Date: (\d{4}-\d\d-\d\d)", block, re.M)
                cl = re.search(r"^Commandline: (.+)$", block, re.M)
                if not (sd and cl):
                    continue
                toks = cl.group(1).split()
                if "install" not in toks or "--reinstall" in toks or re.search(r"unattended|packagekit|aptdaemon", toks[0]):
                    continue
                rb = re.search(r"^Requested-By: .*\((\d+)\)", block, re.M)
                # image build and installer runs have no Requested-By and predate the install day
                if not rb and (not os_inst or sd.group(1) <= os_inst):
                    continue
                by_me = bool(rb and int(rb.group(1)) == me)
                for tok in toks[toks.index("install") + 1:]:
                    if tok.startswith("-"):
                        continue
                    if "/" in tok:
                        if not tok.endswith(".deb"):
                            continue
                        tok = os.path.basename(tok).split("_")[0]
                    name = re.split(r"[=:]", tok)[0]
                    if name:
                        requested.append((name, sd.group(1), by_me))
        return {"pkgs": pkgs, "auto": auto, "first_install": first_inst, "install_months": months,
                "oldest_log": oldest_log, "requested": requested, "os_install": os_inst}
    return _memo(h, "dpkg", build)


_DESKTOP_DIRS = [("system", "/usr/share/applications"), ("local", "/usr/local/share/applications"),
                 ("snap", "/var/lib/snapd/desktop/applications"),
                 ("flatpak", "/var/lib/flatpak/exports/share/applications")]
_CAT_MAP = [("gaming", r"\bGame\b"), ("dev", r"\bDevelopment\b|IDE|TextEditor"), ("comms", r"InstantMessaging|Chat|Email|Telephony|VideoConference"),
            ("creative", r"AudioVideo|Audio|Video|Graphics|Photography"), ("productivity", r"\bOffice\b|WebBrowser|Office"),
            ("utilities", r"Utility|System|Settings|FileManager|TerminalEmulator|Monitor")]


def _desktop(h):
    """Parse .desktop launchers (system, /usr/local, snap, flatpak, user). Names and categories only."""
    def build():
        dirs = list(_DESKTOP_DIRS) + [("user", _data(h, "applications")),
                                      ("flatpak_user", _data(h, "flatpak", "exports", "share", "applications"))]
        out = []
        for src, d in dirs:
            for n in (_ls(d, 3000) or []):
                if not n.endswith(".desktop"):
                    continue
                p = os.path.join(d, n)
                try:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        txt = f.read(64_000)
                    st = os.stat(p)
                except OSError:
                    continue
                sec = txt.split("[Desktop Entry]", 1)[-1].split("\n[", 1)[0]
                kv = {}
                for line in sec.splitlines():
                    k, s, v = line.partition("=")
                    if s and k.strip() in ("Name", "Exec", "Categories", "NoDisplay", "Hidden", "Type", "MimeType",
                                           "OnlyShowIn", "X-KDE-ServiceTypes"):
                        kv.setdefault(k.strip(), v.strip())
                exe = (kv.get("Exec") or "").split()
                exe0 = next((x for x in exe if not x.startswith(("env", "-")) and "=" not in x), "")
                launchable = kv.get("Type", "Application") == "Application" and kv.get("NoDisplay", "").lower() != "true" \
                    and kv.get("Hidden", "").lower() != "true" and not n.startswith(("kcm_", "org.kde.kcm"))
                out.append({"id": n[:-8], "name": kv.get("Name") or n[:-8], "src": src, "exec": exe0,
                            "categories": kv.get("Categories", ""), "mime": kv.get("MimeType", ""),
                            "launchable": launchable, "ctime": _day(st.st_ctime)})
        return out
    return _memo(h, "desktop", build)


def _cls(name, extra=""):
    c = ad._classify(name, "", extra)
    if c != "other":
        return c
    for cat, rx in _CAT_MAP:
        if re.search(rx, extra or ""):
            return cat
    return "other"


def _snaps():
    out = []
    for n in (_ls("/snap", 500) or []):
        if n in ("bin", "README") or not _isdir(os.path.join("/snap", n)):
            continue
        cur = os.path.join("/snap", n, "current")
        try:
            rev = os.readlink(cur)
            ctime = _day(os.stat(os.path.join("/snap", n)).st_ctime)
        except OSError:
            rev, ctime = None, None
        out.append({"name": n, "revision": rev, "installed": ctime})
    return out


def _flatpaks(h):
    out = []
    for scope, root in (("system", "/var/lib/flatpak/app"), ("user", _data(h, "flatpak", "app"))):
        for n in (_ls(root, 1000) or []):
            try:
                ct = _day(os.stat(os.path.join(root, n)).st_ctime)
            except OSError:
                ct = None
            out.append({"id": n, "scope": scope, "installed": ct})
    return out


_BASE_SNAPS = re.compile(r"^(core\d*|snapd|bare|gnome-\d.*|gtk-common-themes|kf5-.*|kde-.*|mesa-.*|firmware-updater|snap-store|snapd-desktop-integration|lxd)$")


def _taxonomy(h):
    """User-facing app inventory: user-requested apt packages, /opt apps, snaps, flatpaks, launchable .desktop
    entries, ~/.local/bin AI CLIs. 'pre' = shipped with the image (not user-requested, not /opt, not user scope)."""
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
        dp = _dpkg(h) or {"pkgs": {}, "requested": [], "first_install": {}}
        req = {}
        for name, d, _me in dp["requested"]:
            if name in dp["pkgs"] and (name not in req or d < req[name]):
                req[name] = d
        for name, d in req.items():
            add(name, _cls(name, dp["pkgs"][name]["section"]), d, "apt_requested", False)
        for n in (_ls("/opt", 200) or []):
            if n not in ("containerd",) and _isdir(os.path.join("/opt", n)):
                try:
                    ct = _day(os.stat(os.path.join("/opt", n)).st_ctime)
                except OSError:
                    ct = None
                add(n, _cls(n), ct, "opt", False)
        for s in _snaps():
            if not _BASE_SNAPS.match(s["name"]):
                add(s["name"], _cls(s["name"]), s["installed"], "snap", False)
        for f in _flatpaks(h):
            nm = f["id"].split(".")[-1]
            add(nm, _cls(f["id"]), f["installed"], "flatpak", False)
        req_names = set(req)
        oi = dp.get("os_install") or ""
        post = {p for p, d in dp["first_install"].items() if d > oi and p in dp["pkgs"]} | req_names

        def post_pkg(*names):
            for n in names:
                n = (n or "").lower()
                if len(n) >= 4 and any(p == n or (len(p) >= 4 and (p.startswith(n) or n.startswith(p))) for p in post):
                    return True
            return False
        for e in _desktop(h):
            if not e["launchable"]:
                continue
            user_scope = e["src"] in ("user", "flatpak_user", "snap", "flatpak")
            from_opt = e["exec"].startswith("/opt/")
            pre = not (user_scope or from_opt or post_pkg(e["id"].split(".")[-1], os.path.basename(e["exec"])))
            add(e["name"], _cls(e["name"], e["categories"]), e["ctime"] if not pre else None, "desktop_" + e["src"], pre)
        for nm in _ls(os.path.join(_U(h), ".local", "bin")) or []:
            if nm.lower() in ("claude", "codex", "hermes", "ollama", "gemini", "aider", "goose", "opencode", "cua-driver",
                              "agent-browser", "copilot"):
                add(nm, "ai", _mtime(os.path.join(_U(h), ".local", "bin", nm)), "local_bin", False)
        allv = list(apps.values())
        user = [v for v in allv if v["category"] in ad.USER_CATS or v["category"] == "other"]
        return {"os_install": dp.get("os_install"), "all": allv, "user": user}
    return _memo(h, "taxonomy", build)


@_lp("apps.dpkg", level="L1", family=APPS, tier="T0", collect="core")
def apps_dpkg(h, facts):
    """dpkg package database present (stat only; parsed by the L2 extractors)."""
    m = h.meta("/var/lib/dpkg/status")
    if not m.get("present"):
        return None
    return {"present": True, "status_bytes": m["bytes"], "status_mtime": _day(m["mtime"]),
            "apt": os.path.exists("/usr/bin/apt"), "os_install": _os_install_date()}


@_lp("apps.packages", level="L2", family=APPS, tier="T0", collect="core", gate="apps.dpkg", timeout_ms=4000)
def apps_packages(h, facts):
    """Installed deb packages: count, manual vs auto, sections, user-requested installs (apt history), log range."""
    dp = _dpkg(h)
    if not dp:
        return None
    pk = dp["pkgs"]
    manual = [p for p in pk if p not in dp["auto"]]
    secs = collections.Counter((v["section"] or "?").split("/")[-1] for v in pk.values())
    req = {n: d for n, d, _ in dp["requested"] if n in pk}
    req_me = sorted({n for n, _, me in dp["requested"] if me and n in pk})
    return {"present": True, "count": len(pk), "manual": len(manual), "auto": len(pk) - len(manual),
            "installed_gb": round(sum(v["kb"] for v in pk.values()) / 1e6, 2),
            "sections_top": dict(secs.most_common(12)), "user_requested": len(req),
            "user_requested_by_me": len(req_me), "user_requested_names": sorted(req)[:60],
            "dpkg_log_oldest": dp["oldest_log"], "os_install": dp["os_install"],
            "install_events_by_month": dict(sorted(dp["install_months"].items()))}


@_lp("apps.snap", level="L1", family=APPS, tier="T0", collect="core")
def apps_snap(h, facts):
    """Snap packages from /snap (no `snap list` spawn); base/runtime snaps flagged."""
    if not _isdir("/snap"):
        return None
    s = _snaps()
    user = [x["name"] for x in s if not _BASE_SNAPS.match(x["name"])]
    return {"present": bool(s), "count": len(s), "user_facing": sorted(user), "snapd": _ex("/usr/lib/snapd/snapd"),
            "user_snap_dirs": len(_ls(os.path.join(_U(h), "snap")) or [])}


@_lp("apps.flatpak", level="L1", family=APPS, tier="T0", collect="core")
def apps_flatpak(h, facts):
    """Flatpak apps (system + user installation dirs; no `flatpak list` spawn)."""
    f = _flatpaks(h)
    if not f and not _ex("/usr/bin/flatpak"):
        return None
    return {"present": True, "binary": _ex("/usr/bin/flatpak"), "count": len(f),
            "apps": sorted(x["id"] for x in f)[:60], "user_scope": sum(1 for x in f if x["scope"] == "user")}


@_lp("apps.desktop_entries", level="L2", family=APPS, tier="T0", collect="core", gate="apps.dpkg")
def apps_desktop_entries(h, facts):
    """Launchable .desktop entries by source and category; user-created launchers listed by name."""
    ents = _desktop(h)
    if not ents:
        return None
    la = [e for e in ents if e["launchable"]]
    by_src = collections.Counter(e["src"] for e in la)
    cats = collections.Counter(_cls(e["name"], e["categories"]) for e in la)
    return {"present": True, "files": len(ents), "launchable": len(la), "by_source": dict(by_src),
            "by_category": dict(cats.most_common()),
            "user_launchers": sorted(e["id"] for e in ents if e["src"] in ("user", "flatpak_user"))[:30]}


@_lp("apps.taxonomy", level="L2", family=APPS, tier="T0", collect="core", gate="apps.dpkg")
def apps_taxonomy(h, facts):
    """Merged user-facing inventory (apt requested + /opt + snap + flatpak + .desktop + ~/.local/bin) by category."""
    t = _taxonomy(h)
    cats = ad.USER_CATS + ["other"]
    return {"present": True, "unique_all": len(t["all"]), "unique_user_facing": len(t["user"]),
            "counts_all": dict(collections.Counter(v["category"] for v in t["all"]).most_common()),
            "counts_user_facing": {c: sum(1 for v in t["user"] if v["category"] == c) for c in cats},
            "names": {c: sorted(v["name"] for v in t["user"] if v["category"] == c)[:60] for c in cats}}


@_lp("apps.taxonomy_user_added", level="L2", family=APPS, tier="T0", collect="core", gate="apps.dpkg")
def apps_taxonomy_user_added(h, facts):
    """Category counts of apps the user added (not shipped with the image)."""
    t = _taxonomy(h)
    cats = ad.USER_CATS + ["other"]
    added = [v for v in t["user"] if not v["pre"]]
    months = collections.Counter(v["date"][:7] for v in added if v["date"])
    return {"present": True, "profile_created": _day(os.stat(_U(h)).st_ctime), "image_date": t["os_install"],
            "user_added_total": len(added), "preinstalled_total": len(t["user"]) - len(added),
            "counts": {c: sum(1 for v in added if v["category"] == c) for c in cats},
            "by_month": dict(sorted(months.items())), "undated": sum(1 for v in added if not v["date"])}


@_lp("apps.install_timeline", level="L2", family=APPS, tier="T0", collect="core", gate="apps.dpkg")
def apps_install_timeline(h, facts):
    """Month spread of user-requested installs (apt history, snap, flatpak, /opt); dpkg dependency installs separate."""
    dp = _dpkg(h) or {}
    t = _taxonomy(h)
    months = collections.Counter(v["date"][:7] for v in t["user"] if not v["pre"] and v["date"])
    return {"present": bool(months or dp.get("install_months")), "distinct_months": len(months),
            "by_month": dict(sorted(months.items())), "first": min(months) if months else None,
            "last": max(months) if months else None,
            "dpkg_install_months": len(dp.get("install_months") or {}), "log_window_starts": dp.get("oldest_log")}


@_lp("apps.autostart", level="L1", family=APPS, tier="T0", collect="core")
def apps_autostart(h, facts):
    """XDG autostart entries (user + /etc/xdg) and enabled systemd --user units (names only)."""
    user = sorted(n[:-8] for n in (_ls(_cfg(h, "autostart")) or []) if n.endswith(".desktop"))
    sysn = len([n for n in (_ls("/etc/xdg/autostart") or []) if n.endswith(".desktop")])
    units = set()
    for w in glob.glob(_cfg(h, "systemd", "user", "*.wants"))[:50]:
        units.update(n for n in (_ls(w) or []) if n.endswith((".service", ".timer", ".socket")))
    own = sorted(n for n in (_ls(_cfg(h, "systemd", "user")) or []) if n.endswith((".service", ".timer")))
    return {"present": True, "user_autostart": user, "system_autostart_count": sysn,
            "systemd_user_enabled": sorted(units)[:40], "systemd_user_unit_files": own[:40]}


_PROC_INTEREST = re.compile(r"claude|codex|hermes|ollama|lm ?studio|cursor|windsurf|copilot|gemini|opencode|aider|cua|"
                            r"agent-browser|docker|discord|steam|slack|teams|^code$|brave|chrome|firefox|tailscale|obs|"
                            r"spotify|tmux|zellij|screen|node|bun|python|signal|telegram|zoom|vnc|plasmashell|kwin", re.I)


@_lp("tasks.nonms", level="L2", family=APPS, tier="T2", collect="extended", gate="apps.dpkg", timeout_ms=3000)
def tasks_nonms(h, facts):
    """This user's scheduled jobs: `crontab -l` entry count and command names, systemd --user timers; lab jobs counted
    separately."""
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
    timers = sorted(n for n in (_ls(_cfg(h, "systemd", "user")) or []) if n.endswith(".timer"))
    if not jobs and not timers:
        return {"present": False}
    return {"present": True, "count": len(jobs) - op, "cron_commands": dict(names.most_common(10)),
            "operator_lab_jobs": op, "systemd_user_timers": timers[:20]}


@_lp("proc.snapshot", level="L1", family=APPS, tier="T1", collect="core")
def proc_snapshot(h, facts):
    """/proc snapshot for the invoking uid only: count, top names, running agents/apps; other users as a count."""
    p = _procs(h)
    n = p["names"]
    return {"present": True, "total": sum(n.values()), "distinct": len(n), "top": dict(n.most_common(20)),
            "interesting": dict(sorted((k, v) for k, v in n.items() if _PROC_INTEREST.search(k))),
            "all_users_total": p["total_all_users"]}


# ================================================================== ai_agents

@_lp("ai.catalog_absent_checks", level="L1", family=AI, tier="T0", collect="core")
def ai_catalog_absent_checks(h, facts):
    """Path stats for AI tools the user may lack (Cursor, Windsurf, Gemini CLI, opencode, aider, LM Studio...)."""
    U = _U(h)
    cat = {
        "cursor": [os.path.join(U, ".cursor"), _cfg(h, "Cursor"), "/opt/Cursor", "/usr/share/cursor"],
        "windsurf": [os.path.join(U, ".codeium"), os.path.join(U, ".windsurf"), _cfg(h, "Windsurf")],
        "gemini_cli": [os.path.join(U, ".gemini")],
        "opencode": [_data(h, "opencode"), _cfg(h, "opencode"), os.path.join(U, ".opencode")],
        "aider": [os.path.join(U, ".aider.conf.yml"), os.path.join(U, ".aider"), os.path.join(U, ".aider.chat.history.md")],
        "continue": [os.path.join(U, ".continue")], "cline": [os.path.join(U, ".cline")],
        "goose": [_cfg(h, "goose"), _data(h, "goose")], "amp": [_cfg(h, "amp")],
        "chatgpt_desktop": [_cfg(h, "ChatGPT")],
        "lm_studio": [os.path.join(U, ".lmstudio"), os.path.join(U, ".cache", "lm-studio"), _cfg(h, "LM Studio")],
        "jan": [_cfg(h, "Jan"), _data(h, "Jan"), os.path.join(U, "jan")], "gpt4all": [_data(h, "nomic.ai")],
        "msty": [_cfg(h, "Msty")], "anythingllm": [_cfg(h, "anythingllm-desktop")],
        "hf_hub": [_cachedir(h, "huggingface", "hub")], "vllm": [_cachedir(h, "vllm")],
        "llama_cpp": [_cachedir(h, "llama.cpp")], "comfyui": [os.path.join(U, "ComfyUI")],
    }
    found = {k: _ex(*v) for k, v in cat.items()}
    hf_models = sum(1 for n in (_ls(_cachedir(h, "huggingface", "hub")) or []) if n.startswith("models--"))
    return {"present": True, "found": sorted(k for k, v in found.items() if v),
            "absent": sorted(k for k, v in found.items() if not v), "hf_hub_models": hf_models}


def _npm_roots(h):
    U = _U(h)
    roots = [os.path.join(U, ".npm-global", "lib", "node_modules"), os.path.join(U, ".local", "lib", "node_modules"),
             "/usr/local/lib/node_modules", "/usr/lib/node_modules"]
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


@_lp("browser.automation", level="L1", family=AI, tier="T0", collect="core")
def browser_automation(h, facts):
    """Browser-automation runtimes on disk: Playwright, puppeteer, camoufox, agent-browser (npm or binary)."""
    U = _U(h)
    out = {}
    for label, p in (("playwright_cache", _cachedir(h, "ms-playwright")), ("puppeteer", _cachedir(h, "puppeteer")),
                     ("camoufox", _cachedir(h, "camoufox")), ("agent_browser_browsers", os.path.join(U, ".agent-browser", "browsers"))):
        names = _ls(p, 50)
        if names is not None:
            out[label] = {"entries": sorted(names)[:20], "mtime": _mtime(p)}
    ab = []
    npm = _npm_pkg(h, "agent-browser")
    if npm:
        ab.append({"where": "npm_" + npm["root"], "version": npm["version"]})
    hb = glob.glob(os.path.join(U, ".hermes", "hermes-agent", "node_modules", "agent-browser", "package.json"))
    if hb:
        ab.append({"where": "hermes_bundled", "version": (_read_json(hb[0]) or {}).get("version")})
    exe = _which(h, "agent-browser")
    if exe:
        ab.append({"where": "binary", "path_kind": "user" if exe.startswith(U) else "system"})
    if _isdir(os.path.join(U, ".agent-browser")):
        out["agent_browser_home"] = {"mtime": _mtime(os.path.join(U, ".agent-browser"))}
    if ab:
        out["agent_browser"] = ab
    return {"present": bool(out), **out}


_mirror("browser_harness.present", ad_agents.browser_harness_present)


@_lp("claude_code.present", level="L1", family=AI, tier="T0", collect="core")
def claude_code_present(h, facts):
    """Claude Code CLI: ~/.claude home, native install (~/.local/bin/claude -> versions/), npm global."""
    U = _U(h)
    home = os.path.join(U, ".claude")
    link = os.path.join(U, ".local", "bin", "claude")
    npm = _npm_pkg(h, "@anthropic-ai/claude-code")
    if not (_isdir(home) or os.path.lexists(link) or npm):
        return None
    ver = None
    try:
        ver = os.path.basename(os.readlink(link))
    except OSError:
        pass
    vers = _ls(_data(h, "claude", "versions")) or []
    return {"present": True, "home": _isdir(home), "native_exe": os.path.exists(link), "exe_version": ver,
            "npm_install": bool(npm), "npm_version": (npm or {}).get("version"), "installed_versions": len(vers),
            "home_mtime": _mtime(home), "running": sum(_running(h, r"^claude$|claude-code").values())}


_mirror("claude_code.config", ad_agents.claude_code_config)
_mirror("claude_code.claude_json", ad_agents.claude_code_claude_json)
_mirror("claude_code.sessions", ad_agents.claude_code_sessions)
_mirror("claude_code.titles", ad_agents.claude_code_titles)


@_lp("claude_desktop.present", level="L1", family=AI, tier="T0", collect="core")
def claude_desktop_present(h, facts):
    """Claude desktop (community Linux builds): ~/.config/Claude and MCP server names from its config."""
    d = _cfg(h, "Claude")
    if not _isdir(d):
        return None
    cfg = _read_json(os.path.join(d, "claude_desktop_config.json"))
    return {"present": True, "installed": _ex("/usr/bin/claude-desktop", "/opt/Claude"), "config_mtime": _mtime(d),
            "mcp_servers": sorted((cfg.get("mcpServers") or {}).keys()) if isinstance(cfg, dict) else []}


@_lp("codex.present", level="L1", family=AI, tier="T0", collect="core")
def codex_present(h, facts):
    """OpenAI Codex CLI: ~/.codex home (or CODEX_HOME), npm global, binary on PATH."""
    home = ad_agents._codex_home(h)
    npm = _npm_pkg(h, "@openai/codex")
    exe = _which(h, "codex")
    if not (_isdir(home) or npm or exe):
        return None
    return {"present": True, "home": _isdir(home), "npm_cli": bool(npm), "npm_version": (npm or {}).get("version"),
            "binary": bool(exe), "home_mtime": _mtime(home)}


@_lp("codex.auth_presence", level="L1", family=AI, tier="T3", collect="core")
def codex_auth_presence(h, facts):
    """Codex auth.json presence/size/mtime (never opened)."""
    return h.meta(os.path.join(ad_agents._codex_home(h), "auth.json"))


_mirror("codex.config", ad_agents.codex_config)
_mirror("codex.usage", ad_agents.codex_usage)
_mirror("codex.extras", ad_agents.codex_extras)
_mirror("codex.chatgpt_catalog", ad_agents.codex_chatgpt_catalog)
_mirror("copilot_cli.present", ad_agents.copilot_cli_present)
_mirror("docker.model_runner", ad_agents.docker_model_runner)


@_lp("cua_driver.present", level="L1", family=AI, tier="T0", collect="core")
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


def _hermes_homes_lx(h):
    env = os.environ.get("HERMES_HOME") or ""
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
    ad._memo(h, "hermes_homes", lambda: _hermes_homes_lx(h))


@_lp("hermes.present", level="L1", family=AI, tier="T0", collect="core")
def hermes_present(h, facts):
    """Hermes Agent: HERMES_HOME / ~/.hermes, install method, agent version, gateway unit, running processes."""
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
    units = sorted(n for n in (_ls(_cfg(h, "systemd", "user")) or []) if "hermes" in n.lower())
    desktop = glob.glob(_cfg(h, "Hermes*"))
    if not (homes or hh["side"] or units or desktop or _which(h, "hermes")):
        return None
    return {"present": True, "HERMES_HOME_set": hh["env_set"], "user_home": hh["primary"] is not None, "homes": homes,
            "side_homes_operator": len(hh["side"]), "cli_on_path": bool(_which(h, "hermes")),
            "systemd_user_units": units, "desktop_userdata_dirs": len(desktop),
            "running": sum(_running(h, r"hermes").values())}


_mirror("hermes.auth_presence", ad_agents.hermes_auth_presence, _prime_hermes)
_mirror("hermes.config", ad_agents.hermes_config, _prime_hermes)
_mirror("hermes.skills", ad_agents.hermes_skills, _prime_hermes)
_mirror("hermes.usage", ad_agents.hermes_usage, _prime_hermes)
_mirror("hermes.titles", ad_agents.hermes_titles, _prime_hermes)


@_lp("l3.mcp_inventory", level="L2", family=AI, tier="T0", collect="core", gate="ai.catalog_absent_checks")
def l3_mcp_inventory(h, facts):
    """Union of whitelisted MCP server names across agents (Claude Code, Codex, Hermes, VS Code, Cursor...)."""
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
    for label, path, keys in [("vscode", _cfg(h, "Code", "User", "mcp.json"), ("servers", "mcpServers")),
                              ("claude_desktop", _cfg(h, "Claude", "claude_desktop_config.json"), ("mcpServers",)),
                              ("cursor", os.path.join(U, ".cursor", "mcp.json"), ("mcpServers",)),
                              ("copilot_cli", os.path.join(U, ".copilot", "mcp-config.json"), ("mcpServers",)),
                              ("gemini_cli", os.path.join(U, ".gemini", "settings.json"), ("mcpServers",)),
                              ("windsurf", os.path.join(U, ".codeium", "windsurf", "mcp_config.json"), ("mcpServers",))]:
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


@_lp("ollama.present", level="L1", family=AI, tier="T0", collect="core")
def ollama_present(h, facts):
    """Ollama: binary, systemd unit, per-user models dir; the service account's home is stat'ed, never listed."""
    exe = _which(h, "ollama")
    unit = _first("/etc/systemd/system/ollama.service", "/usr/lib/systemd/system/ollama.service")
    mroot = ad_agents._ollama_models_root(h)
    svc_home = _isdir("/usr/share/ollama")
    if not (exe or unit or _isdir(os.path.join(_U(h), ".ollama")) or svc_home):
        return None
    return {"present": True, "binary": bool(exe), "systemd_unit": bool(unit), "models_dir": _isdir(mroot),
            "service_account_home": svc_home, "OLLAMA_MODELS_set": bool(os.environ.get("OLLAMA_MODELS")),
            "running": _running(h, r"^ollama")}


_mirror("ollama.models", ad_agents.ollama_models)
