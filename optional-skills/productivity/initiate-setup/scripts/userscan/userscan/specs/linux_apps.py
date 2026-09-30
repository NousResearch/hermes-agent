"""Linux user-level probes: apps, ai_agents, dev, browser, comms_work, files, gaming, media.

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
from userscan.specs import usage_gaming as ug

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


_mirror("browser_harness.present", ad.browser_harness_present)


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


_mirror("claude_code.config", ad.claude_code_config)
_mirror("claude_code.claude_json", ad.claude_code_claude_json)
_mirror("claude_code.sessions", ad.claude_code_sessions)
_mirror("claude_code.titles", ad.claude_code_titles)


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
    home = ad._codex_home(h)
    npm = _npm_pkg(h, "@openai/codex")
    exe = _which(h, "codex")
    if not (_isdir(home) or npm or exe):
        return None
    return {"present": True, "home": _isdir(home), "npm_cli": bool(npm), "npm_version": (npm or {}).get("version"),
            "binary": bool(exe), "home_mtime": _mtime(home)}


@_lp("codex.auth_presence", level="L1", family=AI, tier="T3", collect="core")
def codex_auth_presence(h, facts):
    """Codex auth.json presence/size/mtime (never opened)."""
    return h.meta(os.path.join(ad._codex_home(h), "auth.json"))


_mirror("codex.config", ad.codex_config)
_mirror("codex.usage", ad.codex_usage)
_mirror("codex.extras", ad.codex_extras)
_mirror("codex.chatgpt_catalog", ad.codex_chatgpt_catalog)
_mirror("copilot_cli.present", ad.copilot_cli_present)
_mirror("docker.model_runner", ad.docker_model_runner)


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
    units = sorted(n for n in (_ls(_cfg(h, "systemd", "user")) or []) if "hermes" in n.lower())
    desktop = glob.glob(_cfg(h, "Hermes*"))
    if not (homes or hh["side"] or units or desktop or _which(h, "hermes")):
        return None
    return {"present": True, "HERMES_HOME_set": hh["env_set"], "user_home": hh["primary"] is not None, "homes": homes,
            "side_homes_operator": len(hh["side"]), "cli_on_path": bool(_which(h, "hermes")),
            "systemd_user_units": units, "desktop_userdata_dirs": len(desktop),
            "running": sum(_running(h, r"hermes").values())}


_mirror("hermes.auth_presence", ad.hermes_auth_presence, _prime_hermes)
_mirror("hermes.config", ad.hermes_config, _prime_hermes)
_mirror("hermes.skills", ad.hermes_skills, _prime_hermes)
_mirror("hermes.usage", ad.hermes_usage, _prime_hermes)
_mirror("hermes.titles", ad.hermes_titles, _prime_hermes)


@_lp("l3.mcp_inventory", level="L2", family=AI, tier="T0", collect="core", gate="ai.catalog_absent_checks")
def l3_mcp_inventory(h, facts):
    """Union of whitelisted MCP server names across agents (Claude Code, Codex, Hermes, VS Code, Cursor...)."""
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
    p = ad._hermes_homes(h)["primary"]
    if p and os.path.exists(os.path.join(p["path"], "config.yaml")):
        try:
            by["hermes"] = ad._parse_hermes_yaml(os.path.join(p["path"], "config.yaml"))["mcp_servers"]
        except OSError:
            pass
    user = sorted({n for v in by.values() for n in v if n not in ad.BUNDLED_MCP})
    return {"present": True, "by_agent": {k: v for k, v in by.items() if v}, "user_configured": user,
            "user_configured_total": len(user), "vendor_bundled": sorted({n for v in by.values() for n in v if n in ad.BUNDLED_MCP})}


@_lp("ollama.present", level="L1", family=AI, tier="T0", collect="core")
def ollama_present(h, facts):
    """Ollama: binary, systemd unit, per-user models dir; the service account's home is stat'ed, never listed."""
    exe = _which(h, "ollama")
    unit = _first("/etc/systemd/system/ollama.service", "/usr/lib/systemd/system/ollama.service")
    mroot = ad._ollama_models_root(h)
    svc_home = _isdir("/usr/share/ollama")
    if not (exe or unit or _isdir(os.path.join(_U(h), ".ollama")) or svc_home):
        return None
    return {"present": True, "binary": bool(exe), "systemd_unit": bool(unit), "models_dir": _isdir(mroot),
            "service_account_home": svc_home, "OLLAMA_MODELS_set": bool(os.environ.get("OLLAMA_MODELS")),
            "running": _running(h, r"^ollama")}


_mirror("ollama.models", ad.ollama_models)


# ================================================================== dev

@_lp("apps.pkg_managers", level="L1", family=DEV, tier="T0", collect="core")
def apps_pkg_managers(h, facts):
    """Package managers: apt (inbox), snap, flatpak, brew, nix, npm/pnpm/bun/yarn, cargo, go, uv, pipx."""
    U = _U(h)
    b = {"apt": os.path.exists("/usr/bin/apt"), "snap": os.path.exists("/usr/bin/snap"),
         "flatpak": os.path.exists("/usr/bin/flatpak"), "brew": _ex("/home/linuxbrew/.linuxbrew/bin/brew", os.path.join(U, ".linuxbrew")),
         "nix": _ex("/nix", os.path.join(U, ".nix-profile")), "npm": bool(_which(h, "npm")), "pnpm": bool(_which(h, "pnpm")),
         "bun": bool(_which(h, "bun")), "yarn": bool(_which(h, "yarn")), "cargo": bool(_which(h, "cargo")),
         "go": bool(_which(h, "go")), "uv": bool(_which(h, "uv")), "pipx": bool(_which(h, "pipx")),
         "fnm": bool(_which(h, "fnm")) or _isdir(_data(h, "fnm")), "nvm": _isdir(os.path.join(U, ".nvm"))}
    inbox = {"apt", "snap"}
    return {"present": True, **b, "managers": sorted(k for k, v in b.items() if v and k not in inbox)}


def _ls_pkgs(root, cap=500):
    names = _ls(root, cap)
    if names is None:
        return None
    pk = []
    for n in names:
        if n.startswith("@"):
            pk += [f"{n}/{s}" for s in (_ls(os.path.join(root, n), 100) or [])]
        elif not n.startswith("."):
            pk.append(n)
    return sorted(pk)


@_lp("dev.npm_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
def dev_npm_globals(h, facts):
    """Global JS packages by manager (npm roots incl. fnm/nvm, pnpm, bun, yarn); directory listings, no spawn."""
    U = _U(h)
    out, allp = {}, set()
    for r in _npm_roots(h):
        pk = [p for p in (_ls_pkgs(r) or []) if p not in ("npm", "corepack")]
        if pk:
            key = "npm_user" if r.startswith(U) else "npm_system"
            out.setdefault(key, []).extend(pk)
            allp.update(pk)
    for key, r in (("pnpm", _data(h, "pnpm", "global", "5", "node_modules")),
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


@_lp("dev.lang_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
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


@_lp("dev.uv_tools", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
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


@_lp("dev.git", level="L1", family=DEV, tier="T0", collect="core")
def dev_git(h, facts):
    """git binary on PATH (no spawn); version comes from dev.toolchain_versions."""
    exe = _which(h, "git")
    if not exe:
        return None
    return {"present": True, "on_path": bool(shutil.which("git")), "path_kind": "user" if exe.startswith(_U(h)) else "system",
            "xdg_config": os.path.exists(_cfg(h, "git", "config"))}


_mirror("dev.git_global_config", ad.dev_git_global_config)


@_lp("dev.gh_auth_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_gh_auth_presence(h, facts):
    """GitHub CLI hosts.yml presence/size (means gh logged in; never opened)."""
    return h.meta(_cfg(h, "gh", "hosts.yml"))


@_lp("dev.docker_config_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_docker_config_presence(h, facts):
    """~/.docker/config.json presence/size (may hold registry auths; never opened)."""
    return h.meta(os.path.join(_U(h), ".docker", "config.json"))


_mirror("dev.npmrc_presence", ad.dev_npmrc_presence)

_PATH_TOOLS = [("local_bin", r"/\.local/bin$"), ("home_bin", r"^/home/[^/]+/bin$"), ("bun", r"/\.bun/bin"),
               ("cargo", r"/\.cargo/bin"), ("go", r"/go/bin"), ("fnm", r"fnm"), ("nvm", r"\.nvm"), ("volta", r"\.volta"),
               ("pnpm", r"pnpm"), ("deno", r"\.deno"), ("snap", r"^/snap/bin"), ("flatpak", r"flatpak/exports/bin"),
               ("linuxbrew", r"linuxbrew"), ("nix", r"\.nix-profile|/nix/"), ("cuda", r"cuda"), ("conda", r"conda|miniforge"),
               ("pyenv", r"\.pyenv"), ("hermes", r"hermes"), ("games", r"^/usr(/local)?/games$"), ("opt", r"^/opt/")]


@_lp("dev.path_entries", level="L1", family=DEV, tier="T0", collect="core")
def dev_path_entries(h, facts):
    """PATH entry count and known-tool dir matches for this process (paths not emitted); user bin dirs off PATH."""
    ents = [e for e in os.environ.get("PATH", "").split(":") if e]
    tools = sorted({t for t, rx in _PATH_TOOLS for e in ents if re.search(rx, e)})
    off = [os.path.relpath(d, _U(h)) for d in _extra_bins(h)
           if d.startswith(_U(h)) and _isdir(d) and d not in ents]
    return {"present": True, "entries": len(ents), "user": sum(1 for e in ents if e.startswith(_U(h))),
            "machine": sum(1 for e in ents if not e.startswith(_U(h))), "tools": tools, "user_bin_dirs_off_path": off}


_mirror("dev.ssh_config", ad.dev_ssh_config)

_TOOLCHAIN = ["node", "npm", "pnpm", "yarn", "bun", "deno", "fnm", "nvm", "volta", "uv", "pipx", "conda", "poetry",
              "rustc", "cargo", "rustup", "go", "dotnet", "java", "javac", "mvn", "gradle", "cmake", "ninja", "make", "clang",
              "gcc", "zig", "nvcc", "gh", "docker", "podman", "kubectl", "terraform", "tailscale", "claude", "codex",
              "hermes", "ollama", "code", "cursor", "windsurf", "zed", "nvim", "vim", "emacs", "kate", "python3", "python",
              "tmux", "zellij", "lazygit", "rg", "fd", "fzf", "zoxide", "mcfly", "btop", "htop", "jq", "ssh", "mosh"]


@_lp("dev.toolchain_presence", level="L1", family=DEV, tier="T0", collect="core")
def dev_toolchain_presence(h, facts):
    """which() for language/infra/AI/editor CLIs over PATH plus user tool dirs; off-PATH install dirs."""
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
        ("miniforge", os.path.join(U, "miniforge3")), ("fnm_nodes", _data(h, "fnm", "node-versions")),
        ("nvm", os.path.join(U, ".nvm")), ("cuda", "/usr/local/cuda"), ("sdkman", os.path.join(U, ".sdkman")),
        ("android_sdk", os.path.join(U, "Android", "Sdk")), ("dotnet", os.path.join(U, ".dotnet"))]}
    return {"present": True, "on_path": found, "user_installed": user_found, "off_path": sorted(k for k, v in off.items() if v)}


@_lp("dev.toolchain_versions", level="L2", family=DEV, tier="T0", collect="extended", gate="dev.toolchain_presence",
     timeout_ms=12000)
def dev_toolchain_versions(h, facts):
    """--version spawns (parallel, gated on which) for runtimes, package managers, AI CLIs and infra CLIs.
    `hermes --version` is excluded: it blocked > 10 s; hermes.present reads the version from pyproject."""
    from concurrent.futures import ThreadPoolExecutor
    specs = [("node", ["--version"]), ("npm", ["--version"]), ("uv", ["--version"]), ("gh", ["--version"]),
             ("git", ["--version"]), ("docker", ["--version"]), ("bun", ["--version"]), ("pnpm", ["--version"]),
             ("deno", ["--version"]), ("rustc", ["--version"]), ("cargo", ["--version"]), ("go", ["version"]),
             ("python3", ["--version"]), ("claude", ["--version"]), ("codex", ["--version"]), ("ollama", ["--version"]),
             ("tmux", ["-V"]), ("nvim", ["--version"]), ("kubectl", ["version", "--client=true"]),
             ("tailscale", ["version"]), ("java", ["-version"])]
    todo = [(n, [_which(h, n)] + a) for n, a in specs if _which(h, n)]

    def one(item):
        name, args = item
        rc, txt, ms = _spawn(h, args, timeout=3)
        line = next((l.strip() for l in txt.splitlines() if l.strip()), None)
        return name, {"version": ad._redact(line, 90) if line else None, "rc": rc, "ms": ms}
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, r in ex.map(one, todo):
            out[name] = r
    return {"present": bool(out), "tools": out}


_EDITORS = {
    # name: (install paths, extension dir(s), user dir)
    "vscode": (["/usr/share/code", "/usr/bin/code", "/snap/code", "/var/lib/flatpak/app/com.visualstudio.code"],
               [".vscode/extensions"], "Code"),
    "vscode_insiders": (["/usr/share/code-insiders"], [".vscode-insiders/extensions"], "Code - Insiders"),
    "vscode_server": ([], [".vscode-server/extensions"], None),
    "cursor": (["/opt/Cursor", "/usr/share/cursor", "/usr/bin/cursor"], [".cursor/extensions"], "Cursor"),
    "cursor_server": ([], [".cursor-server/extensions"], None),
    "windsurf": (["/usr/share/windsurf", "/opt/Windsurf"], [".windsurf/extensions"], "Windsurf"),
    "windsurf_server": ([], [".windsurf-server/extensions"], None),
    "vscodium": (["/usr/share/codium", "/snap/codium"], [".vscode-oss/extensions"], "VSCodium"),
    "kiro": (["/usr/share/kiro", "/opt/Kiro"], [".kiro/extensions"], "Kiro"),
}


@_lp("editor.vscode", level="L2", family=DEV, tier="T2", collect="core", gate="dev.toolchain_presence")
def editor_vscode(h, facts):
    """VS Code family incl. Remote-SSH server dirs: extension ids (AI flagged), workspaces, MCP/chat; other editors."""
    U = _U(h)
    out = {}
    for name, (inst, extds, userd) in _EDITORS.items():
        installed = any(os.path.exists(p) for p in inst)
        extd = next((os.path.join(U, e) for e in extds if _isdir(os.path.join(U, e))), None)
        user = _cfg(h, userd, "User") if userd else None
        if not (installed or extd or (user and _isdir(user))):
            continue
        ids = ad._ext_ids(extd) if extd else []
        e = {"installed": installed, "extensions": len(ids), "extension_ids": ids[:80],
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
        if name.endswith("_server"):
            e["remote_target"] = True
            e["user_mtime"] = _mtime(extd) if extd else None
        out[name] = e
    other = {"jetbrains": [x for x in (_ls(_cfg(h, "JetBrains")) or []) if re.match(r"^[A-Za-z]+\d{4}\.\d", x)],
             "zed": _isdir(_cfg(h, "zed")) or _isdir(_data(h, "zed")),
             "neovim_config": _isdir(_cfg(h, "nvim")), "vimrc": _ex(os.path.join(U, ".vimrc"), os.path.join(U, ".vim")),
             "emacs": _ex(os.path.join(U, ".emacs.d"), os.path.join(U, ".emacs"), _cfg(h, "emacs")),
             "helix": _isdir(_cfg(h, "helix")), "sublime": _isdir(_cfg(h, "sublime-text")),
             "kate": _ex(_cfg(h, "katerc")), "kwrite": _ex(_cfg(h, "kwriterc"))}
    other = {k: v for k, v in other.items() if v}
    customised = any(e["extensions"] or e.get("settings_json") or e.get("keybindings_json") for e in out.values()) or \
        bool(other.get("neovim_config") or other.get("emacs"))
    return {"present": bool(out or other), "extensions": sum(e["extensions"] for e in out.values()),
            "customised": customised, "editors": out, "other": other}


# ---------------- repos (deep)

_REPO_SKIP = {"node_modules", ".cache", "__pycache__", ".venv", "venv", "site-packages", ".npm", ".rustup", "snap",
              ".cargo", ".bun", ".vscode-server", ".cursor-server", ".mozilla", "trash", ".local/share/trash",
              "target", "dist", "build", ".next", ".git", "go", ".gradle", ".m2", ".nvm", ".pyenv", "miniconda3",
              "anaconda3", ".docker", ".ollama", ".var", ".steam"}


def _repos_lx(h):
    """Bounded repo walk: home depth 5, 60k dirs, 8 s; classes user_area / tool_managed (dot dirs) / operator."""
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
                if e.name.lower() in _REPO_SKIP:
                    continue
                try:
                    if not e.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                stack.append((e.path, depth + 1))
        repos = []
        for d in hits[:200]:
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
                                   "build.gradle", "Dockerfile", "flake.nix", "AGENTS.md", "CLAUDE.md", ".mcp.json")
                       if os.path.exists(os.path.join(d, m))]
            repos.append({"path": d, "class": cls, "branch": branch, "head_mtime": head_mtime, "remotes": remotes,
                          "local_email": local_email, "markers": markers})
        return {"repos": repos, "visited_dirs": visited[0], "timed_out": time.perf_counter() >= deadline}
    return build


def _commits_lx(h):
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
            rc, txt, _ms = _spawn(h, [git, "-C", x["path"], "log", "--all", "--no-merges",
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
    ad._memo(h, "repos", _repos_lx(h))


def _prime_commits(h):
    _prime_repos(h)
    ad._memo(h, "commits", _commits_lx(h))


# dev.repos and dev.repos.remotes need only the walk; priming commits there made them wait for git log on every repo.
_mirror("dev.repos", ad.dev_repos, _prime_repos)
_mirror("dev.repos.remotes", ad.dev_repos_remotes, _prime_repos)
_mirror("dev.repos.my_commits", ad.dev_repos_my_commits, _prime_commits)
_mirror("dev.repos.commit_hours", ad.dev_repos_commit_hours, _prime_commits)

_BUILTINS = {"cd", "export", "source", ".", "alias", "unalias", "echo", "exit", "history", "set", "unset", "eval", "exec",
             "type", "which", "for", "if", "while", "sudo", "time", "nohup", "env", "clear", "pwd", "ls", "ll", "la", "fg",
             "bg", "jobs", "kill", "z", "zi", "man", "help"}


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


@_lp("dev.shell_history", level="L2", family=DEV, tier="T1", collect="extended", gate="dev.path_entries")
def dev_shell_history(h, facts):
    """bash/zsh/fish/python/node history: line counts, last write, top command names (resolved executables or
    builtins only). Command lines, arguments and paths are never emitted."""
    U = _U(h)
    files = [("bash", os.path.join(U, ".bash_history")), ("zsh", os.environ.get("HISTFILE") or os.path.join(U, ".zsh_history")),
             ("zsh_histfile", os.path.join(U, ".histfile")), ("fish", _data(h, "fish", "fish_history")),
             ("python", os.path.join(U, ".python_history")), ("node", os.path.join(U, ".node_repl_history")),
             ("mcfly_db", _data(h, "mcfly", "history.db")), ("atuin_db", _data(h, "atuin", "history.db"))]
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
                elif label.startswith("zsh"):
                    line = re.sub(r"^: \d+:\d+;", "", line)
                n += 1
                if label in ("bash", "zsh", "zsh_histfile", "fish"):
                    c = _cmd_name(line, known)
                    if c:
                        cmds[c] += 1
        out[label] = {"lines": n, "bytes": st.st_size, "mtime": _day(st.st_mtime)}
        if label in ("bash", "zsh", "zsh_histfile", "fish"):
            total += n
    if not out:
        return {"present": False}
    return {"present": True, "lines": total, "files": out, "top_commands": dict(cmds.most_common(15)),
            "distinct_commands": len(cmds)}


@_lp("dev.docker", level="L1", family=DEV, tier="T0", collect="core")
def dev_docker(h, facts):
    """Docker engine/CLI: binary, socket access for this user, docker group, rootless or Desktop-for-Linux dirs."""
    cli = _which(h, "docker")
    sock = "/var/run/docker.sock"
    rootless = os.path.join(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"), "docker.sock")
    desktop = _isdir(os.path.join(_U(h), ".docker", "desktop"))
    if not (cli or os.path.exists(sock) or desktop):
        return None
    try:
        import grp
        in_group = "docker" in {grp.getgrgid(g).gr_name for g in os.getgroups()}
    except Exception:
        in_group = None
    return {"present": True, "cli": bool(cli), "engine_socket": os.path.exists(sock),
            "socket_access": os.access(sock, os.R_OK | os.W_OK), "docker_group": in_group,
            "rootless_socket": os.path.exists(rootless), "desktop_for_linux": desktop,
            "podman": bool(_which(h, "podman")), "compose_plugin": _ex("/usr/libexec/docker/cli-plugins/docker-compose",
                                                                        "/usr/lib/docker/cli-plugins/docker-compose")}


def _gb(s):
    m = re.match(r"([\d.]+)\s*([kKMGT]?B)", s or "")
    if not m:
        return 0.0
    return float(m.group(1)) * {"B": 1e-9, "kB": 1e-6, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}.get(m.group(2), 0)


@_lp("dev.docker_runtime", level="L2", family=DEV, tier="T2", collect="extended", gate="dev.docker", timeout_ms=8000)
def dev_docker_runtime(h, facts):
    """`docker ps` names/images and `docker images` repos/sizes only; never exec, inspect or logs."""
    d = _fact(facts, "dev.docker") or {}
    cli = _which(h, "docker")
    if not cli or not (d.get("socket_access") or d.get("rootless_socket")):
        return {"present": True, "accessible": False, "count": 0}
    rc, ps, ms1 = _spawn(h, [cli, "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.RunningFor}}"], timeout=5)
    rc2, im, ms2 = _spawn(h, [cli, "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}"], timeout=5)
    if rc != 0 and rc2 != 0:
        return {"present": True, "accessible": False, "count": 0}
    running = [dict(zip(("name", "image", "up"), l.split("\t"))) for l in ps.splitlines() if l.strip()]
    images = [l.split("\t") for l in im.splitlines() if l.strip()]
    return {"present": True, "accessible": True, "count": len(running),
            "running": [{"name": r.get("name"), "image": (r.get("image") or "").split("@")[0][:80], "up": r.get("up")}
                        for r in running[:40]],
            "images": len(images), "images_gb": round(sum(_gb(x[1]) for x in images if len(x) > 1), 2),
            "image_repos": sorted({x[0].rsplit(":", 1)[0] for x in images})[:40], "ms": round(ms1 + ms2, 1)}


@_lp("dev.multiplexers", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_multiplexers(h, facts):
    """tmux / zellij / screen session counts for this user (session names not emitted)."""
    uid = os.getuid()
    out = {}
    # no-tmp: ok — tmux puts its sockets under TMUX_TMPDIR, which falls back to /tmp
    tdir = os.path.join(os.environ.get("TMUX_TMPDIR", "/tmp"), f"tmux-{uid}")
    socks = [os.path.join(tdir, n) for n in (_ls(tdir) or [])]
    tm = _which(h, "tmux")
    if socks or tm:
        n = 0
        for s in socks[:10]:
            rc, txt, _ = _spawn(h, [tm, "-S", s, "ls"], timeout=2) if tm else (None, "", 0)
            if rc == 0:
                n += len([l for l in txt.splitlines() if l.strip()])
        out["tmux"] = {"installed": bool(tm), "sockets": len(socks), "sessions": n,
                       "config": _ex(os.path.join(_U(h), ".tmux.conf"), _cfg(h, "tmux", "tmux.conf"))}
    zj = _which(h, "zellij")
    rt = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    zsocks = [p for p in glob.glob(os.path.join(rt, "zellij", "*", "*")) + glob.glob(os.path.join(tempfile_dir(), f"zellij-{uid}", "*", "*"))
              if not os.path.isdir(p)]
    zres = glob.glob(_cachedir(h, "zellij", "*", "session_info", "*"))
    if zj or zsocks or zres:
        out["zellij"] = {"installed": bool(zj), "live_sessions": len(zsocks), "resurrectable": len(zres),
                         "config": _isdir(_cfg(h, "zellij"))}
    sd = _first(f"/run/screen/S-{os.environ.get('USER', '')}", os.path.join(_U(h), ".screen"))
    if sd:
        out["screen"] = {"sessions": len(_ls(sd) or [])}
    if not out:
        return {"present": False}
    return {"present": True, "count": sum(v.get("sessions", 0) + v.get("live_sessions", 0) for v in out.values()), **out}


def tempfile_dir():
    # no-tmp: ok — zellij puts its sockets under TMPDIR, which falls back to /tmp
    return os.environ.get("TMPDIR") or "/tmp"


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
                 ("browser.history.workspaces", bf.history_workspaces), ("browser.history.work_hosts", bf.history_work_hosts)):
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
    v = bf.files_composition(h, facts)
    if isinstance(v, dict):
        v["home_root"] = _home_root(h)
    return v
_mirror("files.screenshots", bf.files_screenshots, _prime_kf)


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
    desk = bf._kf(h, "Desktop")
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
    root = bf._kf(h, "Downloads")
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
            c = bf._categorize(bf._reg_domain(host)) if host else "other"
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
    ug._cached(("steam", h.l0.get("run_id")), _steam_lx(h))


@_lp("steam.present", level="L1", family=GAMING, tier="T0", collect="core")
def steam_present(h, facts):
    """Steam client data dir (native ~/.local/share/Steam, ~/.steam, flatpak, snap); gates the Steam subtree."""
    sp = _steam_root(h)
    if not sp:
        return None
    via = "flatpak" if "/.var/app/" in sp else "snap" if "/snap/" in sp else "native"
    return {"present": True, "via": via, "proton_prefixes": len(_ls(os.path.join(sp, "steamapps", "compatdata"), 5000) or [])}


for _id, _fn in (("steam.installed", ug.steam_installed), ("steam.playtime", ug.steam_playtime),
                 ("steam.appinfo_genres", ug.steam_appinfo_genres), ("steam.local_sessions", ug.steam_local_sessions),
                 ("steam.non_steam_shortcuts", ug.steam_non_steam_shortcuts), ("steam.screenshots", ug.steam_screenshots),
                 ("steam.login_users", ug.steam_login_users), ("steam.remote_clients", ug.steam_remote_clients)):
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
        s = ug._walk_media(p)
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
