"""macOS system-level probes (families: host, identity, install_age, locale, shell_prefs, security, network,
hardware, health, usage).

Ids match the Windows modules where macOS has a source for the same signal, so derive.py rules fire here too.
Where macOS has no source the id is left unregistered (see tests/darwin-verify.md, UNIMPLEMENTED).

Sources are plists read with plistlib, sysctl via ctypes (no spawn), and a few short spawns (system_profiler,
fdesetup, csrutil, spctl, socketfilterfw, networksetup, tailscale, pmset, last, dscl, bioutil). Stores behind
Transparency Consent and Control (TCC.db, knowledgeC.db) need Full Disk Access for the process that runs the
collector; without it the detector reports readable=false and the extractor returns absent.
"""
from __future__ import annotations

import collections
import ctypes
import ctypes.util
import datetime as dt
import glob
import json
import os
import platform
import plistlib
import re
import sqlite3
import threading
import time

from ..registry import probe

OS = "darwin"
MAC_EPOCH = 978307200  # 2001-01-01 UTC, Core Data / Cocoa absolute time

# ---------------------------------------------------------------- shared helpers

_LOCK = threading.Lock()
_CACHE: dict = {}
_KEYLOCKS: dict = {}


def memo(h, name, fn):
    """Compute fn() once per run, shared by every probe that needs it."""
    key = (h.l0.get("run_id"), name)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        lk = _KEYLOCKS.setdefault(key, threading.Lock())
    with lk:
        if key not in _CACHE:
            try:
                _CACHE[key] = fn()
            except Exception as e:
                _CACHE[key] = {"__error__": f"{type(e).__name__}: {e}"}
    v = _CACHE[key]
    if isinstance(v, dict) and "__error__" in v:
        raise RuntimeError(v["__error__"])
    return v


def home():
    return os.path.expanduser("~")


def H(*parts):
    return os.path.join(home(), *parts)


def isdir(p):
    try:
        return os.path.isdir(p)
    except OSError:
        return False


def ls(p, cap=2000):
    try:
        with os.scandir(p) as it:
            return [e.name for _, e in zip(range(cap), it)]
    except OSError:
        return None


def iso(ts):
    try:
        return dt.datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="minutes") if ts else None
    except (OverflowError, OSError, ValueError, TypeError):
        return None


def day(ts):
    try:
        return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d") if ts else None
    except (OverflowError, OSError, ValueError, TypeError):
        return None


def birth(p):
    try:
        st = os.stat(p)
        return getattr(st, "st_birthtime", None) or st.st_ctime
    except OSError:
        return None


def mtime(p):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return None


def plist(p, maxb=20_000_000):
    """plistlib load (binary or XML). None when missing, unreadable (TCC) or malformed."""
    try:
        if os.path.getsize(p) > maxb:
            return None
        with open(p, "rb") as f:
            return plistlib.load(f)
    except Exception:
        return None


def readable(p):
    """True if the file can be opened for reading. PermissionError here on a user file means TCC (FDA)."""
    try:
        fd = os.open(p, os.O_RDONLY)
        os.close(fd)
        return True
    except OSError:
        return False


def fmeta(p):
    """presence + size + readability, never reads contents. For TCC-protected and T3 files."""
    try:
        st = os.stat(p)
    except OSError:
        return {"present": False}
    ok = readable(p)
    return {"present": True, "bytes": st.st_size, "mtime": day(st.st_mtime), "readable": ok, "needs_fda": not ok}


def run(h, args, timeout_ms=5000):
    return h.run(args, timeout_ms=timeout_ms) or ""


def sp_json(h, *types, timeout_ms=8000):
    """system_profiler -json for one or more data types (one spawn)."""
    txt = run(h, ["system_profiler", "-json", "-detailLevel", "mini", *types], timeout_ms)
    try:
        return json.loads(txt) if txt.strip() else {}
    except ValueError:
        return {}


def sqlite_ro(path, sql, args=(), timeout=2.0):
    """Read-only query on the live file (no copy). Used for large WAL databases (Hermes state.db)
    where a copy would cost seconds; a mode=ro reader never writes the main file."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


_libc = None


def _lib():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"))
    return _libc


def sysctl_str(name):
    try:
        lib = _lib()
        size = ctypes.c_size_t(0)
        if lib.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if lib.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
            return None
        return buf.value.decode("utf-8", "replace")
    except Exception:
        return None


def sysctl_int(name, ctype=ctypes.c_int64):
    try:
        v = ctype()
        size = ctypes.c_size_t(ctypes.sizeof(v))
        if _lib().sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(size), None, 0) != 0:
            return None
        return int(v.value)
    except Exception:
        return None


def boottime():
    class _TV(ctypes.Structure):
        _fields_ = [("sec", ctypes.c_long), ("usec", ctypes.c_int)]
    try:
        t = _TV()
        size = ctypes.c_size_t(ctypes.sizeof(t))
        if _lib().sysctlbyname(b"kern.boottime", ctypes.byref(t), ctypes.byref(size), None, 0) == 0:
            return float(t.sec)
    except Exception:
        pass
    return None


def app_paths():
    """{lowercased bundle name without .app: path} for /Applications (depth 2), ~/Applications, /System/Applications."""
    def build():
        out = {}
        for root in ("/Applications", H("Applications"), "/System/Applications", "/System/Applications/Utilities"):
            for n in ls(root, 3000) or []:
                p = os.path.join(root, n)
                if n.endswith(".app"):
                    out.setdefault(n[:-4].lower(), p)
                elif root == "/Applications" and isdir(p) and not n.startswith("."):
                    for m in ls(p, 200) or []:
                        if m.endswith(".app"):
                            out.setdefault(m[:-4].lower(), os.path.join(p, m))
        return out
    return _shared_memo("app_paths", build)


def has_app(*names):
    """First matching installed app name (case-insensitive, exact bundle name or prefix match)."""
    ap = app_paths()
    for n in names:
        k = n.lower()
        if k in ap:
            return n
    for n in names:
        k = n.lower()
        if any(a.startswith(k) for a in ap):
            return n
    return None


_SHARED: dict = {}


def _shared_memo(name, fn):
    """Process-level memo for pure filesystem inventories that do not depend on the run."""
    with _LOCK:
        if name in _SHARED:
            return _SHARED[name]
    v = fn()
    with _LOCK:
        _SHARED[name] = v
    return v


GLOBAL_PREFS = lambda: plist(H("Library", "Preferences", ".GlobalPreferences.plist")) or {}
SYSCFG = "/Library/Preferences/SystemConfiguration/preferences.plist"
DISABLED = "/var/db/com.apple.xpc.launchd/disabled.plist"


def gprefs(h):
    return memo(h, "gprefs", GLOBAL_PREFS)


def syscfg(h):
    return memo(h, "syscfg", lambda: plist(SYSCFG) or {})


# ================================================================ host

@probe(id="os.build", level="L1", family="host", tier="T0", collect="core", os=OS)
def os_build(h, facts):
    """macOS product version and build (SystemVersion.plist, same data as sw_vers)."""
    sv = plist("/System/Library/CoreServices/SystemVersion.plist") or {}
    ver = sv.get("ProductVersion") or platform.mac_ver()[0]
    if not ver:
        return None
    return {"present": True, "product": sv.get("ProductName", "macOS"), "version": ver,
            "build": sv.get("ProductBuildVersion") or sysctl_str("kern.osversion"),
            "kernel": platform.release(), "major": int(ver.split(".")[0])}


_NAMES = {"11": "Big Sur", "12": "Monterey", "13": "Ventura", "14": "Sonoma", "15": "Sequoia", "26": "Tahoe"}


@probe(id="os.edition", level="L1", family="host", tier="T0", collect="core", os=OS)
def os_edition(h, facts):
    """macOS marketing name (Sequoia, Tahoe...). macOS has no editions; the name stands in for one."""
    ver = platform.mac_ver()[0]
    if not ver:
        return None
    return {"present": True, "edition": "macOS " + _NAMES.get(ver.split(".")[0], ver.split(".")[0]), "version": ver}


@probe(id="os.insider", level="L1", family="host", tier="T0", collect="core", os=OS)
def os_insider(h, facts):
    """Beta/seed enrollment: SoftwareUpdate CatalogURL or seed program key (AppleSeed / developer beta)."""
    su = plist("/Library/Preferences/com.apple.SoftwareUpdate.plist") or {}
    seed = plist("/Library/Preferences/com.apple.seeding.plist") or {}
    cat = str(su.get("CatalogURL") or "")
    beta = "seed" in cat.lower() or bool(seed.get("SeedProgram") or seed.get("ProgramID"))
    return {"present": True, "beta": beta, "catalog_custom": bool(cat),
            "auto_install_macos_updates": su.get("AutomaticallyInstallMacOSUpdates"),
            "auto_download": su.get("AutomaticDownload"),
            "recommended_major_upgrade": su.get("LastRecommendedMajorOSBundleIdentifier")}


@probe(id="host.native_arch", level="L1", family="host", tier="T0", collect="core", os=OS)
def host_native_arch(h, facts):
    """Native CPU arch (hw.optional.arm64) vs interpreter arch."""
    arm = sysctl_int("hw.optional.arm64", ctypes.c_int) == 1
    return {"present": True, "arch": "arm64" if arm else "x86_64", "python_machine": platform.machine(),
            "apple_silicon": arm}


@probe(id="host.python_emulated", level="L1", family="host", tier="T0", collect="core", os=OS)
def host_python_emulated(h, facts):
    """Collector running under Rosetta 2 (sysctl.proc_translated=1): timings are then unreliable."""
    tr = sysctl_int("sysctl.proc_translated", ctypes.c_int)
    return {"present": True, "emulated": tr == 1, "proc_translated": tr}


FDA_SOURCES = {
    "tcc_user": ("Library", "Application Support", "com.apple.TCC", "TCC.db"),
    "knowledgec": ("Library", "Application Support", "Knowledge", "knowledgeC.db"),
    "safari_history": ("Library", "Safari", "History.db"),
    "messages_chat": ("Library", "Messages", "chat.db"),
    "notes_store": ("Library", "Group Containers", "group.com.apple.notes", "NoteStore.sqlite"),
    "mail_dir": ("Library", "Mail"),
    "shared_file_list": ("Library", "Application Support", "com.apple.sharedfilelist"),
}


def fda_state(h):
    def build():
        out = {}
        for k, parts in FDA_SOURCES.items():
            p = H(*parts)
            if not os.path.exists(p):
                out[k] = "absent"
            elif os.path.isdir(p):
                out[k] = "readable" if ls(p, 1) is not None else "blocked"
            else:
                out[k] = "readable" if readable(p) else "blocked"
        out["tcc_system"] = "readable" if readable("/Library/Application Support/com.apple.TCC/TCC.db") else "blocked"
        return out
    return memo(h, "fda", build)


@probe(id="host.fda", level="L1", family="host", tier="T0", collect="core", os=OS)
def host_fda(h, facts):
    """Whether this process has Full Disk Access: open() on TCC-protected stores (TCC.db, knowledgeC, Safari...)."""
    st = fda_state(h)
    blocked = sorted(k for k, v in st.items() if v == "blocked")
    return {"present": True, "full_disk_access": not blocked, "blocked": blocked,
            "readable": sorted(k for k, v in st.items() if v == "readable"),
            "absent": sorted(k for k, v in st.items() if v == "absent")}


# ================================================================ identity

_DEFAULT_HOST = re.compile(r"^.+['’]s (MacBook( Pro| Air)?|iMac|Mac( mini| Studio| Pro)?)( \(\d+\))?$", re.I)


@probe(id="host.name", level="L1", family="identity", tier="T0", collect="core", os=OS)
def host_name(h, facts):
    """ComputerName / LocalHostName with the owner's first name redacted (it is the account name)."""
    s = (syscfg(h).get("System") or {})
    cn = ((s.get("System") or {}).get("ComputerName")) or platform.node()
    lhn = ((s.get("Network") or {}).get("HostNames") or {}).get("LocalHostName")
    red = re.sub(r"^.+?(['’]s )", r"<user>\1", cn or "")
    return {"present": bool(cn), "name": red, "local_host_name_set": bool(lhn),
            "default_pattern": bool(_DEFAULT_HOST.match(cn or ""))}


@probe(id="host.name_default", level="L1", family="identity", tier="T0", collect="core", os=OS)
def host_name_default(h, facts):
    """ComputerName still the Setup Assistant default ("<first name>'s MacBook Pro")."""
    s = (syscfg(h).get("System") or {}).get("System") or {}
    cn = s.get("ComputerName") or ""
    return {"present": True, "default": bool(_DEFAULT_HOST.match(cn))}


@probe(id="acct.account_type", level="L1", family="identity", tier="T0", collect="core", os=OS)
def acct_account_type(h, facts):
    """Current user admin (member of gid 80) or standard."""
    try:
        admin = 80 in os.getgroups()
    except OSError:
        admin = None
    return {"present": True, "admin": admin, "type": "admin" if admin else "standard", "uid": os.getuid()}


@probe(id="profile.count", level="L1", family="identity", tier="T0", collect="core", os=OS)
def profile_count(h, facts):
    """Home directories under /Users (names not emitted)."""
    names = [n for n in (ls("/Users") or []) if not n.startswith(".") and n not in ("Shared", "Guest")]
    return {"present": bool(names), "count": len(names)}


@probe(id="acct.local_users", level="L2", family="identity", tier="T1", collect="extended", gate="profile.count",
       os=OS)
def acct_local_users(h, facts):
    """Local accounts with uid >= 500 (dscl), count only; names never emitted."""
    txt = run(h, ["dscl", ".", "-list", "/Users", "UniqueID"], 4000)
    n = 0
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 500 and not parts[0].startswith("_"):
            n += 1
    lw = plist("/Library/Preferences/com.apple.loginwindow.plist") or {}
    return {"present": n > 0, "count": n, "guest_enabled": lw.get("GuestEnabled"),
            "recent_users": len(lw.get("RecentUsers") or [])}


# ================================================================ install_age

def install_history(h):
    return memo(h, "install_history", lambda: plist("/Library/Receipts/InstallHistory.plist") or [])


def _hist_date(e):
    d = e.get("date")
    if isinstance(d, dt.datetime):
        return d.replace(tzinfo=dt.timezone.utc).timestamp()
    return None


@probe(id="os.install_date", level="L1", family="install_age", tier="T0", collect="core", os=OS)
def os_install_date(h, facts):
    """OS install date: .AppleSetupDone birth time and the first InstallHistory.plist record."""
    sd = birth("/var/db/.AppleSetupDone")
    hist = [t for t in (_hist_date(e) for e in install_history(h)) if t]
    first = min(hist) if hist else None
    cands = [t for t in (sd, first) if t]
    if not cands:
        return None
    return {"present": True, "install_date": iso(min(cands)), "setup_done_birth": iso(sd),
            "install_history_first": iso(first), "install_log_birth": iso(birth("/var/log/install.log"))}


@probe(id="setup.oobe_done", level="L1", family="install_age", tier="T0", collect="core", os=OS)
def setup_oobe_done(h, facts):
    """Setup Assistant completion: /var/db/.AppleSetupDone birth time (stat only; the file is root-only)."""
    b = birth("/var/db/.AppleSetupDone")
    if not b:
        return None
    return {"present": True, "date": iso(b), "mtime": iso(mtime("/var/db/.AppleSetupDone"))}


@probe(id="profile.created", level="L1", family="install_age", tier="T0", collect="core", os=OS)
def profile_created(h, facts):
    """Home directory birth time. Older than the OS install means the home was migrated or restored."""
    b = birth(home())
    if not b:
        return None
    sd = birth("/var/db/.AppleSetupDone")
    pre = round((sd - b) / 86400) if sd and b < sd else 0
    return {"present": True, "created": iso(b), "predates_os_install_days": pre,
            "migrated_home": pre > 30}


@probe(id="setup.source_os_lineage", level="L1", family="install_age", tier="T0", collect="core", os=OS)
def setup_source_os_lineage(h, facts):
    """macOS versions installed on this volume (InstallHistory). count = major-version upgrades only."""
    rows = []
    for e in install_history(h):
        n = str(e.get("displayName") or "")
        if n.startswith("macOS") and e.get("processName") in ("softwareupdated", "macOS Installer", "Setup Assistant",
                                                               "installer", "Installer", "bootinstalld"):
            rows.append((_hist_date(e), str(e.get("displayVersion") or n.split()[-1])))
    rows = sorted(r for r in rows if r[0])
    majors = []
    for _, v in rows:
        m = v.split(".")[0]
        if not majors or majors[-1] != m:
            majors.append(m)
    return {"present": bool(rows), "count": max(0, len(majors) - 1), "majors": majors,
            "updates": [[day(t), v] for t, v in rows][-12:], "oldest": iso(rows[0][0]) if rows else None,
            "note": "minor updates do not count as an upgrade lineage"}


@probe(id="age.footprint_counts", level="L1", family="install_age", tier="T0", collect="core", os=OS)
def age_footprint_counts(h, facts):
    """Accumulation counters: Application Support dirs, preference plists, containers, group containers."""
    lib = H("Library")
    return {"present": True,
            "app_support_dirs": len(ls(os.path.join(lib, "Application Support"), 5000) or []),
            "preference_plists": len([n for n in ls(os.path.join(lib, "Preferences"), 5000) or [] if n.endswith(".plist")]),
            "containers": len(ls(os.path.join(lib, "Containers"), 5000) or []),
            "group_containers": len(ls(os.path.join(lib, "Group Containers"), 5000) or []),
            "launch_agents": len(ls(os.path.join(lib, "LaunchAgents"), 1000) or [])}


def _preferred_wifi(h):
    def build():
        txt = run(h, ["networksetup", "-listpreferredwirelessnetworks", "en0"], 4000)
        lines = [l for l in txt.splitlines() if l.startswith("\t") or l.startswith("    ")]
        return {"count": len(lines), "ok": "Preferred networks" in txt}
    return memo(h, "pref_wifi", build)


@probe(id="net.history", level="L2", family="install_age", tier="T1", collect="extended", gate="os.install_date",
       os=OS)
def net_history(h, facts):
    """Remembered Wi-Fi networks (networksetup preferred list; SSIDs never emitted)."""
    w = _preferred_wifi(h)
    if not w["ok"]:
        return {"present": False}
    return {"present": True, "wireless": w["count"],
            "note": "macOS keeps no first-connected date for non-root readers"}


# ================================================================ locale

@probe(id="locale.user", level="L1", family="locale", tier="T0", collect="core", os=OS)
def locale_user(h, facts):
    """AppleLocale, measurement units, first weekday, 24h clock (GlobalPreferences)."""
    g = gprefs(h)
    loc = g.get("AppleLocale")
    if not loc:
        return None
    return {"present": True, "locale": loc, "measurement_units": g.get("AppleMeasurementUnits"),
            "metric": g.get("AppleMetricUnits"), "temperature": g.get("AppleTemperatureUnit"),
            "first_weekday": g.get("AppleFirstWeekday"), "force_24h": g.get("AppleICUForce24HourTime"),
            "custom_date_formats": bool(g.get("AppleICUDateFormatStrings"))}


@probe(id="locale.geo", level="L1", family="locale", tier="T0", collect="core", os=OS)
def locale_geo(h, facts):
    """Region code from AppleLocale (en_IN -> IN)."""
    loc = gprefs(h).get("AppleLocale") or ""
    m = re.search(r"_([A-Z]{2})", loc)
    return {"present": bool(m), "geo": m.group(1) if m else None}


@probe(id="lang.ui", level="L1", family="locale", tier="T0", collect="core", os=OS)
def lang_ui(h, facts):
    """Primary UI language (AppleLanguages[0])."""
    langs = gprefs(h).get("AppleLanguages") or []
    return {"present": bool(langs), "ui_language": langs[0] if langs else None}


@probe(id="lang.user_list", level="L1", family="locale", tier="T0", collect="core", os=OS)
def lang_user_list(h, facts):
    """Preferred language list."""
    langs = gprefs(h).get("AppleLanguages") or []
    return {"present": bool(langs), "languages": langs[:10], "count": len(langs)}


@probe(id="kbd.layouts", level="L1", family="locale", tier="T0", collect="core", os=OS)
def kbd_layouts(h, facts):
    """Enabled keyboard layouts and input methods (HIToolbox)."""
    t = plist(H("Library", "Preferences", "com.apple.HIToolbox.plist")) or {}
    src = t.get("AppleEnabledInputSources") or []
    layouts = [s.get("KeyboardLayout Name") for s in src if s.get("InputSourceKind") == "Keyboard Layout"]
    ims = [s.get("Bundle ID") for s in src if s.get("InputSourceKind") in ("Input Mode", "Keyboard Input Method")]
    return {"present": bool(src), "layouts": [x for x in layouts if x], "input_methods": [x for x in ims if x],
            "current": t.get("AppleCurrentKeyboardLayoutInputSourceID")}


@probe(id="kbd.scancode_map", level="L1", family="locale", tier="T0", collect="core", os=OS)
def kbd_scancode_map(h, facts):
    """Keyboard remapping: per-keyboard modifier mappings (ByHost GlobalPreferences) and Karabiner-Elements."""
    n = 0
    for p in glob.glob(H("Library", "Preferences", "ByHost", ".GlobalPreferences.*.plist"))[:5]:
        d = plist(p) or {}
        n += sum(1 for k, v in d.items() if k.startswith("com.apple.keyboard.modifiermapping.") and v)
    kar = isdir(H(".config", "karabiner")) or bool(has_app("Karabiner-Elements"))
    return {"present": bool(n or kar), "remapped": bool(n or kar), "modifier_remaps": n, "karabiner": kar}


@probe(id="tz.zone", level="L1", family="locale", tier="T0", collect="core", os=OS)
def tz_zone(h, facts):
    """Time zone from the /etc/localtime link."""
    try:
        tgt = os.readlink("/etc/localtime")
    except OSError:
        return None
    m = re.search(r"zoneinfo/(.+)$", tgt)
    return {"present": bool(m), "zone": m.group(1) if m else tgt, "utc_offset_min": -time.timezone // 60}


@probe(id="tz.auto", level="L1", family="locale", tier="T0", collect="core", os=OS)
def tz_auto(h, facts):
    """Set time zone automatically (com.apple.timezone.auto Active)."""
    p = plist("/Library/Preferences/com.apple.timezone.auto.plist")
    if p is None:
        return None
    return {"present": True, "auto": bool(p.get("Active"))}


# ================================================================ shell_prefs

@probe(id="theme.dark", level="L1", family="shell_prefs", tier="T0", collect="core", os=OS)
def theme_dark(h, facts):
    """Dark mode (AppleInterfaceStyle) and automatic appearance switching."""
    g = gprefs(h)
    return {"present": True, "apps_dark": g.get("AppleInterfaceStyle") == "Dark",
            "auto_switch": bool(g.get("AppleInterfaceStyleSwitchesAutomatically")),
            "accent_color": g.get("AppleAccentColor"), "reduce_transparency": None}


@probe(id="theme.wallpaper", level="L1", family="shell_prefs", tier="T0", collect="core", os=OS)
def theme_wallpaper(h, facts):
    """Desktop wallpaper kind: OS default (dynamic/aerial for this release) vs chosen image or colour."""
    d = plist(H("Library", "Application Support", "com.apple.wallpaper", "Store", "Index.plist")) or {}
    desk = ((d.get("AllSpacesAndDisplays") or {}).get("Desktop") or {})
    ch = ((desk.get("Content") or {}).get("Choices") or [{}])[0]
    prov = str(ch.get("Provider") or "")
    files = [f.get("relative", "") for f in ch.get("Files") or [] if isinstance(f, dict)]
    system = any("/System/Library/Desktop%20Pictures" in f or "/System/Library/Desktop Pictures" in f for f in files)
    kind = ("default" if (not prov or re.search(r"choice\.(sequoia|sonoma|ventura|macintosh|tahoe|dynamic)", prov)
                          and not files) else
            "system_color" if system and "Solid" in " ".join(files) else
            "system_picture" if system else "aerial" if "aerial" in prov else "user_image" if files else "other")
    return {"present": bool(d), "default": kind == "default", "kind": kind, "provider": prov.split(".")[-1] or None,
            "last_set": iso(desk.get("LastSet").timestamp()) if isinstance(desk.get("LastSet"), dt.datetime) else None}


@probe(id="theme.spotlight_suggestions", level="L1", family="shell_prefs", tier="T0", collect="core", os=OS)
def theme_spotlight_suggestions(h, facts):
    """Spotlight / Siri suggestions category enabled (com.apple.Spotlight orderedItems)."""
    s = plist(H("Library", "Preferences", "com.apple.Spotlight.plist")) or {}
    items = s.get("orderedItems")
    on = True
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("name") in ("MENU_SPOTLIGHT_SUGGESTIONS", "MENU_WEBSEARCH"):
                on = on and bool(it.get("enabled"))
    return {"present": True, "suggestions_on": on, "customised": isinstance(items, list)}


_APPLE_DOCK = {"com.apple.finder", "com.apple.launchpad.launcher", "com.apple.apps.launcher", "com.apple.safari",
               "com.apple.mail", "com.apple.maps", "com.apple.photos", "com.apple.facetime", "com.apple.ical",
               "com.apple.addressbook", "com.apple.reminders", "com.apple.notes", "com.apple.freeform", "com.apple.tv",
               "com.apple.music", "com.apple.news", "com.apple.appstore", "com.apple.systempreferences",
               "com.apple.mobilesms", "com.apple.iwork.keynote", "com.apple.iwork.numbers", "com.apple.iwork.pages",
               "com.apple.podcasts", "com.apple.iphonemirroring"}


def _dock(h):
    return memo(h, "dock", lambda: plist(H("Library", "Preferences", "com.apple.dock.plist")) or {})


@probe(id="pins.taskbar", level="L1", family="shell_prefs", tier="T0", collect="core", os=OS)
def pins_taskbar(h, facts):
    """Dock pinned apps (bundle ids), recents, autohide, position."""
    d = _dock(h)
    apps = [((a.get("tile-data") or {}).get("bundle-identifier") or "") for a in d.get("persistent-apps") or []]
    return {"present": bool(d), "count": len(apps), "pins": [a for a in apps if a][:30],
            "autohide": d.get("autohide"), "show_recents": d.get("show-recents"), "orientation": d.get("orientation"),
            "tile_size": d.get("tilesize")}


@probe(id="pins.taskband_oem", level="L2", family="shell_prefs", tier="T0", collect="core", gate="pins.taskbar", os=OS)
def pins_taskband_oem(h, facts):
    """Apple default Dock apps still pinned. oem_pins is set only when the Dock looks factory (>= 8 defaults)."""
    d = _dock(h)
    apps = [((a.get("tile-data") or {}).get("bundle-identifier") or "").lower() for a in d.get("persistent-apps") or []]
    n = sum(1 for a in apps if a in _APPLE_DOCK)
    return {"present": True, "default_pins": n, "other_pins": len(apps) - n, "oem_pins": n if n >= 8 else 0}


@probe(id="shell.explorer_prefs", level="L1", family="shell_prefs", tier="T0", collect="core", os=OS)
def shell_explorer_prefs(h, facts):
    """Finder prefs: show all extensions, hidden files, path/status bar, default view."""
    g = gprefs(h)
    f = plist(H("Library", "Preferences", "com.apple.finder.plist")) or {}
    show_ext = bool(g.get("AppleShowAllExtensions"))
    return {"present": True, "HideFileExt": 0 if show_ext else 1, "show_extensions": show_ext,
            "show_hidden": bool(f.get("AppleShowAllFiles")), "path_bar": f.get("ShowPathbar"),
            "status_bar": f.get("ShowStatusBar"), "view_style": f.get("FXPreferredViewStyle"),
            "desktop_icons_hidden": f.get("CreateDesktop") is False,
            "autocorrect_off": g.get("NSAutomaticSpellingCorrectionEnabled") is False,
            "text_replacements": len(g.get("NSUserDictionaryReplacementItems") or [])}


# ================================================================ security

_REMOTE = ["AnyDesk", "TeamViewer", "RustDesk", "Parsec", "Jump Desktop", "Screens 5", "Screens for Mac",
           "Chrome Remote Desktop Host", "Splashtop Business", "Microsoft Remote Desktop", "Windows App", "NoMachine"]


@probe(id="security.remote_tools", level="L1", family="security", tier="T0", collect="core", os=OS)
def security_remote_tools(h, facts):
    """Third-party remote-access apps installed. Always present (gates the security subtree)."""
    return {"present": True, "found": [n for n in _REMOTE if has_app(n)]}


def _disabled(h):
    return memo(h, "launchd_disabled", lambda: plist(DISABLED) or {})


@probe(id="security.openssh_server", level="L1", family="security", tier="T3", collect="core", os=OS)
def security_openssh_server(h, facts):
    """Remote Login (sshd) enabled: com.openssh.sshd=false in launchd disabled.plist. Keys never read."""
    d = _disabled(h)
    on = d.get("com.openssh.sshd") is False
    return {"present": on, "sshd": on, "sshd_config_bytes": fmeta("/etc/ssh/sshd_config").get("bytes")}


@probe(id="security.rdp", level="L1", family="security", tier="T0", collect="core", os=OS)
def security_rdp(h, facts):
    """Screen Sharing / Remote Management (ARD) enabled (launchd disabled.plist + ARD launchd flag)."""
    d = _disabled(h)
    ss = d.get("com.apple.screensharing") is False
    ard = os.path.exists("/Library/Application Support/Apple/Remote Desktop/RemoteManagement.launchd")
    return {"present": True, "rdp_enabled": ss or ard, "screen_sharing": ss, "remote_management": ard}


@probe(id="acct.autologon", level="L1", family="security", tier="T0", collect="core", os=OS)
def acct_autologon(h, facts):
    """Automatic login configured (loginwindow autoLoginUser present; name not emitted)."""
    lw = plist("/Library/Preferences/com.apple.loginwindow.plist") or {}
    return {"present": True, "autologon": bool(lw.get("autoLoginUser")),
            "kcpassword": os.path.exists("/etc/kcpassword"), "guest_enabled": lw.get("GuestEnabled"),
            "hide_user_list": lw.get("SHOWFULLNAME")}


@probe(id="acct.hello", level="L2", family="security", tier="T0", collect="core", gate="security.remote_tools",
       os=OS)
def acct_hello(h, facts):
    """Touch ID enrolled for this user (bioutil template count)."""
    txt = run(h, ["bioutil", "-c"], 2000)
    m = re.search(r"(\d+) biometric template", txt)
    if not m:
        return {"present": False}
    return {"present": True, "touch_id_templates": int(m.group(1)), "fingerprint": int(m.group(1)) > 0,
            "face": False}


@probe(id="security.firewall", level="L2", family="security", tier="T0", collect="core", gate="security.remote_tools",
       os=OS)
def security_firewall(h, facts):
    """Application Firewall global state and stealth mode (socketfilterfw)."""
    fw = "/usr/libexec/ApplicationFirewall/socketfilterfw"
    g = run(h, [fw, "--getglobalstate"], 3000)
    s = run(h, [fw, "--getstealthmode"], 3000)
    if not g:
        return None
    return {"present": True, "enabled": "enabled" in g.lower() and "disabled" not in g.lower(),
            "stealth": "on" in s.lower() or "enabled" in s.lower()}


@probe(id="security.bitlocker", level="L2", family="security", tier="T0", collect="core",
       gate="security.remote_tools", os=OS)
def security_bitlocker(h, facts):
    """Disk encryption: FileVault state (fdesetup status)."""
    t = run(h, ["fdesetup", "status"], 3000)
    if not t:
        return None
    return {"present": True, "product": "FileVault", "on": "FileVault is On" in t,
            "encrypting": "Encryption in progress" in t}


@probe(id="security.sip", level="L2", family="security", tier="T0", collect="core", gate="security.remote_tools",
       os=OS)
def security_sip(h, facts):
    """System Integrity Protection (csrutil status)."""
    t = run(h, ["csrutil", "status"], 3000)
    if not t:
        return None
    return {"present": True, "enabled": "enabled" in t.lower() and "disabled" not in t.lower(),
            "custom": "Custom Configuration" in t}


@probe(id="security.smartscreen", level="L2", family="security", tier="T0", collect="core",
       gate="security.remote_tools", os=OS)
def security_smartscreen(h, facts):
    """Gatekeeper assessments (spctl --status), the macOS download-reputation check."""
    t = run(h, ["spctl", "--status"], 3000)
    if not t:
        return None
    return {"present": True, "product": "Gatekeeper", "enabled": "enabled" in t}


_AV = ["Malwarebytes", "Sophos Home", "Avast Security", "AVG AntiVirus", "Bitdefender Antivirus for Mac", "Norton 360",
       "Intego", "ESET Endpoint Security", "ClamXAV", "KnockKnock", "LuLu", "Little Snitch", "BlockBlock",
       "CrowdStrike Falcon", "Falcon", "SentinelOne", "Microsoft Defender", "Jamf Protect"]


@probe(id="security.av_products", level="L1", family="security", tier="T0", collect="core", os=OS)
def security_av_products(h, facts):
    """XProtect definitions version plus third-party AV / network-monitor apps installed."""
    xp = plist("/Library/Apple/System/Library/CoreServices/XProtect.bundle/Contents/Info.plist") or {}
    upd = [e for e in install_history(h) if e.get("processName") == "XProtectUpdateService"]
    last = max((_hist_date(e) or 0 for e in upd), default=0)
    return {"present": True, "xprotect_version": xp.get("CFBundleShortVersionString"),
            "xprotect_last_update": day(last) if last else None,
            "third_party": [n for n in _AV if has_app(n)]}


@probe(id="security.telemetry", level="L1", family="security", tier="T0", collect="core", os=OS)
def security_telemetry(h, facts):
    """Share Mac Analytics (CrashReporter AutoSubmit) and share with app developers (ThirdPartyDataSubmit)."""
    d = plist("/Library/Application Support/CrashReporter/DiagnosticMessagesHistory.plist")
    if d is None:
        return None
    auto = bool(d.get("AutoSubmit"))
    return {"present": True, "analytics_on": auto, "third_party_on": bool(d.get("ThirdPartyDataSubmit")),
            "reduced": not auto}


@probe(id="acct.admins", level="L2", family="security", tier="T0", collect="extended", gate="security.remote_tools",
       os=OS)
def acct_admins(h, facts):
    """Members of the admin group (count only)."""
    t = run(h, ["dscl", ".", "-read", "/Groups/admin", "GroupMembership"], 3000)
    m = t.split(":", 1)[1].split() if ":" in t else []
    return {"present": bool(m), "count": len([x for x in m if x != "root"])}


# ---- TCC (camera / microphone / screen recording) and location

TCC_USER = ("Library", "Application Support", "com.apple.TCC", "TCC.db")
TCC_SYS = "/Library/Application Support/com.apple.TCC/TCC.db"
LOCATIOND = "/private/var/db/locationd/clients.plist"


@probe(id="consent.store", level="L1", family="security", tier="T1", collect="core", os=OS)
def consent_store(h, facts):
    """Privacy permission stores: user/system TCC.db (Full Disk Access) and locationd clients.plist."""
    u, s = fmeta(H(*TCC_USER)), fmeta(TCC_SYS)
    loc = fmeta(LOCATIOND)
    return {"present": u["present"] or loc["present"], "tcc_user": u, "tcc_system": s, "locationd": loc}


def _tcc(h):
    def build():
        out = collections.defaultdict(lambda: {"granted": [], "denied": 0, "when": {}})
        read = []
        for label, path in (("user", H(*TCC_USER)), ("system", TCC_SYS)):
            rows = h.sqlite(path, "select service, client, auth_value, last_modified from access")
            if rows is None:
                rows = h.sqlite(path, "select service, client, auth_value, null from access")
            if rows is None:
                continue
            read.append(label)
            for svc, client, auth, when in rows:
                slot = out[svc]
                if auth in (2, 3):
                    slot["granted"].append(client)
                    if when:
                        slot["when"][client] = max(slot["when"].get(client, 0), when)
                else:
                    slot["denied"] += 1
        return {"read": read, "svc": dict(out)}
    return memo(h, "tcc", build)


def _tcc_probe(h, service):
    t = _tcc(h)
    if not t["read"]:
        return {"present": False, "needs_fda": True}
    s = t["svc"].get(service) or {"granted": [], "denied": 0, "when": {}}
    g = sorted(set(s["granted"]), key=lambda c: -(s["when"].get(c) or 0))
    return {"present": True, "granted": len(g), "denied": s["denied"],
            "items": [x.rsplit("/", 1)[-1] for x in g][:15],
            "granted_on": [[x.rsplit("/", 1)[-1], day(s["when"].get(x))] for x in g][:15],
            "latest_grant": day(max(s["when"].values())) if s["when"] else None, "stores_read": t["read"]}


@probe(id="consent.webcam", level="L2", family="security", tier="T1", collect="core", gate="consent.store", os=OS)
def consent_webcam(h, facts):
    """Apps granted camera access (TCC kTCCServiceCamera). Needs Full Disk Access."""
    return _tcc_probe(h, "kTCCServiceCamera")


@probe(id="consent.microphone", level="L2", family="security", tier="T1", collect="core", gate="consent.store", os=OS)
def consent_microphone(h, facts):
    """Apps granted microphone access (TCC kTCCServiceMicrophone). Needs Full Disk Access."""
    return _tcc_probe(h, "kTCCServiceMicrophone")


@probe(id="consent.screen_capture", level="L2", family="security", tier="T1", collect="core", gate="consent.store",
       os=OS)
def consent_screen_capture(h, facts):
    """Apps granted Screen Recording (system TCC kTCCServiceScreenCapture). Needs Full Disk Access."""
    return _tcc_probe(h, "kTCCServiceScreenCapture")


@probe(id="consent.accessibility", level="L2", family="security", tier="T1", collect="core", gate="consent.store",
       os=OS)
def consent_accessibility(h, facts):
    """Apps granted Accessibility control (system TCC kTCCServiceAccessibility): automation tools, launchers."""
    return _tcc_probe(h, "kTCCServiceAccessibility")


@probe(id="consent.location", level="L2", family="security", tier="T1", collect="core", gate="consent.store", os=OS)
def consent_location(h, facts):
    """Location Services clients: authorized / denied counts and authorized bundle ids (locationd clients.plist)."""
    d = plist(LOCATIOND)
    if not isinstance(d, dict):
        return {"present": False}
    auth, denied = [], 0
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        if v.get("Authorized") is True:
            bid = v.get("BundleId") or (k.split(":", 2)[1][1:] if k.count(":") >= 2 else k)
            auth.append(str(bid).rsplit("/", 1)[-1])
        elif v.get("Authorized") is False:
            denied += 1
    return {"present": True, "clients": len(d), "granted": len(auth), "denied": denied, "items": sorted(auth)[:15]}


_PM = ["1Password", "1Password 7", "Bitwarden", "KeePassXC", "Strongbox", "Dashlane", "LastPass", "Proton Pass",
       "Enpass", "NordPass", "Keeper Password Manager", "MacPass"]


@probe(id="pm.desktop", level="L1", family="security", tier="T0", collect="core", os=OS)
def pm_desktop(h, facts):
    """Password manager apps installed (Apple Passwords is built in and not counted)."""
    found = [n for n in _PM if has_app(n)]
    if isdir(H("Library", "Application Support", "Bitwarden CLI")):
        found.append("Bitwarden CLI")
    return {"present": bool(found), "managers": found}


@probe(id="dev.ssh_dir_presence", level="L1", family="security", tier="T3", collect="core", os=OS)
def dev_ssh_dir_presence(h, facts):
    """~/.ssh presence and file count (keys never opened)."""
    names = ls(H(".ssh"), 500)
    if names is None:
        return None
    return {"present": True, "files": len(names), "has_config": "config" in names,
            "private_key_like": sum(1 for n in names if n.startswith("id_") and not n.endswith(".pub"))}


# ================================================================ network

@probe(id="net.hosts_proxy", level="L1", family="network", tier="T0", collect="core", os=OS)
def net_hosts_proxy(h, facts):
    """/etc/hosts custom entry count and proxies configured on any network service."""
    n = 0
    try:
        with open("/etc/hosts", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#") and not re.match(r"^(127\.0\.0\.1|::1|255\.255\.255\.255|fe80::1%lo0)\s+"
                                                                   r"(localhost|broadcasthost)\s*$", s):
                    n += 1
    except OSError:
        pass
    proxies = set()
    for sv in (syscfg(h).get("NetworkServices") or {}).values():
        px = sv.get("Proxies") or {}
        for k in ("HTTPEnable", "HTTPSEnable", "SOCKSEnable", "ProxyAutoConfigEnable", "ProxyAutoDiscoveryEnable"):
            if px.get(k) == 1:
                proxies.add(k.replace("Enable", ""))
    return {"present": True, "hosts_custom_entries": n, "proxies": sorted(proxies)}


_VPN = ["Tailscale", "WireGuard", "Windscribe", "Mullvad VPN", "NordVPN", "ProtonVPN", "Proton VPN",
        "Cloudflare WARP", "OpenVPN Connect", "Tunnelblick", "ExpressVPN", "Surfshark", "ZeroTier One", "Viscosity",
        "GlobalProtect", "Cisco Secure Client", "FortiClient", "Zscaler"]


@probe(id="net.vpn_clients", level="L1", family="network", tier="T0", collect="core", os=OS)
def net_vpn_clients(h, facts):
    """VPN / mesh clients installed and VPN network services configured."""
    apps = [n for n in _VPN if has_app(n)]
    svcs = collections.Counter()
    for sv in (syscfg(h).get("NetworkServices") or {}).values():
        if (sv.get("Interface") or {}).get("Type") == "VPN":
            svcs[(sv.get("Interface") or {}).get("SubType") or "vpn"] += 1
    ts = os.path.exists("/usr/local/bin/tailscale") or bool(has_app("Tailscale"))
    return {"present": bool(apps or svcs or ts), "apps": apps, "vpn_services": dict(svcs), "tailscale_cli": ts}


@probe(id="net.wifi_profiles", level="L2", family="network", tier="T0", collect="core", gate="net.hosts_proxy", os=OS)
def net_wifi_profiles(h, facts):
    """Preferred (remembered) Wi-Fi network count; SSIDs never emitted."""
    w = _preferred_wifi(h)
    if not w["ok"]:
        return {"present": False}
    return {"present": True, "count": w["count"]}


@probe(id="net.adapters", level="L2", family="network", tier="T0", collect="extended", gate="net.hosts_proxy", os=OS)
def net_adapters(h, facts):
    """Configured network services by type (Wi-Fi, Ethernet dongles, Thunderbolt Bridge, VPN, phone tethering)."""
    kinds = collections.Counter()
    tether = 0
    for sv in (syscfg(h).get("NetworkServices") or {}).values():
        it = sv.get("Interface") or {}
        hw = it.get("Hardware") or it.get("Type") or "?"
        kinds["Wi-Fi" if hw == "AirPort" else hw] += 1
        if re.search(r"iphone|pixel|galaxy|android", str(sv.get("UserDefinedName") or ""), re.I):
            tether += 1
    return {"present": bool(kinds), "services": sum(kinds.values()), "by_type": dict(kinds),
            "phone_tethering_services": tether}


@probe(id="net.dns", level="L2", family="network", tier="T0", collect="extended", gate="net.hosts_proxy", os=OS)
def net_dns(h, facts):
    """Resolver kinds from scutil --dns (public, local, Tailscale MagicDNS, router); addresses not emitted."""
    t = run(h, ["scutil", "--dns"], 3000)
    kinds = collections.Counter()
    for ip in set(re.findall(r"nameserver\[\d+\]\s*:\s*(\S+)", t)):
        if ip == "100.100.100.100":
            kinds["tailscale"] += 1
        elif ip in ("1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9", "208.67.222.222"):
            kinds["public"] += 1
        elif re.match(r"^(127\.|::1)", ip):
            kinds["local"] += 1
        elif re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|fe80)", ip):
            kinds["lan"] += 1
        else:
            kinds["other"] += 1
    return {"present": bool(kinds), "resolver_kinds": dict(kinds), "resolvers": len(re.findall(r"^resolver #", t, re.M))}


@probe(id="net.tailscale", level="L2", family="network", tier="T1", collect="extended", gate="net.vpn_clients", os=OS)
def net_tailscale(h, facts):
    """Tailscale backend state and peer counts (tailscale status --json; peer names never emitted)."""
    exe = next((p for p in ("/usr/local/bin/tailscale", "/opt/homebrew/bin/tailscale",
                            "/Applications/Tailscale.app/Contents/MacOS/Tailscale") if os.path.exists(p)), None)
    if not exe:
        return {"present": False}
    t = run(h, [exe, "status", "--json"], 4000)
    try:
        d = json.loads(t)
    except ValueError:
        return {"present": False}
    peers = list((d.get("Peer") or {}).values())
    oses = collections.Counter(str(p.get("OS") or "?") for p in peers)
    return {"present": True, "backend": d.get("BackendState"), "peers": len(peers),
            "online_peers": sum(1 for p in peers if p.get("Online")), "peer_os": dict(oses),
            "version": str(d.get("Version") or "").split("-")[0]}


# ================================================================ hardware

@probe(id="hw.system", level="L1", family="hardware", tier="T0", collect="core", os=OS)
def hw_system(h, facts):
    """Model identifier, chip, memory, core counts (sysctl via ctypes, no spawn)."""
    model = sysctl_str("hw.model")
    if not model:
        return None
    mem = sysctl_int("hw.memsize") or 0
    return {"present": True, "manufacturer": "Apple", "model": model, "chip": sysctl_str("machdep.cpu.brand_string"),
            "ram_gb": round(mem / 2 ** 30), "cpus": sysctl_int("hw.ncpu", ctypes.c_int),
            "laptop": _has_battery(h),
            "virtual": bool(sysctl_int("kern.hv_vmm_present", ctypes.c_int))}


def _has_battery(h):
    """pmset -g batt lists InternalBattery on laptops (Apple silicon model ids do not say MacBook)."""
    return memo(h, "has_battery", lambda: "InternalBattery" in run(h, ["pmset", "-g", "batt"], 2000))


@probe(id="hw.cpu", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system", os=OS)
def hw_cpu(h, facts):
    """Chip and performance/efficiency core split (sysctl perflevels + SPHardwareDataType marketing name)."""
    sp = (sp_json(h, "SPHardwareDataType").get("SPHardwareDataType") or [{}])[0]
    return {"present": True, "chip": sp.get("chip_type") or sysctl_str("machdep.cpu.brand_string"),
            "machine_name": sp.get("machine_name"), "model_number": sp.get("model_number"),
            "p_cores": sysctl_int("hw.perflevel0.physicalcpu", ctypes.c_int),
            "e_cores": sysctl_int("hw.perflevel1.physicalcpu", ctypes.c_int),
            "logical": sysctl_int("hw.logicalcpu", ctypes.c_int), "boot_rom": sp.get("boot_rom_version")}


@probe(id="hw.ram", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system", os=OS)
def hw_ram(h, facts):
    """Unified memory size (hw.memsize). Apple silicon memory is soldered and shared with the GPU."""
    b = sysctl_int("hw.memsize") or 0
    return {"present": b > 0, "total_gb": round(b / 2 ** 30), "unified": sysctl_int("hw.optional.arm64", ctypes.c_int) == 1}


@probe(id="hw.gpu", level="L1", family="hardware", tier="T0", collect="core", os=OS)
def hw_gpu(h, facts):
    """GPU. On Apple silicon it is integrated with unified memory; vram_gb is the GPU-addressable share
    (Metal recommendedMaxWorkingSetSize is about 2/3 of RAM up to 36 GB, 3/4 above)."""
    arm = sysctl_int("hw.optional.arm64", ctypes.c_int) == 1
    ram = (sysctl_int("hw.memsize") or 0) / 2 ** 30
    if not arm:
        return {"present": True, "name": None, "unified_memory": False, "vram_gb": None}
    share = 0.75 if ram > 36 else 0.67
    return {"present": True, "name": (sysctl_str("machdep.cpu.brand_string") or "Apple") + " GPU",
            "unified_memory": True, "vram_gb": round(ram * share, 1), "vram_basis": f"{int(share * 100)}% of unified RAM"}


def _displays(h):
    return memo(h, "sp_displays", lambda: sp_json(h, "SPDisplaysDataType", timeout_ms=8000).get("SPDisplaysDataType") or [])


@probe(id="hw.monitors", level="L2", family="hardware", tier="T0", collect="extended", gate="hw.system", os=OS)
def hw_monitors(h, facts):
    """Connected displays (resolution, built-in vs external) and GPU core count (SPDisplaysDataType)."""
    gpus = _displays(h)
    mons = []
    cores = None
    for g in gpus:
        cores = cores or g.get("sppci_cores")
        for d in g.get("spdisplays_ndrvs") or []:
            mons.append({"name": d.get("_name"), "internal": d.get("spdisplays_connection_type") == "spdisplays_internal",
                         "resolution": d.get("_spdisplays_resolution") or d.get("spdisplays_resolution"),
                         "pixels": d.get("_spdisplays_pixels"), "main": d.get("spdisplays_main") == "spdisplays_yes",
                         "retina": "Retina" in str(d.get("spdisplays_display_type") or d.get("_spdisplays_display-type") or "")})
    return {"present": bool(mons), "count": len(mons), "external": sum(1 for m in mons if not m["internal"]),
            "monitors": mons, "gpu_cores": int(cores) if str(cores or "").isdigit() else cores}


@probe(id="hw.monitor_history", level="L2", family="hardware", tier="T0", collect="extended", gate="hw.system", os=OS)
def hw_monitor_history(h, facts):
    """Distinct displays WindowServer has configured on this install (com.apple.windowserver.displays).
    No first/last dates are stored, so derive's change detector does not use it."""
    d = plist("/Library/Preferences/com.apple.windowserver.displays.plist") or {}
    used = ((d.get("DisplayUUIDMappings_v3") or {}).get("UsedUUIDs")) or []
    cfgs = ((d.get("DisplayAnyUserSets") or {}).get("Configs")) or []
    return {"present": bool(d), "display_uuids_seen": len(used), "arrangements": len(cfgs)}


@probe(id="hw.battery", level="L1", family="hardware", tier="T0", collect="core", os=OS)
def hw_battery(h, facts):
    """Internal battery present (laptop)."""
    if not _has_battery(h):
        return None
    return {"present": True, "laptop": True}


@probe(id="hw.battery_report", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.battery", os=OS)
def hw_battery_report(h, facts):
    """Battery health: cycle count, condition, max capacity %, charge state, power settings, AlDente (SPPowerDataType)."""
    sp = sp_json(h, "SPPowerDataType").get("SPPowerDataType") or []
    bat = next((x for x in sp if x.get("_name") == "spbattery_information"), {})
    pw = next((x for x in sp if x.get("_name") == "sppower_information"), {})
    hi = bat.get("sppower_battery_health_info") or {}
    ci = bat.get("sppower_battery_charge_info") or {}
    cap = str(hi.get("sppower_battery_health_maximum_capacity") or "").rstrip("%")
    if not hi:
        return {"present": False}
    return {"present": True, "cycle_count": hi.get("sppower_battery_cycle_count"),
            "condition": hi.get("sppower_battery_health"),
            "max_capacity_pct": int(cap) if cap.isdigit() else None,
            "charge_pct": ci.get("sppower_battery_state_of_charge"),
            "charging": ci.get("sppower_battery_is_charging") == "TRUE",
            "on_ac": (pw.get("AC Power") or {}).get("Current Power Source") == "TRUE",
            "low_power_mode": {"ac": (pw.get("AC Power") or {}).get("LowPowerMode"),
                               "battery": (pw.get("Battery Power") or {}).get("LowPowerMode")},
            "charge_limiter": [n for n in ("AlDente", "Battery Toolkit", "Bclm") if has_app(n)]}


@probe(id="hw.power_plan", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system", os=OS)
def hw_power_plan(h, facts):
    """pmset settings per power source: powermode (low/high), sleep and display sleep, Power Nap, wake-on-LAN."""
    t = run(h, ["pmset", "-g", "custom"], 3000)
    out, cur = {}, None
    for line in t.splitlines():
        if line.endswith(":") and not line.startswith(" "):
            cur = {"Battery Power:": "battery", "AC Power:": "ac", "UPS Power:": "ups"}.get(line.strip())
            if cur:
                out[cur] = {}
            continue
        m = re.match(r"^\s*(powermode|lowpowermode|highpowermode|sleep|displaysleep|powernap|womp|standby|hibernatemode)"
                     r"\s+(\d+)", line)
        if cur and m:
            out[cur][m.group(1)] = int(m.group(2))
    return {"present": bool(out), **out}


@probe(id="hw.disks", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system", os=OS)
def hw_disks(h, facts):
    """Data volume size and free space (statvfs); external volumes mounted under /Volumes."""
    vols = []
    for name, p in (("Data", "/System/Volumes/Data"),):
        try:
            s = os.statvfs(p)
        except OSError:
            continue
        size = s.f_blocks * s.f_frsize
        free = s.f_bavail * s.f_frsize
        vols.append({"name": name, "size_gb": round(size / 1e9), "free_gb": round(free / 1e9),
                     "free_pct": round(100 * free / size, 1) if size else None})
    ext = [n for n in ls("/Volumes") or [] if not n.startswith(".") and not os.path.islink(os.path.join("/Volumes", n))]
    return {"present": bool(vols), "volumes": vols, "external_mounted": len(ext)}


@probe(id="hw.bluetooth", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.system", os=OS)
def hw_bluetooth(h, facts):
    """Bluetooth power and paired device counts by minor type (names never emitted)."""
    bt = (sp_json(h, "SPBluetoothDataType").get("SPBluetoothDataType") or [{}])[0]
    ctl = bt.get("controller_properties") or {}
    types, conn, total = collections.Counter(), 0, 0
    for key, connected in (("device_connected", True), ("device_not_connected", False)):
        for d in bt.get(key) or []:
            for _n, props in (d.items() if isinstance(d, dict) else []):
                total += 1
                conn += connected
                types[str((props or {}).get("device_minorType") or "?")] += 1
    return {"present": bool(bt), "power": ctl.get("controller_state"), "paired": total, "connected": conn,
            "by_type": dict(types)}


_PERIPH = ["Logi Options+", "Logi Options", "Logitech G HUB", "lghub", "Razer Synapse", "SteelSeries GG",
           "iCUE", "Corsair iCUE", "Elgato Control Center", "Elgato Stream Deck", "Stream Deck", "Wooting",
           "VIA", "QMK Toolbox", "Karabiner-Elements", "Logi Tune"]


@probe(id="periph.rgb_suites", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system", os=OS)
def periph_rgb_suites(h, facts):
    """Peripheral vendor suites (Logitech, Razer, SteelSeries, Corsair, Elgato, keyboard firmware tools)."""
    found = sorted({n for n in _PERIPH if has_app(n)})
    return {"present": bool(found), "suites": found}


_TUNE = ["AlDente", "BetterDisplay", "iStat Menus", "Stats", "Macs Fan Control", "TG Pro", "Lunar", "MonitorControl",
         "Amphetamine", "Hand Mirror", "Rectangle", "Magnet", "Moom", "BetterTouchTool", "Bartender 5", "Ice",
         "Hammerspoon", "Raycast", "Alfred 5", "Keyboard Maestro"]


@probe(id="tuning.mac_utilities", level="L1", family="hardware", tier="T0", collect="core", os=OS)
def tuning_mac_utilities(h, facts):
    """Power-user system utilities: battery/display/fan control, window managers, launchers, automation."""
    found = [n for n in _TUNE if has_app(n)]
    return {"present": bool(found), "found": found}


@probe(id="hw.sleep_study", level="L2", family="hardware", tier="T1", collect="deep", gate="hw.battery", os=OS,
       timeout_ms=10000)
def hw_sleep_study(h, facts):
    """Sleep / wake / dark-wake events per day from pmset -g log (retained about a week; ~2.5 s spawn)."""
    t = run(h, ["pmset", "-g", "log"], 12000)
    rx = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d):\d\d:\d\d [+-]\d{4} (Sleep|Wake|DarkWake)\s{2,}")
    days = collections.defaultdict(collections.Counter)
    wake_hours = [0] * 24
    for line in t.splitlines():
        m = rx.match(line)
        if m:
            days[m.group(1)][m.group(3)] += 1
            if m.group(3) == "Wake":
                wake_hours[int(m.group(2))] += 1
    if not days:
        return {"present": False}
    tot = collections.Counter()
    for c in days.values():
        tot.update(c)
    return {"present": True, "days": len(days), "first_day": min(days), "last_day": max(days),
            "sleeps": tot["Sleep"], "user_wakes": tot["Wake"], "dark_wakes": tot["DarkWake"],
            "user_wakes_per_day": round(tot["Wake"] / len(days), 1), "wake_hours": wake_hours}


# ================================================================ health

_DIAG_DIRS = ("/Library/Logs/DiagnosticReports", "/Library/Logs/DiagnosticReports/Retired",
              "~/Library/Logs/DiagnosticReports", "~/Library/Logs/DiagnosticReports/Retired")
_NAME_RX = re.compile(r"^(.+?)[-_](\d{4}-\d\d-\d\d)-(\d{6})")


def _diag(h):
    def build():
        rows = []
        for d in _DIAG_DIRS:
            d = os.path.expanduser(d)
            for n in ls(d, 5000) or []:
                p = os.path.join(d, n)
                ext = n.rsplit(".", 1)[-1] if "." in n else ""
                if ext not in ("ips", "crash", "panic", "hang", "spin", "diag", "shutdownStall", "cpu_resource"):
                    continue
                m = _NAME_RX.match(n)
                app = m.group(1) if m else n.split(".")[0]
                ts = mtime(p) or 0
                bug = None
                if ext == "ips":
                    try:
                        with open(p, "rb") as f:
                            bug = json.loads(f.readline(2000).decode("utf-8", "replace")).get("bug_type")
                    except (OSError, ValueError):
                        bug = None
                kind = ("panic" if ext == "panic" or bug == "210" else
                        "crash" if ext == "crash" or bug in ("309", "109") else
                        "hang" if ext in ("hang", "spin") or bug in ("298", "228") else
                        "stall" if ext == "shutdownStall" else "resource")
                rows.append((app, kind, ts))
        return rows
    return memo(h, "diag", build)


@probe(id="health.dumps", level="L1", family="health", tier="T1", collect="core", os=OS)
def health_dumps(h, facts):
    """DiagnosticReports counts by kind. count_30d is kernel panics only (the macOS analogue of a bugcheck dump)."""
    rows = _diag(h)
    now = time.time()
    by = collections.Counter(k for _, k, _ in rows)
    by30 = collections.Counter(k for _, k, t in rows if now - t <= 30 * 86400)
    return {"present": True, "reports": len(rows), "by_kind": dict(by), "by_kind_30d": dict(by30),
            "count_30d": by30.get("panic", 0), "panics_total": by.get("panic", 0),
            "oldest": day(min((t for _, _, t in rows), default=0))}


@probe(id="health.wer", level="L2", family="health", tier="T1", collect="core", gate="health.dumps", os=OS)
def health_wer(h, facts):
    """App crashes and hangs by process name (DiagnosticReports file names and .ips bug_type)."""
    rows = _diag(h)
    crashes = collections.Counter(a for a, k, _ in rows if k == "crash")
    hangs = collections.Counter(a for a, k, _ in rows if k == "hang")
    now = time.time()
    c30 = collections.Counter(a for a, k, t in rows if k == "crash" and now - t <= 30 * 86400)
    return {"present": bool(crashes or hangs), "crashes": sum(crashes.values()), "hangs": sum(hangs.values()),
            "crashes_30d": sum(c30.values()),
            "crashes_by_app": [[a, n] for a, n in crashes.most_common(8)],
            "hangs_by_app": [[a, n] for a, n in hangs.most_common(5)],
            "max_repeat": crashes.most_common(1)[0][1] if crashes else 0}


# ================================================================ usage

@probe(id="boot.uptime", level="L1", family="usage", tier="T0", collect="core", os=OS)
def boot_uptime(h, facts):
    """Time since boot (CLOCK_MONOTONIC counts sleep) and awake time (CLOCK_UPTIME_RAW excludes it)."""
    bt = boottime()
    mono = time.clock_gettime(time.CLOCK_MONOTONIC)
    awake = time.clock_gettime(time.CLOCK_UPTIME_RAW)
    return {"present": True, "last_boot": iso(bt), "uptime_h": round(mono / 3600, 1),
            "awake_h": round(awake / 3600, 1), "awake_share": round(awake / mono, 2) if mono else None}


def _wtmp(h):
    """(reboots, shutdowns, wtmp_begins) epoch lists from one `last reboot` (it prints both kinds)."""
    def build():
        t = run(h, ["last", "reboot"], 4000)
        now = dt.datetime.now()
        out = {"reboot": [], "shutdown": []}
        for line in t.splitlines():
            m = re.match(r"^(reboot|shutdown) time\s+\w{3} (\d+) (\w{3}) (\d\d):(\d\d)", line)
            if not m:
                continue
            try:
                d = dt.datetime.strptime(f"{m.group(2)} {m.group(3)} {now.year} {m.group(4)}:{m.group(5)}", "%d %b %Y %H:%M")
            except ValueError:
                continue
            if d > now + dt.timedelta(days=1):
                d = d.replace(year=now.year - 1)
            out[m.group(1)].append(d.timestamp())
        m = re.search(r"wtmp begins \w{3} (\w{3}) +(\d+) [\d:]+ \S+ (\d{4})", t)
        begins = None
        if m:
            try:
                begins = dt.datetime.strptime(" ".join(m.groups()), "%b %d %Y").timestamp()
            except ValueError:
                pass
        return out["reboot"], out["shutdown"], begins
    return memo(h, "wtmp", build)


@probe(id="eventlog.power_history", level="L2", family="usage", tier="T1", collect="extended", gate="boot.uptime", os=OS)
def eventlog_power_history(h, facts):
    """Boots and shutdowns (last reboot / last shutdown, wtmp) and sleep / wake counts since boot (pmset -g stats)."""
    now = time.time()
    boots, shut, begins = _wtmp(h)
    st = run(h, ["pmset", "-g", "stats"], 3000)
    g = lambda k: int(m.group(1)) if (m := re.search(k + r":\s*(\d+)", st)) else None
    return {"present": bool(boots or st), "boots_30d": sum(1 for t in boots if now - t <= 30 * 86400),
            "boots": len(boots), "shutdowns_30d": sum(1 for t in shut if now - t <= 30 * 86400),
            "wtmp_begins": day(begins), "system_log_oldest": day(begins),
            "sleeps_since_boot": g("Sleep Count"), "wakes": g("User Wake Count"),
            "dark_wakes_since_boot": g("Dark Wake Count"),
            "note": "wakes are user wakes since the last boot, not a 30-day window"}


@probe(id="l3.power_event_split", level="L2", family="usage", tier="T1", collect="extended", gate="boot.uptime", os=OS)
def l3_power_event_split(h, facts):
    """Kernel panics (crash) vs shutdown stalls. macOS records no user-visible hard power-off count without
    the unified log, so power_removed is not reported."""
    rows = _diag(h)
    now = time.time()
    return {"present": True, "crash": sum(1 for _, k, _ in rows if k == "panic"),
            "crash_30d": sum(1 for _, k, t in rows if k == "panic" and now - t <= 30 * 86400),
            "shutdown_stalls": sum(1 for _, k, _ in rows if k == "stall"), "button_held": None}


KNOWLEDGE = ("Library", "Application Support", "Knowledge", "knowledgeC.db")
SCREENTIME = ("Library", "Application Support", "com.apple.remotemanagementd", "RMAdminStore-Local.sqlite")


@probe(id="usage.screen_time_store", level="L1", family="usage", tier="T0", collect="core", os=OS)
def usage_screen_time_store(h, facts):
    """knowledgeC.db (app focus, display state) and Screen Time store: presence and readability (Full Disk Access)."""
    k = fmeta(H(*KNOWLEDGE))
    s = fmeta(H(*SCREENTIME))
    biome = isdir(H("Library", "Biome", "streams", "restricted", "App.InFocus"))
    return {"present": k["present"] or s["present"], "knowledgec": k, "screentime": s, "biome_app_infocus": biome}


def _kc(h):
    """App focus and backlight intervals from knowledgeC.db (/app/usage, /display/isBacklit), last 90 days."""
    def build():
        p = H(*KNOWLEDGE)
        cut = time.time() - 90 * 86400 - MAC_EPOCH
        apps = h.sqlite(p, "select ZVALUESTRING, ZSTARTDATE, ZENDDATE from ZOBJECT where ZSTREAMNAME in "
                           "('/app/usage', '/app/inFocus') and ZSTARTDATE > ? and ZENDDATE > ZSTARTDATE", (cut,),
                        timeout_ms=4000)
        if apps is None:
            return None
        disp = h.sqlite(p, "select ZSTARTDATE, ZENDDATE from ZOBJECT where ZSTREAMNAME='/display/isBacklit' "
                           "and ZVALUEINTEGER=1 and ZSTARTDATE > ? and ZENDDATE > ZSTARTDATE", (cut,), timeout_ms=4000) or []
        return {"apps": [(a, s + MAC_EPOCH, e + MAC_EPOCH) for a, s, e in apps if a],
                "display": [(s + MAC_EPOCH, e + MAC_EPOCH) for s, e in disp]}
    return memo(h, "knowledgec", build)


def _app_label(bid):
    return str(bid).rsplit(".", 1)[-1] if bid else "?"


@probe(id="userassist.focus", level="L2", family="usage", tier="T1", collect="extended",
       gate="usage.screen_time_store", os=OS, timeout_ms=5000)
def userassist_focus(h, facts):
    """Per-app foreground hours and sessions over 90 days from knowledgeC /app/usage. Needs Full Disk Access."""
    kc = _kc(h)
    if not kc or not kc["apps"]:
        return {"present": False, "needs_fda": kc is None}
    hrs, n = collections.Counter(), collections.Counter()
    now = time.time()
    h30 = collections.Counter()
    first = min(s for _, s, _ in kc["apps"])
    for a, s, e in kc["apps"]:
        d = min(e - s, 8 * 3600) / 3600
        hrs[a] += d
        n[a] += 1
        if now - s <= 30 * 86400:
            h30[a] += d
    return {"present": True, "apps": len(hrs), "total_focus_h": round(sum(hrs.values()), 1),
            "focus_h_30d": round(sum(h30.values()), 1), "since": day(first),
            "top": [[a, round(x, 1), n[a]] for a, x in hrs.most_common(15)]}


@probe(id="l3.top_apps_by_time", level="L2", family="usage", tier="T1", collect="extended",
       gate="usage.screen_time_store", os=OS, timeout_ms=5000)
def l3_top_apps_by_time(h, facts):
    """Top apps by foreground hours in the last 30 days (knowledgeC). Needs Full Disk Access."""
    kc = _kc(h)
    if not kc or not kc["apps"]:
        return {"present": False, "needs_fda": kc is None}
    now = time.time()
    c = collections.Counter()
    for a, s, e in kc["apps"]:
        if now - s <= 30 * 86400:
            c[a] += min(e - s, 8 * 3600) / 3600
    return {"present": bool(c), "top": [{"app": a, "name": _app_label(a), "hours": round(x, 1)} for a, x in c.most_common(10)]}


@probe(id="usage.screen_on", level="L2", family="usage", tier="T1", collect="extended",
       gate="usage.screen_time_store", os=OS, timeout_ms=5000)
def usage_screen_on(h, facts):
    """Display-on hours per day and start-hour histogram (knowledgeC /display/isBacklit). Needs Full Disk Access."""
    kc = _kc(h)
    if not kc or not kc["display"]:
        return {"present": False, "needs_fda": kc is None}
    per_day = collections.Counter()
    hours = [0] * 24
    for s, e in kc["display"]:
        per_day[day(s)] += min(e - s, 16 * 3600) / 3600
        hours[dt.datetime.fromtimestamp(s).hour] += 1
    return {"present": True, "days": len(per_day), "avg_h_per_day": round(sum(per_day.values()) / len(per_day), 1),
            "max_h_day": round(max(per_day.values()), 1), "hours": hours}


def _mdls_apps(h):
    """kMDItemLastUsedDate / kMDItemUseCount / kMDItemDateAdded for every app bundle, one mdls spawn."""
    def build():
        paths = sorted(set(app_paths().values()))
        paths = [p for p in paths if not p.startswith("/System/")] + [p for p in paths if p.startswith("/System/")]
        out = {}
        for i in range(0, len(paths), 150):
            chunk = paths[i:i + 150]
            t = run(h, ["mdls", "-name", "kMDItemLastUsedDate", "-name", "kMDItemUseCount", "-name", "kMDItemDateAdded",
                        *chunk], 8000)
            vals = re.findall(r"^(kMDItem\w+)\s+=\s+(.*)$", t, re.M)
            for j, p in enumerate(chunk):
                block = dict(vals[j * 3:(j + 1) * 3]) if len(vals) >= (j + 1) * 3 else {}
                def pd(v):
                    try:
                        return dt.datetime.strptime(v, "%Y-%m-%d %H:%M:%S %z").timestamp()
                    except (TypeError, ValueError):
                        return None
                uc = block.get("kMDItemUseCount")
                out[p] = {"last_used": pd(block.get("kMDItemLastUsedDate")), "use_count": int(uc) if (uc or "").isdigit() else 0,
                          "added": pd(block.get("kMDItemDateAdded"))}
        return out
    return memo(h, "mdls_apps", build)


@probe(id="bam.last_run", level="L2", family="usage", tier="T1", collect="extended", gate="boot.uptime", os=OS,
       timeout_ms=8000)
def bam_last_run(h, facts):
    """Last-used date and launch count per app from Spotlight metadata (kMDItemLastUsedDate / kMDItemUseCount).
    No Full Disk Access needed. Launch count is Finder/LaunchServices opens, not focus time."""
    m = _mdls_apps(h)
    now = time.time()
    used = {p: v for p, v in m.items() if v["last_used"]}
    top = sorted(used.items(), key=lambda kv: -kv[1]["use_count"])[:15]
    return {"present": bool(used), "apps_indexed": len(m), "apps_ever_used": len(used),
            "used_7d": sum(1 for v in used.values() if now - v["last_used"] <= 7 * 86400),
            "used_30d": sum(1 for v in used.values() if now - v["last_used"] <= 30 * 86400),
            "never_used": len(m) - len(used),
            "top": [[os.path.basename(p)[:-4], v["use_count"], day(v["last_used"])] for p, v in top]}
