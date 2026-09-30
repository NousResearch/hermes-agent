"""Linux system-level probes: host, identity, install_age, locale, shell_prefs, health, usage. Security and
network live in linux_system_security, hardware in linux_system_hardware; both share this module's helpers.
Signal ids match the Windows modules where Linux has a source.

Per-user rule (CONTRACT.md): only the invoking user's home is read. /etc/passwd, /etc/group, wtmp and
/var/crash are system files; from them this module emits counts, or rows for the invoking user only.
L1 probes read files only. Every subprocess sits in an L2 probe behind a gate.
"""
from __future__ import annotations

import collections
import ctypes
import datetime as dt
import glob
import gzip
import os
import platform
import pwd
import re
import grp
import threading
import time

from ..registry import probe

OS = "linux"


def _home():
    """Invoking user's home at call time (the runner points _home() at an overridden l0 home)."""
    return os.path.expanduser("~")

OPERATOR_RX = re.compile(r"(hn-e2e|ns960|ns923|/lhm\b|/shots\b|user-insights-lab|userscan|/\.?hermes-[\w-]+)", re.I)


def lp(id, **kw):
    kw.setdefault("os", OS)
    return probe(id, **kw)


# ---------------------------------------------------------------- helpers

_GUARD = threading.Lock()


def _cached(h, key, fn):
    """Compute once per run and share between probes (L2 probes run on parallel threads)."""
    with _GUARD:
        cache = h.__dict__.setdefault("_lx_cache", {})
        locks = h.__dict__.setdefault("_lx_locks", {})
        lock = locks.setdefault(key, threading.Lock())
    with lock:
        if key not in cache:
            try:
                cache[key] = fn()
            except Exception as e:
                cache[key] = {"__error__": f"{type(e).__name__}: {e}"}
        v = cache[key]
    if isinstance(v, dict) and "__error__" in v:
        raise RuntimeError(v["__error__"])
    return v


def _fv(facts, pid):
    """Value of another probe; the runner passes records ({status, value}), the contract says bare values."""
    r = facts.get(pid)
    if isinstance(r, dict) and "status" in r and "value" in r:
        return r["value"] if r["status"] == "ok" else None
    return r


def _read(path, limit=1 << 20):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def _read1(path):
    t = _read(path, 4096)
    return t.strip() if t is not None else None


def _kv(text):
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _ini(text):
    """KDE/GTK ini. Section headers like [Containments][1][General] are kept verbatim."""
    out, sec = {}, ""
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", ";")):
            continue
        if s.startswith("[") and s.endswith("]"):
            sec = s[1:-1]
            out.setdefault(sec, {})
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            out.setdefault(sec, {})[k.strip()] = v.strip()
    return out


def _iso(ts):
    if ts is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def _parse_iso(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def _days_ago(ts):
    return None if ts is None else round((time.time() - float(ts)) / 86400.0, 1)


def _which(name, extra=()):
    for d in ("/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin", "/snap/bin", *extra):
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _run(h, args, timeout_ms=3000):
    """h.run under env(1) with LC_ALL=C so parsers see stable English output."""
    return h.run(["env", "LC_ALL=C", "LANG=C", *args], timeout_ms=timeout_ms)


class _Statx(ctypes.Structure):
    _fields_ = [("mask", ctypes.c_uint32), ("blksize", ctypes.c_uint32), ("attributes", ctypes.c_uint64),
                ("nlink", ctypes.c_uint32), ("uid", ctypes.c_uint32), ("gid", ctypes.c_uint32),
                ("mode", ctypes.c_uint16), ("_pad1", ctypes.c_uint16), ("ino", ctypes.c_uint64),
                ("size", ctypes.c_uint64), ("blocks", ctypes.c_uint64), ("attributes_mask", ctypes.c_uint64),
                ("atime_s", ctypes.c_int64), ("atime_ns", ctypes.c_uint32), ("_p2", ctypes.c_int32),
                ("btime_s", ctypes.c_int64), ("btime_ns", ctypes.c_uint32), ("_p3", ctypes.c_int32),
                ("_rest", ctypes.c_uint8 * 160)]


_LIBC = None


def _birth(path):
    """File birth time (epoch s) via statx(2); None when the filesystem does not record it."""
    global _LIBC
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL(None, use_errno=True)
        fn = getattr(_LIBC, "statx", None)
        if fn is None:
            return None
        buf = _Statx()
        if fn(-100, os.fsencode(path), 0, 0x800, ctypes.byref(buf)) != 0:
            return None
        if not buf.mask & 0x800 or buf.btime_s <= 0:
            return None
        return int(buf.btime_s)
    except Exception:
        return None


def _mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _me():
    try:
        return pwd.getpwuid(os.getuid())
    except KeyError:
        return None


def _passwd_rows():
    rows = []
    for line in (_read("/etc/passwd") or "").splitlines()[:5000]:
        p = line.split(":")
        if len(p) >= 7:
            try:
                rows.append({"name": p[0], "uid": int(p[2]), "gid": int(p[3]), "home": p[5], "shell": p[6]})
            except ValueError:
                continue
    return rows


_NOLOGIN = ("nologin", "false", "sync", "halt", "shutdown")


def _human(r):
    return 1000 <= r["uid"] < 60000 and not r["shell"].endswith(_NOLOGIN)


def _enabled_units():
    """Unit names enabled through *.wants/ symlinks under /etc/systemd (no subprocess)."""
    out = set()
    for d in glob.glob("/etc/systemd/system/*.wants")[:100]:
        for n in os.listdir(d)[:500] if os.path.isdir(d) else []:
            out.add(n)
    return out


def _dpkg_names(h):
    """Installed package names from /var/lib/dpkg/info/*.list (a directory listing, no file reads)."""
    def load():
        names = set()
        try:
            with os.scandir("/var/lib/dpkg/info") as it:
                for i, e in enumerate(it):
                    if i > 60000:
                        break
                    if e.name.endswith(".list"):
                        names.add(e.name[:-5].split(":")[0])
        except OSError:
            pass
        return names
    return _cached(h, "dpkg_names", load)


def _os_release():
    return _kv(_read("/etc/os-release") or _read("/usr/lib/os-release") or "")


def _bcp47(loc):
    if not loc:
        return None
    base = loc.split(".")[0].split("@")[0]
    if base in ("C", "POSIX"):
        return base
    return base.replace("_", "-")


# ================================================================ host

@lp("os.edition", level="L1", family="host")
def os_edition(h, facts):
    """Distribution name, version and codename from /etc/os-release."""
    r = _os_release()
    if not r:
        return None
    return {"present": True, "marketing_name": r.get("PRETTY_NAME"), "edition_id": r.get("ID"),
            "id_like": r.get("ID_LIKE"), "version_id": r.get("VERSION_ID"), "codename": r.get("VERSION_CODENAME"),
            "variant": r.get("VARIANT_ID"), "lts": "LTS" in (r.get("VERSION") or "")}


@lp("os.build", level="L1", family="host")
def os_build(h, facts):
    """Kernel release and build string (uname)."""
    u = os.uname()
    flavour = u.release.split("-")[-1] if "-" in u.release else None
    return {"present": True, "build": u.release, "kernel": u.release, "kernel_version": u.version,
            "kernel_flavour": flavour, "display_version": _os_release().get("VERSION_ID")}


_ARCH = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64", "i686": "x86", "i386": "x86"}


@lp("host.native_arch", level="L1", family="host")
def native_arch(h, facts):
    """Machine architecture (uname -m), normalised to the Windows spelling."""
    m = os.uname().machine.lower()
    return {"present": True, "arch": _ARCH.get(m, m), "machine": m}


def _elf_machine(path):
    try:
        with open(path, "rb") as f:
            hdr = f.read(20)
        if hdr[:4] != b"\x7fELF":
            return ""
        (em,) = __import__("struct").unpack("<H" if hdr[5] == 1 else ">H", hdr[18:20])
        return {0x3E: "x64", 0xB7: "arm64", 0x03: "x86", 0x28: "arm"}.get(em, hex(em))
    except OSError:
        return ""


@lp("host.python_emulated", level="L1", family="host")
def python_emulated(h, facts):
    """Interpreter ELF machine vs kernel machine (qemu-user/box64 emulation makes timings untrustworthy)."""
    import sys
    py = _elf_machine(os.path.realpath(sys.executable))
    nat = _ARCH.get(os.uname().machine.lower(), os.uname().machine.lower())
    return {"present": True, "python_arch": py, "native_arch": nat, "emulated": bool(py and nat and py != nat),
            "python": platform.python_version()}


_VIRT_VENDORS = (("qemu", "kvm"), ("kvm", "kvm"), ("vmware", "vmware"), ("virtualbox", "virtualbox"),
                 ("innotek", "virtualbox"), ("microsoft corporation", "hyperv"), ("xen", "xen"),
                 ("amazon ec2", "aws"), ("google", "gce"), ("parallels", "parallels"), ("bochs", "bochs"))


@lp("host.virtualization", level="L1", family="host")
def host_virtualization(h, facts):
    """VM / container / WSL detection from DMI vendor, the cpuinfo hypervisor flag and marker files."""
    vend = " ".join(filter(None, (_read1("/sys/class/dmi/id/sys_vendor"), _read1("/sys/class/dmi/id/product_name"),
                                  _read1("/sys/class/dmi/id/bios_vendor")))).lower()
    kind = next((k for w, k in _VIRT_VENDORS if w in vend), None)
    cpu = _read("/proc/cpuinfo", 8192) or ""
    hyper = bool(re.search(r"^flags\s*:.*\bhypervisor\b", cpu, re.M))
    osrel = (_read1("/proc/sys/kernel/osrelease") or "").lower()
    container = None
    if os.path.exists("/.dockerenv"):
        container = "docker"
    elif os.path.exists("/run/.containerenv"):
        container = "podman"
    return {"present": True, "virtual": bool(kind or hyper), "kind": kind or ("unknown" if hyper else None),
            "hypervisor_flag": hyper, "container": container, "wsl": "microsoft" in osrel}


@lp("host.systemd", level="L1", family="host")
def host_systemd(h, facts):
    """systemd is PID 1 (gate for systemctl/journalctl probes)."""
    if not os.path.isdir("/run/systemd/system"):
        return None
    return {"present": True, "journal_persistent": os.path.isdir("/var/log/journal"),
            "enabled_unit_links": len(_enabled_units())}


# ================================================================ identity

@lp("host.name", level="L1", family="identity")
def host_name(h, facts):
    """Hostname. local_only: identifies the machine, never send off-host."""
    n = os.uname().nodename
    static = _read1("/etc/hostname")
    return {"present": bool(n), "name": n, "local_only": True,
            "pending_rename": bool(static and n and static.lower() != n.lower())}


@lp("host.name_default", level="L1", family="identity")
def host_name_default(h, facts):
    """Hostname still an installer/cloud default (ubuntu, localhost, ip-10-..., a MAC-like id)."""
    n = os.uname().nodename.lower()
    default = bool(re.fullmatch(r"(ubuntu|debian|localhost|fedora|archlinux|raspberrypi|ubuntu-server|"
                                r"ip-\d+-\d+-\d+-\d+.*|[0-9a-f]{12})", n))
    return {"present": True, "default": default}


@lp("acct.account_type", level="L1", family="identity")
def account_type(h, facts):
    """Local vs directory account: invoking user in /etc/passwd or served by sssd/ldap/winbind (nsswitch)."""
    me = _me()
    if me is None:
        return None
    local = any(r["uid"] == me.pw_uid for r in _passwd_rows())
    nss = _kv("\n".join(l.replace(":", "=", 1) for l in (_read("/etc/nsswitch.conf") or "").splitlines()))
    pw_src = nss.get("passwd", "")
    kind = "local" if local else ("sssd" if "sss" in pw_src else "ldap" if "ldap" in pw_src else
                                  "winbind" if "winbind" in pw_src else "directory")
    return {"present": True, "type": kind, "uid": me.pw_uid, "nss_passwd": pw_src.split(),
            "sssd_configured": os.path.exists("/etc/sssd/sssd.conf"), "realm_joined": os.path.exists("/etc/krb5.keytab")}


@lp("profile.count", level="L1", family="identity")
def profile_count(h, facts):
    """Human accounts (uid >= 1000 with a login shell) from /etc/passwd; home dirs are never opened."""
    rows = _passwd_rows()
    humans = [r for r in rows if _human(r)]
    real = [r for r in humans if not OPERATOR_RX.search(r["home"])]
    me = _me()
    return {"present": True, "nonsystem": len(real), "operator_excluded": len(humans) - len(real),
            "passwd_rows": len(rows), "home_dirs": sum(1 for r in real if r["home"].startswith("/home/")),
            "current_found": bool(me and any(r["uid"] == me.pw_uid for r in rows))}


@lp("profile.username_default", level="L1", family="identity")
def profile_username_default(h, facts):
    """Account name is an image default (ubuntu, pi, user, admin, ec2-user)."""
    me = _me()
    if me is None:
        return None
    return {"present": True, "default": me.pw_name.lower() in ("ubuntu", "pi", "user", "admin", "ec2-user", "debian",
                                                               "fedora", "vagrant", "live")}


@lp("acct.local_users", level="L2", family="identity", tier="T1", collect="extended", gate="profile.count")
def local_users(h, facts):
    """Account counts by kind from /etc/passwd plus live-process owners (ps): names never emitted."""
    rows = _passwd_rows()
    me = _me()
    kinds = collections.Counter()
    for r in rows:
        if me and r["uid"] == me.pw_uid:
            kinds["current"] += 1
        elif r["uid"] == 0:
            kinds["root"] += 1
        elif _human(r):
            kinds["other_human"] += 1
        elif r["uid"] < 1000:
            kinds["system"] += 1
        else:
            kinds["other"] += 1
    procs = collections.Counter()
    try:
        for pid in os.listdir("/proc")[:20000]:
            if pid.isdigit():
                try:
                    procs[os.stat("/proc/" + pid).st_uid] += 1
                except OSError:
                    pass
    except OSError:
        pass
    humans_uids = {r["uid"] for r in rows if _human(r)}
    active_humans = sum(1 for u in procs if u in humans_uids and not (me and u == me.pw_uid))
    return {"present": True, "total": len(rows), "kinds": dict(kinds),
            "other_humans_with_processes": active_humans, "uids_with_processes": len(procs)}


# ================================================================ install_age

def _install_candidates():
    c = {}
    for label, p in (("installer_dir", "/var/log/installer"), ("root_fs", "/"), ("lost_found", "/lost+found"),
                     ("machine_id", "/etc/machine-id")):
        b = _birth(p)
        if b:
            c[label] = b
    if "installer_dir" not in c and os.path.isdir("/var/log/installer"):
        m = [x for x in (_mtime(os.path.join("/var/log/installer", n)) for n in os.listdir("/var/log/installer")[:50]) if x]
        if m:
            c["installer_files_mtime"] = min(m)
    return c


@lp("os.install_date", level="L1", family="install_age")
def os_install_date(h, facts):
    """OS install date: birth of /var/log/installer, else the root filesystem birth."""
    c = _install_candidates()
    for k in ("installer_dir", "installer_files_mtime", "root_fs", "lost_found", "machine_id"):
        if k in c:
            return {"present": True, "install_date": _iso(c[k]), "days_ago": _days_ago(c[k]), "source": k,
                    "candidates": {a: _iso(b) for a, b in c.items()}}
    return None


@lp("profile.created", level="L1", family="install_age")
def profile_created(h, facts):
    """Birth of the invoking user's home and ~/.config: user-tenure anchor."""
    folder = _birth(_home())
    cfg = _birth(os.path.join(_home(), ".config"))
    first = min([t for t in (folder, cfg) if t], default=None)
    if first is None:
        return None
    return {"present": True, "created": _iso(first), "folder": _iso(folder), "config_dir": _iso(cfg),
            "days_ago": _days_ago(first)}


def _dpkg_logs():
    """dpkg.log files oldest first."""
    fs = glob.glob("/var/log/dpkg.log*")

    def order(p):
        m = re.search(r"\.(\d+)(\.gz)?$", p)
        return -int(m.group(1)) if m else 0
    return sorted(fs, key=order)


def _first_line(path):
    try:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.readline().strip()
    except (OSError, EOFError):
        return None


@lp("setup.image_date", level="L1", family="install_age")
def setup_image_date(h, facts):
    """Install media build stamp (media-info) and oldest dpkg.log line; image_date only for cloned/OEM images."""
    media = _read1("/var/log/installer/media-info")
    build = None
    if media:
        m = re.search(r"\((\d{8})(?:\.\d+)?\)", media)
        if m:
            build = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
    logs = _dpkg_logs()
    first = _first_line(logs[0]) if logs else None
    dpkg_first = first[:19] if first and re.match(r"\d{4}-\d\d-\d\d", first) else None
    if not build and not dpkg_first:
        return None
    cloned = os.path.isdir("/var/lib/cloud/instance") and not os.path.isdir("/var/log/installer") \
        or os.path.exists("/var/lib/oem-config")
    # An installer-media build date predates every install by months; only a cloned/OEM image carries its build
    # date into the install lineage, so image_date is set for those alone.
    return {"present": True, "image_date": (build or (dpkg_first or "")[:10]) if cloned else None,
            "media_build": build, "dpkg_log_first": dpkg_first, "installer_media": media, "cloned_image": cloned}


@lp("setup.oem", level="L1", family="install_age")
def setup_oem(h, facts):
    """Install method: OEM preload (oem-config) vs subiquity/curtin/ubiquity/autoinstall/cloud image."""
    d = "/var/log/installer"
    names = set(os.listdir(d)[:200]) if os.path.isdir(d) else set()
    oem = any(os.path.exists(p) for p in ("/var/lib/oem-config", "/var/log/oem-config.log", "/usr/sbin/oem-config-firstboot"))
    if "autoinstall-user-data" in names:
        method = "subiquity-autoinstall"
    elif "curtin-install.log" in names or "subiquity-server-debug.log" in names:
        method = "subiquity"
    elif "syslog" in names and "casper.log" in names:
        method = "ubiquity"
    elif os.path.isdir("/var/lib/cloud/instance") and not names:
        method = "cloud-image"
    elif names:
        method = "installer"
    else:
        method = "unknown"
    return {"present": True, "oem_image": oem, "install_method": method,
            "cloud_init": os.path.isdir("/var/lib/cloud/instances")}


@lp("setup.oobe_done", level="L1", family="install_age")
def setup_oobe_done(h, facts):
    """First-login setup finished: GNOME ~/.config/gnome-initial-setup-done birth (KDE has no marker)."""
    p = os.path.join(_home(), ".config/gnome-initial-setup-done")
    t = _birth(p) or _mtime(p)
    if not t:
        return None
    return {"present": True, "date": _iso(t), "source": "gnome-initial-setup-done"}


@lp("setup.source_os_lineage", level="L1", family="install_age")
def setup_source_os_lineage(h, facts):
    """Release upgrades since install: /var/log/dist-upgrade runs and installer release vs current release."""
    runs, oldest = 0, None
    du = "/var/log/dist-upgrade"
    if os.path.isdir(du):
        for n in os.listdir(du)[:200]:
            p = os.path.join(du, n)
            if n == "main.log" or (os.path.isdir(p) and os.path.exists(os.path.join(p, "main.log"))):
                runs += 1
                m = _mtime(os.path.join(p, "main.log") if os.path.isdir(p) else p)
                oldest = m if oldest is None or (m and m < oldest) else oldest
    media = _read1("/var/log/installer/media-info") or ""
    mv = re.search(r"\b(\d\d\.\d\d)", media)
    cur = _os_release().get("VERSION_ID")
    installed = mv.group(1) if mv else None
    upgraded = bool(installed and cur and installed != cur)
    return {"present": True, "count": runs + (1 if upgraded and not runs else 0), "dist_upgrade_runs": runs,
            "installed_release": installed, "current_release": cur, "oldest": _iso(oldest)}


def _xbel_count(path, cap=8 << 20):
    try:
        with open(path, "rb") as f:
            return f.read(cap).count(b"<bookmark ")
    except OSError:
        return None


@lp("age.footprint_counts", level="L1", family="install_age")
def footprint_counts(h, facts):
    """Lived-in counts: dpkg packages, snaps, flatpaks, user .desktop entries, recent-files entries, dot-dirs."""
    snaps = [n for n in (os.listdir("/snap") if os.path.isdir("/snap") else []) if n not in ("bin", "README")]
    fl_sys = h.count_dir("/var/lib/flatpak/app", 5000)
    fl_user = h.count_dir(os.path.join(_home(), ".local/share/flatpak/app"), 5000)
    dots = [n for n in h.list_dir(_home(), 2000) if n.startswith(".")]
    return {"present": True, "dpkg_packages": len(_dpkg_names(h)), "snaps": len(snaps),
            "flatpaks": max(fl_sys, 0) + max(fl_user, 0),
            "user_desktop_entries": max(h.count_dir(os.path.join(_home(), ".local/share/applications"), 5000), 0),
            "recent": _xbel_count(os.path.join(_home(), ".local/share/recently-used.xbel")),
            "home_dot_entries": len(dots)}


def _nm_profiles(h):
    """NetworkManager profiles via nmcli: (type, autoconnect, last-used epoch). Names are never requested."""
    def load():
        if not _which("nmcli"):
            return None
        out = _run(h, ["nmcli", "-t", "-f", "TYPE,AUTOCONNECT,TIMESTAMP", "connection", "show"], 3000)
        if out is None:
            return None
        rows = []
        for line in out.splitlines()[:2000]:
            p = line.split(":")
            if len(p) >= 3:
                try:
                    ts = int(p[2])
                except ValueError:
                    ts = 0
                rows.append({"type": p[0], "autoconnect": p[1] == "yes", "last": ts or None})
        return rows
    return _cached(h, "nm_profiles", load)


def _nm_kind(t):
    t = t.lower()
    if "wireless" in t or t == "wifi":
        return "wireless"
    if "ethernet" in t:
        return "wired"
    if t in ("vpn", "wireguard"):
        return t
    if t in ("bridge", "loopback", "tun", "bond", "vlan"):
        return "virtual"
    if "gsm" in t or "cdma" in t:
        return "mobile"
    return t or "other"


@lp("net.history", level="L2", family="install_age", tier="T1", gate="os.install_date")
def net_history(h, facts):
    """NetworkManager profile counts by type, first-created (profile file birth) and last-used dates. No names."""
    rows = _nm_profiles(h) or []
    d = "/etc/NetworkManager/system-connections"
    births = [b for b in (_birth(os.path.join(d, n)) for n in h.list_dir(d, 1000)) if b]
    if not rows and not births:
        return None
    by_type = collections.Counter(_nm_kind(r["type"]) for r in rows)
    last = [r["last"] for r in rows if r["last"]]
    return {"present": True, "networks": len(rows), "wireless": by_type.get("wireless", 0), "by_type": dict(by_type),
            "first_created": _iso(min(births)) if births else None, "last_created": _iso(max(births)) if births else None,
            "last_connected": _iso(max(last)) if last else None}


@lp("setup.system_log_oldest", level="L2", family="install_age", gate="os.install_date", collect="extended")
def setup_system_log_oldest(h, facts):
    """Oldest system record: first journal boot, wtmp start and oldest dpkg.log line after the image build."""
    out = {}
    if _which("journalctl"):
        txt = _run(h, ["journalctl", "--list-boots", "-q", "--no-pager", "-n", "1000"], 4000) or ""
        lines = [l for l in txt.splitlines() if l.strip()]
        if lines:
            m = re.search(r"(\w{3} \d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", lines[0])
            out["journal_first_boot"] = m.group(1)[4:] if m else None
            out["journal_boots"] = len(lines)
    w = _wtmp(h)
    if w and w.get("begins"):
        out["wtmp_begins"] = w["begins"]
    dates = [v for k, v in out.items() if k in ("journal_first_boot", "wtmp_begins") and v]
    if not out:
        return None
    out["present"] = True
    out["oldest"] = min(dates)[:19] if dates else None
    return out


def _dpkg_log_lines(max_bytes=12 << 20):
    total = 0
    for p in _dpkg_logs():
        try:
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    total += len(line)
                    if total > max_bytes:
                        return
                    yield line
        except (OSError, EOFError):
            continue


def _apt_history(max_bytes=8 << 20):
    """(month, requested_by_uid, action) per apt run from /var/log/apt/history.log*. Package names never kept."""
    runs, total = [], 0
    me = os.getuid()
    for p in sorted(glob.glob("/var/log/apt/history.log*"), reverse=True):
        try:
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8", errors="replace") as f:
                cur = {}
                for line in f:
                    total += len(line)
                    if total > max_bytes:
                        break
                    if line.startswith("Start-Date:"):
                        cur = {"month": line[12:19], "by_me": False, "by_other": False, "action": None}
                        runs.append(cur)
                    elif line.startswith("Requested-By:"):
                        m = re.search(r"\((\d+)\)", line)
                        if m:
                            cur["by_me" if int(m.group(1)) == me else "by_other"] = True
                    elif line.startswith("Commandline:"):
                        m = re.search(r"\b(install|remove|purge|upgrade|full-upgrade|dist-upgrade|autoremove)\b", line)
                        cur["action"] = m.group(1) if m else "other"
                    elif not cur.get("action") and line.startswith(("Install:", "Upgrade:", "Remove:")):
                        cur["action"] = line.split(":", 1)[0].lower()
        except (OSError, EOFError):
            continue
    return runs


@lp("setup.dpkg_history", level="L2", family="install_age", tier="T1", gate="os.install_date", collect="extended")
def setup_dpkg_history(h, facts):
    """Package install/upgrade/remove counts per month from dpkg.log* and apt runs by requester (me vs other)."""
    by_month = collections.defaultdict(collections.Counter)
    first = last = None
    for line in _dpkg_log_lines():
        m = re.match(r"(\d{4}-\d\d)-\d\d \d\d:\d\d:\d\d (install|upgrade|remove|purge) ", line)
        if not m:
            continue
        by_month[m.group(1)][m.group(2)] += 1
        ts = line[:19]
        first = ts if first is None or ts < first else first
        last = ts if last is None or ts > last else last
    apt = _apt_history()
    if not by_month and not apt:
        return None
    apt_month = collections.defaultdict(collections.Counter)
    for r in apt:
        apt_month[r["month"]][r["action"] or "other"] += 1
    cand = _install_candidates()
    inst = cand.get("installer_dir") or cand.get("root_fs")
    inst_month = time.strftime("%Y-%m", time.localtime(inst)) if inst else ""
    installs_months = sorted(k for k, c in by_month.items() if c.get("install") and k >= inst_month)
    return {"present": True, "first": first, "last": last,
            "image_build_entries": sum(sum(c.values()) for k, c in by_month.items() if inst_month and k < inst_month),
            "dpkg_by_month": {k: dict(v) for k, v in sorted(by_month.items())},
            "installs_since_os_install": sum(c.get("install", 0) for k, c in by_month.items() if k >= inst_month),
            "months_spread": len(installs_months),
            "apt_runs": len(apt), "apt_runs_by_me": sum(1 for r in apt if r["by_me"]),
            "apt_runs_by_other_user": sum(1 for r in apt if r["by_other"]),
            "apt_runs_unattended": sum(1 for r in apt if not r["by_me"] and not r["by_other"]),
            "apt_by_month": {k: dict(v) for k, v in sorted(apt_month.items())}}


# ================================================================ locale

def _plasma_locale():
    return _ini(_read(os.path.join(_home(), ".config/plasma-localerc")) or "")


def _user_locale():
    pl = _plasma_locale()
    cfg = (pl.get("Formats") or {}).get("LANG")
    sysd = _kv(_read("/etc/default/locale") or _read("/etc/locale.conf") or "")
    env = os.environ.get("LC_ALL") or os.environ.get("LC_TIME") or os.environ.get("LANG")
    chosen = cfg or sysd.get("LC_TIME") or sysd.get("LANG") or env
    return chosen, {"plasma_formats": cfg, "system_default": sysd.get("LANG"), "env": env}


_IMPERIAL = ("US", "LR", "MM")


@lp("locale.user", level="L1", family="locale")
def locale_user(h, facts):
    """User format locale (Plasma Formats, then /etc/default/locale, then env) with date/time formats."""
    chosen, src = _user_locale()
    if not chosen:
        return None
    d = {"present": True, "locale": _bcp47(chosen), "LocaleName": _bcp47(chosen), "raw": chosen, "sources": src}
    # A child reads the formats: setlocale() is process-wide and this can run inside the Hermes
    # backend. `locale` falls back to C formats for a locale that is not installed, so check first.
    norm = lambda s: s.strip().lower().replace("-", "")
    if norm(chosen) in {norm(n) for n in (h.run(["locale", "-a"]) or "").splitlines()}:
        fmts = (h.run(["locale", "d_fmt", "t_fmt"], env={"LC_ALL": chosen}) or "").splitlines()
        if len(fmts) == 2:
            d["short_date"], d["time_format"] = fmts
    tf = d.get("time_format") or ""
    d["clock_24h"] = ("%H" in tf or "%T" in tf) if tf else None
    country = (d["locale"] or "").split("-")[-1]
    d["metric"] = country not in _IMPERIAL if "-" in (d["locale"] or "") else None
    d["utf8"] = "utf-8" in chosen.lower() or "utf8" in chosen.lower()
    return d


@lp("locale.system", level="L1", family="locale")
def locale_system(h, facts):
    """System default locale (/etc/default/locale) and generated locales."""
    sysd = _kv(_read("/etc/default/locale") or _read("/etc/locale.conf") or "")
    gen = [l.split()[0] for l in (_read("/etc/locale.gen") or "").splitlines() if l.strip() and not l.startswith("#")]
    supported = _read("/var/lib/locales/supported.d/en") or ""
    if not sysd and not gen:
        return None
    return {"present": True, "lang": sysd.get("LANG"), "locale": _bcp47(sysd.get("LANG")),
            "overrides": sorted(k for k in sysd if k.startswith("LC_")), "generated": len(gen) or None,
            "supported_d_lines": len(supported.splitlines()) or None}


@lp("lang.ui", level="L1", family="locale")
def lang_ui(h, facts):
    """UI language: LANGUAGE / Plasma Translations / LANG; installed language packs from dpkg."""
    pl = _plasma_locale()
    tr = (pl.get("Translations") or {}).get("LANGUAGE")
    env_lang = os.environ.get("LANGUAGE")
    chosen = (tr or env_lang or "").split(":")[0] or _user_locale()[0]
    if not chosen:
        return None
    packs = sorted({n.replace("language-pack-", "").split("-base")[0] for n in _dpkg_names(h)
                    if n.startswith("language-pack-") and not n.startswith("language-pack-gnome")
                    and not n.startswith("language-pack-kde")})
    return {"present": True, "language": _bcp47(chosen), "user": _bcp47(chosen),
            "system": _bcp47(_kv(_read("/etc/default/locale") or "").get("LANG")), "packs": packs,
            "source": "plasma" if tr else "LANGUAGE" if env_lang else "LANG"}


@lp("lang.user_list", level="L1", family="locale")
def lang_user_list(h, facts):
    """Ordered UI language preference list (LANGUAGE or Plasma Translations)."""
    pl = _plasma_locale()
    raw = (pl.get("Translations") or {}).get("LANGUAGE") or os.environ.get("LANGUAGE") or ""
    langs = [_bcp47(x) for x in raw.split(":") if x]
    if not langs:
        return None
    return {"present": True, "languages": langs, "count": len(langs)}


def _xkb():
    sysk = _kv(_read("/etc/default/keyboard") or "")
    kx = _ini(_read(os.path.join(_home(), ".config/kxkbrc")) or "").get("Layout") or {}
    return sysk, kx


@lp("kbd.layouts", level="L1", family="locale")
def kbd_layouts(h, facts):
    """Keyboard layouts: Plasma kxkbrc LayoutList, else /etc/default/keyboard XKBLAYOUT; IME frameworks."""
    sysk, kx = _xkb()
    user = [x for x in (kx.get("LayoutList") or "").split(",") if x] if kx.get("Use", "true") != "false" else []
    system = [x for x in (sysk.get("XKBLAYOUT") or "").split(",") if x]
    variants = [x for x in (kx.get("VariantList") or sysk.get("XKBVARIANT") or "").split(",")]
    lays = user or system
    if not lays:
        return None
    out = [{"klid": l, "name": l + (f"({variants[i]})" if i < len(variants) and variants[i] else "")}
           for i, l in enumerate(lays)]
    ime = [n for n, p in (("fcitx5", "/usr/bin/fcitx5"), ("ibus", "/usr/bin/ibus-daemon"), ("fcitx", "/usr/bin/fcitx"))
           if os.path.exists(p)]
    return {"present": True, "layouts": out, "count": len(out), "source": "kxkbrc" if user else "etc_default_keyboard",
            "model": kx.get("Model") or sysk.get("XKBMODEL"), "ime": bool(ime), "ime_frameworks": ime}


@lp("kbd.scancode_map", level="L1", family="locale")
def kbd_scancode_map(h, facts):
    """Key remaps: XKB options (ctrl:nocaps, caps:escape, ...) in kxkbrc or /etc/default/keyboard, keyd/interception."""
    sysk, kx = _xkb()
    opts = [o for o in ((kx.get("Options") or "") + "," + (sysk.get("XKBOPTIONS") or "")).split(",") if o]
    opts = sorted(set(o for o in opts if not o.startswith("grp_led")))
    tools = [n for n, p in (("keyd", "/etc/keyd"), ("interception", "/etc/interception"), ("kmonad", "/usr/bin/kmonad"),
                            ("xremap", "/usr/bin/xremap"), ("hwdb", "/etc/udev/hwdb.d")) if os.path.exists(p)
             and (n != "hwdb" or any(f.endswith(".hwdb") for f in os.listdir(p)))]
    remapped = bool([o for o in opts if not o.startswith(("grp:", "terminate:"))] or tools)
    return {"present": remapped, "remapped": remapped, "options": opts, "tools": tools}


@lp("tz.zone", level="L1", family="locale")
def tz_zone(h, facts):
    """Time zone from /etc/localtime (or /etc/timezone) and the current UTC offset."""
    z = None
    try:
        tgt = os.readlink("/etc/localtime")
        m = re.search(r"zoneinfo/(.+)$", tgt)
        z = m.group(1) if m else None
    except OSError:
        pass
    z = z or _read1("/etc/timezone") or os.environ.get("TZ")
    off = time.localtime().tm_gmtoff // 60
    if not z:
        return None
    return {"present": True, "zone": z, "utc_offset_min": off, "bias_min": -off,
            "abbrev": time.strftime("%Z"), "rtc_local": "LOCAL" in (_read("/etc/adjtime") or "")}


@lp("tz.auto", level="L1", family="locale")
def tz_auto(h, facts):
    """Automatic time: NTP service enabled (timesyncd/chrony/ntpd) and Plasma ktimezoned; no auto time zone source."""
    en = _enabled_units()
    svc = [s for s in ("systemd-timesyncd.service", "chrony.service", "chronyd.service", "ntp.service", "ntpsec.service")
           if s in en]
    return {"present": True, "ntp": bool(svc), "ntp_services": [s.split(".")[0] for s in svc],
            "auto_time_zone": None, "geoclue": os.path.exists("/usr/libexec/geoclue")}


# ================================================================ shell_prefs

def _kdeglobals():
    return _ini(_read(os.path.join(_home(), ".config/kdeglobals")) or "")


def _appletsrc():
    return _ini(_read(os.path.join(_home(), ".config/plasma-org.kde.plasma.desktop-appletsrc"), 2 << 20) or "")


@lp("theme.dark", level="L1", family="shell_prefs")
def theme_dark(h, facts):
    """Dark mode: KDE colour scheme / look-and-feel, GTK prefer-dark, GNOME color-scheme (dconf not parsed)."""
    kg = _kdeglobals()
    scheme = (kg.get("General") or {}).get("ColorScheme") or ""
    laf = (kg.get("KDE") or {}).get("LookAndFeelPackage") or ""
    gtk = _ini(_read(os.path.join(_home(), ".config/gtk-3.0/settings.ini")) or "").get("Settings") or {}
    gtk_dark = gtk.get("gtk-application-prefer-dark-theme") in ("1", "true")
    gtk_theme = gtk.get("gtk-theme-name") or ""
    kde = bool(kg)
    if not kde and not gtk:
        return None
    dark = "dark" in (scheme + laf).lower() or gtk_dark or "dark" in gtk_theme.lower()
    return {"present": True, "apps_dark": dark, "system_dark": dark, "kde_color_scheme": scheme or ("Breeze (default)" if kde else None),
            "kde_look_and_feel": laf or None, "gtk_prefer_dark": gtk_dark, "gtk_theme": gtk_theme or None,
            "icon_theme": (kg.get("Icons") or {}).get("Theme") or gtk.get("gtk-icon-theme-name")}


@lp("theme.wallpaper", level="L1", family="shell_prefs")
def theme_wallpaper(h, facts):
    """Plasma desktop wallpaper: stock (/usr/share/wallpapers or unset) vs custom. Path never emitted."""
    ap = _appletsrc()
    imgs = [v.get("Image") for s, v in ap.items() if s.endswith("[Wallpaper][org.kde.image][General]") and v.get("Image")]
    plugins = [v.get("wallpaperplugin") for s, v in ap.items() if v.get("wallpaperplugin")]
    if not ap:
        return None
    stock = all(i.replace("file://", "").startswith(("/usr/share/wallpapers", "/usr/share/backgrounds")) for i in imgs)
    return {"present": True, "default": stock and all(p in (None, "org.kde.image") for p in plugins),
            "stock": stock, "custom_images": sum(1 for i in imgs if not i.replace("file://", "").startswith("/usr/share")),
            "plugins": sorted(set(p for p in plugins if p))}


@lp("pins.taskbar", level="L1", family="shell_prefs")
def pins_taskbar(h, facts):
    """Plasma task manager pinned launchers (application ids)."""
    ap = _appletsrc()
    pins = []
    for s, v in ap.items():
        if "launchers" in v and v["launchers"]:
            pins += [x.replace("applications:", "").replace("preferred://", "pref:").replace(".desktop", "")
                     for x in v["launchers"].split(",") if x]
    if not ap:
        return None
    return {"present": True, "count": len(pins), "pins": pins[:30]}


@lp("shell.explorer_prefs", level="L1", family="shell_prefs")
def explorer_prefs(h, facts):
    """File manager and dialog prefs: hidden files shown (kdeglobals/dolphinrc), single-click, Dolphin customised."""
    kg = _kdeglobals()
    dr = _ini(_read(os.path.join(_home(), ".config/dolphinrc")) or "")
    fd = kg.get("KFileDialog Settings") or {}
    kde = kg.get("KDE") or {}
    if not kg and not dr:
        return None
    return {"present": True, "show_hidden_dialog": fd.get("Show hidden files") == "true",
            "single_click": kde.get("SingleClick", "true" if kg else None),
            "dolphin_sections": len(dr), "dolphin_show_full_path": (dr.get("General") or {}).get("ShowFullPath")}


@lp("shell.login_shell", level="L1", family="shell_prefs")
def login_shell(h, facts):
    """Login shell from /etc/passwd for the invoking user; zsh/fish framework and prompt presence."""
    me = _me()
    if me is None:
        return None
    fw = [n for n, p in (("oh-my-zsh", ".oh-my-zsh"), ("powerlevel10k", ".p10k.zsh"), ("prezto", ".zprezto"),
                         ("zinit", ".local/share/zinit"), ("starship", ".config/starship.toml"),
                         ("oh-my-bash", ".oh-my-bash"), ("fish", ".config/fish"), ("tmux", ".tmux.conf"))
          if os.path.exists(os.path.join(_home(), p))]
    return {"present": True, "shell": os.path.basename(me.pw_shell), "frameworks": fw,
            "changed_from_default": os.path.basename(me.pw_shell) not in ("bash", "sh")}


@lp("shell.desktop_env", level="L1", family="shell_prefs")
def desktop_env(h, facts):
    """Desktop environment: installed X11/Wayland sessions, display manager, current session type env."""
    xs = [n[:-8] for n in h.list_dir("/usr/share/xsessions", 50) if n.endswith(".desktop")]
    ws = [n[:-8] for n in h.list_dir("/usr/share/wayland-sessions", 50) if n.endswith(".desktop")]
    dm = os.path.basename(_read1("/etc/X11/default-display-manager") or "") or None
    if not xs and not ws and not dm:
        return {"present": False, "headless": True}
    return {"present": True, "x11_sessions": xs, "wayland_sessions": ws, "display_manager": dm,
            "current_desktop": os.environ.get("XDG_CURRENT_DESKTOP"), "session_type": os.environ.get("XDG_SESSION_TYPE"),
            "plasma": os.path.exists("/usr/bin/plasmashell"), "gnome": os.path.exists("/usr/bin/gnome-shell")}


# ================================================================ health

def _crash_app(mangled):
    """apport names encode the executable path with '/' -> '_': _usr_bin_kwin_x11 -> kwin_x11."""
    for sep in ("_libexec_", "_sbin_", "_bin_", "_games_"):
        if sep in mangled:
            return mangled.rsplit(sep, 1)[1]
    return mangled.rsplit("_", 1)[-1]


def _crash_files():
    rows = []
    for n in os.listdir("/var/crash")[:2000] if os.path.isdir("/var/crash") else []:
        p = os.path.join("/var/crash", n)
        m = re.match(r"(.+)\.(\d+)\.crash$", n)
        if m:
            rows.append({"app": _crash_app(m.group(1)), "uid": int(m.group(2)), "mtime": _mtime(p)})
    return rows


def _kdump_dirs():
    out = []
    for n in os.listdir("/var/crash")[:2000] if os.path.isdir("/var/crash") else []:
        if re.match(r"\d{12}$", n) and os.path.isdir(os.path.join("/var/crash", n)):
            out.append(_mtime(os.path.join("/var/crash", n)))
    return out


def _coredumps():
    rows = []
    for n in os.listdir("/var/lib/systemd/coredump")[:5000] if os.path.isdir("/var/lib/systemd/coredump") else []:
        m = re.match(r"core\.(.+?)\.(\d+)\.[0-9a-f]+\.\d+\.(\d+)", n)
        if m:
            rows.append({"app": m.group(1), "uid": int(m.group(2)), "mtime": int(m.group(3)) / 1e6})
    return rows


@lp("health.journal", level="L1", family="health")
def health_journal(h, facts):
    """System journal readable by the invoking user (root, adm or systemd-journal group): gate for journal probes."""
    if not _which("journalctl"):
        return None
    try:
        groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    except KeyError:
        groups = set()
    system = os.geteuid() == 0 or bool(groups & {"adm", "systemd-journal", "wheel"})
    return {"present": True, "system_journal_readable": system, "persistent": os.path.isdir("/var/log/journal")}


@lp("health.dumps", level="L1", family="health", tier="T1")
def health_dumps(h, facts):
    """Crash records: kernel dumps (kdump dirs, pstore), apport .crash and systemd-coredump files; counts and dates."""
    now = time.time()
    me = os.getuid()
    cr = _crash_files()
    cd = _coredumps()
    kd = [m for m in _kdump_dirs() if m]
    pstore = len(h.list_dir("/var/lib/systemd/pstore", 500))
    apps = cr + cd
    if not apps and not kd and not os.path.isdir("/var/crash"):
        return None
    recent = [r for r in apps if r["mtime"] and now - r["mtime"] <= 30 * 86400]
    return {"present": True, "count_30d": sum(1 for m in kd if now - m <= 30 * 86400), "kernel_dumps": len(kd),
            "pstore_records": pstore, "app_crash_files": len(cr), "coredumps": len(cd),
            "app_crashes_30d": len(recent), "mine_30d": sum(1 for r in recent if r["uid"] == me),
            "other_users_30d": sum(1 for r in recent if r["uid"] != me),
            "newest": _iso(max([r["mtime"] for r in apps if r["mtime"]] + kd, default=None)),
            "apport_enabled": _kv(_read("/etc/default/apport") or "").get("enabled") == "1"}


@lp("health.wer", level="L2", family="health", tier="T1", gate="health.dumps", collect="extended")
def health_wer(h, facts):
    """Per-app crashes for the invoking user (apport + systemd-coredump file names); other users counted only."""
    me = os.getuid()
    rows = _crash_files() + _coredumps()
    mine = [r for r in rows if r["uid"] == me]
    by = collections.Counter(r["app"] for r in mine)
    last = {}
    for r in mine:
        last[r["app"]] = max(last.get(r["app"]) or 0, r["mtime"] or 0)
    top = [[a, n, _iso(last.get(a))] for a, n in by.most_common(10)]
    return {"present": bool(rows), "total": len(rows), "mine": len(mine), "other_users": len(rows) - len(mine),
            "crashes_by_app": top, "max_repeat": top[0][1] if top else 0}


_HW_ERR = [("mce", r"mce: \[Hardware Error\]|Machine check"), ("nvidia_xid", r"NVRM: Xid"),
           ("amdgpu", r"amdgpu.*(ring .* timeout|GPU reset|page fault)"), ("i915", r"i915.*(GPU HANG|reset)"),
           ("pcie_aer", r"AER: .*error"), ("edac", r"EDAC .*(CE|UE)"), ("disk_io", r"I/O error, dev|blk_update_request"),
           ("oops_panic", r"Oops|Kernel panic|BUG: |general protection fault"), ("oom_kill", r"Out of memory: Killed process")]


def _kernel_errs(h):
    def load():
        if not _which("journalctl"):
            return None
        pat = "|".join(p for _, p in _HW_ERR)
        out = _run(h, ["journalctl", "-k", "-q", "--no-pager", "-S", "-30d", "-n", "500", "-o", "short-unix",
                       "-g", pat], 5000)
        if out is None:
            return None
        c = collections.Counter()
        for line in out.splitlines():
            for k, p in _HW_ERR:
                if re.search(p, line):
                    c[k] += 1
                    break
        return c
    return _cached(h, "kernel_errs", load)


@lp("health.whea_gpu", level="L2", family="health", tier="T1", gate="health.journal", collect="extended")
def health_whea_gpu(h, facts):
    """Kernel hardware/GPU errors in 30 d (journalctl -k): MCE, NVIDIA Xid, amdgpu/i915 resets, PCIe AER, EDAC."""
    c = _kernel_errs(h)
    if c is None:
        return None
    hw = sum(c.get(k, 0) for k in ("mce", "nvidia_xid", "amdgpu", "i915", "pcie_aer", "edac"))
    return {"present": True, "count_30d": hw, "by_class_30d": dict(c), "window": "30d", "cap": 500}


@lp("health.journal_errors", level="L2", family="health", tier="T1", gate="health.journal", collect="extended")
def journal_errors(h, facts):
    """System journal in 30 d: error-priority count (capped), top 5 identifiers, OOM kills, kernel oops, failed units."""
    idout = _run(h, ["journalctl", "-q", "--no-pager", "-S", "-30d", "-p", "3", "-n", "5000", "-o", "short"], 5000)
    if idout is None:
        return None
    ids = collections.Counter()
    for line in (idout or "").splitlines():
        m = re.match(r"\S+\s+\d+\s+[\d:]+\s+\S+\s+([^\[:\s]+)", line) or re.match(r"\S+\s+\S+\s+([^\[:\s]+)", line)
        if m:
            ids[m.group(1)] += 1
    c = _kernel_errs(h) or {}
    failed = _run(h, ["systemctl", "list-units", "--no-legend", "--no-pager", "--state=failed", "--plain"], 3000) or ""
    n = sum(ids.values())
    return {"present": True, "errors_30d": n, "capped": n >= 5000, "top_identifiers": ids.most_common(5),
            "oom_kills_30d": c.get("oom_kill", 0), "kernel_oops_30d": c.get("oops_panic", 0),
            "disk_io_errors_30d": c.get("disk_io", 0), "failed_units": sum(1 for l in failed.splitlines() if l.strip())}


# ================================================================ usage

@lp("boot.uptime", level="L1", family="usage")
def boot_uptime(h, facts):
    """Uptime from /proc/uptime and boot time; sleep-vs-shutdown habit input."""
    t = (_read1("/proc/uptime") or "").split()
    if not t:
        return None
    up = float(t[0])
    return {"present": True, "uptime_h": round(up / 3600, 1), "boot_time": _iso(time.time() - up),
            "idle_ratio": round(float(t[1]) / up / (os.cpu_count() or 1), 2) if len(t) > 1 and up else None}


def _last(h, args):
    if not _which("last"):
        return None
    return _run(h, ["last", "-w", "--time-format", "iso", *args], 4000)


_ISO_RX = r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:?\d\d)"


def _wtmp(h):
    """System boot/shutdown records from wtmp via last -x: newest first."""
    def load():
        out = _last(h, ["-x", "-n", "5000", "reboot", "shutdown"])
        if out is None:
            return None
        recs = []
        begins = None
        for line in out.splitlines():
            if line.startswith("wtmp begins"):
                m = re.search(_ISO_RX, line)
                begins = m.group(1) if m else line[12:].strip()
                continue
            m = re.match(r"(reboot|shutdown)\s+system (?:boot|down)\s+(\S+)\s+" + _ISO_RX + r"(?:\s+-\s+(\S+))?", line)
            if m:
                recs.append({"kind": m.group(1), "kernel": m.group(2), "t": m.group(3), "end": m.group(4)})
        return {"records": recs, "begins": begins}
    return _cached(h, "wtmp", load)


def _power(h):
    """Boot history + unclean-shutdown split shared by eventlog.power_history and l3.power_event_split."""
    def load():
        w = _wtmp(h)
        if not w:
            return None
        recs = w["records"]
        now = dt.datetime.now().astimezone()
        boots = [r for r in recs if r["kind"] == "reboot"]
        hours = [0] * 24
        boots30 = 0
        unclean = []
        for i, r in enumerate(recs):
            if r["kind"] != "reboot":
                continue
            t = _parse_iso(r["t"])
            if t:
                hours[t.hour] += 1
                boots30 += (now - t).days < 30
            older = recs[i + 1] if i + 1 < len(recs) else None
            if older is not None and older["kind"] == "reboot":
                unclean.append(r["t"])
            elif r.get("end") == "crash":
                unclean.append(r["t"])
        kernels = collections.Counter(r["kernel"] for r in boots)
        resumes = None
        if _which("journalctl"):
            out = _run(h, ["journalctl", "-k", "-q", "--no-pager", "-S", "-30d", "-n", "2000", "-o", "short-unix",
                           "-g", "PM: suspend exit|PM: hibernation exit"], 4000)
            resumes = len([l for l in (out or "").splitlines() if l.strip()]) if out is not None else None
        kd = [m for m in _kdump_dirs() if m]
        pstore = len(h.list_dir("/var/lib/systemd/pstore", 500))
        crash = min(len(unclean), len(kd) + pstore)
        return {"boots": len(boots), "shutdowns": sum(1 for r in recs if r["kind"] == "shutdown"), "boots_30d": boots30,
                "resumes_30d": resumes, "hours": hours, "unclean": unclean, "crash": crash,
                "span": [recs[-1]["t"] if recs else None, recs[0]["t"] if recs else None], "begins": w["begins"],
                "kernels_booted": len(kernels)}
    return _cached(h, "power", load)


@lp("eventlog.power_history", level="L2", family="usage", tier="T1", gate="boot.uptime", collect="extended")
def eventlog_power_history(h, facts):
    """wtmp boots/shutdowns (last -x), boots in 30 d, boot-hour histogram, unclean shutdowns, suspend resumes (journal -k)."""
    p = _power(h)
    if not p:
        return None
    n_unclean = len(p["unclean"])
    return {"present": bool(p["boots"]), "boots": p["boots"], "shutdowns": p["shutdowns"], "boots_30d": p["boots_30d"],
            "resumes_30d": p["resumes_30d"], "os_start_hour_hist": p["hours"], "span": p["span"],
            "system_log_oldest": p["begins"], "unclean_shutdowns": n_unclean, "kernels_booted": p["kernels_booted"],
            "kp41_split": {"crash": p["crash"], "button_held": 0, "power_removed": n_unclean - p["crash"]}}


@lp("l3.power_event_split", level="L2", family="usage", tier="T1", gate="boot.uptime", collect="extended")
def power_event_split(h, facts):
    """Unclean shutdowns (reboot record with no shutdown before it) split into crash (kdump/pstore evidence) vs
    power_removed (no crash record: switch-off, hard reset or VM kill). Linux records no button-held marker."""
    p = _power(h)
    if not p:
        return None
    now = dt.datetime.now().astimezone()
    ages = [(now - t).days for t in (_parse_iso(x) for x in p["unclean"]) if t]
    crash = p["crash"]
    removed = len(p["unclean"]) - crash
    r30 = sum(1 for a in ages if a <= 30)
    return {"present": True, "via": "wtmp", "events": len(p["unclean"]), "crash": crash, "button_held": 0,
            "power_removed": removed, "crash_30d": min(crash, r30), "button_held_30d": 0,
            "power_removed_30d": max(0, r30 - crash), "health_flag_crash_30d": crash > 0 and r30 > 0,
            "power_off_habit_hint": removed >= 3, "timeline": [x[:10] for x in p["unclean"][:60]],
            "note": "linux: cause of an unclean shutdown is not recorded; crash needs a kdump/pstore record"}


def _my_logins(h):
    def load():
        me = _me()
        if me is None:
            return None
        out = _last(h, ["-n", "5000", me.pw_name])
        if out is None:
            return None
        rows = []
        for line in out.splitlines():
            m = re.match(r"(\S+)\s+(\S+)\s+(?:(?!\d{4}-)(\S+)\s+)?" + _ISO_RX + r"\s+(?:-\s+(\S+)(?:\s+\(([^)]+)\))?|(still logged in|gone - no logout))?", line)
            if not m or m.group(1) != me.pw_name:
                continue
            rows.append({"tty": m.group(2), "from": m.group(3) or "", "t": m.group(4), "end": m.group(5), "dur": m.group(6)})
        return rows
    return _cached(h, "my_logins", load)


def _src_kind(frm):
    if re.fullmatch(r":\d+(\.\d+)?", frm):
        return "x_display_terminal"
    if frm in ("-", "0.0.0.0"):
        return "local_tty"
    if frm.startswith("tty") or frm == "login":
        return "local_tty"
    m = re.fullmatch(r"(\d+)\.(\d+)\.\d+\.\d+", frm)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a == 100 and 64 <= b <= 127:
            return "remote_tailscale"
        if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
            return "remote_lan"
        return "remote_public"
    if ":" in frm:
        return "remote_ipv6"
    return "remote_other"


@lp("eventlog.security_logons", level="L2", family="usage", tier="T1", gate="boot.uptime", collect="extended")
def security_logons(h, facts):
    """Invoking user's logins from wtmp (last USER): sessions by source kind, 30 d count, hour histogram, lastlog.
    IP addresses and hostnames are never emitted. x_display_terminal = terminal on an X display (local or VNC)."""
    rows = _my_logins(h)
    if rows is None:
        return None
    now = dt.datetime.now().astimezone()
    kinds, hours = collections.Counter(), [0] * 24
    n30 = 0
    for r in rows:
        k = "local_tty" if r["tty"].startswith("tty") and r["from"] in ("", "-") else _src_kind(r["from"] or "-")
        kinds[k] += 1
        t = _parse_iso(r["t"])
        if t:
            hours[t.hour] += 1
            n30 += (now - t).days < 30
    ll = None
    me = _me()
    if me and _which("lastlog"):
        out = _run(h, ["lastlog", "-u", me.pw_name], 2000) or ""
        lines = out.strip().splitlines()
        if len(lines) >= 2:
            ll = "never" if "Never logged in" in lines[1] else lines[1][-31:].strip()
    return {"present": bool(rows), "events": len(rows), "sessions_30d": n30, "by_source": dict(kinds),
            "interactive_unlock": kinds.get("local_tty", 0), "x_display_terminals": kinds.get("x_display_terminal", 0),
            "remote_interactive": sum(v for k, v in kinds.items() if k.startswith("remote")),
            "hours": hours,
            "span": [rows[-1]["t"] if rows else None, rows[0]["t"] if rows else None], "lastlog": ll}


@lp("lsm.sessions", level="L2", family="usage", tier="T1", gate="boot.uptime", collect="extended")
def lsm_sessions(h, facts):
    """First and latest login of the invoking user in wtmp, wtmp start, and the user's live logind sessions."""
    rows = _my_logins(h)
    w = _wtmp(h)
    if rows is None and not w:
        return None
    live = []
    me = _me()
    if me and _which("loginctl"):
        out = _run(h, ["loginctl", "list-sessions", "--no-legend"], 2000) or ""
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 3 and p[1] == str(me.pw_uid):
                live.append(p[0])
    types = collections.Counter()
    for sid in live[:10]:
        kv = _kv(_run(h, ["loginctl", "show-session", sid, "-p", "Type", "-p", "Remote"], 1500) or "")
        types[(kv.get("Type") or "?") + ("/remote" if kv.get("Remote") == "yes" else "")] += 1
    return {"present": True, "first": rows[-1]["t"] if rows else None, "last": rows[0]["t"] if rows else None,
            "wtmp_begins": (w or {}).get("begins"), "live_sessions": len(live), "live_by_type": dict(types)}


@lp("usage.scheduled_jobs", level="L2", family="usage", tier="T1", gate="boot.uptime", collect="extended")
def scheduled_jobs(h, facts):
    """Scheduled work: own crontab lines, own systemd user timers/enabled services, system timers (counts only)."""
    cron = None
    if _which("crontab"):
        out = _run(h, ["crontab", "-l"], 2000)
        if out is not None:
            lines = [l for l in out.splitlines() if l.strip() and not l.lstrip().startswith("#")]
            cron = {"lines": len(lines), "operator_excluded": sum(1 for l in lines if OPERATOR_RX.search(l)),
                    "reboot_jobs": sum(1 for l in lines if l.startswith("@reboot"))}
            cron["lines"] -= cron["operator_excluded"]
    ut = _run(h, ["systemctl", "--user", "list-timers", "--all", "--no-legend", "--no-pager"], 2000)
    uu = _run(h, ["systemctl", "--user", "list-unit-files", "--state=enabled", "--no-legend", "--no-pager"], 2000)
    st = _run(h, ["systemctl", "list-timers", "--all", "--no-legend", "--no-pager"], 2000)
    own_units = [n for n in h.list_dir(os.path.join(_home(), ".config/systemd/user"), 200) if n.endswith((".service", ".timer"))]
    local_timers = [n for n in h.list_dir("/etc/systemd/system", 1000) if n.endswith(".timer")]
    cnt = lambda o: None if o is None else sum(1 for l in o.splitlines() if l.strip())
    return {"present": True, "crontab": cron, "user_timers": cnt(ut), "user_enabled_units": cnt(uu),
            "user_custom_unit_files": len([n for n in own_units if not OPERATOR_RX.search(n)]),
            "system_timers": cnt(st), "system_local_timers": len(local_timers),
            "cron_d_files": len([n for n in h.list_dir("/etc/cron.d", 200) if not n.startswith(".")])}


def _kam():
    return os.path.join(_home(), ".local/share/kactivitymanagerd/resources/database")


@lp("usage.kde_activity_db", level="L1", family="usage")
def kde_activity_db(h, facts):
    """KDE activity manager usage database present (resource open events per app)."""
    m = h.meta(_kam())
    if not m.get("present"):
        return None
    w = h.meta(_kam() + "-wal")
    return {"present": True, "bytes": m["bytes"], "wal_bytes": w.get("bytes"), "mtime": m["mtime"]}


@lp("usage.kde_activity", level="L2", family="usage", tier="T1", gate="usage.kde_activity_db", collect="extended")
def kde_activity(h, facts):
    """KDE ResourceEvent aggregates per initiating app: events, open hours, 30 d events, hour histogram. Paths never read out."""
    rows = h.sqlite(_kam(), "SELECT initiatingAgent, start, end FROM ResourceEvent ORDER BY start DESC LIMIT 200000")
    if rows is None:
        return {"present": False, "error": "unreadable"}
    now = time.time()
    by, hrs, n30 = collections.Counter(), collections.Counter(), 0
    hours = [0] * 24
    first = last = None
    for agent, s, e in rows:
        if not s:
            continue
        by[agent] += 1
        if e and e > s:
            hrs[agent] += min(e - s, 12 * 3600)
        hours[time.localtime(s).tm_hour] += 1
        n30 += now - s <= 30 * 86400
        first = s if first is None or s < first else first
        last = s if last is None or s > last else last
    return {"present": bool(by), "events": sum(by.values()), "events_30d": n30, "apps": len(by),
            "top_apps": [[a, n, round(hrs[a] / 3600, 1)] for a, n in by.most_common(10)], "hours": hours,
            "first": _iso(first), "last": _iso(last), "via": h.last_via}
