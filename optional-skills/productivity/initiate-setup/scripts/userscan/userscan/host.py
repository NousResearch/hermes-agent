"""L0 host facts + HostAccess facade (all the primitives probes may use). No I/O at import time."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time

OS_NAMES = ("windows", "darwin", "linux")
# l0 keys a run may override, and the environment variables each one stands for.
OVERRIDABLE = {"home": ("HOME", "USERPROFILE"), "localappdata": ("LOCALAPPDATA",),
               "appdata": ("APPDATA",), "hermes_home": ("HERMES_HOME",)}
# Operator lab accounts (CONTRACT.md hard rule 4); list_users drops them unless asked.
_OPERATOR_ACCT_RX = re.compile(r"^(hn-e2e|ns960|ns923.*|lhm|shots|user-insights-lab|hermes-.*)$", re.I)
_PLACEHOLDER_HOMES = ("", "/", "/nonexistent", "/dev/null", "/var/empty")
_VAR_RX = re.compile(r"%([A-Za-z_][A-Za-z0-9_()]*)%")

# ---------------------------------------------------------------- L0 facts


def detect_os() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _pe_machine(path: str) -> str:
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"MZ":
                return ""
            f.seek(0x3C)
            (off,) = struct.unpack("<I", f.read(4))
            f.seek(off)
            if f.read(4) != b"PE\0\0":
                return ""
            (machine,) = struct.unpack("<H", f.read(2))
            return {0x8664: "x64", 0xAA64: "arm64", 0x14C: "x86"}.get(machine, hex(machine))
    except OSError:
        return ""


def _native_machine() -> str:
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32")
            get = getattr(k32, "IsWow64Process2", None)
            if get:
                process_machine = ctypes.c_ushort()
                native_machine = ctypes.c_ushort()
                get(ctypes.c_void_p(-1), ctypes.byref(process_machine), ctypes.byref(native_machine))
                return {0x8664: "x64", 0xAA64: "arm64", 0x14C: "x86"}.get(native_machine.value, "")
        except Exception:
            pass
        return platform.machine().lower()
    return platform.machine().lower()


def _is_admin() -> bool:
    if sys.platform == "win32":
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return os.geteuid() == 0 if hasattr(os, "geteuid") else False


class HostAccess:
    """Primitives every probe may call. `scratch()` is a per-run temp dir, removed at exit."""

    def __init__(self, facts: dict, child_env: dict = None):
        self.l0 = facts
        self._child_env = dict(child_env) if child_env is not None else None
        self.os = facts["os"]
        self.real_os = facts.get("os_detected", detect_os())
        # True when this pass targets an account other than the invoking one: HKCU is not theirs.
        self.foreign = bool(facts.get("foreign_user"))
        self._scratch = None
        self._via = threading.local()

    @property
    def last_via(self):
        return getattr(self._via, "v", None)

    @last_via.setter
    def last_via(self, v):
        self._via.v = v

    def child_env(self, **extra) -> dict:
        """Environment for a spawned child: the one the caller passed (a Hermes backend passes the
        served profile's clean env), else this process's. `extra` entries win."""
        base = self._child_env if self._child_env is not None else os.environ
        return {**base, **extra}

    # -- paths ------------------------------------------------------
    def expand(self, path: str) -> str:
        """~ resolves to l0 home and %VAR% to the l0 value for HOME/USERPROFILE/LOCALAPPDATA/APPDATA/
        HERMES_HOME, so an overridden run targets the overridden home on every OS."""
        home = self.l0.get("home") or os.path.expanduser("~")
        if path == "~" or path.startswith(("~/", "~\\")):
            path = home + path[1:]
        by_env = {}
        for key, names in OVERRIDABLE.items():
            for n in names:
                by_env[n] = self.l0.get(key) or ""

        def sub(m):
            v = by_env.get(m.group(1).upper())
            if v:
                return v
            v = os.environ.get(m.group(1))
            return v if v is not None else m.group(0)
        return os.path.expandvars(_VAR_RX.sub(sub, path))

    def exists(self, path: str) -> bool:
        return os.path.exists(self.expand(path))

    def meta(self, path: str):
        """presence + size + mtime. Safe for T3 files (never opened)."""
        p = self.expand(path)
        try:
            st = os.stat(p)
            return {"present": True, "bytes": st.st_size, "mtime": int(st.st_mtime)}
        except OSError:
            return {"present": False}

    def count_dir(self, path: str, max_entries: int = 20000) -> int:
        p = self.expand(path)
        n = 0
        try:
            with os.scandir(p) as it:
                for _ in it:
                    n += 1
                    if n >= max_entries:
                        break
        except OSError:
            return -1
        return n

    def list_dir(self, path: str, max_entries: int = 500):
        p = self.expand(path)
        try:
            with os.scandir(p) as it:
                return [e.name for _, e in zip(range(max_entries), it)]
        except OSError:
            return []

    # -- registry ---------------------------------------------------
    def reg(self, path: str, value: str = None):
        """Read a value, or enumerate subkey names if value is None. None on miss.
        path = 'HKLM\\SOFTWARE\\...'"""
        if self.os != "windows" or self.real_os != "windows":
            return None
        if self.foreign and path[:4] == "HKCU":
            return None
        import winreg
        root = {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER,
                "HKU": winreg.HKEY_USERS, "HKCR": winreg.HKEY_CLASSES_ROOT}.get(path[:4])
        if root is None:
            return None
        key_path = path[5:]
        try:
            if value is None:
                with winreg.OpenKey(root, key_path) as k:
                    out, i = [], 0
                    while True:
                        try:
                            out.append(winreg.EnumKey(k, i))
                        except OSError:
                            return out
                        i += 1
                        if i > 5000:
                            return out
            with winreg.OpenKey(root, key_path) as k:
                v, _ = winreg.QueryValueEx(k, value)
                return v
        except OSError:
            return None

    # -- sqlite -----------------------------------------------------
    def sqlite(self, path: str, query: str, params=(), timeout_ms: int = 1500):
        """Read-only query on a copy if a WAL is present, else on the live file.
        Returns list of rows (tuples) or None on any failure. Ladder rung is in self.last_via (thread-local)."""
        p = self.expand(path)
        if not os.path.exists(p):
            return None
        if os.path.exists(p + "-wal") or os.path.exists(p + "-journal"):
            return self._sqlite_copy_then_query(p, query, params, timeout_ms)
        for via, uri in (("live", f"file:{p}?mode=ro"), ("immutable", f"file:{p}?mode=ro&immutable=1")):
            try:
                con = sqlite3.connect(uri, uri=True, timeout=timeout_ms / 1000.0)
                con.set_progress_handler(lambda: 1 if timeout_ms else 0, 1000)
                rows = con.execute(query, params).fetchall()
                con.close()
                self.last_via = via
                return rows
            except Exception:
                continue
        return self._sqlite_copy_then_query(p, query, params, timeout_ms)

    def _sqlite_copy_then_query(self, p, query, params, timeout_ms):
        try:
            dst = os.path.join(self.scratch(), os.path.basename(p))
            shutil.copyfile(p, dst)
            for suf in ("-wal", "-journal"):
                if os.path.exists(p + suf):
                    shutil.copyfile(p + suf, dst + suf)
            con = sqlite3.connect(dst, timeout=timeout_ms / 1000.0)
            rows = con.execute(query, params).fetchall()
            con.close()
            self.last_via = "copy"
            return rows
        except Exception:
            return None

    # -- locked files -----------------------------------------------
    def copy_locked(self, path: str, dst_name: str = None, allow_vss: bool = False):
        """Ladder: shutil copy -> esentutl /y -> esentutl /y /vss (only with allow_vss).
        Returns (abs path or None, rung name)."""
        p = self.expand(path)
        dst = os.path.join(self.scratch(), dst_name or os.path.basename(p))
        try:
            shutil.copyfile(p, dst)
            return dst, "shutil"
        except OSError:
            pass
        for rung, args in (("esentutl", ["esentutl", "/y", p, "/d", dst, "/o"]),
                           ("vss", ["esentutl", "/y", p, "/vss", "/d", dst, "/o"])):
            if rung == "vss" and not allow_vss:
                continue
            try:
                r = subprocess.run(args, capture_output=True, timeout=30, stdin=subprocess.DEVNULL,
                                   env=self.child_env())
                if r.returncode == 0 and os.path.exists(dst):
                    return dst, rung
            except Exception:
                continue
        return None, "failed"

    # -- powershell / subprocess -------------------------------------
    def run(self, args, timeout_ms: int = 5000, text: bool = True, env: dict | None = None):
        """Run a command and return its stdout as str (decoded utf-8, errors replaced).
        The `text` argument is kept for backward compatibility and is ignored: every
        caller wants str (regex on bytes raises TypeError). `env` entries go on top of child_env()."""
        try:
            r = subprocess.run(args, capture_output=True, timeout=timeout_ms / 1000.0, stdin=subprocess.DEVNULL,
                               env=self.child_env(**(env or {})), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return (r.stdout or b"").decode("utf-8", "replace")
        except Exception:
            return None

    def powershell(self, script: str, timeout_ms: int = 8000):
        """Single-shot powershell. Prefer ps_probe (batched) when several probes need PS."""
        return self.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout_ms, text=True)

    # -- accounts ----------------------------------------------------
    def list_users(self, include_operator: bool = False):
        """Login accounts on this host as [{"user": name, "home": path}]. Names and homes only.
        windows: ProfileList (SID -> name via LookupAccountSid), `net user` if that is empty.
        linux: passwd entries with uid >= UID_MIN, a shell listed in /etc/shells, a non-placeholder home.
        darwin: dscl . -readall /Users (uid >= 501, real shell, home under /Users), /Users scan fallback.
        Under --os the pretend OS's account store is not on this host, so the real OS's is used."""
        try:
            got = {"windows": self._users_windows, "linux": self._users_linux,
                   "darwin": self._users_darwin}[self.real_os]()
        except Exception:
            got = []
        seen, out = set(), []
        for u in got:
            name = (u.get("user") or "").strip()
            if not name or name.lower() in seen:
                continue
            if not include_operator and _OPERATOR_ACCT_RX.match(name):
                continue
            seen.add(name.lower())
            out.append({"user": name, "home": u.get("home") or ""})
        return out

    def _users_linux(self):
        uid_min = 1000
        try:
            with open("/etc/login.defs", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == "UID_MIN" and parts[1].isdigit():
                        uid_min = int(parts[1])
        except OSError:
            pass
        shells = set()
        try:
            with open("/etc/shells", encoding="utf-8", errors="replace") as fh:
                shells = {s.strip() for s in fh if s.strip() and not s.startswith("#")}
        except OSError:
            pass
        try:
            import pwd
            rows = [(p.pw_name, p.pw_uid, p.pw_dir, p.pw_shell) for p in pwd.getpwall()]
        except Exception:
            rows = []
            with open("/etc/passwd", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    f = line.rstrip("\n").split(":")
                    if len(f) >= 7 and f[2].isdigit():
                        rows.append((f[0], int(f[2]), f[5], f[6]))
        out = []
        for name, uid, home, shell in rows:
            if uid < uid_min or uid >= 65534:
                continue
            base = os.path.basename(shell or "")
            if base in ("nologin", "false", "sync", "") or (shells and shell not in shells):
                continue
            if home.rstrip("/") in _PLACEHOLDER_HOMES or home.startswith(("/proc", "/sys", "/usr/", "/bin")):
                continue
            out.append({"user": name, "home": home})
        return out

    def _users_darwin(self):
        out = []
        raw = self.run(["dscl", ".", "-readall", "/Users", "RecordName", "UniqueID", "UserShell",
                        "NFSHomeDirectory"], 5000) if self.real_os == "darwin" else None
        if raw:
            for rec in raw.split("\n-\n"):
                vals, key = {}, None
                for line in rec.splitlines():
                    if line.startswith(" ") and key:
                        vals[key] = (vals.get(key, "") + " " + line.strip()).strip()
                    elif ":" in line:
                        key, _, v = line.partition(":")
                        vals[key] = v.strip()
                name = (vals.get("RecordName") or "").split()
                uid = (vals.get("UniqueID") or "").strip()
                shell = os.path.basename(vals.get("UserShell") or "")
                home = vals.get("NFSHomeDirectory") or ""
                if not name or not uid.isdigit() or int(uid) < 501:
                    continue
                if shell in ("false", "nologin", "") or not home.startswith("/Users/"):
                    continue
                out.append({"user": name[0], "home": home})
        if not out:
            try:
                with os.scandir("/Users") as it:
                    for _, e in zip(range(200), it):
                        if e.is_dir() and e.name not in ("Shared", "Guest") and not e.name.startswith("."):
                            out.append({"user": e.name, "home": e.path})
            except OSError:
                pass
        return out

    def _users_windows(self):
        base = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList"
        out = []
        for sid in self.reg(base) or []:
            if not sid.startswith(("S-1-5-21-", "S-1-12-1-")) or sid.endswith(".bak"):
                continue
            home = os.path.expandvars(self.reg(base + "\\" + sid, "ProfileImagePath") or "")
            if not home:
                continue
            out.append({"user": _sid_to_name(sid) or os.path.basename(home), "home": home})
        if out:
            return out
        raw = self.run(["net", "user"], 5000) or ""
        lines = raw.splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("---")), None)
        if start is None:
            return out
        root = os.path.join(os.environ.get("SystemDrive", "C:") + "\\", "Users")
        for line in lines[start + 1:]:
            if not line.strip() or line.lower().startswith("the command"):
                break
            for name in re.split(r"\s{2,}", line.strip()):
                if name:
                    out.append({"user": name, "home": os.path.join(root, name)})
        return out

    def process_counts(self):
        """{username: process count} for every account with a visible process, or None when the OS
        does not expose other users' processes to this caller. Counts only; no names, no argv."""
        counts = {}
        if self.real_os == "linux":
            import pwd
            names = {}
            try:
                with os.scandir("/proc") as it:
                    for e in it:
                        if not e.name.isdigit():
                            continue
                        try:
                            uid = e.stat(follow_symlinks=False).st_uid
                        except OSError:
                            continue
                        if uid not in names:
                            try:
                                names[uid] = pwd.getpwuid(uid).pw_name
                            except KeyError:
                                names[uid] = str(uid)
                        counts[names[uid]] = counts.get(names[uid], 0) + 1
            except OSError:
                return None
            return counts
        if self.real_os == "darwin":
            raw = self.run(["ps", "-axo", "user="], 5000)
            if raw is None:
                return None
            for line in raw.splitlines():
                u = line.strip()
                if u:
                    counts[u] = counts.get(u, 0) + 1
            return counts
        if self.real_os == "windows":
            import csv
            raw = self.run(["tasklist", "/v", "/fo", "csv", "/nh"], 15000)
            if raw is None:
                return None
            for row in csv.reader(raw.splitlines()):
                if len(row) >= 7 and row[6] and row[6] != "N/A":
                    u = row[6].split("\\")[-1]
                    counts[u] = counts.get(u, 0) + 1
            return counts
        return None

    # -- scratch -----------------------------------------------------
    def scratch(self) -> str:
        if self._scratch is None:
            base = os.path.join(tempfile.gettempdir(), "userscan", self.l0.get("run_id", "run"))
            os.makedirs(base, exist_ok=True)
            self._scratch = base
        return self._scratch

    def cleanup(self):
        if self._scratch and os.path.isdir(self._scratch):
            shutil.rmtree(self._scratch, ignore_errors=True)
            parent = os.path.dirname(self._scratch)
            try:
                os.rmdir(parent)
            except OSError:
                pass


def per_user_overrides(os_name: str, home: str) -> dict:
    """The l0 overrides that retarget a run at `home`. HERMES_HOME is cleared (the invoking user's
    value must not leak into another account's pass); probes fall back to the default location."""
    if os_name == "windows":
        return {"home": home, "localappdata": os.path.join(home, "AppData", "Local"),
                "appdata": os.path.join(home, "AppData", "Roaming"), "hermes_home": ""}
    return {"home": home, "localappdata": "", "appdata": "", "hermes_home": ""}


def _sid_to_name(sid: str):
    try:
        import ctypes
        from ctypes import wintypes
        adv = ctypes.WinDLL("advapi32")
        psid = ctypes.c_void_p()
        if not adv.ConvertStringSidToSidW(ctypes.c_wchar_p(sid), ctypes.byref(psid)):
            return None
        try:
            name = ctypes.create_unicode_buffer(256)
            dom = ctypes.create_unicode_buffer(256)
            n, d, use = wintypes.DWORD(256), wintypes.DWORD(256), wintypes.DWORD()
            if adv.LookupAccountSidW(None, psid, name, ctypes.byref(n), dom, ctypes.byref(d), ctypes.byref(use)):
                return name.value
        finally:
            ctypes.windll.kernel32.LocalFree(psid)
    except Exception:
        return None
    return None


def collect_l0(run_id: str, allow_vss: bool = False, os_override: str = None, overrides: dict = None) -> dict:
    """L0 facts. `os_override` (windows|darwin|linux) makes probe selection behave as if on that OS;
    `overrides` replaces any of home/localappdata/appdata/hermes_home for this run. When home is
    overridden and the others are not, they are derived from the new home (see per_user_overrides)."""
    detected = detect_os()
    os_name = os_override if os_override in OS_NAMES else detected
    facts = {
        "run_id": run_id,
        "os": os_name,
        "os_detected": detected,
        "os_overridden": os_name != detected,
        "os_release": platform.release(),
        "os_build": platform.version(),
        "native_arch": _native_machine(),
        "python_arch": _pe_machine(sys.executable),
        "admin": _is_admin(),
        "user": os.environ.get("USERNAME") or os.environ.get("USER") or "",
        "home": os.path.expanduser("~"),
        "localappdata": os.environ.get("LOCALAPPDATA", ""),
        "appdata": os.environ.get("APPDATA", ""),
        "hermes_home": os.environ.get("HERMES_HOME", ""),
        "allow_vss": allow_vss,
        "overridden": [],
    }
    ov = {k: v for k, v in (overrides or {}).items() if k in OVERRIDABLE and v is not None}
    if "home" in ov:
        for k, v in per_user_overrides(os_name, ov["home"]).items():
            ov.setdefault(k, v)
    for k, v in ov.items():
        facts[k] = v
    facts["overridden"] = sorted(ov)
    facts["python_emulated"] = bool(facts["python_arch"] and facts["native_arch"]
                                    and facts["python_arch"] != facts["native_arch"]
                                    and facts["python_arch"] in ("x64", "x86")
                                    and facts["native_arch"] == "arm64")
    return facts
