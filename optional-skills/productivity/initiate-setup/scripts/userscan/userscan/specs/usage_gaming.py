"""Usage habit probes (Windows), plus the helpers usage_gaming_games and usage_gaming_hardware share.

Registration only at import time. Routes follow reports/14-triage.json: registry and file reads in core,
the batched PowerShell sidecar for event logs and CIM, SRUM and PnP history in deep.
"""
from __future__ import annotations

import collections
import datetime as dt
import os
import re
import struct
import sys
import threading
import time

from userscan.registry import probe

_T0 = time.time()
_LOCK = threading.Lock()
_CACHE: dict = {}

_NOISE = re.compile(r"(?i)(\\hn-e2e(\\|$)|\\ns960(\\|$)|\\ns923[^\\]*(\\|$)|\\lhm(\\|$)|\\shots(\\|$)|user-insights-lab|"
                    r"\\hermes-(?!agent)[^\\]*\\|\\userscan\\|\\uv\\python\\|\\cache\\scratch\\)")
_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_FT_EPOCH = dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc)


# ----------------------------------------------------------------- helpers

def _noise(path: str) -> bool:
    return bool(path) and bool(_NOISE.search(path))


def _base(p):
    if not p:
        return p
    return p.replace("/", "\\").rstrip("\\").rsplit("\\", 1)[-1]


def _ft(ft):
    if not ft or ft <= 0 or ft > 0x7FFFFFFFFFFFFFFF:
        return None
    try:
        return (_FT_EPOCH + dt.timedelta(microseconds=ft // 10)).astimezone()
    except (OverflowError, ValueError):
        return None


def _iso(d):
    if d is None:
        return None
    if isinstance(d, (int, float)):
        if d <= 0:
            return None
        d = dt.datetime.fromtimestamp(d).astimezone()
    return d.isoformat(timespec="seconds")


def _days_ago(ts):
    return None if not ts else round((time.time() - ts) / 86400, 1)


def _grid():
    return [[0] * 24 for _ in range(7)]


def _grid_out(g, scale=1.0, nd=1):
    return {_DOW[i]: [round(v / scale, nd) for v in g[i]] for i in range(7)}


def _env(name, default=""):
    return os.environ.get(name, default)


def _paths():
    home = os.path.expanduser("~")
    return {
        "HOME": home, "LAD": _env("LOCALAPPDATA", os.path.join(home, "AppData", "Local")),
        "RAD": _env("APPDATA", os.path.join(home, "AppData", "Roaming")),
        "PD": _env("ProgramData", r"C:\ProgramData"), "PF": _env("ProgramFiles", r"C:\Program Files"),
        "PF86": _env("ProgramFiles(x86)", r"C:\Program Files (x86)"), "WIN": _env("SystemRoot", r"C:\Windows"),
    }


def _ex(p):
    try:
        return os.path.exists(p)
    except (OSError, ValueError):
        return False


def _mtime(p):
    try:
        return int(os.path.getmtime(p))
    except OSError:
        return None


def _read(p, limit=8_000_000):
    try:
        with open(p, "rb") as f:
            raw = f.read(limit)
    except OSError:
        return None
    for enc in ("utf-8-sig", "utf-16", "mbcs", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _winreg():
    import winreg
    return winreg


def _hive(path):
    w = _winreg()
    return {"HKLM": w.HKEY_LOCAL_MACHINE, "HKCU": w.HKEY_CURRENT_USER}[path[:4]], path[5:]


def _reg_values(path, max_values=5000):
    """{name: value} for one key, or {} on miss."""
    w = _winreg()
    root, sub = _hive(path)
    out = {}
    try:
        with w.OpenKey(root, sub) as k:
            i = 0
            while i < max_values:
                try:
                    n, v, _ = w.EnumValue(k, i)
                except OSError:
                    break
                out[n] = v
                i += 1
    except OSError:
        return {}
    return out


def _reg_keys(h, path, limit=5000):
    return (h.reg(path) or [])[:limit]


def _cached(key, fn):
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = {"lock": threading.Lock(), "done": False, "value": None}
        slot = _CACHE[key]
    with slot["lock"]:
        if not slot["done"]:
            try:
                slot["value"] = fn()
            except Exception as e:
                slot["value"] = {"error": f"{type(e).__name__}: {e}"}
            slot["done"] = True
    return slot["value"]


def _uninstall(h):
    def load():
        out = []
        for base in (r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                     r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
                     r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"):
            for sk in _reg_keys(h, base, 3000):
                v = _reg_values(base + "\\" + sk, 60)
                if v.get("DisplayName"):
                    out.append({"name": str(v["DisplayName"]), "version": str(v.get("DisplayVersion") or ""),
                                "date": str(v.get("InstallDate") or "")})
        return out
    return _cached(("uninstall", h.l0.get("run_id")), load) or []


def _unmatch(h, pattern):
    rx = re.compile(pattern, re.I)
    seen, out = set(), []
    idx = _uninstall(h)
    for e in idx if isinstance(idx, list) else []:
        if rx.search(e["name"]) and e["name"] not in seen:
            seen.add(e["name"])
            out.append({"name": e["name"], "version": e["version"] or None, "install_date": e["date"] or None})
    return out


# ================================================================= USAGE (L1, registry + files)

@probe(id="boot.uptime", level="L1", family="usage", tier="T0", collect="core")
def boot_uptime(h, facts):
    """Uptime from GetTickCount64 and derived boot time; sleep-vs-shutdown habit input."""
    import ctypes
    k32 = ctypes.WinDLL("kernel32")
    k32.GetTickCount64.restype = ctypes.c_ulonglong
    ms = int(k32.GetTickCount64())
    boot = dt.datetime.now().astimezone() - dt.timedelta(milliseconds=ms)
    return {"present": True, "uptime_h": round(ms / 3.6e6, 1), "boot_time": _iso(boot)}


@probe(id="wu.active_hours", level="L1", family="usage", tier="T0", collect="core")
def wu_active_hours(h, facts):
    """Windows Update active hours (user-set or learned): the hours the machine expects to be in use."""
    v = _reg_values(r"HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings", 200)
    if not v:
        return None
    keys = ("ActiveHoursStart", "ActiveHoursEnd", "SmartActiveHoursState", "SmartActiveHoursStart", "SmartActiveHoursEnd")
    out = {k: v.get(k) for k in keys if k in v}
    if not out:
        return None
    return {"present": True, **out}


@probe(id="srum.present", level="L1", family="usage", tier="T0", collect="core")
def srum_present(h, facts):
    """SRUM database presence and size; gates the SRUM subtree."""
    m = h.meta(r"C:\Windows\System32\sru\SRUDB.dat")
    return m if m.get("present") else None


@probe(id="userassist.focus", level="L1", family="usage", tier="T1", collect="core")
def userassist_focus(h, facts):
    """Lifetime per-app run count and focus time from UserAssist (ROT13 names); cheapest screen-time source."""
    import codecs
    w = _winreg()
    root = r"Software\Microsoft\Windows\CurrentVersion\Explorer\UserAssist"
    apps = collections.defaultdict(lambda: [0, 0, 0, None])
    noise = 0
    try:
        w.OpenKey(w.HKEY_CURRENT_USER, root).Close()
    except OSError:
        return None
    for g in _reg_keys(h, "HKCU\\" + root, 50):
        for name, data in _reg_values("HKCU\\" + root + "\\" + g + "\\Count", 3000).items():
            name = codecs.decode(name, "rot13")
            if not isinstance(data, bytes) or len(data) < 68 or name.startswith("UEME_"):
                continue
            if _noise(name):
                noise += 1
                continue
            runc, focc, focms = struct.unpack_from("<III", data, 4)
            last = _ft(struct.unpack_from("<Q", data, 60)[0])
            a = apps[_base(name)]
            a[0] += runc
            a[1] += focc
            a[2] += focms
            if last and (a[3] is None or last > a[3]):
                a[3] = last
    if not apps:
        return None
    lasts = [a[3] for a in apps.values() if a[3]]
    by_focus = sorted(apps.items(), key=lambda kv: -kv[1][2])
    by_run = sorted(apps.items(), key=lambda kv: -kv[1][0])
    return {"present": True, "entries": len(apps), "noise_filtered": noise,
            "total_focus_h": round(sum(a[2] for a in apps.values()) / 3.6e6, 1),
            "top_focus_h": [[n, round(a[2] / 3.6e6, 2)] for n, a in by_focus[:15] if a[2]],
            "top_runs": [[n, a[0]] for n, a in by_run[:10] if a[0]],
            "oldest_last_run": _iso(min(lasts)) if lasts else None,
            "newest_last_run": _iso(max(lasts)) if lasts else None}


@probe(id="featureusage.appswitched", level="L1", family="usage", tier="T1", collect="core")
def featureusage_appswitched(h, facts):
    """Taskbar switch/launch counts per app (lifetime) from Explorer FeatureUsage."""
    root = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\FeatureUsage"
    out = {}
    for sub in ("AppSwitched", "AppLaunch", "ShowJumpView"):
        c = collections.Counter()
        for n, v in _reg_values(root + "\\" + sub, 3000).items():
            if isinstance(v, int) and not _noise(n):
                c[_base(n)] += v
        if c:
            out[sub] = {"entries": len(c), "total": sum(c.values()), "top": c.most_common(10)}
    return {"present": True, **out} if out else None


@probe(id="bam.last_run", level="L1", family="usage", tier="T1", collect="core")
def bam_last_run(h, facts):
    """Background Activity Moderator: last-run time per exe across SIDs; counts in 24 h / 7 d."""
    root = r"HKLM\SYSTEM\CurrentControlSet\Services\bam\State\UserSettings"
    sids = _reg_keys(h, root, 100)
    if not sids:
        return None
    rows = {}
    noise = 0
    own = _base(sys.executable).lower()
    for sid in sids:
        for n, v in _reg_values(root + "\\" + sid, 2000).items():
            if not isinstance(v, bytes) or len(v) < 8:
                continue
            t = _ft(struct.unpack_from("<Q", v, 0)[0])
            if not t:
                continue
            if _noise(n) or t.timestamp() >= _T0 - 5 and _base(n).lower() in (own, "uv.exe", "conhost.exe", "sshd.exe"):
                noise += 1
                continue
            b = _base(n)
            if b not in rows or t > rows[b]:
                rows[b] = t
    if not rows:
        return None
    now = dt.datetime.now().astimezone()
    srt = sorted(rows.items(), key=lambda kv: kv[1], reverse=True)
    return {"present": True, "sids": len(sids), "exes": len(rows), "noise_filtered": noise,
            "used_24h": sum(1 for _, t in srt if now - t < dt.timedelta(days=1)),
            "used_7d": sum(1 for _, t in srt if now - t < dt.timedelta(days=7)),
            "oldest": _iso(srt[-1][1]), "recent": [[a, _iso(t)] for a, t in srt[:10]]}


@probe(id="pca.launchdic", level="L1", family="usage", tier="T1", collect="core")
def pca_launchdic(h, facts):
    """Program Compatibility Assistant launch dictionary: distinct exes launched with last-launch dates."""
    txt = _read(os.path.join(_paths()["WIN"], r"appcompat\pca\PcaAppLaunchDic.txt"), 2_000_000)
    if not txt:
        return None
    rows, noise = [], 0
    for line in txt.splitlines()[:20000]:
        if "|" not in line:
            continue
        path, ts = line.rsplit("|", 1)
        if _noise(path):
            noise += 1
            continue
        try:
            t = dt.datetime.strptime(ts.strip()[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).astimezone()
        except ValueError:
            continue
        rows.append((_base(path), t))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1], reverse=True)
    now = dt.datetime.now().astimezone()
    return {"present": True, "entries": len(rows), "noise_filtered": noise,
            "oldest": _iso(rows[-1][1]), "newest": _iso(rows[0][1]),
            "launched_30d": sum(1 for _, t in rows if now - t < dt.timedelta(days=30)),
            "recent": [[a, _iso(t)] for a, t in rows[:10]]}


@probe(id="prefetch.stat", level="L1", family="usage", tier="T0", collect="core")
def prefetch_stat(h, facts):
    """Prefetch directory stat (admin): file count, distinct exes, oldest and newest mtime."""
    d = os.path.join(_paths()["WIN"], "Prefetch")
    n, exes, oldest, newest = 0, set(), None, None
    try:
        with os.scandir(d) as it:
            for e in it:
                if n >= 5000:
                    break
                if not e.name.lower().endswith(".pf"):
                    continue
                n += 1
                exes.add(e.name.rsplit("-", 1)[0])
                try:
                    m = e.stat().st_mtime
                except OSError:
                    continue
                oldest = m if oldest is None or m < oldest else oldest
                newest = m if newest is None or m > newest else newest
    except OSError:
        return None
    if not n:
        return None
    return {"present": True, "files": n, "distinct_exes": len(exes), "oldest": _iso(oldest), "newest": _iso(newest)}


def _mam(data):
    import ctypes
    from ctypes import byref, c_ulong
    ntdll = ctypes.WinDLL("ntdll")
    sig, usize = struct.unpack_from("<II", data, 0)
    fmt = (sig >> 24) & 0x0F
    wsa, wsb = c_ulong(0), c_ulong(0)
    ntdll.RtlGetCompressionWorkSpaceSize(ctypes.c_ushort(fmt), byref(wsa), byref(wsb))
    ws = ctypes.create_string_buffer(wsa.value)
    out = ctypes.create_string_buffer(usize)
    fin = c_ulong(0)
    src = data[8:]
    rc = ntdll.RtlDecompressBufferEx(ctypes.c_ushort(fmt), out, c_ulong(usize), src, c_ulong(len(src)), byref(fin), ws)
    if rc != 0:
        raise OSError(f"RtlDecompressBufferEx 0x{rc & 0xffffffff:x}")
    return out.raw[: fin.value]


@probe(id="prefetch.run_counts", level="L2", family="usage", tier="T1", collect="deep", gate="prefetch.stat",
       timeout_ms=20000)
def prefetch_run_counts(h, facts):
    """Per-exe run counts and last-8 run times from every .pf (MAM decompress); launch-hour grid."""
    d = os.path.join(_paths()["WIN"], "Prefetch")
    runs, errors = collections.Counter(), 0
    grid = _grid()
    files = [e.path for e in os.scandir(d) if e.name.lower().endswith(".pf")][:3000]
    for f in files:
        try:
            with open(f, "rb") as fh:
                data = fh.read(4_000_000)
            if data[:3] == b"MAM":
                data = _mam(data)
            ver = struct.unpack_from("<I", data, 0)[0]
            if data[4:8] != b"SCCA":
                errors += 1
                continue
            exe = data[0x10:0x10 + 60].decode("utf-16-le", "replace").split("\0")[0]
            if ver >= 26:
                metrics_off = struct.unpack_from("<I", data, 0x54)[0]
                rc_off = 0xD0 if (ver == 26 or metrics_off >= 0x130) else 0xC8
                times = [struct.unpack_from("<Q", data, 0x80 + 8 * i)[0] for i in range(8)]
            else:
                rc_off = 0x98
                times = [struct.unpack_from("<Q", data, 0x80)[0]]
            runs[exe] += struct.unpack_from("<I", data, rc_off)[0]
            for x in times:
                t = _ft(x)
                if t:
                    grid[t.weekday()][t.hour] += 1
        except Exception:
            errors += 1
    if not runs:
        return None
    return {"present": True, "files": len(files), "parse_errors": errors, "total_runs": sum(runs.values()),
            "top_run_counts": runs.most_common(20), "launch_grid": _grid_out(grid, 1, 0)}


# ----------------------------------------------------------------- SRUM (deep): one shared load, many probes

_SRUM_TABLES = {
    "timeline": ("{5C8CF1C7-7257-4F13-B223-970EF5939312}",
                 ["AppId", "UserId", "EndTime", "DurationMS", "InFocusS", "UserInputS", "KeyboardInputS", "AudioOutS"]),
    "resource": ("{D10CA2FE-6FCF-4F6D-848E-B2E99266FA89}", ["TimeStamp", "AppId", "UserId", "ForegroundCycleTime"]),
    "network": ("{973F5D5C-1D90-4944-BE8E-24B94231A174}", ["AppId", "BytesSent", "BytesRecvd"]),
}


class _Ese:
    MOVE_FIRST = -2147483648

    def __init__(self, path, tmpdir, tag):
        import ctypes
        from ctypes import byref, c_size_t, c_ulong
        self.ct = ctypes
        self.e = ctypes.WinDLL("esent.dll")
        self.path = path.encode("mbcs")
        with open(path, "rb") as f:
            hdr = f.read(512)
        self.page_size = struct.unpack_from("<I", hdr, 236)[0]
        self.inst, self.ses, self.db = c_size_t(0), c_size_t(0), c_ulong(0)
        os.makedirs(tmpdir, exist_ok=True)
        tmp = (tmpdir.rstrip("\\") + "\\").encode("mbcs")
        e = self.e
        self._chk(e.JetCreateInstanceA(byref(self.inst), tag.encode()), "CreateInstance")
        for pid, s, n in ((0, tmp, 0), (1, tmp, 0), (2, tmp, 0), (34, b"Off", 0), (64, None, self.page_size)):
            self._chk(e.JetSetSystemParameterA(byref(self.inst), c_size_t(0), c_ulong(pid), c_size_t(n), s), f"Param{pid}")
        self._chk(e.JetInit(byref(self.inst)), "Init")
        self._chk(e.JetBeginSessionA(self.inst, byref(self.ses), None, None), "BeginSession")
        self._chk(e.JetAttachDatabaseA(self.ses, self.path, c_ulong(1)), "Attach")
        self._chk(e.JetOpenDatabaseA(self.ses, self.path, None, byref(self.db), c_ulong(1)), "OpenDb")

    @staticmethod
    def _chk(rc, what):
        if rc < 0:
            raise OSError(f"ESE {what} rc={rc}")

    def _retrieve(self, t, colid, size=64):
        ct = self.ct
        buf = ct.create_string_buffer(size)
        act = ct.c_ulong(0)
        rc = self.e.JetRetrieveColumn(self.ses, t, ct.c_ulong(colid), buf, ct.c_ulong(size), ct.byref(act), ct.c_ulong(0), None)
        if rc == 1004:
            return None
        if rc == 1006:
            return self._retrieve(t, colid, act.value)
        if rc < 0:
            raise OSError(f"Retrieve rc={rc}")
        return buf.raw[: act.value]

    def _columns(self, t):
        ct = self.ct

        class COLLIST(ct.Structure):
            _fields_ = [("cbStruct", ct.c_ulong), ("tableid", ct.c_size_t), ("cRecord", ct.c_ulong)] + [
                (n, ct.c_ulong) for n in (
                    "idPres", "idName", "idColid", "idColtyp", "idCountry", "idLangid", "idCp", "idCollate",
                    "idCbMax", "idGrbit", "idDefault", "idBaseTable", "idBaseColumn", "idDefName")]
        cl = COLLIST()
        cl.cbStruct = ct.sizeof(cl)
        self._chk(self.e.JetGetTableColumnInfoA(self.ses, t, None, ct.byref(cl), ct.c_ulong(ct.sizeof(cl)), ct.c_ulong(1)), "ColInfo")
        cols = {}
        tt = ct.c_size_t(cl.tableid)
        rc = self.e.JetMove(self.ses, tt, ct.c_long(self.MOVE_FIRST), ct.c_ulong(0))
        while rc >= 0:
            name = self._retrieve(tt, cl.idName, 256).split(b"\0")[0].decode("mbcs")
            cid = struct.unpack("<I", self._retrieve(tt, cl.idColid))[0]
            typ = struct.unpack("<I", self._retrieve(tt, cl.idColtyp))[0]
            cols[name] = (cid, typ)
            rc = self.e.JetMove(self.ses, tt, ct.c_long(1), ct.c_ulong(0))
        self.e.JetCloseTable(self.ses, tt)
        return cols

    @staticmethod
    def _decode(raw, typ):
        if raw is None:
            return None
        if typ in (1, 2):
            return raw[0]
        fmt = {3: "<h", 17: "<H", 4: "<i", 14: "<I", 5: "<q", 15: "<q", 6: "<f", 7: "<d"}.get(typ)
        if fmt:
            return struct.unpack(fmt, raw)[0]
        if typ == 8:
            return dt.datetime(1899, 12, 30, tzinfo=dt.timezone.utc) + dt.timedelta(days=struct.unpack("<d", raw)[0])
        return raw

    def rows(self, name, want, max_rows=400000):
        ct = self.ct
        t = ct.c_size_t(0)
        if self.e.JetOpenTableA(self.ses, self.db, name.encode(), None, ct.c_ulong(0), ct.c_ulong(4), ct.byref(t)) < 0:
            return None
        cols = self._columns(t)
        use = [(c, cols[c]) for c in want if c in cols]
        out = []
        rc = self.e.JetMove(self.ses, t, ct.c_long(self.MOVE_FIRST), ct.c_ulong(0))
        while rc >= 0 and len(out) < max_rows:
            out.append({c: self._decode(self._retrieve(t, cid, 64), typ) for c, (cid, typ) in use})
            rc = self.e.JetMove(self.ses, t, ct.c_long(1), ct.c_ulong(0))
        self.e.JetCloseTable(self.ses, t)
        return out

    def close(self):
        ct = self.ct
        self.e.JetCloseDatabase(self.ses, self.db, ct.c_ulong(0))
        self.e.JetDetachDatabaseA(self.ses, self.path)
        self.e.JetEndSession(self.ses, ct.c_ulong(0))
        self.e.JetTerm(self.inst)


def _sid_str(b):
    try:
        rev, n = b[0], b[1]
        auth = int.from_bytes(b[2:8], "big")
        subs = struct.unpack_from("<%dI" % n, b, 8)
        return "S-%d-%d-" % (rev, auth) + "-".join(str(x) for x in subs)
    except Exception:
        return "?"


def _srum_name(blob, idtype):
    if blob is None:
        return None
    if idtype == 3:
        return _sid_str(blob)
    try:
        s = blob.decode("utf-16-le").rstrip("\0")
    except UnicodeDecodeError:
        return blob.hex()[:16]
    if "!" in s:
        parts = s.split("!")
        if s.startswith("!!") and len(parts) > 2:
            return parts[2]
        if len(parts) > 2 and parts[2]:
            return parts[0].split("_")[0] + "/" + parts[2]
    return _base(s) if "\\" in s else s


def _srum(h):
    """Copy (ladder incl. VSS when allowed), open with esent.dll, read idmap + 3 tables once per run."""
    def load():
        res = {"t": {}}
        t = time.perf_counter()
        allow = bool(h.l0.get("allow_vss", "--allow-vss" in sys.argv))
        dst, rung = h.copy_locked(r"C:\Windows\System32\sru\SRUDB.dat", "SRUDB.dat", allow_vss=allow)
        res["copy"] = {"rung": rung, "allow_vss": allow, "bytes": os.path.getsize(dst) if dst else None,
                       "ms": round((time.perf_counter() - t) * 1000, 1)}
        if not dst:
            return res
        t = time.perf_counter()
        tmp = os.path.join(h.scratch(), "esetmp")
        ese = _Ese(dst, tmp, "userscan_" + h.l0.get("run_id", "x"))
        res["open"] = {"page_size": ese.page_size, "ms": round((time.perf_counter() - t) * 1000, 1)}
        try:
            t = time.perf_counter()
            idm = ese.rows("SruDbIdMapTable", ["IdType", "IdIndex", "IdBlob"]) or []
            res["idmap"] = {r["IdIndex"]: _srum_name(r.get("IdBlob"), r.get("IdType")) for r in idm}
            users = {r["IdIndex"]: _srum_name(r.get("IdBlob"), 3) for r in idm if r.get("IdType") == 3}
            res["human"] = {k for k, v in users.items() if (v or "").startswith("S-1-5-21-")}
            res["t"]["idmap_ms"] = round((time.perf_counter() - t) * 1000, 1)
            res["idmap_rows"] = len(idm)
            for key, (tbl, cols) in _SRUM_TABLES.items():
                t = time.perf_counter()
                res[key] = ese.rows(tbl, cols)
                res["t"][key + "_ms"] = round((time.perf_counter() - t) * 1000, 1)
        finally:
            ese.close()
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                os.remove(dst)
            except OSError:
                pass
        return res
    return _cached(("srum", h.l0.get("run_id")), load)


@probe(id="srum.copy", level="L2", family="usage", tier="T0", collect="deep", gate="srum.present", timeout_ms=60000)
def srum_copy(h, facts):
    """SRUM copy ladder result: shutil, esentutl, esentutl /vss (only with --allow-vss)."""
    s = _srum(h)
    c = s.get("copy") or {}
    if "error" in s:
        return {"present": False, "error": s["error"], **c}
    return {"present": c.get("rung") not in (None, "failed"), **c}


@probe(id="srum.open", level="L2", family="usage", tier="T0", collect="deep", gate="srum.present", timeout_ms=60000)
def srum_open(h, facts):
    """esent.dll attach of the SRUM copy (read-only, recovery off)."""
    s = _srum(h)
    if "error" in s or "open" not in s:
        return {"present": False, "error": s.get("error")}
    return {"present": True, **s["open"]}


@probe(id="srum.idmap", level="L2", family="usage", tier="T0", collect="deep", gate="srum.present", timeout_ms=60000)
def srum_idmap(h, facts):
    """SRUM id map: app/user id counts; the join table for every SRUM extractor."""
    s = _srum(h)
    if "idmap" not in s:
        return {"present": False, "error": s.get("error")}
    return {"present": True, "ids": s["idmap_rows"], "human_sids": len(s["human"]), "ms": s["t"].get("idmap_ms")}


def _spread(grid, days, start, end, amount):
    total = (end - start).total_seconds()
    if total <= 0:
        grid[start.weekday()][start.hour] += amount
        days[start.date().isoformat()] += amount
        return
    cur = start
    while cur < end:
        nxt = min(end, cur.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1))
        share = amount * (nxt - cur).total_seconds() / total
        grid[cur.weekday()][cur.hour] += share
        days[cur.date().isoformat()] += share
        cur = nxt


_NOT_FOCUS = re.compile(r"(?i)^(LogonUI\.exe|.*LockApp\.exe|.*\.scr|csrss\.exe)$")


@probe(id="srum.app_timeline", level="L2", family="usage", tier="T1", collect="deep", gate="srum.present", timeout_ms=60000)
def srum_app_timeline(h, facts):
    """7 days of per-app focus and input seconds for human SIDs: screen time, input/focus ratio, hour grid."""
    s = _srum(h)
    rows = s.get("timeline")
    if not rows:
        return {"present": False, "error": s.get("error")}
    idmap, human = s["idmap"], s["human"]
    focus, inp, kbd, audio, idle = (collections.Counter() for _ in range(5))
    gf, gi = _grid(), _grid()
    df, di = collections.Counter(), collections.Counter()
    tmin = tmax = None
    for r in rows:
        if r.get("UserId") not in human:
            continue
        app = idmap.get(r.get("AppId")) or "<unmapped>"
        f = r.get("InFocusS") or 0
        u = r.get("UserInputS") or 0
        if _NOT_FOCUS.match(app):
            idle[app] += f
            f = 0
        focus[app] += f
        inp[app] += u
        kbd[app] += r.get("KeyboardInputS") or 0
        audio[app] += r.get("AudioOutS") or 0
        end = _ft(r.get("EndTime"))
        if end is None:
            continue
        start = end - dt.timedelta(milliseconds=r.get("DurationMS") or 0)
        tmin = start if tmin is None or start < tmin else tmin
        tmax = end if tmax is None or end > tmax else tmax
        if f:
            _spread(gf, df, start, end, f)
        if u:
            _spread(gi, di, start, end, u)
    tf, ti = sum(focus.values()), sum(inp.values())
    return {"present": True, "rows": len(rows), "span": [_iso(tmin), _iso(tmax)],
            "total_focus_h": round(tf / 3600, 1), "total_input_h": round(ti / 3600, 2),
            "input_focus_ratio": round(ti / tf, 3) if tf else None,
            "top_focus_h": [[a, round(v / 3600, 2)] for a, v in focus.most_common(15) if v],
            "top_input_min": [[a, round(v / 60, 1)] for a, v in inp.most_common(10) if v],
            "top_keyboard_min": [[a, round(v / 60, 1)] for a, v in kbd.most_common(5) if v],
            "top_audio_h": [[a, round(v / 3600, 2)] for a, v in audio.most_common(5) if v],
            "lock_screensaver_h": round(sum(idle.values()) / 3600, 1),
            "focus_grid_h": _grid_out(gf, 3600, 2), "input_grid_min": _grid_out(gi, 60, 1),
            "focus_h_per_day": {d: round(v / 3600, 1) for d, v in sorted(df.items())},
            "ms": s["t"].get("timeline_ms")}


@probe(id="srum.app_resource", level="L2", family="usage", tier="T1", collect="deep", gate="srum.present", timeout_ms=60000)
def srum_app_resource(h, facts):
    """30 days of hourly foreground cycles per app: machine-on hours grid, days used, top apps by share."""
    s = _srum(h)
    rows = s.get("resource")
    if not rows:
        return {"present": False, "error": s.get("error")}
    idmap, human = s["idmap"], s["human"]
    fg = collections.Counter()
    on_hours, user_hours = set(), set()
    for r in rows:
        ts = r.get("TimeStamp")
        lt = ts.astimezone().replace(minute=0, second=0, microsecond=0) if isinstance(ts, dt.datetime) else None
        if lt:
            on_hours.add(lt)
        if r.get("UserId") not in human:
            continue
        c = r.get("ForegroundCycleTime") or 0
        fg[idmap.get(r.get("AppId")) or "<unmapped>"] += c
        if lt and c > 0:
            user_hours.add(lt)
    g_on, g_user = _grid(), _grid()
    for x in on_hours:
        g_on[x.weekday()][x.hour] += 1
    for x in user_hours:
        g_user[x.weekday()][x.hour] += 1
    tot = max(1, sum(fg.values()))
    return {"present": True, "rows": len(rows),
            "span": [_iso(min(on_hours)) if on_hours else None, _iso(max(on_hours)) if on_hours else None],
            "record_hours": len(on_hours), "user_foreground_hours": len(user_hours),
            "days_with_records": len({x.date() for x in on_hours}),
            "days_with_user_foreground": len({x.date() for x in user_hours}),
            "top_foreground_pct": [[a, round(c / tot * 100, 1)] for a, c in fg.most_common(15)],
            "hours": [sum(g_user[d][hr] for d in range(7)) for hr in range(24)],
            "record_hours_grid": _grid_out(g_on, 1, 0), "user_hours_grid": _grid_out(g_user, 1, 0),
            "ms": s["t"].get("resource_ms")}


_SENSITIVE_NET = re.compile(r"(?i)(torrent|qbit|utorrent|bittorrent|transmission|deluge|tixati|vuze|frostwire|"
                            r"aria2|jdownloader|warp|wireguard|openvpn|nordvpn|expressvpn|protonvpn|mullvad|surfshark)")


@probe(id="srum.network_usage", level="L2", family="usage", tier="T1", collect="deep", gate="srum.present", timeout_ms=60000)
def srum_network_usage(h, facts):
    """30-day bytes per app; torrent and VPN clients are folded into one filtered total."""
    s = _srum(h)
    rows = s.get("network")
    if not rows:
        return {"present": False, "error": s.get("error")}
    net = collections.Counter()
    filtered = unmapped = 0
    for r in rows:
        app = s["idmap"].get(r.get("AppId"))
        b = (r.get("BytesSent") or 0) + (r.get("BytesRecvd") or 0)
        if not app:
            unmapped += b
        elif _SENSITIVE_NET.search(app):
            filtered += b
        else:
            net[app] += b
    return {"present": True, "rows": len(rows), "total_gb": round((sum(net.values()) + filtered + unmapped) / 1e9, 1),
            "unmapped_gb": round(unmapped / 1e9, 1),
            "filtered_sensitive_gb": round(filtered / 1e9, 1),
            "top_gb": [[a, round(b / 1e9, 2)] for a, b in net.most_common(10)], "ms": s["t"].get("network_ms")}


# ----------------------------------------------------------------- usage: event logs (PS sidecar)

_EVT_NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"


def _wevt(h, log, xpath, count, newest_first=False, timeout_ms=20000):
    """wevtutil qe as XML (no message rendering), parsed into ElementTree events. None when the query fails."""
    import xml.etree.ElementTree as ET
    args = ["wevtutil", "qe", log, "/q:" + xpath, "/f:xml", f"/c:{count}"]
    if newest_first:
        args.append("/rd:true")
    out = h.run(args, timeout_ms=timeout_ms, text=False)
    if out is None:
        return None
    try:
        return list(ET.fromstring("<r>" + re.sub(r"<\?xml[^>]*\?>", "", out) + "</r>"))
    except ET.ParseError:
        return None


def _evt_fields(ev):
    sysn = ev.find(_EVT_NS + "System")
    prov = sysn.find(_EVT_NS + "Provider").get("Name", "")
    eid = int(sysn.find(_EVT_NS + "EventID").text)
    ts = sysn.find(_EVT_NS + "TimeCreated").get("SystemTime", "")
    try:
        t = dt.datetime.fromisoformat(ts.rstrip("Z")[:26]).replace(tzinfo=dt.timezone.utc).astimezone()
    except ValueError:
        t = None
    data = {}
    ed = ev.find(_EVT_NS + "EventData")
    if ed is not None:
        for d in ed:
            data[d.get("Name")] = d.text
    return prov, eid, t, data


_POWER_PROV = {"Microsoft-Windows-Kernel-General", "Microsoft-Windows-Kernel-Power", "EventLog", "User32"}


@probe(id="eventlog.power_history", level="L2", family="usage", tier="T1", collect="extended", gate="boot.uptime",
       timeout_ms=20000)
def eventlog_power_history(h, facts):
    """System-log boots, shutdowns, sleep/resume; each Kernel-Power 41 classed crash / button_held / power_removed."""
    ids = (12, 13, 41, 42, 107, 506, 507, 1074, 6005, 6006, 6008)
    evs = _wevt(h, "System", "*[System[(" + " or ".join(f"EventID={i}" for i in ids) + ")]]", 20000)
    if evs is None:
        return {"present": False, "error": "wevtutil failed"}
    counts = collections.Counter()
    hours = [0] * 24
    kp = []
    now = dt.datetime.now().astimezone()
    boots30 = res30 = 0
    first = last = None
    for ev in evs:
        try:
            prov, eid, t, data = _evt_fields(ev)
        except (AttributeError, ValueError):
            continue
        if prov not in _POWER_PROV or t is None:
            continue
        counts[prov.replace("Microsoft-Windows-", "") + "/" + str(eid)] += 1
        first = t if first is None or t < first else first
        last = t if last is None or t > last else last
        recent = (now - t).days < 30
        if prov == "Microsoft-Windows-Kernel-General" and eid == 12:
            hours[t.hour] += 1
            boots30 += recent
        if prov == "Microsoft-Windows-Kernel-Power" and eid in (107, 507):
            res30 += recent
        if prov == "Microsoft-Windows-Kernel-Power" and eid == 41:
            bc = int(data.get("BugcheckCode") or 0)
            pb = int(data.get("PowerButtonTimestamp") or 0)
            cls = "crash" if bc else "button_held" if pb else "power_removed"
            kp.append({"t": _iso(t), "class": cls, "bugcheck": f"0x{bc:X}"})
    kp.sort(key=lambda x: x["t"])
    split = collections.Counter(x["class"] for x in kp)
    return {"present": bool(counts), "events": sum(counts.values()), "counts": dict(counts),
            "span": [_iso(first), _iso(last)], "boots_30d": boots30, "resumes_30d": res30,
            "os_start_hour_hist": hours,
            "kp41_split": {k: split.get(k, 0) for k in ("crash", "button_held", "power_removed")}, "kp41": kp[-40:]}


@probe(id="eventlog.security_logons", level="L2", family="usage", tier="T1", collect="deep", gate="boot.uptime",
       needs_admin=True, timeout_ms=20000)
def eventlog_security_logons(h, facts):
    """Security-log logons by type (2/11 console, 7 unlock, 10 RDP); 4800/4801 only if audited."""
    evs = _wevt(h, "Security", "*[System[(EventID=4624 or EventID=4800 or EventID=4801)]]", 5000, newest_first=True)
    if evs is None:
        return {"present": False, "error": "wevtutil failed"}
    by, types = collections.Counter(), collections.Counter()
    first = last = None
    for ev in evs:
        try:
            _prov, eid, t, data = _evt_fields(ev)
        except (AttributeError, ValueError):
            continue
        by[str(eid)] += 1
        if eid == 4624:
            types[str(data.get("LogonType"))] += 1
        if t:
            first = t if first is None or t < first else first
            last = t if last is None or t > last else last
    return {"present": bool(by), "events": sum(by.values()), "by_id": dict(by), "logon_types_4624": dict(types),
            "interactive_unlock": types["2"] + types["7"] + types["11"], "remote_interactive": types["10"],
            "span": [_iso(first), _iso(last)]}
