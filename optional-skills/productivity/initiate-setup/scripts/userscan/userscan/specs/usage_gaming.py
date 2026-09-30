"""Usage habits, gaming, media, hardware and health probes (Windows).

Registration only at import time. Routes follow reports/14-triage.json: registry and file reads in core,
the batched PowerShell sidecar for event logs and CIM, SRUM and PnP history in deep.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import re
import struct
import sys
import threading
import time

from userscan.registry import probe, ps_probe

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


# ================================================================= HARDWARE

@probe(id="hw.system", level="L1", family="hardware", tier="T0", collect="core")
def hw_system(h, facts):
    """SMBIOS system/board/BIOS from registry, visible RAM (GlobalMemoryStatusEx), fixed-drive free space."""
    import ctypes
    b = _reg_values(r"HKLM\HARDWARE\DESCRIPTION\System\BIOS", 100)

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]
    ms = MEMSTAT()
    ms.dwLength = ctypes.sizeof(ms)
    ram = None
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        ram = round(ms.total / 2 ** 30, 1)
    drives = []
    k32 = ctypes.windll.kernel32
    mask = k32.GetLogicalDrives()
    for i in range(26):
        if not mask & (1 << i):
            continue
        root = chr(65 + i) + ":\\"
        if k32.GetDriveTypeW(ctypes.c_wchar_p(root)) != 3:
            continue
        free, total = ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)
        if k32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(root), None, ctypes.byref(total), ctypes.byref(free)) and total.value:
            drives.append([root[:2], round(total.value / 1e9, 1), round(free.value / 1e9, 1),
                           round(100 * free.value / total.value, 1)])
    return {"present": True, "manufacturer": b.get("SystemManufacturer"), "product": b.get("SystemProductName"),
            "family": b.get("SystemFamily"), "board_vendor": b.get("BaseBoardManufacturer"),
            "board": b.get("BaseBoardProduct"), "bios_vendor": b.get("BIOSVendor"), "bios_version": b.get("BIOSVersion"),
            "bios_date": b.get("BIOSReleaseDate"), "visible_ram_gb": ram, "memory_load_pct": ms.dwMemoryLoad if ram else None,
            "fixed_drives_total_free_gb_pct": drives}


@probe(id="hw.cpu", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system")
def hw_cpu(h, facts):
    """CPU name, vendor, MHz and logical processor count from HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor."""
    base = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor"
    cores = _reg_keys(h, base, 1024)
    v = _reg_values(base + "\\0", 50)
    if not v:
        return {"present": False}
    return {"present": True, "name": (v.get("ProcessorNameString") or "").strip(), "vendor": v.get("VendorIdentifier"),
            "mhz": v.get("~MHz"), "logical_processors": len(cores), "native_arch": h.l0.get("native_arch")}


@probe(id="hw.gpu", level="L1", family="hardware", tier="T0", collect="core")
def hw_gpu(h, facts):
    """Display adapters from the Display class registry key: name, driver version, VRAM (qwMemorySize)."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    out = []
    for sk in _reg_keys(h, base, 50):
        if not sk.isdigit():
            continue
        v = _reg_values(base + "\\" + sk, 400)
        name = v.get("DriverDesc")
        if not name:
            continue
        mem = v.get("HardwareInformation.qwMemorySize")
        if mem is None:
            m2 = v.get("HardwareInformation.MemorySize")
            mem = int.from_bytes(m2[:8], "little") if isinstance(m2, (bytes, bytearray)) else m2
        out.append({"name": name, "driver": v.get("DriverVersion"), "driver_date": v.get("DriverDate"),
                    "vram_gb": round(mem / 2 ** 30, 1) if isinstance(mem, int) and mem > 0 else None,
                    "vendor": v.get("ProviderName")})
    real = [g for g in out if not re.search(r"Microsoft Basic|Remote Display|Virtual|Parsec|Meta Virtual", g["name"], re.I)]
    vr = [g["vram_gb"] for g in real if g["vram_gb"]]
    return {"present": True, "max_vram_gb": max(vr) if vr else None, "adapters": real, "other": len(out) - len(real)} if out else None


@probe(id="hw.power_plan", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system")
def hw_power_plan(h, facts):
    """Active power scheme GUID and friendly name from registry; Modern Standby flag."""
    known = {"381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
             "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver", "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate Performance"}
    base = r"HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
    g = h.reg(base, "ActivePowerScheme")
    if not g:
        return {"present": False}
    fname = h.reg(base + "\\" + g, "FriendlyName")
    if fname and fname.startswith("@"):
        fname = None
    ov = {"ded574b5-45a0-4f42-8737-46345c09c238": "Best performance", "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
          "00000000-0000-0000-0000-000000000000": "Balanced"}
    ac = h.reg(base, "ActiveOverlayAcPowerScheme")
    cs = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Power", "CsEnabled")
    return {"present": True, "guid": g, "name": known.get(g.lower()) or fname or "custom",
            "overlay_ac": ov.get((ac or "").lower(), ac), "modern_standby": cs}


@probe(id="hw.battery", level="L1", family="hardware", tier="T0", collect="core")
def hw_battery(h, facts):
    """Battery present (GetSystemPowerStatus + ACPI PNP0C0A); laptop vs desktop."""
    import ctypes

    class SPS(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = SPS()
    ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    acpi = bool(_reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\ACPI\PNP0C0A", 10))
    has = acpi or (ok and s.flag not in (128, 255))
    if not has:
        return None
    return {"present": True, "on_ac": s.ac == 1 if ok else None, "charge_pct": s.pct if ok and s.pct <= 100 else None,
            "battery_saver": bool(s.saver) if ok else None}


@probe(id="boot.fastboot", level="L1", family="hardware", tier="T0", collect="core")
def boot_fastboot(h, facts):
    """Fast Startup (HiberbootEnabled): needed to read 'shutdown' events correctly."""
    v = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power", "HiberbootEnabled")
    hib = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Power", "HibernateEnabled")
    if v is None and hib is None:
        return None
    return {"present": True, "fast_startup": v, "hibernate_enabled": hib}


@probe(id="tuning.afterburner", level="L1", family="hardware", tier="T0", collect="core")
def tuning_afterburner(h, facts):
    """MSI Afterburner: autostart, fan curve, logging, per-GPU profiles (power limit, offsets, VF curve)."""
    ab = os.path.join(_paths()["PF86"], "MSI Afterburner")
    if not _ex(ab):
        return None
    out = {"present": True}
    t = _read(os.path.join(ab, "Profiles", "MSIAfterburner.cfg"), 500_000)
    if t:
        def k(n):
            m = re.search(rf"^{n}=(.*)$", t, re.M)
            return m.group(1).strip() if m else None
        out.update(start_with_windows=k("StartWithWindows"), sw_fan_curve=k("SwAutoFanControl"),
                   hw_log=k("EnableHwMonitoringLog") or k("EnableLog"))
    gpus = []
    for f in h.list_dir(os.path.join(ab, "Profiles"), 100):
        if f.startswith("VEN_") and not f.startswith("VEN_0000"):
            tt = _read(os.path.join(ab, "Profiles", f), 500_000) or ""

            def g(n):
                m = re.search(rf"^{n}=(\S+)", tt, re.M)
                return m.group(1) if m else None
            gpus.append({"vendor": {"10DE": "NVIDIA", "8086": "Intel", "1002": "AMD"}.get(f[4:8], f[4:8]), "dev": f[13:17],
                         "power_limit": g("PowerLimit"), "core_offset": g("CoreClkBoost"), "fan_mode": g("FanMode"),
                         "vf_curve": bool(g("VFCurve"))})
    out["gpu_profiles"] = gpus
    out["version"] = (_unmatch(h, r"MSI Afterburner") or [{}])[0].get("version")
    return out


@probe(id="tuning.rtss", level="L1", family="hardware", tier="T0", collect="core")
def tuning_rtss(h, facts):
    """RivaTuner Statistics Server presence and per-game profile count."""
    rt = os.path.join(_paths()["PF86"], "RivaTuner Statistics Server")
    if not _ex(rt):
        return None
    prof = [f for f in h.list_dir(os.path.join(rt, "Profiles"), 2000) if f.endswith(".cfg") and f.lower() != "global"]
    return {"present": True, "profiles": len(prof)}


@probe(id="periph.rgb_suites", level="L2", family="hardware", tier="T0", collect="core", gate="hw.system")
def periph_rgb_suites(h, facts):
    """RGB / peripheral vendor suites from the uninstall index (Razer, Logi, Corsair, ASUS Aura, OpenRGB, ...)."""
    hits = _unmatch(h, r"Razer|Logi|G HUB|SteelSeries|Corsair|iCUE|HyperX|NGENUITY|Patriot Viper|Armoury|\bAura\b|OpenRGB|"
                       r"SignalRGB|Wallpaper Engine|DS4Windows|8BitDo|Xbox Accessories|Stream Deck|Elgato|Wooting|NZXT")
    names = sorted({re.sub(r"\s+[\d.]+$", "", x["name"]) for x in hits})
    return {"present": bool(names), "count": len(hits), "suites": names[:25]}


@probe(id="hw.audio", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.system")
def hw_audio(h, facts):
    """Audio endpoints from MMDevices registry: active render/capture counts and adapter interface names."""
    base = r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio"
    out = {"present": False}
    for kind in ("Render", "Capture"):
        keys = _reg_keys(h, base + "\\" + kind, 300)
        active = []
        for k in keys:
            st = h.reg(base + "\\" + kind + "\\" + k, "DeviceState")
            if st == 1:
                iface = h.reg(base + "\\" + kind + "\\" + k + "\\Properties", "{b3f8fa53-0004-438e-9003-51a46e139bfc},6")
                active.append(re.sub(r"(?i)\b[A-Z][a-z]+'s\b", "<name>'s", iface or "?"))
        out[kind.lower()] = {"total": len(keys), "active": len(active), "active_ifaces": sorted(set(active))[:12]}
        out["present"] = out["present"] or bool(keys)
    return out


@probe(id="hw.bluetooth", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.system")
def hw_bluetooth(h, facts):
    """Paired Bluetooth devices by class (gamepad/audio/HID) from BTHPORT registry; device names not emitted."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Services\BTHPORT\Parameters\Devices"
    keys = _reg_keys(h, base, 500)
    major = {0: "Misc", 1: "Computer", 2: "Phone", 3: "LAN", 4: "Audio/Video", 5: "Peripheral", 6: "Imaging",
             7: "Wearable", 8: "Toy", 9: "Health", 31: "Uncategorized"}
    cls = collections.Counter()
    for k in keys:
        cod = h.reg(base + "\\" + k, "COD") or 0
        if isinstance(cod, bytes):
            cod = int.from_bytes(cod[:4], "little")
        maj, mn = (cod >> 8) & 0x1F, (cod >> 2) & 0x3F
        sub = None
        if maj == 5:
            sub = {0x10: "Keyboard", 0x20: "Mouse", 0x30: "Keyboard+Mouse", 0x01: "Joystick", 0x02: "Gamepad"}.get(mn & 0x33)
        elif maj == 4:
            sub = {1: "Headset", 2: "Handsfree", 4: "Microphone", 5: "Loudspeaker", 6: "Headphones"}.get(mn)
        if not sub:
            raw = h.reg(base + "\\" + k, "Name")
            nm = raw.decode("utf-8", "replace").strip("\0") if isinstance(raw, bytes) else ""
            for rx, lbl in ((r"(?i)controller|gamepad|dualsense|dualshock|joy-?con", "Gamepad"),
                            (r"(?i)mouse|mx master|deathadder|mx anywhere", "Mouse"), (r"(?i)keyboard|keys\b", "Keyboard"),
                            (r"(?i)WH-|WF-|buds|airpods|headphone|headset", "Headphones"), (r"(?i)speaker|soundbar|kanto", "Speaker")):
                if re.search(rx, nm):
                    sub = lbl + "(name)"
                    break
        cls[major.get(maj, str(maj)) + ("/" + sub if sub else "")] += 1
    le = sum(len(_reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\BTHLE\\" + d, 50))
             for d in _reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\BTHLE", 300))
    if not keys and not le:
        return {"present": False}
    return {"present": True, "classic_paired": len(keys), "ble_instances": le, "by_class": dict(cls)}


@probe(id="hw.usb_history", level="L2", family="hardware", tier="T1", collect="extended", gate="hw.system")
def hw_usb_history(h, facts):
    """USB device ids/instances ever seen and USB storage count (Enum\\USB, USBSTOR); top vendor ids only."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Enum\USB"
    ids = _reg_keys(h, base, 2000)
    inst = 0
    vids = collections.Counter()
    for i in ids:
        inst += len(_reg_keys(h, base + "\\" + i, 200))
        m = re.search(r"VID_([0-9A-Fa-f]{4})", i)
        if m:
            vids[m.group(1).upper()] += 1
    stor = sum(len(_reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\USBSTOR\\" + d, 100))
               for d in _reg_keys(h, r"HKLM\SYSTEM\CurrentControlSet\Enum\USBSTOR", 500))
    vend = {"046D": "Logitech", "1532": "Razer", "1B1C": "Corsair", "1038": "SteelSeries", "0B05": "ASUS", "045E": "Microsoft",
            "054C": "Sony", "057E": "Nintendo", "28DE": "Valve", "3434": "Keychron", "0951": "Kingston/HyperX", "03F0": "HP/HyperX",
            "8087": "Intel", "0BDA": "Realtek", "0955": "NVIDIA", "1462": "MSI", "05AC": "Apple", "04E8": "Samsung", "0781": "SanDisk"}
    if not ids:
        return {"present": False}
    return {"present": True, "device_ids_ever": len(ids), "instances_ever": inst, "usbstor_instances": stor,
            "top_vendors": [[v, vend.get(v), n] for v, n in vids.most_common(12)]}


@probe(id="hw.monitor_mode", level="L2", family="hardware", tier="T0", collect="extended", gate="hw.system")
def hw_monitor_mode(h, facts):
    """Current resolution/refresh per display (EnumDisplaySettings); falls back to GraphicsDrivers\\Configuration."""
    import ctypes
    from ctypes import wintypes as W

    class DEVMODEW(ctypes.Structure):
        _fields_ = [("dmDeviceName", ctypes.c_wchar * 32), ("dmSpecVersion", W.WORD), ("dmDriverVersion", W.WORD),
                    ("dmSize", W.WORD), ("dmDriverExtra", W.WORD), ("dmFields", W.DWORD), ("dmPositionX", W.LONG),
                    ("dmPositionY", W.LONG), ("dmDisplayOrientation", W.DWORD), ("dmDisplayFixedOutput", W.DWORD),
                    ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                    ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short), ("dmFormName", ctypes.c_wchar * 32),
                    ("dmLogPixels", W.WORD), ("dmBitsPerPel", W.DWORD), ("dmPelsWidth", W.DWORD), ("dmPelsHeight", W.DWORD),
                    ("dmDisplayFlags", W.DWORD), ("dmDisplayFrequency", W.DWORD), ("dmICMMethod", W.DWORD),
                    ("dmICMIntent", W.DWORD), ("dmMediaType", W.DWORD), ("dmDitherType", W.DWORD), ("dmReserved1", W.DWORD),
                    ("dmReserved2", W.DWORD), ("dmPanningWidth", W.DWORD), ("dmPanningHeight", W.DWORD)]

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [("cb", W.DWORD), ("DeviceName", ctypes.c_wchar * 32), ("DeviceString", ctypes.c_wchar * 128),
                    ("StateFlags", W.DWORD), ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128)]
    u32 = ctypes.windll.user32
    modes = []
    i = 0
    while i < 16:
        dd = DISPLAY_DEVICEW()
        dd.cb = ctypes.sizeof(dd)
        if not u32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            break
        i += 1
        if not dd.StateFlags & 1:
            continue
        dm = DEVMODEW()
        dm.dmSize = ctypes.sizeof(dm)
        if u32.EnumDisplaySettingsW(dd.DeviceName, -1, ctypes.byref(dm)):
            modes.append({"w": dm.dmPelsWidth, "h": dm.dmPelsHeight, "hz": dm.dmDisplayFrequency, "adapter": dd.DeviceString})
    if modes:
        return {"present": True, "via": "EnumDisplaySettings", "displays": modes}
    base = r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\Configuration"
    best, best_ts = None, -1
    for cfg in _reg_keys(h, base, 200):
        ts = h.reg(base + "\\" + cfg, "Timestamp") or 0
        if isinstance(ts, int) and ts > best_ts:
            best, best_ts = cfg, ts
    if not best:
        return {"present": False}
    out = []
    for t in _reg_keys(h, base + "\\" + best, 20):
        for m in _reg_keys(h, base + "\\" + best + "\\" + t, 20):
            v = _reg_values(base + "\\" + best + "\\" + t + "\\" + m, 100)
            if v.get("PrimSurfSize.cx"):
                num, den = v.get("VSyncFreq.Numerator"), v.get("VSyncFreq.Denominator")
                out.append({"w": v["PrimSurfSize.cx"], "h": v.get("PrimSurfSize.cy"),
                            "hz": round(num / den, 1) if num and den else None})
    return {"present": bool(out), "via": "GraphicsDrivers\\Configuration", "committed": _iso(_ft(best_ts)), "displays": out}


@probe(id="hw.nvidia_smi", level="L2", family="hardware", tier="T0", collect="extended", gate="hw.gpu")
def hw_nvidia_smi(h, facts):
    """nvidia-smi query: VRAM, power draw/limit, temperature, pstate, driver."""
    P = _paths()
    exe = next((p for p in (os.path.join(P["WIN"], "System32", "nvidia-smi.exe"),
                            os.path.join(P["PF"], "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")) if _ex(p)), None)
    if not exe:
        return {"present": False}
    fields = "name,memory.total,power.draw,power.limit,power.default_limit,power.max_limit,temperature.gpu,pstate,driver_version,fan.speed"
    out = h.run([exe, "--query-gpu=" + fields, "--format=csv,noheader,nounits"], timeout_ms=5000, text=False)
    if not out:
        return {"present": False, "exe": True}
    gpus = []
    for line in out.strip().splitlines()[:8]:
        vals = [x.strip() for x in line.split(",")]
        gpus.append({k: (None if v in ("[N/A]", "N/A", "[Not Supported]") else v) for k, v in zip(fields.split(","), vals)})
    vram = None
    try:
        vram = round(max(float(g["memory.total"]) for g in gpus if g.get("memory.total")) / 1024, 1)
    except ValueError:
        pass
    return {"present": bool(gpus), "vram_gb": vram, "gpus": gpus}


ps_probe("hw.monitors", r"""
try {
  function U($a) { if ($a) { (($a | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ }) -join '').Trim() } }
  $ids = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID -ErrorAction SilentlyContinue | ForEach-Object {
    [ordered]@{ mfr = (U $_.ManufacturerName); product = (U $_.ProductCodeID); name = (U $_.UserFriendlyName); year = $_.YearOfManufacture; active = $_.Active } })
  $size = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.MaxHorizontalImageSize) { [math]::Round([math]::Sqrt([math]::Pow($_.MaxHorizontalImageSize,2)+[math]::Pow($_.MaxVerticalImageSize,2))/2.54,1) } })
  $tech = @(Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorConnectionParams -ErrorAction SilentlyContinue | ForEach-Object {
    switch ([int64]$_.VideoOutputTechnology) { 0 {'VGA'} 4 {'DVI'} 5 {'HDMI'} 10 {'DP'} 11 {'DP-embedded'} 2147483648 {'Internal'} default { [string]$_.VideoOutputTechnology } } })
  [ordered]@{ present = ($ids.Count -gt 0); count = $ids.Count; monitors = $ids; diag_in = $size; outputs = $tech }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="extended", gate="hw.system", timeout_ms=6000,
         doc="Connected monitors from EDID (WmiMonitorID): maker, model, year, size, connection; no serials.")

ps_probe("hw.ram", r"""
try {
  $m = @(Get-CimInstance Win32_PhysicalMemory -ErrorAction SilentlyContinue)
  $a = @(Get-CimInstance Win32_PhysicalMemoryArray -ErrorAction SilentlyContinue)
  [ordered]@{ present = ($m.Count -gt 0); modules = $m.Count
    total_gb = [math]::Round((($m | Measure-Object Capacity -Sum).Sum) / 1GB, 1)
    slots = ($a | Measure-Object MemoryDevices -Sum).Sum
    speed = @($m | ForEach-Object { $_.Speed } | Select-Object -Unique)
    configured = @($m | ForEach-Object { $_.ConfiguredClockSpeed } | Select-Object -Unique)
    rated_mhz = ($m | Measure-Object Speed -Minimum).Minimum; configured_mhz = ($m | Measure-Object ConfiguredClockSpeed -Minimum).Minimum
    smbios_type = @($m | ForEach-Object { $_.SMBIOSMemoryType } | Select-Object -Unique)
    mfr = @($m | ForEach-Object { ([string]$_.Manufacturer).Trim() } | Select-Object -Unique)
    part = @($m | ForEach-Object { ([string]$_.PartNumber).Trim() } | Select-Object -Unique) }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="extended", gate="hw.system", timeout_ms=6000,
         doc="Installed RAM modules, slots, rated vs configured speed (Win32_PhysicalMemory).")

ps_probe("hw.disks", r"""
try {
  $pd = @(Get-PhysicalDisk -ErrorAction SilentlyContinue | ForEach-Object {
    $rc = $_ | Get-StorageReliabilityCounter -ErrorAction SilentlyContinue
    [ordered]@{ name = $_.FriendlyName; media = [string]$_.MediaType; bus = [string]$_.BusType; size_gb = [math]::Round($_.Size/1GB)
      health = [string]$_.HealthStatus; wear_pct = $rc.Wear; temp_c = $rc.Temperature; temp_max_c = $rc.TemperatureMax
      power_on_h = $rc.PowerOnHours; read_err_uncorrected = $rc.ReadErrorsUncorrected; write_err_uncorrected = $rc.WriteErrorsUncorrected } })
  $vol = @(Get-Volume -ErrorAction SilentlyContinue | Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' -and $_.Size } | ForEach-Object {
    [ordered]@{ letter = [string]$_.DriveLetter; size_gb = [math]::Round($_.Size/1GB,1); free_gb = [math]::Round($_.SizeRemaining/1GB,1)
      free_pct = [math]::Round(100*$_.SizeRemaining/$_.Size,1) } })
  [ordered]@{ present = ($pd.Count -gt 0); count = $pd.Count; disks = $pd; volumes = $vol }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="deep", gate="hw.system", timeout_ms=15000,
         doc="Physical disks with health, wear, temperature and error counters (Get-PhysicalDisk).")

ps_probe("hw.gpu_history", r"""
try {
  $g = @(Get-PnpDevice -Class Display -ErrorAction SilentlyContinue | ForEach-Object {
    $fi = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_FirstInstallDate -ErrorAction SilentlyContinue).Data
    $la = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_LastArrivalDate -ErrorAction SilentlyContinue).Data
    [ordered]@{ name = $_.FriendlyName; pci = (($_.InstanceId -split '\\')[1] -replace '&REV.*$',''); present = $_.Present
      first = $(if ($fi) { ([datetime]$fi).ToString('s') }); last = $(if ($la) { ([datetime]$la).ToString('s') }) } })
  [ordered]@{ present = ($g.Count -gt 0); ever = $g.Count; now = @($g | Where-Object { $_.present }).Count
    swaps = @($g | Where-Object { -not $_.present -and $_.name -notmatch 'Basic|Remote|Virtual' }).Count; adapters = $g }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="deep", gate="hw.gpu", timeout_ms=15000,
         doc="Every display adapter ever installed with first-install/last-arrival dates (GPU swaps).")

ps_probe("hw.monitor_history", r"""
try {
  $h = @(Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.InstanceId -match '^DISPLAY\\' -and $_.InstanceId -notmatch 'DEFAULT_MONITOR' } | ForEach-Object {
    $fi = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_FirstInstallDate -ErrorAction SilentlyContinue).Data
    $la = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_LastArrivalDate -ErrorAction SilentlyContinue).Data
    [pscustomobject]@{ pnp = ($_.InstanceId -split '\\')[1]; present = [bool]$_.Present; first = $fi; last = $la } } |
    Group-Object pnp | ForEach-Object { $f = @($_.Group | Where-Object first | Sort-Object first); $l = @($_.Group | Where-Object last | Sort-Object last -Descending)
      [ordered]@{ pnp = $_.Name; present = [bool]($_.Group | Where-Object present)
        first = $(if ($f) { ([datetime]$f[0].first).ToString('s') }); last = $(if ($l) { ([datetime]$l[0].last).ToString('s') }) } })
  [ordered]@{ present = ($h.Count -gt 0); distinct_monitors_ever = $h.Count; changes = @($h | Where-Object { -not $_.present }).Count; monitors = $h }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T0", collect="deep", gate="hw.system", timeout_ms=15000,
         doc="Every monitor model (PnP id) ever connected with first/last dates.")

ps_probe("hw.peripherals", r"""
try {
  $cls = 'Keyboard','Mouse','HIDClass','Camera','Image','Biometric','XnaComposite','Printer','MEDIA','SmartCardReader'
  $pp = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $cls -contains $_.Class })
  $by = @{}; foreach ($d in $pp) { $by[[string]$d.Class] = 1 + [int]$by[[string]$d.Class] }
  $vids = @{}; foreach ($d in $pp) { if ($d.InstanceId -match 'VID_([0-9A-F]{4})') { $vids[$matches[1]] = 1 + [int]$vids[$matches[1]] } }
  $pads = @($pp | Where-Object { $_.Class -eq 'XnaComposite' -or $_.FriendlyName -match 'game ?controller|gamepad|xbox|dualsense|dualshock' }).Count
  $cams = @($pp | Where-Object { $_.Class -in 'Camera','Image' }).Count
  [ordered]@{ present = ($pp.Count -gt 0); by_class = $by; hid_vids = $vids; game_controllers = $pads; cameras = $cams }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="hardware", tier="T1", collect="deep", gate="hw.system", timeout_ms=15000,
         doc="Present peripherals by PnP class, USB vendor ids, controller and camera counts.")


def _xml_local(path):
    import xml.etree.ElementTree as ET
    try:
        return ET.parse(path).getroot()
    except Exception:
        return None


def _ln(el):
    return el.tag.rsplit("}", 1)[-1]


@probe(id="hw.battery_report", level="L2", family="hardware", tier="T1", collect="deep", gate="hw.battery", timeout_ms=15000)
def hw_battery_report(h, facts):
    """powercfg /batteryreport (XML into scratch): design vs full capacity, cycle count, recent AC/DC usage."""
    out = os.path.join(h.scratch(), "battery.xml")
    h.run(["powercfg", "/batteryreport", "/xml", "/output", out], timeout_ms=15000, text=False)
    root = _xml_local(out)
    if root is None:
        return {"present": False}
    bat = next((e for e in root.iter() if _ln(e) == "Battery"), None)
    if bat is None:
        return {"present": False}
    kv = {_ln(c): (c.text or "").strip() for c in bat}
    design, full = int(kv.get("DesignCapacity") or 0), int(kv.get("FullChargeCapacity") or 0)
    usage = [e for e in root.iter() if _ln(e) == "UsageEntry"]
    ac = sum(1 for e in usage if e.get("Ac") == "1")
    try:
        os.remove(out)
    except OSError:
        pass
    return {"present": True, "design_mwh": design, "full_mwh": full, "health_pct": round(100 * full / design, 1) if design else None,
            "cycles": kv.get("CycleCount"), "chemistry": kv.get("Chemistry"), "recent_usage_entries": len(usage),
            "recent_ac": ac, "recent_dc": len(usage) - ac}


@probe(id="hw.sleep_study", level="L2", family="hardware", tier="T1", collect="deep", gate="hw.battery", timeout_ms=20000)
def hw_sleep_study(h, facts):
    """powercfg /sleepstudy (XML into scratch): Modern Standby sessions, low-power share, drain."""
    out = os.path.join(h.scratch(), "sleepstudy.xml")
    h.run(["powercfg", "/sleepstudy", "/xml", "/output", out], timeout_ms=20000, text=False)
    root = _xml_local(out)
    if root is None:
        return {"present": False}
    inst = [e for e in root.iter() if _ln(e) == "ScenarioInstance"]
    types = collections.Counter(e.get("Type") or "?" for e in inst)
    lp = []
    for e in inst:
        d, l = int(e.get("Duration") or 0), int(e.get("LowPowerStateTime") or 0)
        if d:
            lp.append(100 * l / d)
    try:
        os.remove(out)
    except OSError:
        pass
    return {"present": bool(inst), "sessions": len(inst), "types": dict(types),
            "lowpower_pct_median": round(sorted(lp)[len(lp) // 2], 1) if lp else None,
            "lowpower_pct_min": round(min(lp), 1) if lp else None}


# ================================================================= HEALTH

@probe(id="health.dumps", level="L1", family="health", tier="T1", collect="core")
def health_dumps(h, facts):
    """Crash dump config (CrashControl) and dump files: Minidump count/newest, MEMORY.DMP, LiveKernelReports."""
    P = _paths()
    cc = _reg_values(r"HKLM\SYSTEM\CurrentControlSet\Control\CrashControl", 100)
    mini, newest, mini30 = 0, None, 0
    try:
        with os.scandir(os.path.join(P["WIN"], "Minidump")) as it:
            for e in it:
                if mini >= 1000:
                    break
                if e.is_file():
                    mini += 1
                    m = e.stat().st_mtime
                    newest = m if newest is None or m > newest else newest
                    mini30 += time.time() - m <= 30 * 86400
    except OSError:
        pass
    lkr = {}
    lroot = os.path.join(P["WIN"], "LiveKernelReports")
    for d in h.list_dir(lroot, 50):
        p = os.path.join(lroot, d)
        if os.path.isdir(p):
            n = sum(1 for f in h.list_dir(p, 500) if f.lower().endswith(".dmp"))
            if n:
                lkr[d] = n
        elif d.lower().endswith(".dmp"):
            lkr["root"] = lkr.get("root", 0) + 1
    md = h.meta(os.path.join(P["WIN"], "MEMORY.DMP"))
    if not cc and not mini and not lkr:
        return None
    return {"present": True, "crash_dump_enabled": cc.get("CrashDumpEnabled"), "auto_reboot": cc.get("AutoReboot"),
            "minidumps": mini, "count_30d": mini30, "minidump_newest": _iso(newest), "memory_dmp_bytes": md.get("bytes"),
            "live_kernel_reports": lkr}


_WER_NOISE = re.compile(r"(?i)^(svchost\.exe|TrustedInstaller\.exe|TiWorker\.exe|MoUsoCoreWorker\.exe|WinStore\.App\.exe|"
                        r"Microsoft\.WindowsStore.*|MicrosoftEdgeUpdate.*|MicrosoftEdge_X64_.*|setup\.exe|MsiExec\.exe|"
                        r"wermgr\.exe|StoreDesktopExtension.*|backgroundTaskHost\.exe|RuntimeBroker\.exe|SearchHost\.exe|"
                        r"MpSigStub\.exe|MsMpEng\.exe|WindowsPackageManagerServer\.exe)$")


@probe(id="health.wer", level="L2", family="health", tier="T1", collect="extended", gate="health.dumps")
def health_wer(h, facts):
    """WER report dirs: crashes/hangs per app (Report.wer AppName), kernel reports; svchost/Store noise split out."""
    roots = [os.path.join(_paths()["PD"], "Microsoft", "Windows", "WER", r) for r in ("ReportArchive", "ReportQueue")]
    types = collections.Counter()
    apps = collections.Counter()
    last = {}
    kernel = collections.Counter()
    noise = total = 0
    oldest = newest = None
    for rt in roots:
        for n in h.list_dir(rt, 1000):
            p = os.path.join(rt, n)
            total += 1
            typ = n.split("_", 1)[0]
            types[typ] += 1
            m = _mtime(p)
            if m:
                oldest = m if oldest is None or m < oldest else oldest
                newest = m if newest is None or m > newest else newest
            app, evt = None, None
            try:
                with open(os.path.join(p, "Report.wer"), "rb") as f:
                    txt = f.read(200_000).decode("utf-16", "replace")
                ma = re.search(r"^AppPath=([^\r\n]+)", txt, re.M) or re.search(r"^AppName=([^\r\n]+)", txt, re.M)
                me = re.search(r"^EventType=([^\r\n]+)", txt, re.M)
                app, evt = (_base(ma.group(1).strip()) if ma else None), (me.group(1).strip() if me else None)
            except OSError:
                pass
            if re.search(r"Kernel|BlueScreen", typ, re.I) or (evt and re.search(r"LiveKernel|BlueScreen", evt, re.I)):
                kernel[evt or typ] += 1
                continue
            if re.search(r"AppCrash|AppHang|BEX|Critical", typ, re.I):
                m2 = re.match(r"^[^_]+_([^_]+?\.exe)_", n, re.I)
                if m2:
                    app = m2.group(1)
                app = app or "?"
                if _WER_NOISE.match(app):
                    noise += 1
                    continue
                apps[app] += 1
                last[app] = max(last.get(app) or 0, m or 0)
    if not total:
        return {"present": False}
    types = collections.Counter({k: v for k, v in types.items() if not re.fullmatch(r"[0-9a-fA-F-]{36}", k)})
    return {"present": True, "reports": total, "by_type": dict(types.most_common(8)), "system_noise_crashes": noise,
            "max_repeat": apps.most_common(1)[0][1] if apps else 0,
            "crashes_by_app": [[a, c, _iso(last.get(a))] for a, c in apps.most_common(12)], "kernel_reports": dict(kernel),
            "oldest": _iso(oldest), "newest": _iso(newest)}


@probe(id="health.whea_gpu", level="L2", family="health", tier="T1", collect="extended", gate="hw.system", timeout_ms=20000)
def health_whea_gpu(h, facts):
    """Hardware error events: WHEA, nvlddmkm, display TDR 4101, bugcheck 1001 codes, storage warnings (wevtutil)."""
    provs = ("Microsoft-Windows-WHEA-Logger", "nvlddmkm", "Display", "Microsoft-Windows-WER-SystemErrorReporting",
             "disk", "Ntfs", "stornvme")
    xp = "*[System[Provider[" + " or ".join(f"@Name='{p}'" for p in provs) + "]]]"
    evs = _wevt(h, "System", xp, 3000, newest_first=True)
    if evs is None:
        return {"present": False, "error": "wevtutil failed"}
    grp = {k: {"count": 0, "by_id": collections.Counter(), "newest": None} for k in ("whea", "nvlddmkm", "tdr_4101", "storage_warn_err")}
    bug = []
    now = dt.datetime.now().astimezone()
    c30 = 0
    for ev in evs:
        try:
            prov, eid, t, data = _evt_fields(ev)
            lvl = int(ev.find(_EVT_NS + "System").find(_EVT_NS + "Level").text or 4)
        except (AttributeError, ValueError, TypeError):
            continue
        key = None
        if prov == "Microsoft-Windows-WHEA-Logger":
            key = "whea"
        elif prov == "nvlddmkm":
            key = "nvlddmkm"
        elif prov == "Display" and eid == 4101:
            key = "tdr_4101"
        elif prov in ("disk", "Ntfs", "stornvme") and 1 <= lvl <= 3:
            key = "storage_warn_err"
        elif prov == "Microsoft-Windows-WER-SystemErrorReporting" and eid == 1001:
            m = re.search(r"0x[0-9a-fA-F]+", " ".join(v or "" for v in data.values()))
            bug.append({"t": _iso(t), "code": m.group(0) if m else None})
            c30 += bool(t and (now - t).days < 30)
            continue
        if not key:
            continue
        g = grp[key]
        g["count"] += 1
        g["by_id"][str(eid)] += 1
        if t and (g["newest"] is None or _iso(t) > g["newest"]):
            g["newest"] = _iso(t)
        if key != "storage_warn_err" and t and (now - t).days < 30:
            c30 += 1
    for g in grp.values():
        g["by_id"] = dict(g["by_id"])
    return {"present": True, "count_30d": c30, **grp, "bugchecks": bug[:20]}


ps_probe("health.reliability", r"""
try {
  $r = @(Get-CimInstance Win32_ReliabilityRecords -ErrorAction SilentlyContinue)
  $m = @(Get-CimInstance Win32_ReliabilityStabilityMetrics -ErrorAction SilentlyContinue | Sort-Object TimeGenerated)
  $s = @($r | Sort-Object TimeGenerated)
  function T($grp) { $o = [ordered]@{}; foreach ($x in @($grp | Sort-Object Count -Descending | Select-Object -First 8)) { $o[[string]$x.Name] = $x.Count }; $o }
  [ordered]@{ present = ($r.Count -gt 0); records = $r.Count
    oldest = $(if ($s) { $s[0].TimeGenerated.ToString('s') }); newest = $(if ($s) { $s[-1].TimeGenerated.ToString('s') })
    by_source = (T ($r | Group-Object SourceName)); top_products = (T ($r | Where-Object ProductName | Group-Object ProductName))
    stability_latest = $(if ($m) { [math]::Round($m[-1].SystemStabilityIndex, 2) })
    stability_min = $(if ($m) { [math]::Round(($m | Measure-Object SystemStabilityIndex -Minimum).Minimum, 2) }) }
} catch { [ordered]@{ present = $false; error = $_.Exception.Message } }
""", family="health", tier="T1", collect="deep", gate="hw.system", timeout_ms=20000,
         doc="Reliability Monitor records by source/product and the stability index (latest, min).")
