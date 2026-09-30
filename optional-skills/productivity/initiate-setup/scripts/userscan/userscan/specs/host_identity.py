"""Host, identity, install age, locale, shell prefs, security posture and network probes (Windows).

Signal ids follow reports/14-triage.json. Every route here is registry, stat or a single small spawn,
except the PowerShell sidecar probes (CIM-only signals) and the deep tier.
No I/O at import time.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import sys

from userscan.registry import probe, ps_probe

# ---------------------------------------------------------------- helpers

_NOISE_RX = re.compile(r"(^|[\\/])(hn-e2e|ns960|ns923[^\\/]*|lhm|shots|user-insights-lab|hermes-[^\\/]*)([\\/]|$)", re.I)
_NT_CV = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion"
_NOW = None


def _now():
    global _NOW
    if _NOW is None:
        _NOW = _dt.datetime.now(_dt.timezone.utc)
    return _NOW


def _is_noise(path: str) -> bool:
    return bool(path) and bool(_NOISE_RX.search(path))


def _root(path: str):
    import winreg
    head, _, sub = path.partition("\\")
    return {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER,
            "HKU": winreg.HKEY_USERS, "HKCR": winreg.HKEY_CLASSES_ROOT}.get(head), sub


def _vals(path: str, limit: int = 3000):
    """All values of a key as {name: data}, or None if the key is missing."""
    if sys.platform != "win32":
        return None
    import winreg
    root, sub = _root(path)
    try:
        k = winreg.OpenKey(root, sub)
    except OSError:
        return None
    out = {}
    with k:
        for i in range(limit):
            try:
                n, v, _t = winreg.EnumValue(k, i)
            except OSError:
                break
            out[n] = v
    return out


def _subs(h, path: str):
    return h.reg(path) or []


def _key_exists(path: str) -> bool:
    if sys.platform != "win32":
        return False
    import winreg
    root, sub = _root(path)
    try:
        winreg.CloseKey(winreg.OpenKey(root, sub))
        return True
    except OSError:
        return False


def _value_present(path: str, name: str) -> bool:
    """Presence of a registry value without reading its data (T3-safe)."""
    import ctypes
    from ctypes import wintypes
    root, sub = _root(path)
    adv = ctypes.WinDLL("advapi32")
    fn = adv.RegGetValueW
    fn.argtypes = [wintypes.HKEY, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    fn.restype = wintypes.LONG
    cb = wintypes.DWORD(0)
    rc = fn(wintypes.HKEY(root), sub, name, 0x0000FFFF, None, None, ctypes.byref(cb))
    return rc == 0


def _lastwrite(path: str):
    if sys.platform != "win32":
        return None
    import winreg
    root, sub = _root(path)
    try:
        with winreg.OpenKey(root, sub) as k:
            return _ft_iso(winreg.QueryInfoKey(k)[2])
    except OSError:
        return None


def _ft_iso(ft):
    if not isinstance(ft, int) or ft <= 0:
        return None
    try:
        return (_dt.datetime(1601, 1, 1, tzinfo=_dt.timezone.utc)
                + _dt.timedelta(microseconds=ft // 10)).isoformat(timespec="seconds")
    except OverflowError:
        return None


def _unix_iso(s):
    if not isinstance(s, int) or s <= 0:
        return None
    return _dt.datetime.fromtimestamp(s, _dt.timezone.utc).isoformat(timespec="seconds")


def _systemtime(b):
    if not isinstance(b, (bytes, bytearray)) or len(b) < 16:
        return None
    y, mo, _dow, d, hh, mi, s, _ms = [int.from_bytes(b[i:i + 2], "little") for i in range(0, 16, 2)]
    try:
        return _dt.datetime(y, mo, d, hh, mi, s).isoformat()
    except ValueError:
        return None


def _days_ago(iso):
    if not iso:
        return None
    t = _dt.datetime.fromisoformat(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return round((_now() - t).total_seconds() / 86400, 2)


def _within(iso, days):
    d = _days_ago(iso)
    return d is not None and d <= days


def _birth(h, path: str):
    """Creation time (Windows st_birthtime / st_ctime) as iso, or None."""
    try:
        st = os.stat(h.expand(path))
    except OSError:
        return None
    t = getattr(st, "st_birthtime", None) or st.st_ctime
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat(timespec="seconds")


def _s32(x):
    return x - (1 << 32) if isinstance(x, int) and x >= (1 << 31) else x


def _run_text(h, args, timeout_ms):
    """h.run returns bytes or str depending on its text flag; normalise to str (None on failure)."""
    out = h.run(args, timeout_ms)
    if out is None:
        return None
    return out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)


def _current_sid(h):
    """SID of the collecting user, matched from ProfileList by profile path. Cached on h."""
    if hasattr(h, "_hi_sid"):
        return h._hi_sid
    home = os.path.normcase(os.path.normpath(h.l0.get("home", "")))
    sid = None
    base = _NT_CV + r"\ProfileList"
    for s in _subs(h, base):
        p = h.reg(base + "\\" + s, "ProfileImagePath") or ""
        if os.path.normcase(os.path.normpath(os.path.expandvars(p))) == home:
            sid = s
            break
    h._hi_sid = sid
    return sid


def _sid_kind(sid):
    if not sid:
        return None
    if sid.startswith("S-1-12-1-"):
        return "azuread"
    if sid.startswith("S-1-5-21-"):
        return "local_or_domain"
    return "other"


# ================================================================ host

@probe(id="host.native_arch", level="L1", family="host", tier="T0", collect="core")
def native_arch(h, facts):
    """Machine architecture from IsWow64Process2 nativeMachine (L0 fact)."""
    return {"present": True, "arch": h.l0.get("native_arch")}


@probe(id="host.python_emulated", level="L1", family="host", tier="T0", collect="core")
def python_emulated(h, facts):
    """Collector interpreter arch from its PE header vs machine arch; emulated x64 Python on ARM64 is 2-6x slower."""
    return {"present": True, "python_arch": h.l0.get("python_arch"),
            "native_arch": h.l0.get("native_arch"), "emulated": bool(h.l0.get("python_emulated"))}


@probe(id="os.build", level="L1", family="host", tier="T0", collect="core")
def os_build(h, facts):
    """OS build as CurrentBuild.UBR plus DisplayVersion (not BuildLabEx)."""
    b = h.reg(_NT_CV, "CurrentBuild")
    if not b:
        return None
    ubr = h.reg(_NT_CV, "UBR")
    return {"present": True, "build": f"{b}.{ubr}" if ubr is not None else str(b),
            "current_build": int(b), "display_version": h.reg(_NT_CV, "DisplayVersion")}


@probe(id="os.edition", level="L1", family="host", tier="T0", collect="core")
def os_edition(h, facts):
    """Windows edition and marketing name (ProductName says 'Windows 10' on 11; derive from build)."""
    name = h.reg(_NT_CV, "ProductName") or ""
    b = int(h.reg(_NT_CV, "CurrentBuild") or 0)
    return {"present": bool(name), "marketing_name": name.replace("Windows 10", "Windows 11") if b >= 22000 else name,
            "edition_id": h.reg(_NT_CV, "EditionID"),
            "composition_edition_id": h.reg(_NT_CV, "CompositionEditionID"),
            "installation_type": h.reg(_NT_CV, "InstallationType")}


@probe(id="os.insider", level="L1", family="host", tier="T0", collect="core")
def os_insider(h, facts):
    """Windows Insider ring enrollment (WindowsSelfHost keys)."""
    a = r"HKLM\SOFTWARE\Microsoft\WindowsSelfHost\Applicability"
    branch, ring = h.reg(a, "BranchName"), h.reg(a, "Ring")
    return {"present": True, "insider": bool(branch or ring), "branch": branch, "ring": ring,
            "ui_branch": h.reg(r"HKLM\SOFTWARE\Microsoft\WindowsSelfHost\UI\Selection", "UIBranch")}


# ================================================================ identity

_CRED_PROVIDERS = {
    "{D6886603-9D2F-4EB2-B667-1971041FA96B}": "PIN", "{60B78E88-EAD8-445C-9CFD-0B87F74EA6CD}": "Password",
    "{8AF662BF-65A0-4D0A-A540-A338A999D36F}": "Face", "{BEC09223-B018-416D-A0AC-523971B639F5}": "Fingerprint",
    "{F8A1793B-7873-4046-B2A7-1F318747F427}": "FIDO", "{1B283861-754F-4022-AD47-A5EAAA618894}": "SmartCard",
    "{27FBDB57-B613-4AF2-9D7E-4FA7A66C21AD}": "TrustedSignal",
    "{C5D7540A-CD51-453B-B22B-05305BA03F07}": "CloudExperienceCredProv",
}
_LOGONUI = r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Authentication\LogonUI"


@probe(id="acct.account_type", level="L1", family="identity", tier="T0", collect="core")
def account_type(h, facts):
    """Local vs Microsoft vs Entra account: SID prefix, LogonUI domain prefix, IdentityCRL/IdentityStore counts (no names)."""
    sid = _current_sid(h)
    llu = (h.reg(_LOGONUI, "LastLoggedOnSAMUser") or h.reg(_LOGONUI, "LastLoggedOnUser") or "").lower()
    msa_prefix = llu.startswith("microsoftaccount\\")
    aad_prefix = llu.startswith("azuread\\")
    crl = len(_subs(h, r"HKCU\Software\Microsoft\IdentityCRL\StoredIdentities"))
    providers = {}
    if sid:
        base = r"HKLM\SOFTWARE\Microsoft\IdentityStore\Cache" + "\\" + sid + r"\IdentityCache"
        for s in _subs(h, base)[:20]:
            pn = str(h.reg(base + "\\" + s, "ProviderName"))
            providers[pn] = providers.get(pn, 0) + 1
    kind = _sid_kind(sid)
    if kind == "azuread" or aad_prefix:
        t = "entra"
    elif msa_prefix or any("microsoftaccount" in p.lower() for p in providers):
        t = "microsoft"
    elif kind == "local_or_domain":
        t = "local"
    else:
        t = "unknown"
    return {"present": True, "type": t, "sid_kind": kind, "logonui_msa_prefix": msa_prefix,
            "logonui_azuread_prefix": aad_prefix,
            "last_logon_is_collecting_user": (llu.split("\\")[-1] == h.l0.get("user", "").lower()) if llu else None,
            "identitycrl_stored_identities": crl, "identitystore_providers": providers}


@probe(id="host.name", level="L1", family="identity", tier="T0", collect="core")
def host_name(h, facts):
    """Computer name. local_only: identifies the machine, never send off-host."""
    n = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\ComputerName\ActiveComputerName", "ComputerName")
    pending = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\ComputerName\ComputerName", "ComputerName")
    return {"present": bool(n), "name": n, "local_only": True,
            "pending_rename": bool(pending and n and pending.lower() != n.lower())}


_DEFAULT_USERS = {"public", "default", "default user", "all users", "defaultapppool"}


@probe(id="profile.count", level="L1", family="identity", tier="T0", collect="core")
def profile_count(h, facts):
    """Number of real user profiles (ProfileList non-system SIDs, C:\\Users dirs); operator side homes excluded."""
    base = _NT_CV + r"\ProfileList"
    nonsys, noise = 0, 0
    for s in _subs(h, base):
        if s in ("S-1-5-18", "S-1-5-19", "S-1-5-20") or not s.startswith(("S-1-5-21-", "S-1-12-1-")):
            continue
        p = os.path.expandvars(h.reg(base + "\\" + s, "ProfileImagePath") or "")
        if _is_noise(p):
            noise += 1
            continue
        nonsys += 1
    users = []
    try:
        with os.scandir(r"C:\Users") as it:
            for _, e in zip(range(200), it):
                if e.is_dir() and e.name.lower() not in _DEFAULT_USERS:
                    users.append(e.name)
    except OSError:
        pass
    users_real = [n for n in users if not _is_noise("\\" + n)]
    return {"present": True, "nonsystem": nonsys, "operator_excluded": noise,
            "c_users_nondefault": len(users_real), "current_found": _current_sid(h) is not None}


@probe(id="profile.last_load", level="L1", family="identity", tier="T0", collect="core", gate="profile.count")
def profile_last_load(h, facts):
    """Last interactive profile load time of the collecting user (ProfileList FILETIME)."""
    sid = _current_sid(h)
    if not sid:
        return None
    k = _NT_CV + r"\ProfileList" + "\\" + sid
    lo, hi = h.reg(k, "LocalProfileLoadTimeLow"), h.reg(k, "LocalProfileLoadTimeHigh")
    if lo is None or hi is None:
        return None
    t = _ft_iso((hi << 32) | lo)
    return {"present": True, "last_load": t, "days_ago": _days_ago(t)}


@probe(id="profile.rid", level="L1", family="identity", tier="T0", collect="core", gate="profile.count")
def profile_rid(h, facts):
    """RID of the collecting user; 1001 = first account created at OOBE."""
    sid = _current_sid(h)
    if not sid:
        return None
    rid = int(sid.rsplit("-", 1)[-1])
    return {"present": True, "rid": rid, "first_oobe_user": rid == 1001}


_AGENT_ACCT_RX = re.compile(r"^(CodexSandbox\w*|ClaudeSandbox\w*|WsiAccount|sshd|docker-users)$", re.I)


@probe(id="acct.local_users", level="L2", family="identity", tier="T1", collect="extended", gate="profile.count")
def local_users(h, facts):
    """Local account counts via NetUserEnum (no PowerShell); names never emitted, agent sandbox accounts counted by kind."""
    import ctypes
    from ctypes import wintypes

    class USER_INFO_20(ctypes.Structure):
        _fields_ = [("name", wintypes.LPWSTR), ("full_name", wintypes.LPWSTR), ("comment", wintypes.LPWSTR),
                    ("flags", wintypes.DWORD), ("user_id", wintypes.DWORD)]
    net = ctypes.WinDLL("netapi32")
    net.NetUserEnum.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                ctypes.POINTER(wintypes.DWORD)]
    net.NetApiBufferFree.argtypes = [ctypes.c_void_p]
    buf = ctypes.c_void_p()
    read, total, resume = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD(0)
    rc = net.NetUserEnum(None, 20, 2, ctypes.byref(buf), 0xFFFFFFFF, ctypes.byref(read), ctypes.byref(total), ctypes.byref(resume))
    if rc not in (0, 234) or not buf:
        return {"present": False, "error": rc}
    me = h.l0.get("user", "").lower()
    out = {"total": 0, "enabled": 0, "enabled_kinds": {"current": 0, "builtin": 0, "agent_or_tool": 0, "other_human": 0}}
    try:
        arr = ctypes.cast(buf, ctypes.POINTER(USER_INFO_20 * read.value)).contents
        for u in arr:
            out["total"] += 1
            if u.flags & 0x2:  # UF_ACCOUNTDISABLE
                continue
            out["enabled"] += 1
            name = (u.name or "")
            if name.lower() == me:
                kind = "current"
            elif u.user_id in (500, 501, 503, 504):
                kind = "builtin"
            elif _AGENT_ACCT_RX.match(name):
                kind = "agent_or_tool"
            else:
                kind = "other_human"
            out["enabled_kinds"][kind] += 1
    finally:
        net.NetApiBufferFree(buf)
    out["present"] = True
    return out


# ================================================================ install_age

@probe(id="age.footprint_counts", level="L1", family="install_age", tier="T0", collect="core")
def footprint_counts(h, facts):
    """Lived-in counts: Prefetch, Recent .lnk, Uninstall entries, AppX package dirs."""
    recent = h.list_dir(os.path.join(h.l0.get("appdata", ""), r"Microsoft\Windows\Recent"), 5000)
    return {"present": True,
            "prefetch": h.count_dir(r"C:\Windows\Prefetch", 5000),
            "recent_lnk": sum(1 for n in recent if n.lower().endswith(".lnk")),
            "uninstall_hklm": len(_subs(h, r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")),
            "uninstall_hklm_wow64": len(_subs(h, r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall")),
            "uninstall_hkcu": len(_subs(h, r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall")),
            "packages_dirs": h.count_dir(os.path.join(h.l0.get("localappdata", ""), "Packages"), 5000)}


_MS_PKG_RX = re.compile(r"^(Microsoft|MicrosoftWindows|Windows|MicrosoftCorporationII|Clipchamp|MSTeams)\b", re.I)


@probe(id="apps.appx_provisioned", level="L1", family="install_age", tier="T0", collect="core")
def appx_provisioned(h, facts):
    """Image-provisioned AppX packages (AppxAllUserStore); non-Microsoft ones mark an OEM image, deprovisioned ones a debloat."""
    base = r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Appx\AppxAllUserStore"
    apps = _subs(h, base + r"\Applications")
    if not apps:
        return None
    oem = sorted({a.split("_")[0] for a in apps if not _MS_PKG_RX.match(a)})
    return {"present": True, "provisioned": len(apps), "oem_provisioned": len(oem), "oem_families": oem[:20],
            "deprovisioned": len(_subs(h, base + r"\Deprovisioned"))}


@probe(id="os.install_date", level="L1", family="install_age", tier="T0", collect="core")
def os_install_date(h, facts):
    """Current OS image install date (last install, not first use)."""
    d = h.reg(_NT_CV, "InstallDate")
    iso = _unix_iso(d)
    return {"present": bool(iso), "install_date": iso, "days_ago": _days_ago(iso),
            "install_time": _ft_iso(h.reg(_NT_CV, "InstallTime"))}


@probe(id="profile.created", level="L1", family="install_age", tier="T0", collect="core", gate="profile.count")
def profile_created(h, facts):
    """Creation time of the user profile folder and NTUSER.DAT: user-tenure anchor."""
    home = h.l0.get("home", "")
    folder, ntuser = _birth(h, home), _birth(h, os.path.join(home, "NTUSER.DAT"))
    first = min([t for t in (folder, ntuser) if t], default=None)
    return {"present": bool(first), "created": first, "folder": folder, "ntuser": ntuser, "days_ago": _days_ago(first)}


@probe(id="setup.image_date", level="L1", family="install_age", tier="T0", collect="core")
def setup_image_date(h, facts):
    """Factory/capture date of the image: CloneTag, setupapi.setup.log head, Windows dir and Panther ctimes."""
    clone = h.reg(r"HKLM\SYSTEM\Setup", "CloneTag")
    if isinstance(clone, list):
        clone = clone[0] if clone else None
    first = None
    try:
        with open(r"C:\Windows\INF\setupapi.setup.log", "r", encoding="utf-8", errors="replace") as f:
            m = re.search(r"Section start (\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)", f.read(8192))
            first = m.group(1) if m else None
    except OSError:
        pass
    dates = []
    if isinstance(clone, str):
        try:
            dates.append(_dt.datetime.strptime(clone.strip(), "%a %b %d %H:%M:%S %Y").isoformat())
        except ValueError:
            pass
    if first:
        dates.append(first.replace("/", "-").replace(" ", "T"))
    win = _birth(h, r"C:\Windows")
    if win:
        dates.append(win[:19])
    return {"present": True, "image_date": min(dates) if dates else None, "clonetag": clone, "setupapi_first": first,
            "windows_dir_created": win,
            "software_hive_created": _birth(h, r"C:\Windows\System32\config\SOFTWARE"),
            "panther_setupact_created": _birth(h, r"C:\Windows\Panther\setupact.log"),
            "panther_unattend": h.exists(r"C:\Windows\Panther\unattend.xml"),
            "panther_miglog": h.exists(r"C:\Windows\Panther\MigLog.xml")}


@probe(id="setup.oem", level="L1", family="install_age", tier="T0", collect="core")
def setup_oem(h, facts):
    """OEM preload marker: OEMInformation Manufacturer/Model and C:\\Recovery\\OEM."""
    k = r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\OEMInformation"
    mfr = h.reg(k, "Manufacturer")
    rec = _birth(h, r"C:\Recovery\OEM")
    return {"present": True, "oem_image": bool(mfr), "manufacturer": mfr, "model": h.reg(k, "Model"),
            "recovery_oem_created": rec}


@probe(id="setup.oobe_done", level="L1", family="install_age", tier="T0", collect="core")
def setup_oobe_done(h, facts):
    """OOBE key LastWriteTime: when the user took ownership of the machine."""
    t = _lastwrite(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\OOBE")
    return {"present": bool(t), "date": t, "days_ago": _days_ago(t)}


@probe(id="setup.source_os_lineage", level="L1", family="install_age", tier="T0", collect="core")
def source_os_lineage(h, facts):
    """Prior OS installs from HKLM\\SYSTEM\\Setup\\Source OS keys (survive resets and upgrades)."""
    rows = []
    for s in _subs(h, r"HKLM\SYSTEM\Setup"):
        if not s.startswith("Source OS"):
            continue
        k = r"HKLM\SYSTEM\Setup" + "\\" + s
        rows.append({"date": (_unix_iso(h.reg(k, "InstallDate")) or "")[:10] or None,
                     "edition": h.reg(k, "EditionID"), "build": h.reg(k, "CurrentBuild"),
                     "version": h.reg(k, "DisplayVersion") or h.reg(k, "ReleaseId")})
    rows.sort(key=lambda r: r["date"] or "")
    return {"present": True, "count": len(rows), "oldest": rows[0]["date"] if rows else None,
            "editions": sorted({r["edition"] for r in rows if r["edition"]}), "lineage": rows[:20]}


@probe(id="setup.sysreset", level="L1", family="install_age", tier="T0", collect="core")
def setup_sysreset(h, facts):
    """C:\\$SysReset presence and ctime: 'Reset this PC' marker."""
    t = _birth(h, r"C:\$SysReset")
    return {"present": True, "reset": bool(t), "created": t}


@probe(id="setup.windows_old", level="L1", family="install_age", tier="T0", collect="core")
def setup_windows_old(h, facts):
    """In-place upgrade residue: C:\\Windows.old, C:\\$WINDOWS.~BT."""
    return {"present": True, "windows_old": h.exists(r"C:\Windows.old"), "windows_bt": h.exists(r"C:\$WINDOWS.~BT")}


_TYPES = {71: "wireless", 6: "wired", 23: "vpn", 243: "mobile_broadband", 53: "virtual"}
_CATS = {0: "public", 1: "private", 2: "domain"}


def _network_profiles(h):
    if hasattr(h, "_hi_netprof"):
        return h._hi_netprof
    base = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\NetworkList\Profiles"
    rows = []
    for s in _subs(h, base)[:500]:
        v = _vals(base + "\\" + s) or {}
        rows.append({"type": _TYPES.get(v.get("NameType"), str(v.get("NameType"))),
                     "category": _CATS.get(v.get("Category"), str(v.get("Category"))),
                     "created": _systemtime(v.get("DateCreated")),
                     "last": _systemtime(v.get("DateLastConnected"))})
    h._hi_netprof = rows
    return rows


@probe(id="net.history", level="L2", family="install_age", tier="T1", collect="core", gate="os.install_date")
def net_history(h, facts):
    """NetworkList profiles: network count by type and first/last DateCreated (first-online marker). No SSIDs."""
    rows = _network_profiles(h)
    if not rows:
        return None
    by_type, by_cat = {}, {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    created = sorted(r["created"] for r in rows if r["created"])
    last = sorted(r["last"] for r in rows if r["last"])
    return {"present": True, "networks": len(rows), "by_type": by_type, "by_category": by_cat,
            "first_created": created[0] if created else None, "last_created": created[-1] if created else None,
            "last_connected": last[-1] if last else None}


def _utf16_strings(b, minlen=6):
    return [x.decode("utf-16le") for x in re.findall(rb"(?:[\x20-\x7e]\x00){%d,}" % minlen, b)]


@probe(id="pins.taskband_oem", level="L2", family="install_age", tier="T0", collect="core", gate="pins.taskbar")
def taskband_oem(h, facts):
    """Packaged-app pins in the Taskband blob; OEM AUMIDs still present = untouched OEM taskbar layout."""
    fav = h.reg(r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Taskband", "Favorites")
    if not isinstance(fav, (bytes, bytearray)):
        return None
    aumids = sorted({s.split("_")[0] + "!" + s.split("!")[-1] for s in _utf16_strings(fav) if "!" in s and "_" in s})
    oem = [a for a in aumids if not _MS_PKG_RX.match(a)]
    return {"present": True, "packaged_pins": len(aumids), "oem_pins": oem, "oem_layout_present": bool(oem),
            "taskband_lastwrite": _lastwrite(r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Taskband")}


@probe(id="setup.system_log_oldest", level="L2", family="install_age", tier="T0", collect="extended", gate="os.install_date")
def system_log_oldest(h, facts):
    """Oldest retained System event (measures log retention; image-date corroboration only)."""
    txt = _run_text(h, ["wevtutil", "qe", "System", "/c:1", "/rd:false", "/f:xml"], 5000) or ""
    m = re.search(r"SystemTime='([^']+)'", txt)
    if not m:
        return None
    info = _run_text(h, ["wevtutil", "gli", "System"], 3000) or ""
    n = re.search(r"numberOfLogRecords:\s*(\d+)", info)
    return {"present": True, "oldest": m.group(1)[:19], "records": int(n.group(1)) if n else None}


# ================================================================ locale

_INTL = r"HKCU\Control Panel\International"


@probe(id="kbd.layouts", level="L1", family="locale", tier="T0", collect="core")
def kbd_layouts(h, facts):
    """Keyboard layouts from Preload + Substitutes (GetKeyboardLayoutList is empty over ssh)."""
    pre = _vals(r"HKCU\Keyboard Layout\Preload") or {}
    sub = _vals(r"HKCU\Keyboard Layout\Substitutes") or {}
    out, seen = [], set()
    for k in sorted(pre, key=lambda x: int(x) if x.isdigit() else 0):
        klid = sub.get(pre[k], pre[k])
        if klid.upper() in seen:
            continue
        seen.add(klid.upper())
        out.append({"klid": klid, "name": h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Keyboard Layouts" + "\\" + klid, "Layout Text")})
    return {"present": bool(out), "layouts": out, "count": len(out), "preload_entries": len(pre),
            "ime": any(str(x["klid"]).upper().startswith("E0") for x in out)}


@probe(id="kbd.scancode_map", level="L1", family="locale", tier="T0", collect="core")
def kbd_scancode_map(h, facts):
    """Scancode Map key remaps (e.g. CapsLock->Ctrl): power-user hint."""
    v = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Keyboard Layout", "Scancode Map")
    n = (int.from_bytes(v[8:12], "little") - 1) if isinstance(v, (bytes, bytearray)) and len(v) >= 12 else 0
    return {"present": True, "remapped": n > 0, "remaps": max(n, 0)}


def _lcid_name(lcid):
    import ctypes
    buf = ctypes.create_unicode_buffer(85)
    ctypes.windll.kernel32.LCIDToLocaleName(lcid, buf, 85, 0)
    return buf.value


@probe(id="lang.ui", level="L1", family="locale", tier="T0", collect="core")
def lang_ui(h, facts):
    """User and system UI language plus installed language packs: reply language."""
    import ctypes
    k32 = ctypes.windll.kernel32
    user = _lcid_name(k32.GetUserDefaultUILanguage())
    return {"present": True, "language": user, "user": user,
            "system": _lcid_name(k32.GetSystemDefaultUILanguage()),
            "packs": _subs(h, r"HKLM\SYSTEM\CurrentControlSet\Control\MUI\UILanguages")}


@probe(id="lang.user_list", level="L1", family="locale", tier="T0", collect="core")
def lang_user_list(h, facts):
    """User language preference list (Control Panel\\International\\User Profile Languages)."""
    langs = h.reg(_INTL + r"\User Profile", "Languages") or []
    return {"present": bool(langs), "languages": list(langs)}


@probe(id="locale.geo", level="L1", family="locale", tier="T0", collect="core")
def locale_geo(h, facts):
    """Home location (country) from Geo Name/Nation."""
    name, nation = h.reg(_INTL + r"\Geo", "Name"), h.reg(_INTL + r"\Geo", "Nation")
    return {"present": bool(name or nation), "name": name, "nation_id": nation}


@probe(id="locale.system", level="L1", family="locale", tier="T0", collect="core")
def locale_system(h, facts):
    """System locale, install language and ANSI code page (UTF-8 beta flag)."""
    nls = r"HKLM\SYSTEM\CurrentControlSet\Control\Nls"
    acp = h.reg(nls + r"\CodePage", "ACP")
    return {"present": True, "default": h.reg(nls + r"\Language", "Default"),
            "install_language": h.reg(nls + r"\Language", "InstallLanguage"),
            "acp": acp, "utf8": acp == "65001"}


@probe(id="locale.user", level="L1", family="locale", tier="T0", collect="core")
def locale_user(h, facts):
    """User date/time/number/currency format for replies."""
    v = _vals(_INTL) or {}
    if not v:
        return None
    keep = ("LocaleName", "sShortDate", "sShortTime", "sTimeFormat", "iMeasure", "sCurrency", "iPaperSize", "iFirstDayOfWeek", "sDecimal")
    d = {k: v.get(k) for k in keep if k in v}
    t = v.get("sShortTime") or v.get("sTimeFormat") or ""
    d["clock_24h"] = "H" in t
    d["metric"] = v.get("iMeasure") == "0"
    d["present"] = True
    return d


@probe(id="tz.auto", level="L1", family="locale", tier="T0", collect="core")
def tz_auto(h, facts):
    """Automatic time zone service and NTP server."""
    start = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services\tzautoupdate", "Start")
    return {"present": True, "auto_tz": {2: True, 3: True, 4: False}.get(start), "tzautoupdate_start": start,
            "ntp": h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services\W32Time\Parameters", "NtpServer")}


@probe(id="tz.zone", level="L1", family="locale", tier="T0", collect="core")
def tz_zone(h, facts):
    """Time zone key name and signed bias in minutes."""
    k = r"HKLM\SYSTEM\CurrentControlSet\Control\TimeZoneInformation"
    name = h.reg(k, "TimeZoneKeyName")
    return {"present": bool(name), "zone": name, "bias_min": _s32(h.reg(k, "Bias")),
            "active_bias_min": _s32(h.reg(k, "ActiveTimeBias"))}


# ================================================================ shell_prefs

@probe(id="pins.taskbar", level="L1", family="shell_prefs", tier="T0", collect="core")
def pins_taskbar(h, facts):
    """Taskbar .lnk pins (desktop apps the user pinned)."""
    d = os.path.join(h.l0.get("appdata", ""), r"Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar")
    names = sorted(n[:-4] for n in h.list_dir(d, 200) if n.lower().endswith(".lnk"))
    return {"present": True, "pins": names, "count": len(names)}


_EXPLORER = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer"


@probe(id="shell.explorer_prefs", level="L1", family="shell_prefs", tier="T0", collect="core")
def explorer_prefs(h, facts):
    """Explorer/taskbar preferences: HideFileExt, Hidden, search box mode, auto-hide, widgets, dev toggles."""
    adv = _vals(_EXPLORER + r"\Advanced") or {}
    keep = ("HideFileExt", "Hidden", "ShowSuperHidden", "TaskbarAl", "TaskbarDa", "TaskbarMn", "ShowTaskViewButton",
            "LaunchTo", "TaskbarEndTask", "ShowCopilotButton", "Start_IrisRecommendations")
    d = {k: adv.get(k) for k in keep if k in adv}
    d["SearchboxTaskbarMode"] = h.reg(r"HKCU\Software\Microsoft\Windows\CurrentVersion\Search", "SearchboxTaskbarMode")
    sr = h.reg(_EXPLORER + r"\StuckRects3", "Settings")
    if isinstance(sr, (bytes, bytearray)) and len(sr) > 12:
        d["taskbar_autohide"] = bool(sr[8] & 0x01)
        d["taskbar_edge"] = {0: "left", 1: "top", 2: "right", 3: "bottom"}.get(sr[12], sr[12])
    d["classic_context_menu"] = _key_exists(r"HKCU\Software\Classes\CLSID\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}\InprocServer32")
    d["dev_mode"] = h.reg(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock", "AllowDevelopmentWithoutDevLicense")
    d["sudo"] = h.reg(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Sudo", "Enabled")
    d["long_paths"] = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\FileSystem", "LongPathsEnabled")
    d["ps_execution_policy_cu"] = h.reg(r"HKCU\Software\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell", "ExecutionPolicy")
    d["ps_execution_policy_lm"] = h.reg(r"HKLM\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell", "ExecutionPolicy")
    d["present"] = True
    return d


@probe(id="theme.dark", level="L1", family="shell_prefs", tier="T0", collect="core")
def theme_dark(h, facts):
    """Dark mode for apps and system, transparency."""
    k = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    a, s = h.reg(k, "AppsUseLightTheme"), h.reg(k, "SystemUsesLightTheme")
    return {"present": True, "apps_dark": a == 0, "system_dark": s == 0, "transparency": h.reg(k, "EnableTransparency")}


@probe(id="theme.spotlight_suggestions", level="L1", family="shell_prefs", tier="T0", collect="core")
def spotlight_suggestions(h, facts):
    """Spotlight / suggestions / silent app installs left on: weak untouched-defaults hint."""
    cdm = _vals(r"HKCU\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager") or {}
    return {"present": True, "lock_spotlight": cdm.get("RotatingLockScreenEnabled"),
            "silent_installs": cdm.get("SilentInstalledAppsEnabled"),
            "suggestions_on": sum(1 for k, v in cdm.items() if k.startswith("SubscribedContent-") and k.endswith("Enabled") and v == 1),
            "system_pane_suggestions": cdm.get("SystemPaneSuggestionsEnabled")}


@probe(id="theme.wallpaper", level="L1", family="shell_prefs", tier="T0", collect="core")
def theme_wallpaper(h, facts):
    """Wallpaper type only (solid/picture/slideshow/spotlight) and whether history holds stock or user images; never paths."""
    wps = _vals(_EXPLORER + r"\Wallpapers") or {}
    cur = h.reg(r"HKCU\Control Panel\Desktop", "WallPaper") or ""
    bt = {0: "picture", 1: "solid_color", 2: "slideshow", 3: "spotlight"}.get(wps.get("BackgroundType"), wps.get("BackgroundType"))

    def cls(p):
        pl = (p or "").lower()
        if not pl:
            return "none"
        if "\\windows\\web\\" in pl:
            return "stock"
        if "transcodedwallpaper" in pl:
            return "user_copy"
        return "user"
    hist = [cls(wps.get(f"BackgroundHistoryPath{i}")) for i in range(5) if wps.get(f"BackgroundHistoryPath{i}")]
    return {"present": True, "type": bt, "current": cls(cur), "history_stock": hist.count("stock"),
            "history_user": sum(1 for x in hist if x.startswith("user"))}


# ================================================================ security

@probe(id="acct.autologon", level="L1", family="security", tier="T0", collect="core")
def acct_autologon(h, facts):
    """AutoAdminLogon flag and DefaultPassword presence (value never read)."""
    k = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"
    return {"present": True, "auto_admin_logon": h.reg(k, "AutoAdminLogon"),
            "default_password_present": _value_present(k, "DefaultPassword")}


@probe(id="acct.hello", level="L1", family="security", tier="T0", collect="core")
def acct_hello(h, facts):
    """Windows Hello: NGC PIN credentials, WinBio enrolled factors, last sign-in credential provider."""
    pins = len(_subs(h, _LOGONUI + r"\NgcPin\Credentials"))
    factors = 0
    wb = r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\WinBio\AccountInfo"
    for s in _subs(h, wb)[:20]:
        factors |= int(h.reg(wb + "\\" + s, "EnrolledFactors") or 0)
    lp = str(h.reg(_LOGONUI, "LastLoggedOnProvider") or "").upper()
    return {"present": True, "pin_credentials": pins, "face": bool(factors & 0x2), "fingerprint": bool(factors & 0x8),
            "last_provider": _CRED_PROVIDERS.get(lp, "other" if lp else None)}


_CS = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore"


@probe(id="consent.store", level="L1", family="security", tier="T1", collect="core")
def consent_store(h, facts):
    """ConsentStore capability keys (HKCU + HKLM) and global allow state of the key capabilities."""
    cu, lm = _subs(h, "HKCU\\" + _CS), _subs(h, "HKLM\\" + _CS)
    if not cu and not lm:
        return None
    glob = {c: h.reg("HKLM\\" + _CS + "\\" + c, "Value") for c in ("webcam", "microphone", "location", "graphicsCaptureWithoutBorder", "passkeys")}
    return {"present": True, "hkcu_capabilities": len(cu), "hklm_capabilities": len(lm), "machine_value": glob}


def _consent_apps(h, root, cap):
    base = root + "\\" + _CS + "\\" + cap
    apps, noise = [], 0
    for sub in _subs(h, base)[:400]:
        if sub == "NonPackaged":
            for s2 in _subs(h, base + r"\NonPackaged")[:400]:
                path = s2.replace("#", "\\")
                if _is_noise(path):
                    noise += 1
                    continue
                v = _vals(base + r"\NonPackaged" + "\\" + s2) or {}
                apps.append((os.path.basename(path), False, v))
        else:
            v = _vals(base + "\\" + sub) or {}
            apps.append((sub.split("_")[0], True, v))
    rows = {}
    for name, packaged, v in apps:
        start, stop = _ft_iso(v.get("LastUsedTimeStart")), _ft_iso(v.get("LastUsedTimeStop"))
        key = name.lower()
        r = {"app": name, "packaged": packaged, "value": v.get("Value"), "start": start, "stop": stop}
        prev = rows.get(key)
        if prev is None or (start or "") > (prev["start"] or ""):
            rows[key] = r
    return list(rows.values()), noise


def _consent_summary(h, caps, hklm_top=False):
    rows, noise = [], 0
    for cap in caps:
        r, n = _consent_apps(h, "HKCU", cap)
        rows += r
        noise += n
    if not rows and not noise:
        return None
    merged = {}
    for r in rows:
        k = r["app"].lower()
        if k not in merged or (r["start"] or "") > (merged[k]["start"] or ""):
            merged[k] = r
    rows = list(merged.values())
    used = sorted((r for r in rows if r["start"]), key=lambda r: r["start"], reverse=True)
    recent = []
    for r in used[:8]:
        d = _days_ago(r["start"])
        sess = None
        if r["stop"]:
            sess = round((_dt.datetime.fromisoformat(r["stop"]) - _dt.datetime.fromisoformat(r["start"])).total_seconds(), 1)
        recent.append({"app": r["app"], "days_ago": d, "last_session_s": sess, "in_use_now": not r["stop"]})
    out = {"present": True, "apps": len(rows), "allow": sum(1 for r in rows if r["value"] == "Allow"),
           "deny": sum(1 for r in rows if r["value"] == "Deny"), "ever_used": len(used),
           "used_7d": sum(1 for r in used if _within(r["start"], 7)),
           "used_30d": sum(1 for r in used if _within(r["start"], 30)),
           "recent": recent, "operator_excluded": noise}
    if hklm_top:
        sysrows = []
        for cap in caps:
            r, _n = _consent_apps(h, "HKLM", cap)
            sysrows += [x for x in r if x["start"]]
        sysrows.sort(key=lambda r: r["start"], reverse=True)
        out["system_recent"] = [{"app": r["app"], "days_ago": _days_ago(r["start"])} for r in sysrows[:4]]
    return out


@probe(id="consent.webcam", level="L2", family="security", tier="T1", collect="core", gate="consent.store")
def consent_webcam(h, facts):
    """Apps that used the camera (last session per app); HKLM system use (Hello face) listed separately."""
    return _consent_summary(h, ["webcam"], hklm_top=True)


@probe(id="consent.microphone", level="L2", family="security", tier="T1", collect="core", gate="consent.store")
def consent_microphone(h, facts):
    """Apps that used the microphone: voice chat, recording, streaming, agents."""
    return _consent_summary(h, ["microphone"])


@probe(id="consent.location", level="L2", family="security", tier="T1", collect="core", gate="consent.store")
def consent_location(h, facts):
    """Apps that read location; HKLM system readers listed separately."""
    return _consent_summary(h, ["location"], hklm_top=True)


@probe(id="consent.passkeys", level="L2", family="security", tier="T1", collect="core", gate="consent.store")
def consent_passkeys(h, facts):
    """Apps that used Windows passkeys."""
    return _consent_summary(h, ["passkeys"])


@probe(id="consent.screen_capture", level="L2", family="security", tier="T1", collect="core", gate="consent.store")
def consent_screen_capture(h, facts):
    """Apps that used border-less or programmatic screen capture (computer-use and capture tools)."""
    return _consent_summary(h, ["graphicsCaptureWithoutBorder", "graphicsCaptureProgrammatic"])


@probe(id="dev.ssh_dir_presence", level="L1", family="security", tier="T3", collect="core")
def ssh_dir_presence(h, facts):
    """~/.ssh file kinds by stat only (keys never opened); known_hosts line count, config Host count."""
    d = os.path.join(h.l0.get("home", ""), ".ssh")
    names = h.list_dir(d, 200)
    if not names:
        return None
    pub = [n for n in names if n.endswith(".pub")]
    priv = [n for n in names if n.startswith("id_") and not n.endswith(".pub")]
    out = {"present": True, "files": len(names), "private_keys": len(priv), "public_keys": len(pub),
           "config": h.exists(os.path.join(d, "config")), "authorized_keys": h.meta(os.path.join(d, "authorized_keys")).get("bytes"),
           "known_hosts_lines": None, "config_host_entries": None}
    kh = os.path.join(d, "known_hosts")
    try:
        with open(kh, "rb") as f:
            out["known_hosts_lines"] = sum(1 for ln in f if ln.strip())
    except OSError:
        pass
    try:
        with open(os.path.join(d, "config"), encoding="utf-8", errors="replace") as f:
            out["config_host_entries"] = sum(1 for ln in f if re.match(r"\s*Host\s+", ln, re.I))
    except OSError:
        pass
    return out


_PM_PATHS = {
    "1Password": [r"%LOCALAPPDATA%\1Password", r"%ProgramFiles%\1Password"],
    "Bitwarden": [r"%APPDATA%\Bitwarden", r"%LOCALAPPDATA%\Programs\Bitwarden"],
    "KeePass": [r"%ProgramFiles%\KeePass Password Safe 2", r"%ProgramFiles(x86)%\KeePass Password Safe 2"],
    "KeePassXC": [r"%ProgramFiles%\KeePassXC", r"%LOCALAPPDATA%\KeePassXC"],
    "Proton Pass": [r"%LOCALAPPDATA%\Programs\ProtonPass", r"%LOCALAPPDATA%\Programs\Proton Pass"],
    "Dashlane": [r"%LOCALAPPDATA%\Dashlane", r"%ProgramFiles%\Dashlane"],
    "Enpass": [r"%ProgramFiles%\Enpass", r"%LOCALAPPDATA%\Programs\Enpass"],
    "NordPass": [r"%LOCALAPPDATA%\Programs\nordpass", r"%ProgramFiles%\NordPass"],
    "LastPass": [r"%ProgramFiles%\LastPass", r"%ProgramFiles(x86)%\LastPass"],
    "RoboForm": [r"%ProgramFiles%\Siber Systems\AI RoboForm", r"%ProgramFiles(x86)%\Siber Systems\AI RoboForm"],
    "Keeper": [r"%LOCALAPPDATA%\Programs\keeper-password-manager"],
}


@probe(id="pm.desktop", level="L1", family="security", tier="T0", collect="core")
def pm_desktop(h, facts):
    """Desktop password manager installs by directory presence (vaults never touched)."""
    found = sorted(n for n, paths in _PM_PATHS.items() if any(h.exists(p) for p in paths))
    return {"present": bool(found), "managers": found, "count": len(found)}


@probe(id="security.defender_exclusions_count", level="L1", family="security", tier="T0", collect="core")
def defender_exclusions_count(h, facts):
    """Defender exclusion value counts per kind (admin token needed, else unknown); tamper registry value."""
    base = r"HKLM\SOFTWARE\Microsoft\Windows Defender"
    if not _key_exists(base):
        return None
    out = {"present": True}
    for kind in ("Paths", "Processes", "Extensions", "IpAddresses"):
        v = _vals(base + r"\Exclusions" + "\\" + kind)
        out[kind.lower()] = len(v) if v is not None else "unknown"
    out["tamper_protection_reg"] = h.reg(base + r"\Features", "TamperProtection")
    out["disable_antispyware_policy"] = h.reg(r"HKLM\SOFTWARE\Policies\Microsoft\Windows Defender", "DisableAntiSpyware")
    return out


@probe(id="security.firewall", level="L1", family="security", tier="T0", collect="core")
def security_firewall(h, facts):
    """Firewall EnableFirewall per profile from registry (Get-NetFirewallProfile dropped: 7.4 s on ARM64)."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters\FirewallPolicy"
    out = {}
    for label, key in (("domain", "DomainProfile"), ("private", "StandardProfile"), ("public", "PublicProfile")):
        v = h.reg(base + "\\" + key, "EnableFirewall")
        out[label] = None if v is None else bool(v)
    if all(v is None for v in out.values()):
        return None
    out["present"] = True
    out["all_on"] = all(out[k] for k in ("domain", "private", "public"))
    return out


@probe(id="security.openssh_server", level="L1", family="security", tier="T3", collect="core")
def openssh_server(h, facts):
    """OpenSSH server: service start type, DefaultShell, non-secret sshd_config directives, admin keys file size only."""
    start = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services\sshd", "Start")
    pd = os.environ.get("ProgramData", r"C:\ProgramData")
    shell = h.reg(r"HKLM\SOFTWARE\OpenSSH", "DefaultShell")
    directives = {}
    try:
        with open(os.path.join(pd, "ssh", "sshd_config"), encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 2000:
                    break
                parts = line.strip().split(None, 1)
                if not parts or parts[0].startswith("#"):
                    continue
                k = parts[0].lower()
                if k in ("port", "passwordauthentication", "pubkeyauthentication", "permitrootlogin", "kbdinteractiveauthentication"):
                    directives[k] = parts[1] if len(parts) > 1 else ""
                elif k in ("allowusers", "allowgroups", "match"):
                    directives[k] = "<set>"
    except OSError:
        pass
    aak = h.meta(os.path.join(pd, "ssh", "administrators_authorized_keys"))
    return {"present": start is not None, "sshd_start": start, "sshd_auto": start == 2,
            "ssh_agent_start": h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services\ssh-agent", "Start"),
            "default_shell": os.path.basename(shell) if shell else None, "sshd_config": directives,
            "admin_authorized_keys_bytes": aak.get("bytes") if aak.get("present") else None}


@probe(id="security.rdp", level="L1", family="security", tier="T0", collect="core")
def security_rdp(h, facts):
    """Remote Desktop accepting connections, NLA, port; client MRU count."""
    ts = r"HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server"
    deny = h.reg(ts, "fDenyTSConnections")
    if deny is None:
        return None
    return {"present": True, "rdp_on": deny == 0, "nla": h.reg(ts + r"\WinStations\RDP-Tcp", "UserAuthentication"),
            "port": h.reg(ts + r"\WinStations\RDP-Tcp", "PortNumber"),
            "remote_assistance": h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\Remote Assistance", "fAllowToGetHelp"),
            "client_mru": len(_vals(r"HKCU\Software\Microsoft\Terminal Server Client\Default") or {})}


_TOOL_RX = {
    "remote_access": re.compile(r"tailscale|anydesk|parsec|chrome remote desktop|teamviewer|rustdesk|sunshine|moonlight|realvnc|tightvnc|ultravnc|tigervnc|splashtop|hamachi|ngrok|cloudflared|nomachine|jump desktop|screenconnect|netbird", re.I),
    "vpn": re.compile(r"wireguard|openvpn|nordvpn|proton ?vpn|mullvad|expressvpn|cisco (anyconnect|secure client)|globalprotect|forticlient|surfshark|windscribe|private internet access|cloudflare warp|cloudflare one|zerotier|pritunl|softether|tunnelbear|zscaler|netskope", re.I),
    "virtualization": re.compile(r"virtualbox|vmware|qemu|docker desktop|podman|multipass|vagrant|sandboxie", re.I),
}
_SVC = {"Tailscale": "remote_access", "AnyDesk": "remote_access", "Parsec": "remote_access", "chromoting": "remote_access",
        "TeamViewer": "remote_access", "RustDesk": "remote_access", "SunshineService": "remote_access",
        "CloudflareWARP": "vpn", "WireGuardManager": "vpn", "OpenVPNService": "vpn", "ZeroTierOneService": "vpn",
        "MullvadVPN": "vpn", "nordvpn-service": "vpn", "ProtonVPN Service": "vpn", "com.docker.service": "virtualization",
        "vmms": "virtualization"}


def _inventory(h):
    if hasattr(h, "_hi_inv"):
        return h._hi_inv
    hits = {k: [] for k in _TOOL_RX}
    seen, n = set(), 0
    for root in (r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                 r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
                 r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall"):
        for s in _subs(h, root)[:1500]:
            name = h.reg(root + "\\" + s, "DisplayName")
            if not isinstance(name, str):
                continue
            n += 1
            for cat, rx in _TOOL_RX.items():
                if rx.search(name) and name not in seen:
                    seen.add(name)
                    hits[cat].append({"name": name, "version": h.reg(root + "\\" + s, "DisplayVersion"),
                                      "install_date": h.reg(root + "\\" + s, "InstallDate")})
    svcs = {}
    for svc, cat in _SVC.items():
        st = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services" + "\\" + svc, "Start")
        if st is not None:
            svcs[svc] = {"category": cat, "start": st}
    h._hi_inv = {"scanned": n, "hits": hits, "services": svcs}
    return h._hi_inv


@probe(id="security.remote_tools", level="L1", family="security", tier="T0", collect="core")
def remote_tools(h, facts):
    """Remote-access tools and virtualization from Uninstall keys + service keys (AnyDesk, Parsec, Tailscale, Sunshine, Docker...)."""
    inv = _inventory(h)
    svc = {k: v["start"] for k, v in inv["services"].items() if v["category"] in ("remote_access", "virtualization")}
    return {"present": True, "uninstall_scanned": inv["scanned"], "remote_access": inv["hits"]["remote_access"],
            "virtualization": inv["hits"]["virtualization"], "services": svc,
            "tailscale_cli": h.exists(r"%ProgramFiles%\Tailscale\tailscale.exe")}


@probe(id="security.secureboot", level="L1", family="security", tier="T0", collect="core")
def secureboot(h, facts):
    """UEFI Secure Boot state from registry."""
    v = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\SecureBoot\State", "UEFISecureBootEnabled")
    return {"present": v is not None, "secure_boot": bool(v) if v is not None else None}


@probe(id="security.smartscreen", level="L1", family="security", tier="T0", collect="core")
def smartscreen(h, facts):
    """SmartScreen and Smart App Control state (absent SmartScreenEnabled = default on)."""
    ex = h.reg(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer", "SmartScreenEnabled")
    pol = h.reg(r"HKLM\SOFTWARE\Policies\Microsoft\Windows\System", "EnableSmartScreen")
    sac = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy", "VerifiedAndReputablePolicyState")
    return {"present": True, "explorer": ex if ex is not None else "default_on", "policy": pol,
            "edge_user": h.reg(r"HKCU\Software\Microsoft\Edge\SmartScreenEnabled", ""),
            "smart_app_control": {0: "off", 1: "on", 2: "evaluation"}.get(sac, sac)}


@probe(id="security.telemetry", level="L1", family="security", tier="T0", collect="core")
def telemetry(h, facts):
    """Diagnostic data level, advertising ID, tailored experiences: privacy-conscious toggles."""
    return {"present": True,
            "allow_telemetry": h.reg(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\DataCollection", "AllowTelemetry"),
            "policy_allow_telemetry": h.reg(r"HKLM\SOFTWARE\Policies\Microsoft\Windows\DataCollection", "AllowTelemetry"),
            "diagtrack_start": h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services\DiagTrack", "Start"),
            "advertising_id": h.reg(r"HKCU\Software\Microsoft\Windows\CurrentVersion\AdvertisingInfo", "Enabled"),
            "tailored_experiences": h.reg(r"HKCU\Software\Microsoft\Windows\CurrentVersion\Privacy", "TailoredExperiencesWithDiagnosticDataEnabled"),
            "recall_disabled_policy": h.reg(r"HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsAI", "DisableAIDataAnalysis")}


@probe(id="security.uac", level="L1", family="security", tier="T0", collect="core")
def uac(h, facts):
    """UAC slider level from EnableLUA, ConsentPromptBehaviorAdmin, PromptOnSecureDesktop."""
    k = r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System"
    lua, c, s = h.reg(k, "EnableLUA"), h.reg(k, "ConsentPromptBehaviorAdmin"), h.reg(k, "PromptOnSecureDesktop")
    if lua == 0:
        lvl = "off"
    elif c == 2:
        lvl = "4_always_notify"
    elif c == 5 and s == 1:
        lvl = "3_default"
    elif c == 5 and s == 0:
        lvl = "2_no_dim"
    elif c == 0:
        lvl = "1_never_notify"
    else:
        lvl = f"custom_c{c}_s{s}"
    return {"present": lua is not None, "level": lvl, "enable_lua": lua, "consent_prompt_admin": c,
            "secure_desktop": s, "filter_admin_token": h.reg(k, "FilterAdministratorToken")}


@probe(id="acct.admins", level="L2", family="security", tier="T0", collect="extended", gate="security.uac")
def acct_admins(h, facts):
    """Administrators group member count by kind via NetLocalGroupGetMembers (no net.exe spawn); names never emitted."""
    import ctypes
    from ctypes import wintypes
    adv, net = ctypes.WinDLL("advapi32"), ctypes.WinDLL("netapi32")
    psid = ctypes.c_void_p()
    if not adv.ConvertStringSidToSidW(ctypes.c_wchar_p("S-1-5-32-544"), ctypes.byref(psid)):
        return None
    name, dom = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
    cn, cd, use = wintypes.DWORD(256), wintypes.DWORD(256), wintypes.DWORD()
    ok = adv.LookupAccountSidW(None, psid, name, ctypes.byref(cn), dom, ctypes.byref(cd), ctypes.byref(use))
    ctypes.windll.kernel32.LocalFree(psid)
    if not ok:
        return None

    class LGMI2(ctypes.Structure):
        _fields_ = [("sid", ctypes.c_void_p), ("use", wintypes.DWORD), ("name", wintypes.LPWSTR)]
    net.NetLocalGroupGetMembers.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
                                            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                            ctypes.c_void_p]
    buf = ctypes.c_void_p()
    read, total = wintypes.DWORD(), wintypes.DWORD()
    rc = net.NetLocalGroupGetMembers(None, name.value, 2, ctypes.byref(buf), 0xFFFFFFFF, ctypes.byref(read), ctypes.byref(total), None)
    if rc != 0 or not buf:
        return {"present": False, "error": rc}
    me = _current_sid(h)
    kinds = {"builtin_admin": 0, "current_user": 0, "local_other": 0, "group": 0, "domain_or_entra": 0}
    try:
        arr = ctypes.cast(buf, ctypes.POINTER(LGMI2 * read.value)).contents
        for m in arr:
            s = ctypes.c_wchar_p()
            sid = ""
            if adv.ConvertSidToStringSidW(ctypes.c_void_p(m.sid), ctypes.byref(s)):
                sid = s.value or ""
                ctypes.windll.kernel32.LocalFree(s)
            if me and sid == me:
                kinds["current_user"] += 1
            elif sid.endswith("-500") and sid.startswith("S-1-5-21-"):
                kinds["builtin_admin"] += 1
            elif m.use in (2, 4):  # SidTypeGroup, SidTypeAlias
                kinds["group"] += 1
            elif sid.startswith("S-1-12-1-") or (me and sid.startswith("S-1-5-21-") and sid.rsplit("-", 1)[0] != me.rsplit("-", 1)[0]):
                kinds["domain_or_entra"] += 1
            else:
                kinds["local_other"] += 1
    finally:
        net.NetApiBufferFree(buf)
    return {"present": True, "count": read.value, "kinds": kinds, "current_is_member": kinds["current_user"] > 0}


@probe(id="security.bcdedit", level="L2", family="security", tier="T0", collect="extended", gate="security.uac", needs_admin=True)
def bcdedit(h, facts):
    """Boot flags: testsigning, hypervisorlaunchtype (Docker/WSL readiness), isolatedcontext. Admin only."""
    out = _run_text(h, ["bcdedit", "/enum", "{current}"], 4000)
    if not out:
        return None
    flags = {}
    for line in out.splitlines():
        m = re.match(r"^(testsigning|nointegritychecks|hypervisorlaunchtype|flightsigning|debug|bootdebug|isolatedcontext)\s+(\S+)", line.strip(), re.I)
        if m:
            flags[m.group(1).lower()] = m.group(2)
    return {"present": True, "lines": len(out.splitlines()), **flags}


_PS_AV = r"""
try {
  $av = @(Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntiVirusProduct -ErrorAction Stop | ForEach-Object {
    $s = [int]$_.productState
    [ordered]@{ name = $_.displayName; state = ('0x{0:X6}' -f $s); enabled = (($s -band 0x1000) -ne 0); up_to_date = (($s -band 0x10) -eq 0) } })
  [ordered]@{ present = ($av.Count -gt 0); count = $av.Count; third_party = @($av | Where-Object { $_.name -notmatch 'Windows Defender|Microsoft Defender' }).Count; products = $av }
} catch { [ordered]@{ present = $false; error = $_.Exception.GetType().Name } }
"""
ps_probe("security.av_products", _PS_AV, level="L2", family="security", tier="T0", collect="extended", gate="security.uac",
         doc="Registered antivirus products (SecurityCenter2); third-party AV presence.")

_PS_DEFENDER = r"""
try {
  $s = Get-MpComputerStatus -ErrorAction Stop
  [ordered]@{ present = $true; rtp = $s.RealTimeProtectionEnabled; av = $s.AntivirusEnabled; mode = [string]$s.AMRunningMode;
    tamper = $s.IsTamperProtected; sig_age_d = $s.AntivirusSignatureAge; quick_scan_age_d = $s.QuickScanAge;
    full_scan_age_d = $(if ($s.FullScanAge -ge 4294967295) { 'never' } else { $s.FullScanAge }); smart_app_control = [string]$s.SmartAppControlState }
} catch { [ordered]@{ present = $false; error = $_.Exception.GetType().Name } }
"""
ps_probe("security.defender_status_ps", _PS_DEFENDER, level="L2", family="security", tier="T0", collect="deep",
         gate="security.defender_exclusions_count",
         doc="Get-MpComputerStatus route (2-4 s on ARM64). Fallback and cross-check for security.defender_status.")

_SAC_STATE = {0: "Off", 1: "On", 2: "Evaluation"}


def _ft_bytes_days(b):
    """Whole days since a FILETIME stored as REG_BINARY (Get-MpComputerStatus *Age semantics)."""
    if not isinstance(b, (bytes, bytearray)) or len(b) < 8:
        return None, None
    iso = _ft_iso(int.from_bytes(b[:8], "little"))
    d = _days_ago(iso)
    return iso, (None if d is None else int(d))


@probe(id="security.defender_status", level="L2", family="security", tier="T0", collect="core",
       gate="security.defender_exclusions_count")
def defender_status(h, facts):
    """Defender real-time, tamper, signature and scan ages from the registry (same keys as the Get-MpComputerStatus route)."""
    base = r"HKLM\SOFTWARE\Microsoft\Windows Defender"
    pol = r"HKLM\SOFTWARE\Policies\Microsoft\Windows Defender"
    running = h.reg(base, "IsServiceRunning")
    if running is None:
        return None
    running = running == 1
    rtm_off = 1 in (h.reg(base + r"\Real-Time Protection", "DisableRealtimeMonitoring"),
                    h.reg(pol + r"\Real-Time Protection", "DisableRealtimeMonitoring"))
    av_off = 1 in (h.reg(base, "DisableAntiVirus"), h.reg(base, "DisableAntiSpyware"), h.reg(pol, "DisableAntiSpyware"))
    passive = 1 in (h.reg(base, "PassiveMode"),
                    h.reg(r"HKLM\SOFTWARE\Policies\Microsoft\Windows Advanced Threat Protection", "ForceDefenderPassiveMode"))
    tp = h.reg(base + r"\Features", "TamperProtection")
    sig_iso, sig_age = _ft_bytes_days(h.reg(base + r"\Signature Updates", "AVSignatureApplied"))
    scan_iso, scan_age = _ft_bytes_days(h.reg(base + r"\Scan", "LastScanRun"))
    scan_type = h.reg(base + r"\Scan", "LastScanType")
    sac = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy", "VerifiedAndReputablePolicyState")
    excl = {}
    for kind in h.reg(base + r"\Exclusions") or []:
        v = _vals(base + r"\Exclusions" + "\\" + kind)
        excl[kind.lower()] = len(v) if v is not None else "unknown"
    return {
        "present": True,
        "rtp": running and not av_off and not rtm_off,
        "av": running and not av_off,
        "mode": "Not running" if not running else ("Passive Mode" if passive else "Normal"),
        "tamper": None if tp is None else bool(tp & 1),
        "sig_age_d": sig_age,
        # LastScanRun holds only the most recent scan; the other scan type's age is unknown, not "never".
        "quick_scan_age_d": scan_age if scan_type == 1 else None,
        "full_scan_age_d": scan_age if scan_type == 2 else None,
        "smart_app_control": _SAC_STATE.get(sac),
        "route": "registry",
        "service_running": running,
        "product_status": h.reg(base, "ProductStatus"),
        "pua": h.reg(base, "PUAProtection"),
        "maps": h.reg(base + r"\Spynet", "SpyNetReporting"),
        "sig_version": h.reg(base + r"\Signature Updates", "AVSignatureVersion"),
        "engine_version": h.reg(base + r"\Signature Updates", "EngineVersion"),
        "sig_applied": sig_iso,
        "last_scan": scan_iso,
        "last_scan_type": {1: "quick", 2: "full", 3: "custom"}.get(scan_type, scan_type),
        "exclusion_counts": excl,
    }

_PS_SERVICES = r"""
$n = 'Tailscale','AnyDesk','Parsec','chromoting','TeamViewer','RustDesk','SunshineService','sshd','ssh-agent','TermService','WinRM','RemoteRegistry','WireGuardManager','OpenVPNService','CloudflareWARP','ZeroTierOneService','WinDefend','MpsSvc','wscsvc','vmms','com.docker.service','WbioSrvc','NgcSvc','BDESVC'
$r = [ordered]@{}
Get-Service -Name $n -ErrorAction SilentlyContinue | ForEach-Object { $r[$_.Name] = [string]$_.Status }
[ordered]@{ present = ($r.Count -gt 0); running = @($r.Keys | Where-Object { $r[$_] -eq 'Running' }); stopped = @($r.Keys | Where-Object { $r[$_] -ne 'Running' }) }
"""
ps_probe("security.services_state", _PS_SERVICES, level="L2", family="security", tier="T0", collect="extended", gate="security.uac",
         doc="Running state of remote-access, security and virtualization services (SCM).")

_PS_VBS = r"""
try {
  $g = Get-CimInstance -Namespace root/Microsoft/Windows/DeviceGuard -ClassName Win32_DeviceGuard -ErrorAction Stop
  $hv = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\DeviceGuard\Scenarios\HypervisorEnforcedCodeIntegrity' -Name Enabled -ErrorAction SilentlyContinue).Enabled
  [ordered]@{ present = $true; vbs_status = $g.VirtualizationBasedSecurityStatus; services_running = @($g.SecurityServicesRunning);
    services_configured = @($g.SecurityServicesConfigured); hvci_reg = $hv }
} catch { [ordered]@{ present = $false; error = $_.Exception.GetType().Name } }
"""
ps_probe("security.vbs_hvci", _PS_VBS, level="L2", family="security", tier="T0", collect="extended", gate="security.uac",
         doc="VBS/HVCI runtime status (Win32_DeviceGuard) plus HVCI registry flag.")

_PS_BITLOCKER = r"""
try {
  $v = @(Get-BitLockerVolume -ErrorAction Stop | ForEach-Object { [ordered]@{ mount = $_.MountPoint; type = [string]$_.VolumeType;
    status = [string]$_.VolumeStatus; protection = [string]$_.ProtectionStatus; method = [string]$_.EncryptionMethod;
    protectors = @($_.KeyProtector | ForEach-Object { [string]$_.KeyProtectorType }) } })
  [ordered]@{ present = $true; volumes = $v }
} catch { [ordered]@{ present = $false; error = $_.Exception.GetType().Name } }
"""
ps_probe("security.bitlocker", _PS_BITLOCKER, level="L2", family="security", tier="T0", collect="deep", gate="security.uac",
         needs_admin=True, doc="BitLocker status per volume, protector types only. Admin only; non-admin = unknown.")

_PS_CERTS = r"""
$rx = 'mitmproxy|fiddler|charles|burp|portswigger|mkcert|development|testing|test only|zscaler|netskope|fortinet|umbrella|kaspersky|avast|eset|bitdefender'
$cu = @(Get-ChildItem Cert:\CurrentUser\Root); $lm = @(Get-ChildItem Cert:\LocalMachine\Root)
[ordered]@{ present = $true; cu_roots = $cu.Count; lm_roots = $lm.Count;
  flagged = @(($cu + $lm) | Where-Object { $_.Subject -match $rx } | ForEach-Object { $_.Subject -replace '^.*?CN=([^,]+).*$', '$1' } | Select-Object -Unique -First 10) }
"""
ps_probe("security.cert_roots", _PS_CERTS, level="L2", family="security", tier="T0", collect="deep", gate="security.uac",
         doc="Root store counts; flags interception/test/dev roots by CN.")

_PS_DEFPREFS = r"""
try {
  $p = Get-MpPreference -ErrorAction Stop
  $rx = [regex]::Escape($env:USERPROFILE)
  $noise = '(^|\\)(hn-e2e|ns960|ns923[^\\]*|lhm|shots|user-insights-lab|hermes-[^\\]*)(\\|$)'
  $paths = @($p.ExclusionPath | Where-Object { $_ } | ForEach-Object { $_ -replace $rx, '%USERPROFILE%' })
  [ordered]@{ present = $true; exclusion_paths = @($paths | Where-Object { $_ -notmatch $noise });
    operator_excluded = @($paths | Where-Object { $_ -match $noise }).Count;
    exclusion_processes = @($p.ExclusionProcess | Where-Object { $_ }).Count; exclusion_extensions = @($p.ExclusionExtension | Where-Object { $_ });
    pua = $p.PUAProtection; maps = $p.MAPSReporting; cfa = $p.EnableControlledFolderAccess; network_protection = $p.EnableNetworkProtection }
} catch { [ordered]@{ present = $false; error = $_.Exception.GetType().Name } }
"""
ps_probe("security.defender_prefs", _PS_DEFPREFS, level="L2", family="security", tier="T2", collect="deep",
         gate="security.defender_exclusions_count", doc="Defender preferences; exclusion paths are T2 (reveal game drives, lab dirs).")

_PS_TPM = r"""
try {
  $t = Get-Tpm -ErrorAction Stop
  $w = Get-CimInstance -Namespace root/cimv2/Security/MicrosoftTpm -ClassName Win32_Tpm -ErrorAction SilentlyContinue
  [ordered]@{ present = [bool]$t.TpmPresent; ready = [bool]$t.TpmReady; manufacturer = ([string]$t.ManufacturerIdTxt).Trim([char]0, ' '); spec = [string]$w.SpecVersion }
} catch { [ordered]@{ present = $false; error = $_.Exception.GetType().Name } }
"""
ps_probe("security.tpm", _PS_TPM, level="L2", family="security", tier="T0", collect="deep", gate="security.uac",
         needs_admin=True, doc="TPM presence/readiness and spec version.")


# ================================================================ network

@probe(id="net.hosts_proxy", level="L1", family="network", tier="T0", collect="core")
def hosts_proxy(h, facts):
    """hosts file active-entry counts (block/loopback) and user proxy flags; entries never emitted."""
    hp = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "drivers", "etc", "hosts")
    n = blk = loop = 0
    try:
        with open(hp, encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f):
                if i > 200000:
                    break
                s = ln.split("#", 1)[0].strip()
                if not s:
                    continue
                n += 1
                ip = s.split()[0]
                if ip in ("0.0.0.0", "::"):
                    blk += 1
                elif ip.startswith("127.") or ip == "::1":
                    loop += 1
    except OSError:
        return None
    isk = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings"
    ps = h.reg(isk, "ProxyServer")
    return {"present": True, "hosts_entries": n, "hosts_block": blk, "hosts_loopback": loop,
            "hosts_blocklist": blk >= 100, "proxy_enable": h.reg(isk, "ProxyEnable"),
            "proxy_kind": None if not ps else ("loopback" if re.search(r"127\.0\.0\.1|localhost", ps) else "remote"),
            "pac": h.reg(isk, "AutoConfigURL") is not None}


_VPN_DIRS = {"Tailscale": r"%ProgramFiles%\Tailscale", "Cloudflare WARP": r"%ProgramFiles%\Cloudflare\Cloudflare WARP",
             "WireGuard": r"%ProgramFiles%\WireGuard", "OpenVPN": r"%ProgramFiles%\OpenVPN",
             "OpenVPN Connect": r"%ProgramFiles%\OpenVPN Connect", "ZeroTier": r"%ProgramData%\ZeroTier",
             "Mullvad": r"%ProgramFiles%\Mullvad VPN", "ProtonVPN": r"%ProgramFiles%\Proton\VPN"}


@probe(id="net.vpn_clients", level="L1", family="network", tier="T0", collect="core")
def vpn_clients(h, facts):
    """VPN/mesh clients (Tailscale, WARP, WireGuard, OpenVPN, ZeroTier...) by install dir, Uninstall and service keys."""
    inv = _inventory(h)
    dirs = sorted(k for k, p in _VPN_DIRS.items() if h.exists(p))
    svc = {k: v["start"] for k, v in inv["services"].items() if v["category"] == "vpn" or k == "Tailscale"}
    return {"present": True, "dirs": dirs, "uninstall": inv["hits"]["vpn"], "services": svc,
            "count": len(set(dirs) | {x["name"] for x in inv["hits"]["vpn"]})}


@probe(id="net.wifi_profiles", level="L1", family="network", tier="T0", collect="core")
def wifi_profiles(h, facts):
    """Saved Wi-Fi profile count from Wlansvc XML file count per interface; files never opened, SSIDs never read."""
    base = os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), r"Microsoft\Wlansvc\Profiles\Interfaces")
    ifs = h.list_dir(base, 50)
    if not ifs:
        return {"present": h.exists(base), "profiles": 0, "interfaces": 0, "readable": h.exists(base)}
    per = [sum(1 for n in h.list_dir(os.path.join(base, i), 500) if n.lower().endswith(".xml")) for i in ifs]
    return {"present": True, "count": sum(per), "interfaces": len(ifs), "profiles_max_per_interface": max(per)}


_NET_CLASS = r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e972-e325-11ce-bfc1-08002be10318}"
_NET_CONN = r"HKLM\SYSTEM\CurrentControlSet\Control\Network\{4D36E972-E325-11CE-BFC1-08002BE10318}"


def _adapter_kind(desc: str, iftype):
    d = desc.lower()
    if "tailscale" in d or "wireguard" in d or "tap-" in d or "wintun" in d or "warp" in d or "zerotier" in d:
        return "tunnel"
    if "hyper-v" in d or "virtual" in d or "vmware" in d or "virtualbox" in d or "vethernet" in d:
        return "virtual"
    if "bluetooth" in d:
        return "bluetooth"
    if iftype == 71 or "wi-fi" in d or "wireless" in d or "wlan" in d or "802.11" in d:
        return "wifi"
    if iftype == 6 or "ethernet" in d or "gbe" in d or "lan" in d:
        return "ethernet"
    if iftype == 243 or "mobile" in d or "wwan" in d:
        return "mobile"
    return "other"


@probe(id="net.adapters", level="L2", family="network", tier="T0", collect="extended", gate="net.hosts_proxy")
def net_adapters(h, facts):
    """Network adapters from the Network class registry (Get-NetAdapter is 4.6 s on ARM64); hidden/miniport adapters skipped."""
    rows = []
    for s in _subs(h, _NET_CLASS)[:200]:
        if not s.isdigit():
            continue
        v = _vals(_NET_CLASS + "\\" + s) or {}
        ch = v.get("Characteristics") or 0
        guid = v.get("NetCfgInstanceId")
        desc = v.get("DriverDesc") or ""
        if not guid or ch & 0x8 or re.search(r"WAN Miniport|Kernel Debug|Teredo|6to4|IP-HTTPS|ISATAP|Microsoft Wi-Fi Direct|RAS Async", desc, re.I):
            continue
        name = h.reg(_NET_CONN + "\\" + guid + r"\Connection", "Name")
        ift = v.get("*IfType")
        try:
            ift = int(ift) if ift is not None else None
        except (TypeError, ValueError):
            ift = None
        tcp = r"HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces" + "\\" + guid.lower()
        ip = h.reg(tcp, "DhcpIPAddress") or h.reg(tcp, "IPAddress")
        if isinstance(ip, list):
            ip = ip[0] if ip else None
        has_ip = bool(ip) and ip not in ("0.0.0.0",)
        rows.append({"name": name, "desc": desc, "kind": _adapter_kind(desc, ift), "physical": bool(ch & 0x4),
                     "has_ipv4": has_ip})
    if not rows:
        return None
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"present": True, "count": len(rows), "by_kind": kinds, "adapters": rows[:20]}


@probe(id="net.connection_profiles", level="L2", family="network", tier="T0", collect="extended", gate="net.hosts_proxy")
def connection_profiles(h, facts):
    """Network category (public/private/domain) by network type from NetworkList (Get-NetConnectionProfile is 0.9 s on ARM64); no names."""
    rows = _network_profiles(h)
    if not rows:
        return None
    grid = {}
    for r in rows:
        grid.setdefault(r["type"], {}).setdefault(r["category"], 0)
        grid[r["type"]][r["category"]] += 1
    newest = sorted((r for r in rows if r["last"]), key=lambda r: r["last"], reverse=True)[:3]
    return {"present": True, "by_type_category": grid,
            "private_physical": sum(1 for r in rows if r["category"] == "private" and r["type"] in ("wireless", "wired")),
            "most_recent": [{"type": r["type"], "category": r["category"], "last": r["last"]} for r in newest]}


_DNS_KNOWN = {"1.1.1.1": "cloudflare", "1.0.0.1": "cloudflare", "8.8.8.8": "google", "8.8.4.4": "google",
              "9.9.9.9": "quad9", "149.112.112.112": "quad9", "208.67.222.222": "opendns", "208.67.220.220": "opendns",
              "100.100.100.100": "tailscale_magicdns", "94.140.14.14": "adguard", "94.140.15.15": "adguard",
              "45.90.28.0": "nextdns", "45.90.30.0": "nextdns", "76.76.2.0": "controld", "76.76.10.0": "controld"}


def _dns_class(ip: str) -> str:
    if ip in _DNS_KNOWN:
        return _DNS_KNOWN[ip]
    if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|fe80|fec0|fd)", ip, re.I):
        return "private_lan"
    if ip.startswith("127.") or ip == "::1":
        return "loopback"
    if ip.startswith("100.") and 64 <= int(ip.split(".")[1]) <= 127:
        return "cgnat_or_tailnet"
    return "other_public"


@probe(id="net.dns", level="L2", family="network", tier="T0", collect="extended", gate="net.hosts_proxy")
def net_dns(h, facts):
    """DNS resolver classes per interface from Tcpip Interfaces registry (raw IPs never emitted) plus DoH table size."""
    base = r"HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces"
    classes, static = {}, 0
    n_if = 0
    for g in _subs(h, base)[:100]:
        ns = h.reg(base + "\\" + g, "NameServer") or ""
        dhcp = h.reg(base + "\\" + g, "DhcpNameServer") or ""
        servers = [x for x in re.split(r"[ ,]+", ns or dhcp) if x]
        if not servers:
            continue
        n_if += 1
        static += 1 if ns else 0
        for ip in servers:
            c = _dns_class(ip)
            classes[c] = classes.get(c, 0) + 1
    doh = len(_subs(h, r"HKLM\SYSTEM\CurrentControlSet\Services\Dnscache\Parameters\DohWellKnownServers"))
    return {"present": bool(n_if), "interfaces_with_dns": n_if, "static_dns_interfaces": static,
            "resolver_classes": classes, "doh_known_servers": doh,
            "custom_resolver": any(c not in ("private_lan", "other_public", "loopback") for c in classes)}


@probe(id="net.tailscale", level="L2", family="network", tier="T1", collect="extended", gate="net.vpn_clients")
def net_tailscale(h, facts):
    """Tailscale state via 'tailscale status --json': peer counts by OS and online count; peer names never emitted."""
    import json
    exe = h.expand(r"%ProgramFiles%\Tailscale\tailscale.exe")
    if not os.path.exists(exe):
        return None
    out = _run_text(h, [exe, "status", "--json"], 3000)
    try:
        j = json.loads(out) if out else None
    except ValueError:
        j = None
    if not isinstance(j, dict):
        return {"present": True, "status": "unparsed"}
    peers = list((j.get("Peer") or {}).values())
    by_os = {}
    for p in peers:
        o = p.get("OS") or "?"
        by_os[o] = by_os.get(o, 0) + 1
    return {"present": True, "version": (j.get("Version") or "").split("-")[0], "backend": j.get("BackendState"),
            "peers": len(peers), "online": sum(1 for p in peers if p.get("Online")), "peers_by_os": by_os,
            "exit_node_in_use": any(p.get("ExitNode") for p in peers),
            "magicdns": bool((j.get("CurrentTailnet") or {}).get("MagicDNSEnabled"))}


# ================================================================ usage (L3 rule as a probe)

_KP41_Q = "*[System[Provider[@Name='Microsoft-Windows-Kernel-Power'] and (EventID=41)]]"


def _kp41_events(h):
    import xml.etree.ElementTree as ET
    out = _run_text(h, ["wevtutil", "qe", "System", "/q:" + _KP41_Q, "/c:500", "/rd:true", "/f:xml"], 8000)
    if out is None:
        return None
    txt = out.strip()
    if not txt:
        return []
    ns = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
    try:
        root = ET.fromstring("<r>" + txt + "</r>")
    except ET.ParseError:
        return None
    rows = []
    for ev in root.findall("e:Event", ns):
        tc = ev.find("e:System/e:TimeCreated", ns)
        data = {d.get("Name"): (d.text or "") for d in ev.findall("e:EventData/e:Data", ns)}
        rows.append({"time": (tc.get("SystemTime") if tc is not None else "")[:19],
                     "bugcheck": int(data.get("BugcheckCode") or 0),
                     "button": int(data.get("PowerButtonTimestamp") or 0)})
    return rows


@probe(id="l3.power_event_split", level="L2", family="usage", tier="T1", collect="extended", gate="os.build")
def power_event_split(h, facts):
    """Kernel-Power 41 split per event: crash (BugcheckCode!=0), button_held (PowerButtonTimestamp!=0), else power_removed.
    Only crash is a health signal; power_removed is a power-off habit (wall/PSU switch or outage). Never flag by count."""
    rows, via = None, "wevtutil"
    rec = facts.get("eventlog.power_history")
    ph = rec.get("value") if isinstance(rec, dict) and rec.get("status") == "ok" else None
    if isinstance(ph, dict) and isinstance(ph.get("kp41"), list) and isinstance(ph.get("kp41_split"), dict):
        total = sum(int(v or 0) for v in ph["kp41_split"].values())
        if total == len(ph["kp41"]):
            rows = []
            for e in ph["kp41"]:
                bc = str(e.get("bugcheck") or "0")
                rows.append({"time": str(e.get("t") or "")[:19], "bugcheck": int(bc, 16) if bc.lower().startswith("0x") else int(bc or 0),
                             "button": 1 if e.get("class") == "button_held" else 0})
            via = "eventlog.power_history"
    if rows is None:
        rows = _kp41_events(h)
    if rows is None:
        return None
    counts = {"crash": 0, "button_held": 0, "power_removed": 0}
    last30 = {"crash": 0, "button_held": 0, "power_removed": 0}
    codes, events = {}, []
    for r in rows:
        cls = "crash" if r["bugcheck"] else ("button_held" if r["button"] else "power_removed")
        counts[cls] += 1
        d = _days_ago(r["time"]) if r["time"] else None
        if d is not None and d <= 30:
            last30[cls] += 1
        if r["bugcheck"]:
            codes[hex(r["bugcheck"])] = codes.get(hex(r["bugcheck"]), 0) + 1
        events.append({"date": r["time"][:10], "class": cls, "bugcheck": hex(r["bugcheck"]) if r["bugcheck"] else None})
    return {"present": True, "via": via, "events": len(rows), **counts,
            "crash_30d": last30["crash"], "button_held_30d": last30["button_held"], "power_removed_30d": last30["power_removed"],
            "bugcheck_codes": codes, "health_flag_crash_30d": last30["crash"] > 0,
            "power_off_habit_hint": counts["power_removed"] >= 3, "timeline": events[:60]}
