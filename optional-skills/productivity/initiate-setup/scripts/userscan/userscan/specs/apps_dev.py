"""Installed software (apps) probes, plus the helpers apps_dev_agents and apps_dev_devenv share.

Registration only at import time. Shared reads (uninstall keys, appx repository, process snapshot,
Claude/Codex/Hermes stores) are memoised per run so several probes can reuse one read.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from ..registry import probe

try:
    import winreg  # noqa: F401
except ImportError:  # non-Windows import (e.g. `run.py --list` on the Mac)
    winreg = None

try:
    import tomllib
except ImportError:
    tomllib = None

APPS, AI, DEV = "apps", "ai_agents", "dev"
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# ------------------------------------------------------------------ shared helpers

_LOCK = threading.Lock()
_CACHE: dict = {}
_KEYLOCKS: dict = {}


def _memo(h, name, fn):
    key = (h.l0.get("run_id"), name)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        lk = _KEYLOCKS.setdefault(key, threading.Lock())
    with lk:
        if key in _CACHE:
            return _CACHE[key]
        v = fn()
        _CACHE[key] = v
        return v


def _U(h):
    return h.l0.get("home") or os.path.expanduser("~")


def _LA(h):
    return h.l0.get("localappdata") or os.environ.get("LOCALAPPDATA", "")


def _RA(h):
    return h.l0.get("appdata") or os.environ.get("APPDATA", "")


def _PD():
    return os.environ.get("ProgramData", r"C:\ProgramData")


def _PF():
    return os.environ.get("ProgramW6432") or os.environ.get("ProgramFiles") or r"C:\Program Files"


def _PF86():
    return os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"


def _day(t):
    try:
        return dt.datetime.fromtimestamp(float(t)).strftime("%Y-%m-%d") if t else None
    except (OverflowError, OSError, ValueError):
        return None


def _iso(ts):
    """unix s/ms/us/ns or ISO string -> local ISO minutes."""
    if ts is None or ts == "":
        return None
    if isinstance(ts, str):
        try:
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().isoformat(timespec="minutes")
        except ValueError:
            try:
                ts = float(ts)
            except ValueError:
                return None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts > 1e17:
        ts /= 1e9
    elif ts > 1e14:
        ts /= 1e6
    elif ts > 1e11:
        ts /= 1e3
    try:
        return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="minutes")
    except (OverflowError, OSError, ValueError):
        return None


def _mtime(p):
    try:
        return _day(os.stat(p).st_mtime)
    except OSError:
        return None


def _ctime(p):
    try:
        return _day(os.stat(p).st_ctime)
    except OSError:
        return None


def _isdir(p):
    try:
        return os.path.isdir(p)
    except OSError:
        return False


def _ls(p, cap=500):
    try:
        with os.scandir(p) as it:
            return [e.name for _, e in zip(range(cap), it)]
    except OSError:
        return None


def _read_json(p, maxb=5_000_000, jsonc=False):
    try:
        if os.path.getsize(p) > maxb:
            return None
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            txt = f.read()
        if jsonc:
            txt = _strip_jsonc(txt)
        return json.loads(txt)
    except (OSError, ValueError):
        return None


def _strip_jsonc(txt):
    out, i, n, in_str = [], 0, len(txt), False
    while i < n:
        c = txt[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(txt[i + 1]); i += 2; continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True; out.append(c)
        elif txt.startswith("//", i):
            j = txt.find("\n", i); i = n if j < 0 else j; continue
        elif txt.startswith("/*", i):
            j = txt.find("*/", i + 2); i = n if j < 0 else j + 2; continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _top(counter, n=10):
    return dict(collections.Counter({k: v for k, v in counter.items() if k is not None}).most_common(n))


_RED = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
        (re.compile(r"[A-Za-z]:\\[^\s\"']*"), "<path>"),
        (re.compile(r"(?<![\w])/(?:Users|home)/[^\s\"']*"), "<path>"),
        (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"), "<token>"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>")]


def _redact(s, n=70):
    if not isinstance(s, str):
        return None
    for rx, rep in _RED:
        s = rx.sub(rep, s)
    s = " ".join(s.split())
    return s[:n] + ("..." if len(s) > n else "")


def _copy_db(h, src, tag):
    """Copy an sqlite db (+wal/shm/journal) into scratch under a unique name. Returns path or None."""
    try:
        dst = os.path.join(h.scratch(), f"{tag}-{threading.get_ident()}-{os.path.basename(src)}")
        shutil.copyfile(src, dst)
        for suf in ("-wal", "-journal"):
            if os.path.exists(src + suf):
                shutil.copyfile(src + suf, dst + suf)
        return dst
    except OSError:
        return None


def _q(path, sql, args=()):
    import sqlite3
    con = sqlite3.connect(path, timeout=2)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def _cols(path, table):
    try:
        return {r[1] for r in _q(path, f"pragma table_info({table})")}
    except Exception:
        return set()


def _rmdb(p):
    for suf in ("", "-wal", "-journal", "-shm"):
        try:
            os.remove(p + suf)
        except OSError:
            pass


_PATHIDX = None
_PATHIDX_LOCK = threading.Lock()


def _path_index():
    """One listdir per PATH dir (interpreter dir removed: uv run prepends it). Cached per process."""
    global _PATHIDX
    with _PATHIDX_LOCK:
        if _PATHIDX is None:
            exe_dir = os.path.normcase(os.path.dirname(sys.executable).rstrip("\\/"))
            dirs, seen = [], set()
            for p in os.environ.get("PATH", "").split(os.pathsep):
                p = os.path.expandvars(p.strip().strip('"'))
                key = os.path.normcase(p.rstrip("\\/"))
                if not p or key == exe_dir or key in seen:
                    continue
                seen.add(key)
                names = _ls(p, 20000)
                if names:
                    dirs.append((p, {n.lower(): n for n in names}))
            exts = [e.lower() for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]
            _PATHIDX = (dirs, exts)
        return _PATHIDX


def _which(name):
    """shutil.which equivalent over the cached PATH index."""
    dirs, exts = _path_index()
    n = name.lower()
    cands = [n] if os.path.splitext(n)[1] in exts else [n + e for e in exts]
    for d, names in dirs:
        for c in cands:
            if c in names:
                return os.path.join(d, names[c])
    return None


def _spawn(h, args, timeout=8.0, env=None):
    """(rc, stdout, ms). Never used with bash.exe or pwsh. `env` defaults to h.child_env()."""
    exe = args[0]
    if exe.lower().endswith((".cmd", ".bat")):
        args = ["cmd.exe", "/d", "/c"] + args
    t = time.perf_counter()
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout, env=env or h.child_env(), creationflags=NOWIN,
                           stdin=subprocess.DEVNULL)
        out = (r.stdout or b"") or (r.stderr or b"")
        txt = out.decode("utf-8", "replace")
        if txt.count("\x00") > len(txt) // 4:
            txt = out.decode("utf-16-le", "replace")
        return r.returncode, txt.replace("\x00", "").strip(), round((time.perf_counter() - t) * 1000, 1)
    except Exception:
        return None, "", round((time.perf_counter() - t) * 1000, 1)


# operator noise: scratch/test artifacts that belong to the lab operator, not the user
OPERATOR_TOP_RE = re.compile(r"^(hn-e2e|ns960|ns923.*|lhm|shots|user-insights-lab|hermes-.+|hermes-lab)$", re.I)
OPERATOR_TASK_RE = re.compile(r"^(hermes-|lab-run|hn-|ns960|ns923|shot-live|shots|cua-driver-serve|ensure-sshd|user-insights|uil-|lhm)", re.I)


def _is_operator_path(h, p):
    """True if p is under %USERPROFILE%\\<op>, %LOCALAPPDATA%\\<op> or C:\\<op> for an operator top dir."""
    pn = os.path.normcase(os.path.abspath(p))
    for base in (_U(h), _LA(h), "C:\\"):
        b = os.path.normcase(os.path.abspath(base)).rstrip("\\/") + os.sep
        if pn.startswith(b):
            first = pn[len(b):].split(os.sep, 1)[0]
            if first and OPERATOR_TOP_RE.match(first):
                return True
    return False


# ------------------------------------------------------------------ registry helpers

def _open(hive, path, flag=0):
    try:
        return winreg.OpenKey(hive, path, 0, winreg.KEY_READ | flag)
    except OSError:
        return None


def _rv(k, name):
    try:
        return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


def _subkeys(k, cap=5000):
    i = 0
    while i < cap:
        try:
            yield winreg.EnumKey(k, i)
        except OSError:
            return
        i += 1


def _values(k, cap=2000):
    i = 0
    while i < cap:
        try:
            yield winreg.EnumValue(k, i)
        except OSError:
            return
        i += 1


def _ft_day(ft100ns):
    if not ft100ns:
        return None
    try:
        return (dt.datetime(1601, 1, 1) + dt.timedelta(microseconds=ft100ns // 10)).strftime("%Y-%m-%d")
    except OverflowError:
        return None


def _install_date(s):
    if isinstance(s, str):
        s = s.strip()
        if re.fullmatch(r"\d{8}", s):
            try:
                dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
                return f"{s[:4]}-{s[4:6]}-{s[6:]}"
            except ValueError:
                return None
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
            m, d, y = s.split("/")
            return f"{y}-{int(m):02d}-{int(d):02d}"
    return None


# ------------------------------------------------------------------ PE version info (ctypes)

_VER = None
_VER_LOCK = threading.Lock()
_COMPANY: dict = {}


def _verdll():
    global _VER
    with _VER_LOCK:
        if _VER is None:
            import ctypes
            from ctypes import wintypes
            v = ctypes.WinDLL("version")
            v.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
            v.GetFileVersionInfoSizeW.restype = wintypes.DWORD
            v.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
            v.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.POINTER(wintypes.UINT)]
            _VER = v
        return _VER


def _verbuf(path):
    import ctypes
    v = _verdll()
    n = v.GetFileVersionInfoSizeW(path, None)
    if not n:
        return None
    buf = ctypes.create_string_buffer(n)
    if not v.GetFileVersionInfoW(path, 0, n, buf):
        return None
    return buf


def _file_version(path, kind="product"):
    if not path or not os.path.isfile(path):
        return None
    try:
        import ctypes
        from ctypes import wintypes
        buf = _verbuf(path)
        if buf is None:
            return None
        p = ctypes.c_void_p(); ln = wintypes.UINT()
        if not _verdll().VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(ln)) or ln.value < 52:
            return None
        f = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
        ms, ls = (f[4], f[5]) if kind == "product" else (f[2], f[3])
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception:
        return None


def _company(path):
    if not path:
        return None
    key = path.lower()
    if key in _COMPANY:
        return _COMPANY[key]
    res = None
    try:
        import ctypes
        from ctypes import wintypes
        buf = _verbuf(path) if os.path.isfile(path) else None
        if buf is not None:
            v = _verdll()
            p = ctypes.c_void_p(); ln = wintypes.UINT()
            langs = []
            if v.VerQueryValueW(buf, r"\VarFileInfo\Translation", ctypes.byref(p), ctypes.byref(ln)) and ln.value:
                arr = ctypes.cast(p, ctypes.POINTER(wintypes.WORD * (ln.value // 2))).contents
                langs = [(arr[i], arr[i + 1]) for i in range(0, len(arr) - 1, 2)]
            langs += [(0x0409, 0x04B0), (0x0409, 0x04E4)]
            for lang, cp in langs:
                q = f"\\StringFileInfo\\{lang:04x}{cp:04x}\\CompanyName"
                if v.VerQueryValueW(buf, q, ctypes.byref(p), ctypes.byref(ln)) and ln.value:
                    res = ctypes.wstring_at(p, ln.value).split("\x00")[0].strip() or None
                    if res:
                        break
    except Exception:
        res = None
    _COMPANY[key] = res
    return res


def _exe_from_cmd(cmd):
    if not cmd or not isinstance(cmd, str):
        return None
    c = os.path.expandvars(cmd.strip())
    if c.startswith('"'):
        e = c[1:].split('"', 1)[0]
    else:
        m = re.match(r"(.+?\.(exe|dll|sys|cmd|bat|ps1|com))\b", c, re.I)
        e = m.group(1) if m else c.split(" ")[0]
    sr = os.environ.get("SystemRoot", r"C:\Windows")
    if e.lower().startswith("\\systemroot\\"):
        e = os.path.join(sr, e[12:])
    elif e.lower().startswith("system32\\"):
        e = os.path.join(sr, e)
    elif e.startswith("\\??\\"):
        e = e[4:]
    return e


# ------------------------------------------------------------------ taxonomy (from report 02)

OVERRIDES = [
    ("productivity", r"officehub|microsoft 365 copilot"),
    ("runtime", r"applicationcompatibilityenhancements|officepushnotification|office\.actionsserver|steamworks common"),
    ("hardware_vendor", r"asustek|asus\b|b9eced6f|armoury|rog |glidex|storycube|proart|screenxpert|dolby|intelligo|cirrus ?logic|realtek|appup\.intel|intel\(r\)|nvidiacorp|mediatek|lgelectronics|lg monitor|logi(tech)?\b|razer|patriot|openrgb|msi afterburner|rivatuner|ene\b|physx|nvidia (graphics|hd audio|app|control|display|frameview|localsystem)|nvidia broadcast|cpuid"),
    ("ai", r"ollama|openai|codex|chatgpt|claude|anthropic|hermes|nousresearch|\bcua\b|lm ?studio|gpt4all|comfyui|pinokio|anythingllm|jan\.ai|cursor|windsurf|copilot\b|perplexity|msty|koboldcpp|browser-use"),
    ("gaming", r"\bsteam\b|cd projekt|steam app \d+|\\steamapps\\|\\games\\|epic games|epic online|easyanticheat|easy anti-cheat|battleye|elytra|rockstar|riot|xbox|gamingapp|gameassist|\bgame(s)?\b|palworld|cyberpunk|helldivers|death stranding|dead cells|faster than light|disco elysium|hotline miami|stickman|universe sandbox|\bpeak\b|control resonant|deadpool|\bf1 \d\d|minecraft|gog galaxy|battle\.net|ubisoft|ea app|playnite|vortex|curseforge|solitaire"),
]
RULES = [
    ("runtime", r"visual c\+\+|redistributable|vcredist|\.net (runtime|framework|desktop)|desktop runtime|windows desktop runtime|webview2|directx|vulkanrt|openal|winappruntime|windowsappruntime|vclibs|ui\.xaml|\bcrt\b|application compatibility|compatibility fix|extension$|videoextension|imageextension|mediaextensions|d3dmappinglayers|webexperience|widgetsplatform|language ?experience|ink\.handwriting|steamworks common"),
    ("dev", r"visual studio|vscode|vs code|\bgit\b|github|node\.?js|nodejs|docker|wsl|subsystemforlinux|windowsterminal|terminal|powershell|openssh|ripgrep|python|jetbrains|pycharm|android studio|cmake|llvm|rust|golang|jdk|java|postman|wireshark|putty|winscp|devhome|dev home|sysinternals|msys|conda|\bsdk\b|cuda|hyper-v|virtualbox|vmware|neovim|sublime|notepad\+\+|devtoys|msbuild"),
    ("creative", r"adobe|photoshop|lightroom|premiere|illustrator|gimp|krita|inkscape|blender|davinci|obs ?studio|obs-studio|streamlabs|audacity|reaper|ableton|fl studio|handbrake|kdenlive|shotcut|paint|affinity|figma|canva|capcut|clipchamp|vlc|videolan|mpv|potplayer|zunemusic|media player|photos|camera|screensketch|snipping|soundrecorder|sketchup|maxon|red giant|cinema 4d|unity|unreal|godot|omniverse|autodesk|ffmpeg|sharex|voicemeeter"),
    ("comms", r"discord|slack|\bteams\b|msteams|zoom|skype|telegram|whatsapp|signal|thunderbird|outlook|messenger|webex|mattermost|beeper|yourphone|phone link|crossdevice|mixedrealitylink"),
    ("productivity", r"microsoft 365|\boffice\b|officehub|word|excel|powerpoint|onenote|notion|obsidian|logseq|evernote|todoist|todos|onedrive|dropbox|google drive|brave|chrome|firefox|\bedge\b|microsoftedge|opera|vivaldi|zen browser|acrobat|pdf|sumatra|libreoffice|sticky ?notes|notepad|calculator|journal|whiteboard|1password|bitwarden|keepass|everything|powertoys|flow launcher|raycast|bing(news|search|weather)|news|weather|aimgr|local ai manager|powerautomate"),
    ("utilities", r"7-?zip|winrar|peazip|nanazip|qbittorrent|torrent|ccleaner|revo|windirstat|wiztree|rufus|ventoy|crystaldisk|macrium|anydesk|teamviewer|rustdesk|parsec|tailscale|cloudflare|warp|wireguard|openvpn|mullvad|nordvpn|equalizer ?apo|peace\b|pawnio|autohotkey|sharpkeys|twinkle tray|translucenttb|startallback|explorerpatcher|quicklook|winget|desktopappinstaller|appinstaller|chocolatey|scoop|unigetui|quickassist|family|gethelp|feedbackhub|sechealth|alarms|clock|store|startexperiences|decompyle|pyinstxtractor"),
]
_CLS = [(c, re.compile(p, re.I)) for c, p in OVERRIDES + RULES]
USER_CATS = ["dev", "ai", "gaming", "creative", "comms", "productivity", "utilities", "hardware_vendor"]
_ALIASES = {"microsoft vs code": "visual studio code", "vscode": "visual studio code"}
_NOISE = {"microsoft", "version", "desktop", "app", "installer", "en", "us", "user", "the", "x64", "x86", "arm64"}


def _classify(name, publisher="", extra=""):
    s = f"{name} {publisher or ''} {extra or ''}"
    for c, rx in _CLS:
        if rx.search(s):
            return c
    return "other"


def _norm(name, appx=False):
    n = (name or "").strip()
    if appx and "." in n:
        n = re.sub(r"([a-z])([A-Z])", r"\1 \2", n.split(".")[-1])
    n = n.lower()
    n = _ALIASES.get(n, n)
    n = re.sub(r"\(.*?\)|v?\d+(\.\d+)+\S*|\b64-bit\b|\b32-bit\b", " ", n)
    return "".join(w for w in re.split(r"[^a-z0-9]+", n) if w and w not in _NOISE)


# ------------------------------------------------------------------ shared inventories

UNINSTALL = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
APPX_REPO = r"Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages"
APPX_PROV = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Appx\AppxAllUserStore\Applications"


def _uninstall(h):
    def build():
        views = [("HKLM", winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
                 ("HKLM_WOW6432", winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
                 ("HKCU", winreg.HKEY_CURRENT_USER, 0)]
        items, stats = [], {}
        for label, hive, flag in views:
            root = _open(hive, UNINSTALL, flag)
            if root is None:
                stats[label] = None
                continue
            st = {"keys": 0, "listed": 0, "system_component": 0, "updates": 0, "no_name": 0}
            for sk in _subkeys(root):
                st["keys"] += 1
                try:
                    k = winreg.OpenKey(root, sk)
                except OSError:
                    continue
                with k:
                    name = _rv(k, "DisplayName")
                    if not name:
                        st["no_name"] += 1; continue
                    if _rv(k, "SystemComponent") == 1:
                        st["system_component"] += 1; continue
                    if _rv(k, "ParentKeyName") or _rv(k, "ReleaseType") in ("Update", "Hotfix", "Security Update"):
                        st["updates"] += 1; continue
                    pub = _rv(k, "Publisher")
                    loc = _rv(k, "InstallLocation") or ""
                    st["listed"] += 1
                    items.append({"name": name, "publisher": pub, "version": _rv(k, "DisplayVersion"),
                                  "install_date": _install_date(_rv(k, "InstallDate")),
                                  "key_lastwrite": _ft_day(winreg.QueryInfoKey(k)[2]), "view": label,
                                  "category": _classify(name, pub, f"{sk} {loc}")})
            stats[label] = st
        return {"stats": stats, "items": items}
    return _memo(h, "uninstall", build)


def _uninst_match(h, pattern):
    rx = re.compile(pattern, re.I)
    return [i for i in _uninstall(h)["items"] if rx.search(i["name"] or "")]


def _appx(h):
    def build():
        k = _open(winreg.HKEY_CURRENT_USER, APPX_REPO)
        if k is None:
            return None
        prov = set()
        pk = _open(winreg.HKEY_LOCAL_MACHINE, APPX_PROV)
        if pk is not None:
            for fn in _subkeys(pk):
                prov.add(fn.split("_")[0].lower())
        windir = os.environ.get("SystemRoot", r"C:\Windows").lower()
        counts = {"total": 0, "framework": 0, "resource": 0, "system": 0, "kept": 0, "provisioned": 0, "user_added": 0}
        items, full = [], []
        for fullname in _subkeys(k):
            counts["total"] += 1
            parts = fullname.split("_")
            if len(parts) != 5:
                continue
            name, ver, arch, resid, pubid = parts
            full.append({"name": name, "version": ver, "arch": arch})
            try:
                s = winreg.OpenKey(k, fullname)
            except OSError:
                continue
            with s:
                if _rv(s, "Framework") == 1:
                    counts["framework"] += 1; continue
                if resid and resid != "~":
                    counts["resource"] += 1; continue
                root = (_rv(s, "PackageRootFolder") or "").lower()
                if not root or root.startswith(windir):
                    counts["system"] += 1; continue
                dn = _rv(s, "DisplayName") or ""
            if dn.startswith(("@", "ms-resource")):
                dn = None
            provisioned = name.lower() in prov
            counts["kept"] += 1
            counts["provisioned" if provisioned else "user_added"] += 1
            items.append({"name": name, "display": dn, "version": ver, "arch": arch, "provisioned": provisioned,
                          "first_seen": _ctime(os.path.join(_LA(h), "Packages", f"{name}_{pubid}")),
                          "category": _classify(f"{name} {dn or ''}")})
        return {"provisioned_apps": len(prov), "counts": counts, "items": items, "all": full}
    return _memo(h, "appx", build)


def _appx_match(h, pattern):
    a = _appx(h)
    if not a:
        return []
    rx = re.compile(pattern, re.I)
    seen, out = set(), []
    for p in a["all"]:
        if rx.search(p["name"]) and (p["name"], p["version"], p["arch"]) not in seen:
            seen.add((p["name"], p["version"], p["arch"]))
            out.append(p)
    return out


def _procs(h):
    def build():
        import ctypes
        from ctypes import wintypes

        class PE(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                        ("szExeFile", ctypes.c_wchar * 260)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
        k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        snap = k32.CreateToolhelp32Snapshot(2, 0)
        names = collections.Counter()
        if not snap or snap == wintypes.HANDLE(-1).value:
            return names
        e = PE(); e.dwSize = ctypes.sizeof(PE)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        n = 0
        while ok and n < 20000:
            names[e.szExeFile.lower()] += 1
            n += 1
            ok = k32.Process32NextW(snap, ctypes.byref(e))
        k32.CloseHandle(snap)
        return names
    return _memo(h, "procs", build)


def _running(h, pattern):
    rx = re.compile(pattern, re.I)
    return {k: v for k, v in _procs(h).items() if rx.search(k)}


# ================================================================== apps

@probe(id="apps.uninstall", level="L1", family=APPS, tier="T0", collect="core")
def apps_uninstall(h, facts):
    """Uninstall keys (HKLM 64/32-bit views + HKCU): every installed program with installer metadata."""
    u = _uninstall(h)
    items = u["items"]
    return {"present": True, "views": u["stats"], "listed": len(items),
            "with_install_date": sum(1 for i in items if i["install_date"]),
            "items": [{k: i[k] for k in ("name", "publisher", "version", "install_date", "category")} for i in items]}


@probe(id="apps.appx_registry", level="L1", family=APPS, tier="T0", collect="core")
def apps_appx_registry(h, facts):
    """Store/MSIX packages from the HKCU AppModel repository (replaces Get-AppxPackage)."""
    a = _appx(h)
    if not a:
        return None
    return {"present": True, "provisioned_apps": a["provisioned_apps"], "counts": a["counts"],
            "user_added": sorted(i["name"] for i in a["items"] if not i["provisioned"]),
            "items": [{k: i[k] for k in ("name", "version", "arch", "provisioned", "first_seen", "category")}
                      for i in a["items"]]}


@probe(id="apps.autostart", level="L1", family=APPS, tier="T0", collect="core")
def apps_autostart(h, facts):
    """Run/RunOnce keys + Startup folders with StartupApproved enabled/disabled state."""
    approved = {}
    for hive, path in [(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"),
                       (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"),
                       (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run32"),
                       (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\StartupFolder")]:
        k = _open(hive, path)
        if k is None:
            continue
        for n, v, _ in _values(k):
            if isinstance(v, bytes) and v:
                approved[n.lower()] = "disabled" if v[0] & 1 else "enabled"
    items = []
    for label, hive, path, flag in [
            ("HKLM_Run", winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", winreg.KEY_WOW64_64KEY),
            ("HKLM_RunOnce", winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce", winreg.KEY_WOW64_64KEY),
            ("HKLM_WOW_Run", winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", winreg.KEY_WOW64_32KEY),
            ("HKCU_Run", winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", 0),
            ("HKCU_RunOnce", winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce", 0)]:
        k = _open(hive, path, flag)
        if k is None:
            continue
        for n, v, _ in _values(k, 200):
            exe = _exe_from_cmd(v if isinstance(v, str) else "")
            items.append({"name": n, "source": label, "state": approved.get(n.lower(), "enabled"),
                          "company": _company(exe), "category": _classify(f"{n} {exe or ''}")})
    for label, p in [("startup_folder_user", os.path.join(_RA(h), r"Microsoft\Windows\Start Menu\Programs\Startup")),
                     ("startup_folder_common", os.path.join(_PD(), r"Microsoft\Windows\Start Menu\Programs\StartUp"))]:
        for n in _ls(p, 200) or []:
            if n.lower() == "desktop.ini":
                continue
            items.append({"name": os.path.splitext(n)[0], "source": label,
                          "state": approved.get(n.lower(), "enabled"), "company": None,
                          "category": _classify(n)})
    seen, dedup = set(), []
    for i in items:
        key = (i["name"].lower(), i["source"].startswith("HKLM"))
        if key not in seen:
            seen.add(key); dedup.append(i)
    persistent = [i for i in dedup if "RunOnce" not in i["source"]]
    return {"present": True, "total": len(persistent), "enabled": sum(1 for i in persistent if i["state"] == "enabled"),
            "disabled": sum(1 for i in persistent if i["state"] == "disabled"),
            "run_once": len(dedup) - len(persistent), "items": dedup}


@probe(id="apps.lad_programs", level="L1", family=APPS, tier="T0", collect="core")
def apps_lad_programs(h, facts):
    """%LOCALAPPDATA%\\Programs: per-user installs (VS Code, Ollama, Cua, GIMP) that skip HKLM."""
    names = _ls(os.path.join(_LA(h), "Programs"), 300)
    if names is None:
        return None
    names = sorted(n for n in names if n.lower() != "common")
    return {"present": True, "count": len(names), "names": names}


@probe(id="apps.progdirs", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_progdirs(h, facts):
    """Top-level Program Files / ProgramData directory names with category."""
    out = {}
    for label, p in [("ProgramFiles", _PF()), ("ProgramFilesX86", _PF86()),
                     ("LocalAppData_Programs", os.path.join(_LA(h), "Programs")), ("ProgramData", _PD())]:
        try:
            with os.scandir(p) as it:
                dirs = [e.name for _, e in zip(range(600), it) if e.is_dir(follow_symlinks=False)]
        except OSError:
            out[label] = None
            continue
        by = collections.Counter(_classify(d) for d in dirs)
        out[label] = {"count": len(dirs), "by_category": dict(by), "names": sorted(dirs)[:120]}
    return {"present": True, **out}


_SM_SKIP_DIR = re.compile(r"^(accessibility|accessories|administrative tools|system tools|windows powershell|startup|maintenance)(\\|$)|\\startup\\", re.I)
_SM_SKIP_NAME = re.compile(r"uninstall|readme|documentation|website|help|release notes|faq|reset|sdk|skin format|localization|samples|setup -|command prompt|file explorer|administrative tools|control panel|^run$|task manager|settings|language preferences|install additional", re.I)


def _walk_lnk(root, depth, cap=800):
    out = []
    if not _isdir(root):
        return None
    base = root.rstrip("\\").count("\\")
    for dp, dns, fns in os.walk(root):
        if dp.count("\\") - base >= depth:
            dns[:] = []
        for f in fns:
            if f.lower().endswith((".lnk", ".url", ".appref-ms")):
                p = os.path.join(dp, f)
                out.append((os.path.relpath(p, root), _ctime(p)))
                if len(out) >= cap:
                    return out
    return out


def _known_app_keys(h):
    keys = {_norm(i["name"]) for i in _uninstall(h)["items"]}
    a = _appx(h)
    if a:
        keys |= {_norm(i["name"], appx=True) for i in a["items"]}
        keys |= {_norm(i["display"]) for i in a["items"] if i["display"]}
    keys |= {_norm(n) for n in (_ls(os.path.join(_LA(h), "Programs")) or [])}
    return {k for k in keys if len(k) >= 3}


def _matches_known(n, known):
    k = _norm(n)
    if not k:
        return False
    if k in known:
        return True
    return any(len(a) >= 4 and (k.startswith(a) or a.startswith(k)) for a in known if len(k) >= 4)


@probe(id="apps.startmenu", level="L2", family=APPS, tier="T2", collect="core", gate="apps.uninstall")
def apps_startmenu(h, facts):
    """Start Menu / taskbar / desktop shortcut names; keeps names that match a known app or category."""
    known = _known_app_keys(h)
    roots = {"common": (os.path.join(_PD(), r"Microsoft\Windows\Start Menu\Programs"), 3),
             "user": (os.path.join(_RA(h), r"Microsoft\Windows\Start Menu\Programs"), 3),
             "taskbar_pinned": (os.path.join(_RA(h), r"Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar"), 1),
             "desktop_user": (os.path.join(_U(h), "Desktop"), 1),
             "desktop_public": (r"C:\Users\Public\Desktop", 1)}
    out = {"present": True}
    for label, (p, depth) in roots.items():
        items = _walk_lnk(p, depth)
        if items is None:
            out[label] = None
            continue
        kept, dropped = [], 0
        for rel, _c in items:
            nm = os.path.splitext(os.path.basename(rel))[0]
            if _SM_SKIP_DIR.search(rel) or _SM_SKIP_NAME.search(nm):
                continue
            if _matches_known(nm, known) or _classify(nm) != "other":
                kept.append(nm)
            else:
                dropped += 1
        out[label] = {"count": len(items), "app_names": sorted(set(kept))[:120], "unmatched_dropped": dropped}
    return out


def _steam_games(h):
    steam = None
    for hive, path, val in [(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath")]:
        k = _open(hive, path)
        if k is not None and _rv(k, val):
            steam = _rv(k, val); break
    if not steam:
        return []
    try:
        with open(os.path.join(steam, "steamapps", "libraryfolders.vdf"), encoding="utf-8", errors="replace") as f:
            libs = re.findall(r'"path"\s+"([^"]+)"', f.read())
    except OSError:
        libs = [steam]
    names = []
    for lib in libs[:10]:
        sa = os.path.join(lib.replace("\\\\", "\\"), "steamapps")
        for n in _ls(sa, 2000) or []:
            if n.startswith("appmanifest_") and n.endswith(".acf"):
                try:
                    with open(os.path.join(sa, n), encoding="utf-8", errors="replace") as f:
                        m = re.search(r'"name"\s+"([^"]+)"', f.read(4096))
                    if m and "redistributable" not in m.group(1).lower():
                        names.append(m.group(1))
                except OSError:
                    pass
    return names


def _winget(h):
    def build():
        src = os.path.join(_LA(h), r"Packages\Microsoft.DesktopAppInstaller_8wekyb3d8bbwe\LocalState\Microsoft.Winget.Source_8wekyb3d8bbwe\installed.db")
        if not os.path.exists(src):
            return None
        dst = _copy_db(h, src, "winget")
        if not dst:
            return {"error": "copy_failed"}
        try:
            rows = _q(dst, """select m.rowid, i.id, n.name, v.version from manifest m
                              join ids i on i.rowid=m.id join names n on n.rowid=m.name
                              join versions v on v.rowid=m.version""")
            meta = {}
            for mid, key, val in _q(dst, "select manifest, metadata, value from manifest_metadata"):
                meta.setdefault(mid, {})[str(key)] = val
        except Exception as e:
            return {"error": type(e).__name__}
        finally:
            _rmdb(dst)
        items = []
        for mid, pid, name, ver in rows:
            md = meta.get(mid, {})
            inst = md.get("7")
            items.append({"id": pid, "name": name, "version": ver, "arch": md.get("8"),
                          "pinned": bool(md.get("9")) if md.get("9") not in (None, "0", 0) else False,
                          "installed": _day(int(inst)) if inst and str(inst).isdigit() else None,
                          "category": _classify(f"{pid} {name}")})
        return {"bytes": os.path.getsize(src), "items": items}
    return _memo(h, "winget", build)


def _taxonomy(h):
    def build():
        apps = {}

        def add(name, pub, cat, date, src, pre=False):
            n = _norm(name, appx=(src == "appx"))
            if not n:
                return
            if n not in apps:
                for k in apps:
                    a_, b_ = (n, k) if len(n) <= len(k) else (k, n)
                    if len(a_) >= 4 and b_.startswith(a_):
                        n = k; break
            a = apps.setdefault(n, {"name": name, "category": cat, "date": None, "sources": set(), "pre": pre})
            a["sources"].add(src)
            a["pre"] = a["pre"] and pre
            if date and (not a["date"] or date < a["date"]):
                a["date"] = date

        for i in _uninstall(h)["items"]:
            add(i["name"], i["publisher"], i["category"], i["install_date"] or i["key_lastwrite"], "uninstall")
        a = _appx(h)
        for i in (a or {}).get("items", []):
            add(i["name"], None, i["category"], i["first_seen"], "appx", pre=i["provisioned"])
        lap = os.path.join(_LA(h), "Programs")
        for n in _ls(lap) or []:
            if n.lower() != "common" and _isdir(os.path.join(lap, n)):
                add(n, None, _classify(n), _ctime(os.path.join(lap, n)), "localappdata_programs")
        for g in _steam_games(h):
            add(g, None, "gaming", None, "steam_manifest")
        w = _winget(h)
        for i in (w or {}).get("items", []) or []:
            add(i["name"] or i["id"], None, i["category"], i["installed"], "winget")
        for root in (os.path.join(_RA(h), r"Microsoft\Windows\Start Menu\Programs"),
                     os.path.join(_PD(), r"Microsoft\Windows\Start Menu\Programs")):
            for rel, c in _walk_lnk(root, 3) or []:
                if not rel.lower().endswith(".lnk") or _SM_SKIP_DIR.search(rel):
                    continue
                nm = os.path.splitext(os.path.basename(rel))[0]
                if not _SM_SKIP_NAME.search(nm):
                    add(nm, None, _classify(nm), c, "start_menu")
        for nm in _ls(os.path.join(_U(h), ".local", "bin")) or []:
            base = os.path.splitext(nm)[0]
            if base.lower() in ("claude", "codex", "hermes", "ollama", "gemini", "aider", "goose", "opencode"):
                add(base, None, "ai", None, "local_bin")
        prof = _ctime(_U(h))
        for v in apps.values():
            if prof and v["date"] and v["date"] < prof:
                v["pre"] = True
        allv = list(apps.values())
        user = [v for v in allv if v["category"] in USER_CATS or v["category"] == "other"]
        return {"profile_created": prof,
                "image_date": _ctime(os.path.join(_PF(), "Windows NT")),
                "all": allv, "user": user}
    return _memo(h, "taxonomy", build)


@probe(id="apps.taxonomy", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_taxonomy(h, facts):
    """Merged app inventory (uninstall + appx + LAD Programs + Steam + winget + Start menu) by category."""
    t = _taxonomy(h)
    cats = USER_CATS + ["other"]
    all_counts = collections.Counter(v["category"] for v in t["all"])
    user_counts = {c: sum(1 for v in t["user"] if v["category"] == c) for c in cats}
    names = {c: sorted(v["name"] for v in t["user"] if v["category"] == c)[:60] for c in cats}
    return {"present": True, "unique_all": len(t["all"]), "unique_user_facing": len(t["user"]),
            "counts_all": dict(all_counts.most_common()), "counts_user_facing": user_counts, "names": names}


@probe(id="apps.taxonomy_user_added", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_taxonomy_user_added(h, facts):
    """Category counts of apps the user added (not provisioned, not dated before the profile existed)."""
    t = _taxonomy(h)
    cats = USER_CATS + ["other"]
    added = [v for v in t["user"] if not v["pre"]]
    months = collections.Counter(v["date"][:7] for v in added if v["date"])
    return {"present": True, "profile_created": t["profile_created"], "image_date": t["image_date"],
            "user_added_total": len(added), "preinstalled_total": len(t["user"]) - len(added),
            "counts": {c: sum(1 for v in added if v["category"] == c) for c in cats},
            "by_month": dict(sorted(months.items())), "undated": sum(1 for v in added if not v["date"])}


@probe(id="apps.vendor_software", level="L2", family=APPS, tier="T0", collect="core", gate="apps.uninstall")
def apps_vendor_software(h, facts):
    """Hardware-vendor utilities (ASUS, NVIDIA, MSI, Realtek, Logitech...) from the uninstall inventory."""
    items = [i for i in _uninstall(h)["items"] if i["category"] == "hardware_vendor"]
    a = _appx(h)
    appx = [i["name"] for i in (a or {}).get("items", []) if i["category"] == "hardware_vendor"]
    pubs = collections.Counter((i["publisher"] or "?").split(",")[0].strip() for i in items)
    return {"present": bool(items or appx), "uninstall_count": len(items), "appx_count": len(appx),
            "publishers": dict(pubs.most_common(10)),
            "names": sorted({i["name"] for i in items})[:80], "appx_names": sorted(appx)[:40]}


@probe(id="oem.ai_preloads", level="L1", family=APPS, tier="T0", collect="core")
def oem_ai_preloads(h, facts):
    """OEM-shipped AI apps (ASUS AI Image Agent, AI framework services...). Must not count toward AI-user score."""
    rx = r"AI Image Agent|Translate Agent|AIFramework|AI ?Creator|StoryCube|Virtual Assistant|AI Noise|Dell Optimizer|HP AI|Lenovo AI|Galaxy AI|MyASUS AI"
    items = [{"name": i["name"], "version": i["version"], "install_date": i["install_date"]}
             for i in _uninst_match(h, rx)]
    dirs = [p for p in (os.path.join(_PF(), "ASUS", "AICreator_n1x"),) if _isdir(p)]
    return {"present": bool(items or dirs), "items": items, "asus_aicreator_dir": bool(dirs)}


_PROC_INTEREST = re.compile(r"claude|codex|hermes|ollama|lm ?studio|^lms|cursor|windsurf|copilot|gemini|opencode|aider|cua|agent-browser|chatgpt|nvidia app|broadcast|docker|discord|onedrive|steam|slack|teams|code\.exe|brave|chrome|msedge|firefox|tailscale|obs|spotify", re.I)


@probe(id="proc.snapshot", level="L1", family=APPS, tier="T1", collect="core")
def proc_snapshot(h, facts):
    """Toolhelp32 process snapshot: total count, top process names, running agents and apps."""
    p = _procs(h)
    if not p:
        return None
    interest = {k: v for k, v in p.items() if _PROC_INTEREST.search(k)}
    return {"present": True, "total": sum(p.values()), "distinct": len(p),
            "top": dict(p.most_common(20)), "interesting": dict(sorted(interest.items()))}


@probe(id="apps.services_nonms", level="L2", family=APPS, tier="T0", collect="extended", gate="apps.uninstall",
       timeout_ms=4000)
def apps_services_nonms(h, facts):
    """Non-Microsoft Win32 services (ImagePath CompanyName), per-user instances skipped."""
    k = _open(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services")
    if k is None:
        return None
    windir = os.environ.get("SystemRoot", r"C:\Windows").lower()
    start_map = {0: "boot", 1: "system", 2: "auto", 3: "manual", 4: "disabled"}
    total, items = 0, []
    for sk in _subkeys(k, 3000):
        try:
            s = winreg.OpenKey(k, sk)
        except OSError:
            continue
        with s:
            typ = _rv(s, "Type") or 0
            if not (typ & 0x30) or typ & 0x80 or re.search(r"_[0-9a-f]{4,}$", sk, re.I):
                continue
            total += 1
            exe = _exe_from_cmd(_rv(s, "ImagePath"))
            if not exe:
                continue
            comp = _company(exe)
            if exe.lower().startswith(windir) and (comp is None or "microsoft" in comp.lower()):
                continue
            dn = _rv(s, "DisplayName") or sk
            if isinstance(dn, str) and dn.startswith("@"):
                dn = sk
            items.append({"name": sk, "display": dn, "company": comp, "start": start_map.get(_rv(s, "Start")),
                          "ms_outside_windir": bool(comp and "microsoft" in comp.lower()),
                          "category": _classify(f"{sk} {dn} {comp or ''}")})
    by_co = collections.Counter((i["company"] or "?") for i in items)
    return {"present": True, "win32_services": total, "non_ms": len(items),
            "auto": sum(1 for i in items if i["start"] == "auto"), "by_company": dict(by_co.most_common(15)),
            "items": items}


@probe(id="apps.winget_db", level="L2", family=APPS, tier="T0", collect="extended", gate="apps.uninstall")
def apps_winget_db(h, facts):
    """winget installed.db (copied): packages winget installed, with install date and arch."""
    w = _winget(h)
    if not w or "items" not in w:
        return w
    items = w["items"]
    return {"present": True, "count": len(items),
            "by_day": dict(collections.Counter(i["installed"] for i in items if i["installed"]).most_common(10)),
            "by_arch": dict(collections.Counter(i["arch"] or "?" for i in items)),
            "pinned": sum(1 for i in items if i["pinned"]),
            "items": [{k: i[k] for k in ("id", "version", "arch", "installed", "category")} for i in items]}


@probe(id="tasks.nonms", level="L2", family=APPS, tier="T2", collect="extended", gate="apps.uninstall",
       needs_admin=True, timeout_ms=4000)
def tasks_nonms(h, facts):
    """Scheduled task XML outside \\Microsoft (admin); operator-lab tasks counted but excluded."""
    import xml.etree.ElementTree as ET
    root = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "Tasks")
    if not os.access(root, os.R_OK):
        return None
    ns = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
    total = ms = op = errors = 0
    items = []
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        if rel.count(os.sep) >= 4:
            dns[:] = []
        is_ms = rel.lower().startswith("microsoft")
        for f in fns:
            total += 1
            if total > 3000:
                break
            if is_ms:
                ms += 1; continue
            if OPERATOR_TASK_RE.match(f):
                op += 1; continue
            p = os.path.join(dp, f)
            try:
                with open(p, "rb") as fh:
                    raw = fh.read(200_000)
                enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
                x = ET.fromstring(raw.decode(enc, "replace").lstrip("\ufeff").encode("utf-8")
                                  .replace(b'encoding="UTF-16"', b""))
            except Exception:
                errors += 1; continue
            ri = x.find(ns + "RegistrationInfo")
            author = ri.findtext(ns + "Author") if ri is not None else None
            en = x.find(f"{ns}Settings/{ns}Enabled")
            enabled = (en.text.strip().lower() == "true") if en is not None and en.text else True
            cmds = [e.findtext(ns + "Command") for e in x.iter(ns + "Exec")]
            exe = _exe_from_cmd(cmds[0]) if cmds and cmds[0] else None
            trig = x.find(ns + "Triggers")
            name = f if rel == "." else os.path.join(rel, f)
            items.append({"task": name, "enabled": enabled, "exe": os.path.basename(exe) if exe else None,
                          "company": _company(exe), "author_is_account": bool(author and "\\" in author),
                          "triggers": sorted({t.tag.replace(ns, "") for t in trig}) if trig is not None else [],
                          "category": _classify(f"{name} {exe or ''}")})
    return {"present": True, "total_task_files": total, "microsoft_folder": ms, "operator_lab_excluded": op,
            "non_ms": len(items), "parse_errors": errors, "items": items}


@probe(id="apps.winget_cli", level="L2", family=APPS, tier="T0", collect="deep", gate="apps.uninstall",
       timeout_ms=30000)
def apps_winget_cli(h, facts):
    """`winget list` row counts by source (2-18 s; deep only)."""
    exe = _which("winget")
    if not exe:
        return None
    rc, txt, ms = _spawn(h, [exe, "list", "--disable-interactivity", "--accept-source-agreements"], timeout=40)
    lines = [l for l in txt.splitlines() if l.strip()]
    hdr = next((i for i, l in enumerate(lines) if l.startswith("Name") and "Id" in l), None)
    if hdr is None:
        return {"present": False, "rc": rc}
    col = lines[hdr].find("Source")
    by = collections.Counter()
    for l in lines[hdr + 2:]:
        s = l[col:].strip() if col > 0 and len(l) > col else ""
        by[s or "(none)"] += 1
    return {"present": True, "rows": sum(by.values()), "by_source": dict(by), "ms": ms}
