"""Installed software (apps), AI agents (ai_agents) and dev environment (dev) probes.

Registration only at import time. Shared reads (uninstall keys, appx repository, process snapshot,
Claude/Codex/Hermes stores) are memoised per run so several probes can reuse one read.
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from userscan.registry import probe

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


def _spawn(args, timeout=8.0, env=None):
    """(rc, stdout, ms). Never used with bash.exe or pwsh."""
    exe = args[0]
    if exe.lower().endswith((".cmd", ".bat")):
        args = ["cmd.exe", "/d", "/c"] + args
    t = time.perf_counter()
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout, env=env, creationflags=NOWIN,
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
    rc, txt, ms = _spawn([exe, "list", "--disable-interactivity", "--accept-source-agreements"], timeout=40)
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


# ================================================================== ai_agents

def _detect(paths):
    return [p for p in paths if p and os.path.exists(p)]


@probe(id="ai.catalog_absent_checks", level="L1", family=AI, tier="T0", collect="core")
def ai_catalog_absent_checks(h, facts):
    """Path stats for AI tools the user may lack (Cursor, Windsurf, Gemini CLI, opencode, aider, LM Studio...)."""
    U, LA, RA = _U(h), _LA(h), _RA(h)
    npm = os.path.join(RA, "npm", "node_modules")
    cat = {
        "cursor": [os.path.join(U, ".cursor"), os.path.join(RA, "Cursor"), os.path.join(LA, "Programs", "cursor")],
        "windsurf": [os.path.join(U, ".codeium"), os.path.join(U, ".windsurf"), os.path.join(RA, "Windsurf"),
                     os.path.join(LA, "Programs", "Windsurf")],
        "gemini_cli": [os.path.join(U, ".gemini"), os.path.join(npm, "@google", "gemini-cli")],
        "opencode": [os.path.join(U, ".local", "share", "opencode"), os.path.join(U, ".config", "opencode"),
                     os.path.join(npm, "opencode-ai")],
        "aider": [os.path.join(U, ".aider.conf.yml"), os.path.join(U, ".aider"), os.path.join(U, ".aider.chat.history.md")],
        "continue": [os.path.join(U, ".continue")],
        "cline": [os.path.join(U, ".cline"), os.path.join(U, "Documents", "Cline")],
        "chatgpt_desktop": [os.path.join(LA, "Programs", "ChatGPT"), os.path.join(RA, "ChatGPT")],
        "lm_studio": [os.path.join(U, ".lmstudio"), os.path.join(U, ".cache", "lm-studio"), os.path.join(RA, "LM Studio"),
                      os.path.join(LA, "Programs", "LM Studio")],
        "jan": [os.path.join(RA, "Jan"), os.path.join(U, "jan")],
        "gpt4all": [os.path.join(LA, "nomic.ai")],
        "msty": [os.path.join(RA, "Msty")],
        "anythingllm": [os.path.join(RA, "anythingllm-desktop")],
        "hf_hub": [os.path.join(U, ".cache", "huggingface", "hub")],
    }
    found = {k: len(_detect(v)) > 0 for k, v in cat.items()}
    if _appx_match(h, r"^OpenAI\.ChatGPT"):
        found["chatgpt_desktop"] = True
    hf = os.path.join(U, ".cache", "huggingface", "hub")
    hf_models = sum(1 for n in (_ls(hf) or []) if n.startswith("models--"))
    return {"present": True, "found": sorted(k for k, v in found.items() if v),
            "absent": sorted(k for k, v in found.items() if not v), "hf_hub_models": hf_models}


@probe(id="browser.automation", level="L1", family=AI, tier="T0", collect="core")
def browser_automation(h, facts):
    """Browser-automation runtimes on disk: Playwright, camoufox, agent-browser, puppeteer."""
    U, LA, RA = _U(h), _LA(h), _RA(h)
    out = {}
    for label, p in [("playwright", os.path.join(LA, "ms-playwright")), ("playwright_cache", os.path.join(U, ".cache", "ms-playwright")),
                     ("puppeteer", os.path.join(U, ".cache", "puppeteer")), ("camoufox", os.path.join(LA, "camoufox")),
                     ("agent_browser_browsers", os.path.join(U, ".agent-browser", "browsers"))]:
        names = _ls(p, 50)
        if names is not None:
            out[label] = {"entries": sorted(names)[:20], "mtime": _mtime(p)}
    installs = []
    for loc in ([os.path.join(LA, "hermes", "node", "node_modules", "agent-browser"),
                 os.path.join(RA, "npm", "node_modules", "agent-browser")]):
        pj = _read_json(os.path.join(loc, "package.json"))
        if isinstance(pj, dict):
            installs.append({"where": "hermes_bundled" if "hermes" in loc.lower() else "npm_global",
                             "version": pj.get("version")})
    if installs:
        out["agent_browser"] = installs
    return {"present": bool(out), **out}


@probe(id="browser_harness.present", level="L1", family=AI, tier="T0", collect="core")
def browser_harness_present(h, facts):
    """browser-harness config dir and runtime session pid-file count."""
    bh = os.path.join(_U(h), ".config", "browser-harness")
    if not _isdir(bh):
        return None
    pids = glob.glob(os.path.join(bh, "runtime", "*.pid"))
    return {"present": True, "runtime_sessions": len(pids[:500]), "mtime": _mtime(bh)}


# ---------------- Claude Code

def _claude_json(h):
    """Whitelisted view of ~/.claude.json. Secret-bearing keys are reduced to booleans immediately."""
    def build():
        cj = _read_json(os.path.join(_U(h), ".claude.json"), maxb=20_000_000)
        if not isinstance(cj, dict):
            return None
        projects = cj.get("projects") or {}
        out = {"numStartups": cj.get("numStartups"), "installMethod": cj.get("installMethod"),
               "firstStartTime": _iso(cj.get("firstStartTime")), "autoUpdates": cj.get("autoUpdates"),
               "hasCompletedOnboarding": cj.get("hasCompletedOnboarding"),
               "project_count": len(projects) if isinstance(projects, dict) else 0,
               "global_mcp_servers": sorted((cj.get("mcpServers") or {}).keys()),
               "project_mcp_servers": sorted({n for v in (projects.values() if isinstance(projects, dict) else [])
                                              if isinstance(v, dict) for n in (v.get("mcpServers") or {}).keys()}),
               "has_oauth_account": "oauthAccount" in cj, "has_primary_api_key": "primaryApiKey" in cj}
        del cj
        return out
    return _memo(h, "claude_json", build)


@probe(id="claude_code.present", level="L1", family=AI, tier="T0", collect="core")
def claude_code_present(h, facts):
    """Claude Code CLI: ~/.claude home and native claude.exe (with file version)."""
    U = _U(h)
    home = os.path.join(U, ".claude")
    exe = os.path.join(U, ".local", "bin", "claude.exe")
    npm = os.path.join(_RA(h), "npm", "node_modules", "@anthropic-ai", "claude-code")
    if not (_isdir(home) or os.path.exists(exe) or _isdir(npm)):
        return None
    vers = _ls(os.path.join(U, ".local", "share", "claude", "versions")) or []
    return {"present": True, "home": _isdir(home), "native_exe": os.path.exists(exe),
            "exe_version": _file_version(exe), "npm_install": _isdir(npm), "installed_versions": len(vers),
            "home_mtime": _mtime(home)}


@probe(id="claude_code.config", level="L2", family=AI, tier="T0", collect="core", gate="claude_code.present")
def claude_code_config(h, facts):
    """Claude Code settings (whitelisted keys), permission rule counts, skills/commands/agents/plugins."""
    home = os.path.join(_U(h), ".claude")
    if not _isdir(home):
        return None
    s = _read_json(os.path.join(home, "settings.json")) or {}
    sl = _read_json(os.path.join(home, "settings.local.json")) or {}
    km = _read_json(os.path.join(home, "plugins", "known_marketplaces.json")) or {}
    ip = _read_json(os.path.join(home, "plugins", "installed_plugins.json")) or {}
    plugins = (ip.get("plugins") if isinstance(ip.get("plugins"), dict) else ip) if isinstance(ip, dict) else {}
    return {"present": True, "settings_keys": sorted(s.keys()) if isinstance(s, dict) else [],
            "settings": {k: s.get(k) for k in ("model", "autoUpdatesChannel", "theme", "outputStyle")
                         if isinstance(s, dict) and isinstance(s.get(k), (str, bool, int))},
            "hooks_events": sorted((s.get("hooks") or {}).keys()) if isinstance(s, dict) else [],
            "allow_rules": len(((s.get("permissions") or {}).get("allow") or [])) + len(((sl.get("permissions") or {}).get("allow") or [])),
            "skills": len(glob.glob(os.path.join(home, "skills", "*", "SKILL.md"))),
            "commands": len(glob.glob(os.path.join(home, "commands", "*.md"))),
            "agents": len(glob.glob(os.path.join(home, "agents", "*.md"))),
            "plugin_marketplaces": sorted(km.keys()) if isinstance(km, dict) else [],
            "installed_plugins": sorted(plugins.keys())[:30] if isinstance(plugins, dict) else [],
            "claude_md": os.path.exists(os.path.join(home, "CLAUDE.md"))}


@probe(id="claude_code.claude_json", level="L2", family=AI, tier="T1", collect="core", gate="claude_code.present")
def claude_code_claude_json(h, facts):
    """~/.claude.json whitelist: numStartups, installMethod, firstStartTime, MCP names; secrets as booleans."""
    cj = _claude_json(h)
    if not cj:
        return None
    return {"present": True, **cj}


def _cc_scan(h):
    def build():
        pdir = os.path.join(_U(h), ".claude", "projects")
        r = {"projects": 0, "sessions": 0, "sessions_30d": 0, "bytes": 0, "user_msgs": 0, "assistant_msgs": 0,
             "skipped_large": 0}
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        models, entry, versions, days = collections.Counter(), collections.Counter(), collections.Counter(), set()
        first = last = None
        titles = []
        files_seen = 0
        for pd in sorted(glob.glob(os.path.join(pdir, "*")))[:500]:
            files = glob.glob(os.path.join(pd, "*.jsonl"))
            if not files:
                continue
            r["projects"] += 1
            for f in files:
                files_seen += 1
                if files_seen > 5000:
                    break
                r["sessions"] += 1
                try:
                    sz = os.path.getsize(f)
                except OSError:
                    continue
                r["bytes"] += sz
                if sz > 60_000_000:
                    r["skipped_large"] += 1
                    continue
                title = None
                flast = None
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for n, line in enumerate(fh):
                        if n > 200_000:
                            break
                        try:
                            j = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(j, dict):
                            continue
                        t = j.get("type")
                        ts = j.get("timestamp")
                        if isinstance(ts, str):
                            flast = ts if flast is None or ts > flast else flast
                            first = ts if first is None or ts < first else first
                            last = ts if last is None or ts > last else last
                            days.add(ts[:10])
                        if j.get("entrypoint"):
                            entry[j["entrypoint"]] += 1
                        if j.get("version"):
                            versions[j["version"]] += 1
                        if t == "user" and not j.get("isMeta") and not j.get("toolUseResult"):
                            r["user_msgs"] += 1
                        elif t == "assistant":
                            r["assistant_msgs"] += 1
                            m = (j.get("message") or {}).get("model") if isinstance(j.get("message"), dict) else None
                            if m:
                                models[m] += 1
                        elif t == "custom-title" and j.get("customTitle"):
                            title = j["customTitle"]
                        elif t == "ai-title" and j.get("aiTitle") and not title:
                            title = j["aiTitle"]
                        elif t == "summary" and j.get("summary") and not title:
                            title = j["summary"]
                if flast and flast[:10] >= cutoff:
                    r["sessions_30d"] += 1
                if title:
                    titles.append((os.path.getmtime(f), title))
        hist = os.path.join(_U(h), ".claude", "history.jsonl")
        hinfo = None
        if os.path.exists(hist) and os.path.getsize(hist) < 50_000_000:
            hts = []
            with open(hist, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        v = json.loads(line).get("timestamp")
                    except (ValueError, AttributeError):
                        continue
                    if v:
                        hts.append(v)
            hinfo = {"entries": len(hts), "first": _iso(min(hts)) if hts else None, "last": _iso(max(hts)) if hts else None}
        r.update({"first": _iso(first), "last": _iso(last), "active_days": len(days), "models": _top(models),
                  "entrypoints": _top(entry), "cli_versions": _top(versions, 5), "prompt_history": hinfo})
        return {"agg": r, "titles": [t for _, t in sorted(titles, reverse=True)]}
    return _memo(h, "cc_scan", build)


@probe(id="claude_code.sessions", level="L2", family=AI, tier="T1", collect="extended", gate="claude_code.present",
       timeout_ms=5000)
def claude_code_sessions(h, facts):
    """Claude Code JSONL aggregates: session/message counts, date range, active days, models, entrypoints."""
    if not _isdir(os.path.join(_U(h), ".claude", "projects")):
        return None
    agg = _cc_scan(h)["agg"]
    return {"present": agg["sessions"] > 0, **agg}


@probe(id="claude_code.titles", level="L2", family=AI, tier="T2", collect="deep", gate="claude_code.present")
def claude_code_titles(h, facts):
    """Most recent Claude Code session titles (redacted, truncated). T2 content, opt-in."""
    if not _isdir(os.path.join(_U(h), ".claude", "projects")):
        return None
    t = _cc_scan(h)["titles"]
    return {"present": bool(t), "count_with_title": len(t), "recent": [_redact(x) for x in t[:5]]}


@probe(id="claude_desktop.present", level="L1", family=AI, tier="T0", collect="core")
def claude_desktop_present(h, facts):
    """Claude desktop app: installed now, or evidence of a past install (Claude-3p dir)."""
    LA, RA = _LA(h), _RA(h)
    uninst = [i["version"] for i in _uninst_match(h, r"^Claude\b")]
    appx = _appx_match(h, r"Claude|Anthropic")
    dirs = {"roaming_Claude": _isdir(os.path.join(RA, "Claude")),
            "local_AnthropicClaude": _isdir(os.path.join(LA, "AnthropicClaude")),
            "local_Claude-3p": _isdir(os.path.join(LA, "Claude-3p")),
            "programdata_Claude": _isdir(os.path.join(_PD(), "Claude"))}
    installed = bool(uninst or appx or dirs["local_AnthropicClaude"] or dirs["roaming_Claude"])
    cfg = _read_json(os.path.join(RA, "Claude", "claude_desktop_config.json"))
    mcp = sorted((cfg.get("mcpServers") or {}).keys()) if isinstance(cfg, dict) else []
    past = dirs["local_Claude-3p"] or dirs["programdata_Claude"]
    if not (installed or past):
        return None
    return {"present": True, "installed": installed, "past_install_evidence": bool(past and not installed),
            "uninstall_versions": uninst, "appx": [f"{p['name']} {p['version']}" for p in appx],
            "dirs": dirs, "mcp_servers": mcp, "claude_3p_mtime": _mtime(os.path.join(LA, "Claude-3p"))}


# ---------------- Codex

def _codex_home(h):
    return os.environ.get("CODEX_HOME") or os.path.join(_U(h), ".codex")


@probe(id="codex.present", level="L1", family=AI, tier="T0", collect="core")
def codex_present(h, facts):
    """OpenAI Codex: ~/.codex home and Codex Desktop MSIX (appx repository)."""
    home = _codex_home(h)
    appx = _appx_match(h, r"^OpenAI\.Codex")
    bindir = os.path.join(_LA(h), "OpenAI", "Codex", "bin")
    npm = _isdir(os.path.join(_RA(h), "npm", "node_modules", "@openai", "codex"))
    if not (_isdir(home) or appx or npm):
        return None
    return {"present": True, "home": _isdir(home), "desktop_appx": [{"version": p["version"], "arch": p["arch"]} for p in appx],
            "bundled_cli_builds": len(_ls(bindir) or []), "npm_cli": npm, "home_mtime": _mtime(home)}


@probe(id="codex.auth_presence", level="L1", family=AI, tier="T3", collect="core")
def codex_auth_presence(h, facts):
    """Codex auth.json presence/size/mtime (never opened). Presence means signed in."""
    return h.meta(os.path.join(_codex_home(h), "auth.json"))


@probe(id="codex.config", level="L2", family=AI, tier="T0", collect="core", gate="codex.present")
def codex_config(h, facts):
    """Codex config.toml whitelist: model, provider, sandbox, MCP/plugin names. MCP env tables excluded."""
    p = os.path.join(_codex_home(h), "config.toml")
    if not os.path.exists(p) or tomllib is None:
        return None
    try:
        with open(p, "rb") as f:
            c = tomllib.load(f)
    except Exception as e:
        return {"present": True, "error": type(e).__name__}
    return {"present": True, "model": c.get("model"), "model_provider": c.get("model_provider"),
            "model_reasoning_effort": c.get("model_reasoning_effort"), "approval_policy": c.get("approval_policy"),
            "sandbox_mode": c.get("sandbox_mode"), "windows_sandbox": (c.get("windows") or {}).get("sandbox"),
            "profiles": sorted((c.get("profiles") or {}).keys()),
            "model_providers": sorted((c.get("model_providers") or {}).keys()),
            "mcp_servers": sorted((c.get("mcp_servers") or {}).keys()),
            "plugins_enabled": sorted(k for k, v in (c.get("plugins") or {}).items() if isinstance(v, dict) and v.get("enabled")),
            "trusted_projects": len(c.get("projects") or {})}


@probe(id="codex.usage", level="L2", family=AI, tier="T1", collect="extended", gate="codex.present", timeout_ms=5000)
def codex_usage(h, facts):
    """Codex rollouts (count/bytes/date range/turn models) + state_*.sqlite thread counts + turn durations."""
    home = _codex_home(h)
    if not _isdir(home):
        return None
    files = sorted(glob.glob(os.path.join(home, "sessions", "*", "*", "*", "rollout-*.jsonl")) +
                   glob.glob(os.path.join(home, "archived_sessions", "*.jsonl")))[:3000]
    models, efforts, orig, clis = (collections.Counter() for _ in range(4))
    dates, b, turns = [], 0, 0
    for f in files:
        try:
            b += os.path.getsize(f)
        except OSError:
            continue
        m = re.search(r"rollout-(\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d)", f)
        if m:
            dates.append(m.group(1))
        if os.path.getsize(f) > 80_000_000:
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                head = line[:120]
                if i == 0 and '"session_meta"' in head:
                    try:
                        p = json.loads(line)["payload"]
                        orig[p.get("originator")] += 1; clis[p.get("cli_version")] += 1
                    except Exception:
                        pass
                elif '"turn_context"' in head:
                    turns += 1
                    try:
                        p = json.loads(line)["payload"]
                        models[p.get("model")] += 1
                        efforts[p.get("effort") or p.get("reasoning_effort")] += 1
                    except Exception:
                        pass
    dates.sort()
    cutoff = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")
    out = {"present": True, "sessions_30d": sum(1 for d in dates if d[:10] >= cutoff),
           "rollouts": {"files": len(files), "bytes": b, "first": dates[0] if dates else None,
                        "last": dates[-1] if dates else None, "turn_contexts": turns, "models": _top(models),
                        "efforts": _top(efforts), "originators": _top(orig), "cli_versions": _top(clis, 5)}}
    sdb = sorted(glob.glob(os.path.join(home, "state_*.sqlite")), key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if sdb:
        dst = _copy_db(h, sdb[-1], "codex-state")
        if dst:
            try:
                cs = _cols(dst, "threads")
                by_src = collections.Counter()
                for (s,) in _q(dst, "select source from threads"):
                    if isinstance(s, str) and s.startswith("{"):
                        try:
                            s = "subagent" if "subagent" in s.lower() else next(iter(json.loads(s)))
                        except Exception:
                            s = "object"
                    by_src[s] += 1
                a, z = _q(dst, "select min(created_at), max(updated_at) from threads")[0]
                out["threads"] = {"count": sum(by_src.values()), "first": _iso(a), "last": _iso(z), "by_source": dict(by_src),
                                  "by_provider": dict(_q(dst, "select model_provider, count(*) from threads group by 1")) if "model_provider" in cs else None,
                                  "by_model": dict(_q(dst, "select model, count(*) from threads group by 1")) if "model" in cs else None,
                                  "tokens_used_total": _q(dst, "select sum(tokens_used) from threads")[0][0] if "tokens_used" in cs else None}
                try:
                    out["threads"]["subagent_edges"] = _q(dst, "select count(*) from thread_spawn_edges")[0][0]
                except Exception:
                    pass
            except Exception as e:
                out["threads"] = {"error": type(e).__name__}
            finally:
                _rmdb(dst)
    th = sorted(glob.glob(os.path.join(home, "thread_history_*.sqlite")))
    if th:
        dst = _copy_db(h, th[-1], "codex-th")
        if dst:
            try:
                out["turns"] = {"count": _q(dst, "select count(*) from thread_turns")[0][0],
                                "agent_minutes": round((_q(dst, "select sum(duration_ms) from thread_turns")[0][0] or 0) / 60000, 1),
                                "items": _q(dst, "select count(*) from thread_items")[0][0]}
            except Exception as e:
                out["turns"] = {"error": type(e).__name__}
            finally:
                _rmdb(dst)
    return out


@probe(id="codex.extras", level="L2", family=AI, tier="T1", collect="extended", gate="codex.present")
def codex_extras(h, facts):
    """Codex skill/plugin counts, dictation-session count, logs db row count and range."""
    home = _codex_home(h)
    if not _isdir(home):
        return None
    sk = os.path.join(home, "skills")
    pc = os.path.join(home, "plugins", "cache")
    out = {"present": True,
           "skills_system": len(glob.glob(os.path.join(sk, ".system", "*", "SKILL.md"))),
           "skills_user": len(glob.glob(os.path.join(sk, "*", "SKILL.md"))),
           "plugin_cache": sum(len(_ls(m) or []) for m in glob.glob(os.path.join(pc, "*"))),
           "dictation_sessions": len(_ls(os.path.join(home, "dictation-history"), 5000) or []),
           "agents_md": os.path.exists(os.path.join(home, "AGENTS.md"))}
    lg = sorted(glob.glob(os.path.join(home, "logs_*.sqlite")))
    if lg:
        dst = _copy_db(h, lg[-1], "codex-logs")
        if dst:
            try:
                n, a, z = _q(dst, "select count(*), min(ts), max(ts) from logs")[0]
                out["logs_db"] = {"rows": n, "first": _iso(a), "last": _iso(z)}
            except Exception as e:
                out["logs_db"] = {"error": type(e).__name__}
            finally:
                _rmdb(dst)
    return out


@probe(id="codex.chatgpt_catalog", level="L2", family=AI, tier="T2", collect="deep", gate="codex.present")
def codex_chatgpt_catalog(h, facts):
    """ChatGPT cloud conversations visible to Codex: count/date range by host kind + recent titles (T2)."""
    dev = os.path.join(_codex_home(h), "sqlite", "codex-dev.db")
    if not os.path.exists(dev):
        return None
    dst = _copy_db(h, dev, "codex-dev")
    if not dst:
        return None
    try:
        hosts = dict(_q(dst, "select host_id, host_kind from local_thread_catalog_hosts"))
        rows = _q(dst, "select host_id, source_kind, count(*), min(source_created_at), max(source_created_at) from local_thread_catalog group by 1,2")
        cat = [{"host_kind": hosts.get(hid, "?"), "source_kind": sk, "count": n, "first": _iso(a), "last": _iso(z)}
               for hid, sk, n, a, z in rows]
        titles = [_redact(t) for (t,) in _q(dst, "select display_title from local_thread_catalog where source_kind='chatgpt' order by source_updated_at desc limit 5")]
    except Exception as e:
        return {"present": True, "error": type(e).__name__}
    finally:
        _rmdb(dst)
    return {"present": bool(cat), "catalog": cat, "recent_titles": titles}


@probe(id="consent.system_ai_models", level="L1", family=AI, tier="T1", collect="core")
def consent_system_ai_models(h, facts):
    """CapabilityAccessManager systemAIModels consent: global value and apps that used on-device AI models."""
    base = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\systemAIModels"
    out = {}
    for label, hive in (("HKCU", winreg.HKEY_CURRENT_USER), ("HKLM", winreg.HKEY_LOCAL_MACHINE)):
        k = _open(hive, base)
        if k is None:
            continue
        apps = used = 0
        for sk in list(_subkeys(k, 500)):
            if sk == "NonPackaged":
                nk = _open(hive, base + "\\NonPackaged")
                subs = ["NonPackaged\\" + s for s in (list(_subkeys(nk, 500)) if nk else [])]
            else:
                subs = [sk]
            for s in subs:
                ak = _open(hive, base + "\\" + s)
                if ak is None:
                    continue
                apps += 1
                if (_rv(ak, "LastUsedTimeStart") or 0) > 0:
                    used += 1
        out[label] = {"value": _rv(k, "Value"), "apps": apps, "apps_used": used}
    if not out:
        return None
    return {"present": True, **out}


@probe(id="copilot_cli.present", level="L1", family=AI, tier="T0", collect="core")
def copilot_cli_present(h, facts):
    """GitHub Copilot CLI: ~/.copilot, firstLaunchAt, log/session counts, MCP server names."""
    home = os.path.join(_U(h), ".copilot")
    if not _isdir(home):
        return None
    c = _read_json(os.path.join(home, "config.json"), jsonc=True) or {}
    mc = _read_json(os.path.join(home, "mcp-config.json"), jsonc=True) or {}
    return {"present": True, "first_launch": _iso(c.get("firstLaunchAt")) if isinstance(c, dict) else None,
            "logs": len(_ls(os.path.join(home, "logs")) or []),
            "sessions": len(_ls(os.path.join(home, "session-state")) or []) + len(_ls(os.path.join(home, "history-session-state")) or []),
            "mcp_servers": sorted((mc.get("mcpServers") or {}).keys()) if isinstance(mc, dict) else []}


@probe(id="copilot_m365.present", level="L1", family=AI, tier="T0", collect="core")
def copilot_m365_present(h, facts):
    """Microsoft 365 Copilot (OfficeHub) and Windows Copilot appx + running process."""
    m365 = _appx_match(h, r"^Microsoft\.MicrosoftOfficeHub$")
    win = _appx_match(h, r"^Microsoft\.Copilot$")
    run = _running(h, r"^m365copilot\.exe$|^copilot\.exe$")
    if not (m365 or win):
        return None
    return {"present": True, "m365_version": m365[0]["version"] if m365 else None,
            "windows_copilot_version": win[0]["version"] if win else None, "running": run}


@probe(id="cua_driver.present", level="L1", family=AI, tier="T0", collect="core")
def cua_driver_present(h, facts):
    """cua-driver (computer-use): concrete release dirs (junctions fail over ssh), version check, running."""
    home = os.path.join(_U(h), ".cua-driver")
    if not _isdir(home) and not _isdir(os.path.join(_LA(h), "Programs", "Cua")):
        return None
    rel = sorted(_ls(os.path.join(home, "packages", "releases")) or [])
    vc = _read_json(os.path.join(home, "version_check.json")) or {}
    return {"present": True, "releases": rel[-5:], "releases_installed": sorted(_ls(os.path.join(home, ".release_installed")) or [])[-5:],
            "first_install": _mtime(os.path.join(home, ".installation_recorded")),
            "latest_available": vc.get("latest_version") if isinstance(vc, dict) else None,
            "running": bool(_running(h, r"^cua-driver"))}


@probe(id="docker.model_runner", level="L1", family=AI, tier="T0", collect="core")
def docker_model_runner(h, facts):
    """Docker Model Runner models.json model count."""
    dm = _read_json(os.path.join(_U(h), ".docker", "models", "models.json"))
    if dm is None:
        return None
    lst = dm.get("models", []) if isinstance(dm, dict) else dm
    return {"present": True, "models": len(lst) if isinstance(lst, (list, dict)) else 0}


# ---------------- Hermes

def _hermes_homes(h):
    def build():
        env = h.l0.get("hermes_home") or os.environ.get("HERMES_HOME") or ""
        LA, U = _LA(h), _U(h)
        cands = []
        for role, p in (("HERMES_HOME", env), ("localappdata", os.path.join(LA, "hermes")),
                        ("dot_hermes", os.path.join(U, ".hermes"))):
            if p and _isdir(p) and all(os.path.normcase(p) != os.path.normcase(c["path"]) for c in cands):
                cands.append({"role": role, "path": p, "has_state": os.path.exists(os.path.join(p, "state.db"))})
        primary = next((c for c in cands if c["role"] == "HERMES_HOME"), None) or \
            next((c for c in cands if c["has_state"]), None)
        side = [p for p in glob.glob(os.path.join(LA, "hermes-*")) if _isdir(p)][:50]
        return {"env_set": bool(env), "cands": cands, "primary": primary, "side": side}
    return _memo(h, "hermes_homes", build)


@probe(id="hermes.present", level="L1", family=AI, tier="T0", collect="core")
def hermes_present(h, facts):
    """Hermes Agent: HERMES_HOME/default homes, MSIX builds, desktop userData. Side homes flagged as operator."""
    hh = _hermes_homes(h)
    appx = _appx_match(h, r"^NousResearch\.|Hermes")
    ud = glob.glob(os.path.join(_RA(h), "Hermes*"))
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
        if tomllib and os.path.exists(pp):
            try:
                with open(pp, "rb") as f:
                    ver = tomllib.load(f).get("project", {}).get("version")
            except Exception:
                pass
        homes.append({"role": c["role"], "primary": hh["primary"] is c, "state_db_bytes": (h.meta(os.path.join(p, "state.db")).get("bytes")),
                      "mtime": _mtime(p), "install_method": im, "agent_version": ver,
                      "git_checkout": _isdir(os.path.join(p, "hermes-agent", ".git"))})
    if not (homes or appx or ud or hh["side"]):
        return None
    return {"present": True, "HERMES_HOME_set": hh["env_set"], "user_home": hh["primary"] is not None,
            "homes": homes, "side_homes_operator": len(hh["side"]),
            "appx": [{"name": p["name"], "version": p["version"], "arch": p["arch"]} for p in appx],
            "desktop_userdata_dirs": len(ud), "running": sum(_running(h, r"^hermes").values())}


@probe(id="hermes.auth_presence", level="L1", family=AI, tier="T3", collect="core")
def hermes_auth_presence(h, facts):
    """Primary Hermes home auth.json and .env presence/size (never opened)."""
    p = _hermes_homes(h)["primary"]
    if not p:
        return None
    a, e = h.meta(os.path.join(p["path"], "auth.json")), h.meta(os.path.join(p["path"], ".env"))
    return {"present": a["present"] or e["present"], "auth_json": a, "env": e}


def _parse_hermes_yaml(p):
    out = {"model": {}, "toolsets": [], "mcp_servers": [], "platform_toolsets": [], "top_level_keys": 0}
    cur = None
    with open(p, encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh):
            if n > 5000:
                break
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            ind = len(line) - len(line.lstrip(" "))
            s = line.strip()
            if ind == 0:
                out["top_level_keys"] += 1
                cur = s.split(":", 1)[0]
                rest = s.split(":", 1)[1].strip() if ":" in s else ""
                if cur == "model" and rest and rest != "{}":
                    out["model"]["default"] = rest.strip("'\"")
                continue
            if cur == "model" and ind == 2 and ":" in s:
                k, v = s.split(":", 1)
                v = v.strip().strip("'\"")
                if k == "base_url":
                    m = re.match(r"https?://([^/:]+)", v)
                    host = m.group(1) if m else ""
                    v = "local" if re.match(r"^(127\.|localhost|0\.0\.0\.0|192\.168\.|10\.)", host) else host
                if k in ("default", "provider", "base_url", "api_mode"):
                    out["model"][k] = v
            elif cur == "toolsets" and s.startswith("- "):
                out["toolsets"].append(s[2:].strip("'\""))
            elif cur == "mcp_servers" and ind == 2 and s.endswith(":"):
                out["mcp_servers"].append(s[:-1].strip("'\""))
            elif cur == "platform_toolsets" and ind == 2 and ":" in s:
                out["platform_toolsets"].append(s.split(":", 1)[0])
    return out


@probe(id="hermes.config", level="L2", family=AI, tier="T0", collect="core", gate="hermes.present")
def hermes_config(h, facts):
    """Primary Hermes config.yaml whitelist: model/provider, base_url host kind, toolsets, MCP names. .env never read."""
    p = _hermes_homes(h)["primary"]
    if not p or not os.path.exists(os.path.join(p["path"], "config.yaml")):
        return None
    try:
        return {"present": True, **_parse_hermes_yaml(os.path.join(p["path"], "config.yaml"))}
    except OSError:
        return None


@probe(id="hermes.skills", level="L2", family=AI, tier="T1", collect="extended", gate="hermes.present")
def hermes_skills(h, facts):
    """Primary Hermes skills: count, categories, archived; top-used names only if in the bundled catalog."""
    p = _hermes_homes(h)["primary"]
    if not p:
        return None
    sk = os.path.join(p["path"], "skills")
    if not _isdir(sk):
        return None
    total, cats = 0, set()
    for dp, dns, fns in os.walk(sk):
        rel = os.path.relpath(dp, sk)
        dns[:] = [d for d in dns if not d.startswith(".")] if rel.count(os.sep) < 4 else []
        if "SKILL.md" in fns:
            total += 1
            if rel != ".":
                cats.add(rel.split(os.sep)[0])
        if total > 5000:
            break
    catalog = set()
    for base in ("skills", "optional-skills"):
        for f in glob.glob(os.path.join(p["path"], "hermes-agent", base, "**", "SKILL.md"), recursive=True)[:3000]:
            catalog.add(os.path.basename(os.path.dirname(f)))
    top, top_plugin, tracked, used = [], [], 0, 0
    u = _read_json(os.path.join(sk, ".usage.json"))
    if isinstance(u, dict):
        items = u.get("skills", u) if isinstance(u.get("skills", u), dict) else {}
        tracked = len(items)

        def score(v):
            if isinstance(v, dict):
                return v.get("use_count") or v.get("count") or v.get("uses") or 0
            return v if isinstance(v, (int, float)) else 0
        ranked = [k for k, v in sorted(items.items(), key=lambda kv: -score(kv[1])) if score(v) > 0]
        top = [k for k in ranked if k in catalog][:5]
        top_plugin = [k.split(":", 1)[1] for k in ranked if ":" in k][:5]
        used = len(ranked)
    cats = {d for d in (_ls(sk) or []) if not d.startswith(".") and _isdir(os.path.join(sk, d))}
    return {"present": True, "skills": total, "categories": len(cats), "skills_used": used,
            "top_used_plugin": top_plugin,
            "archived": len(glob.glob(os.path.join(sk, ".archive", "*", "SKILL.md"))),
            "usage_tracked": tracked, "top_used_catalog": top, "catalog_available": bool(catalog),
            "plugins": len(_ls(os.path.join(p["path"], "plugins")) or [])}


def _hermes_db_summary(h, home, tag, titles=False):
    st = os.path.join(home, "state.db")
    if not os.path.exists(st):
        return None
    dst = _copy_db(h, st, tag)
    if not dst:
        return {"error": "copy_failed"}
    try:
        cs = _cols(dst, "sessions")
        if titles:
            if "title" not in cs:
                return {"titles": []}
            rows = _q(dst, "select title from sessions where title is not null and title != '' "
                           + ("and parent_session_id is null " if "parent_session_id" in cs else "")
                           + "order by started_at desc limit 5")
            return {"titles": [_redact(t) for (t,) in rows]}
        a, z = _q(dst, "select min(started_at), max(coalesce(ended_at, started_at)) from sessions")[0]
        r = {"sessions": _q(dst, "select count(*) from sessions")[0][0],
             "messages": _q(dst, "select count(*) from messages")[0][0],
             "first": _iso(a), "last": _iso(z),
             "by_source": dict(_q(dst, "select coalesce(source,'?'), count(*) from sessions group by 1")),
             "by_month": dict(_q(dst, "select strftime('%Y-%m', started_at, 'unixepoch'), count(*) from sessions group by 1")),
             "by_model": dict(_q(dst, "select coalesce(model,'?'), count(*) from sessions group by 1 order by 2 desc limit 10"))}
        r["sessions_30d"] = _q(dst, "select count(*) from sessions where started_at >= ?", (time.time() - 30 * 86400,))[0][0]
        if "parent_session_id" in cs:
            r["subagent_sessions"] = _q(dst, "select count(*) from sessions where parent_session_id is not null")[0][0]
        if "tool_call_count" in cs:
            r["tool_calls"] = _q(dst, "select sum(tool_call_count) from sessions")[0][0]
        if "estimated_cost_usd" in cs:
            r["est_cost_usd"] = _q(dst, "select round(sum(estimated_cost_usd), 2) from sessions")[0][0]
        try:
            r["model_calls"] = [{"model": m, "provider": p, "calls": c} for m, p, c in
                                _q(dst, "select model, billing_provider, sum(api_call_count) from session_model_usage group by 1,2 order by 3 desc limit 10")]
        except Exception:
            pass
        return r
    except Exception as e:
        return {"error": type(e).__name__}
    finally:
        _rmdb(dst)


@probe(id="hermes.usage", level="L2", family=AI, tier="T1", collect="extended", gate="hermes.present", timeout_ms=5000)
def hermes_usage(h, facts):
    """Primary Hermes state.db (copied with -wal): session/message/tool-call counts, date range, sources, models."""
    hh = _hermes_homes(h)
    p = hh["primary"]
    if not p:
        return None
    r = _hermes_db_summary(h, p["path"], "hermes-state")
    if r is None:
        return None
    prof = []
    for db in glob.glob(os.path.join(p["path"], "profiles", "*", "state.db"))[:20]:
        s = _hermes_db_summary(h, os.path.dirname(db), "hermes-prof") or {}
        prof.append({"sessions": s.get("sessions"), "first": s.get("first"), "last": s.get("last")})
    others = []
    for c in hh["cands"]:
        if c is not p and c["has_state"]:
            s = _hermes_db_summary(h, c["path"], "hermes-other") or {}
            others.append({"role": c["role"], "sessions": s.get("sessions"), "last": s.get("last")})
    return {"present": True, "home_role": p["role"], **r, "profiles": prof, "other_homes": others,
            "side_homes_excluded": len(hh["side"])}


@probe(id="hermes.titles", level="L2", family=AI, tier="T2", collect="deep", gate="hermes.present")
def hermes_titles(h, facts):
    """Most recent top-level Hermes session titles (redacted). T2 content, opt-in."""
    p = _hermes_homes(h)["primary"]
    if not p:
        return None
    r = _hermes_db_summary(h, p["path"], "hermes-titles", titles=True)
    if not r or not r.get("titles"):
        return None
    return {"present": True, "recent": r["titles"]}


# ---------------- MCP inventory, NVIDIA, Ollama

BUNDLED_MCP = {"node_repl"}


@probe(id="l3.mcp_inventory", level="L2", family=AI, tier="T0", collect="core", gate="ai.catalog_absent_checks")
def l3_mcp_inventory(h, facts):
    """Union of whitelisted MCP server names across agents; vendor-bundled servers separated."""
    U, RA = _U(h), _RA(h)
    by = {}
    cj = _claude_json(h)
    if cj:
        by["claude_code"] = sorted(set(cj["global_mcp_servers"]) | set(cj["project_mcp_servers"]))
    cp = os.path.join(_codex_home(h), "config.toml")
    if tomllib and os.path.exists(cp):
        try:
            with open(cp, "rb") as f:
                by["codex"] = sorted((tomllib.load(f).get("mcp_servers") or {}).keys())
        except Exception:
            pass
    for label, path, keys in [("vscode", os.path.join(RA, "Code", "User", "mcp.json"), ("servers", "mcpServers")),
                              ("claude_desktop", os.path.join(RA, "Claude", "claude_desktop_config.json"), ("mcpServers",)),
                              ("cursor", os.path.join(U, ".cursor", "mcp.json"), ("mcpServers",)),
                              ("copilot_cli", os.path.join(U, ".copilot", "mcp-config.json"), ("mcpServers",)),
                              ("gemini_cli", os.path.join(U, ".gemini", "settings.json"), ("mcpServers",)),
                              ("windsurf", os.path.join(U, ".codeium", "windsurf", "mcp_config.json"), ("mcpServers",))]:
        j = _read_json(path, jsonc=True)
        if isinstance(j, dict):
            for k in keys:
                if isinstance(j.get(k), dict):
                    by[label] = sorted(j[k].keys()); break
    p = _hermes_homes(h)["primary"]
    if p and os.path.exists(os.path.join(p["path"], "config.yaml")):
        try:
            by["hermes"] = _parse_hermes_yaml(os.path.join(p["path"], "config.yaml"))["mcp_servers"]
        except OSError:
            pass
    user = sorted({n for v in by.values() for n in v if n not in BUNDLED_MCP})
    bundled = sorted({n for v in by.values() for n in v if n in BUNDLED_MCP})
    nv = _isdir(os.path.join(_LA(h), "NVIDIA Corporation", "NVIDIA App", "McpServer"))
    return {"present": True, "by_agent": {k: v for k, v in by.items() if v}, "user_configured": user,
            "user_configured_total": len(user), "vendor_bundled": bundled + (["nvidia_app_mcp_server"] if nv else [])}


@probe(id="nvidia.app_mcp_server", level="L1", family=AI, tier="T0", collect="core")
def nvidia_app_mcp_server(h, facts):
    """NVIDIA App McpServer dir + server.json key names and non-secret fields (token never emitted)."""
    mcp = os.path.join(_LA(h), "NVIDIA Corporation", "NVIDIA App", "McpServer")
    if not _isdir(mcp):
        return None
    sj = _read_json(os.path.join(mcp, "server.json"))
    keys, info = [], {}
    if isinstance(sj, dict):
        keys = sorted(sj.keys())
        info = {k: sj[k] for k in ("name", "title", "version") if isinstance(sj.get(k), str)}
        del sj
    return {"present": True, "mtime": _mtime(mcp), "files": len(_ls(mcp) or []), "server_json_keys": keys,
            "has_token_key": "token" in keys, **info}


def _ollama_models_root(h):
    return os.environ.get("OLLAMA_MODELS") or os.path.join(_U(h), ".ollama", "models")


@probe(id="ollama.present", level="L1", family=AI, tier="T0", collect="core")
def ollama_present(h, facts):
    """Ollama install (per-user exe, uninstall version), models dir, running process."""
    exe = os.path.join(_LA(h), "Programs", "Ollama", "ollama.exe")
    mroot = _ollama_models_root(h)
    if not (os.path.exists(exe) or _isdir(mroot) or _isdir(os.path.join(_U(h), ".ollama"))):
        return None
    un = _uninst_match(h, r"^Ollama")
    return {"present": True, "exe": os.path.exists(exe), "version": un[0]["version"] if un else _file_version(exe),
            "install_date": un[0]["install_date"] if un else None, "models_dir": _isdir(mroot),
            "OLLAMA_MODELS_set": bool(os.environ.get("OLLAMA_MODELS")), "running": _running(h, r"^ollama")}


@probe(id="ollama.models", level="L2", family=AI, tier="T0", collect="core", gate="ollama.present")
def ollama_models(h, facts):
    """Ollama model manifests (name, size, pulled date) and blob count."""
    mroot = _ollama_models_root(h)
    man = os.path.join(mroot, "manifests")
    models = []
    for f in glob.glob(os.path.join(man, "*", "*", "*", "*"))[:500]:
        if not os.path.isfile(f):
            continue
        parts = f[len(man) + 1:].split(os.sep)
        j = _read_json(f, maxb=1_000_000) or {}
        size = sum(l.get("size", 0) for l in j.get("layers", []) if isinstance(l, dict)) + ((j.get("config") or {}).get("size", 0) or 0)
        name = ("" if parts[1] == "library" else parts[1] + "/") + parts[2] + ":" + parts[3]
        if parts[0] != "registry.ollama.ai":
            name = parts[0] + "/" + name
        models.append({"name": name, "gb": round(size / 1e9, 2), "pulled": _mtime(f)})
    blobs = _ls(os.path.join(mroot, "blobs"), 20000)
    return {"present": True, "count": len(models), "total_gb": round(sum(m["gb"] for m in models), 2),
            "models": sorted(models, key=lambda m: -m["gb"])[:30], "blobs": len(blobs) if blobs is not None else None,
            "manifests_mtime": _mtime(man)}


@probe(id="ollama.app_db", level="L2", family=AI, tier="T1", collect="extended", gate="ollama.present")
def ollama_app_db(h, facts):
    """Ollama app db.sqlite: chat/message counts, signed-in flag (users row count only), first-run flags."""
    db = os.path.join(_LA(h), "Ollama", "db.sqlite")
    if not os.path.exists(db):
        return None
    dst = _copy_db(h, db, "ollama")
    if not dst:
        return None
    try:
        out = {"present": True}
        for t in ("chats", "messages"):
            try:
                out[t] = _q(dst, f"select count(*) from {t}")[0][0]
            except Exception:
                out[t] = None
        try:
            out["signed_in"] = _q(dst, "select count(*) from users")[0][0] > 0
        except Exception:
            out["signed_in"] = None
        cs = _cols(dst, "settings")
        want = [c for c in ("has_completed_first_run", "claude_desktop_used", "codex_desktop_used", "turbo_enabled",
                            "websearch_enabled", "airplane_mode", "expose") if c in cs]
        if want:
            row = _q(dst, "select %s from settings limit 1" % ",".join(want))
            out["settings"] = dict(zip(want, row[0])) if row else None
        return out
    except Exception as e:
        return {"present": True, "error": type(e).__name__}
    finally:
        _rmdb(dst)


@probe(id="ollama.logs", level="L2", family=AI, tier="T1", collect="extended", gate="ollama.present")
def ollama_logs(h, facts):
    """Ollama server/app logs: API call counts by path, active days, first/last; decides installed-but-unused."""
    ld = os.path.join(_LA(h), "Ollama")
    files = glob.glob(os.path.join(ld, "*.log"))[:20]
    if not files:
        return None
    rx = re.compile(r'\[GIN\] (\d{4}/\d\d/\d\d) - (\d\d:\d\d:\d\d) \|.*\|\s+(GET|POST|HEAD|DELETE)\s+"([^"?]+)')
    api, days, ver = collections.Counter(), set(), set()
    first = last = None
    for lf in files:
        try:
            with open(lf, encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh):
                    if n > 500_000:
                        break
                    m = rx.search(line)
                    if m:
                        d = m.group(1) + " " + m.group(2)
                        days.add(m.group(1))
                        first = d if first is None or d < first else first
                        last = d if last is None or d > last else last
                        api[m.group(3) + " " + m.group(4)] += 1
                    v = re.search(r"version=(\d+\.\d+\.\d+)", line)
                    if v:
                        ver.add(v.group(1))
        except OSError:
            continue
    infer = sum(v for k, v in api.items() if re.search(r"/api/(chat|generate|embed)|/v1/(chat|completions)", k))
    return {"present": True, "files": len(files), "api_calls": _top(api, 12), "inference_calls": infer,
            "first": first, "last": last, "active_days": len(days), "versions_logged": sorted(ver)}


# ================================================================== dev

@probe(id="apps.pkg_managers", level="L1", family=DEV, tier="T0", collect="core")
def apps_pkg_managers(h, facts):
    """Package managers present: winget (alias + DesktopAppInstaller), scoop, choco, npm/uv/pipx on PATH."""
    U, LA = _U(h), _LA(h)
    dai = _appx_match(h, r"^Microsoft\.DesktopAppInstaller$")
    out = {"winget": os.path.exists(os.path.join(LA, "Microsoft", "WindowsApps", "winget.exe")),
           "winget_appinstaller_version": dai[0]["version"] if dai else None,
           "scoop": _isdir(os.path.join(U, "scoop")) or _isdir(os.path.join(_PD(), "scoop")),
           "choco": _isdir(os.path.join(_PD(), "chocolatey")),
           "npm": bool(_which("npm")), "uv": bool(_which("uv")), "pipx": bool(_which("pipx")),
           "pnpm": bool(_which("pnpm")), "bun": bool(_which("bun"))}
    return {"present": True, **out}


@probe(id="dev.devmode_sudo_longpaths", level="L1", family=DEV, tier="T0", collect="core")
def dev_devmode_sudo_longpaths(h, facts):
    """Developer Mode, Windows sudo and LongPathsEnabled registry toggles."""
    dm = _open(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock")
    fs = _open(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
    su = _open(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Sudo")
    sudo_mode = {0: "disabled", 1: "forceNewWindow", 2: "disableInput", 3: "normal"}
    sv = _rv(su, "Enabled") if su else None
    return {"present": True, "developer_mode": _rv(dm, "AllowDevelopmentWithoutDevLicense") if dm else None,
            "long_paths": _rv(fs, "LongPathsEnabled") if fs else None,
            "sudo": sudo_mode.get(sv, sv) if sv is not None else None}


@probe(id="dev.docker", level="L1", family=DEV, tier="T0", collect="core")
def dev_docker(h, facts):
    """Docker Desktop install + settings-store.json whitelisted keys (no docker spawn)."""
    inst = os.path.join(_PF(), "Docker", "Docker")
    un = _uninst_match(h, r"^Docker Desktop")
    if not (_isdir(inst) or un):
        return None
    sel = {}
    for p in (os.path.join(_RA(h), "Docker", "settings-store.json"), os.path.join(_RA(h), "Docker", "settings.json")):
        j = _read_json(p)
        if isinstance(j, dict):
            sel = {k: j[k] for k in j if k in ("wslEngineEnabled", "WslEngineEnabled", "autoStart", "AutoStart",
                                               "kubernetesEnabled", "KubernetesEnabled", "UseContainerdSnapshotter",
                                               "useContainerdSnapshotter", "memoryMiB", "MemoryMiB", "cpus", "Cpus",
                                               "EnableDockerAI", "EnableInference")
                   and isinstance(j[k], (bool, int, str))}
            break
    cli = os.path.join(inst, "resources", "bin", "docker.exe")
    return {"present": True, "desktop_version": un[0]["version"] if un else None, "install_date": un[0]["install_date"] if un else None,
            "cli_version": (lambda v: None if v in (None, "0.0.0.0") else v)(_file_version(cli)), "settings": sel,
            "service_running": bool(_running(h, r"^com\.docker\.service|^docker desktop"))}


@probe(id="dev.docker_config_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_docker_config_presence(h, facts):
    """~/.docker/config.json presence/size (may hold registry auths; never opened)."""
    return h.meta(os.path.join(_U(h), ".docker", "config.json"))


@probe(id="dev.gh_auth_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_gh_auth_presence(h, facts):
    """GitHub CLI hosts.yml presence/size (means gh logged in; never opened)."""
    return h.meta(os.path.join(_RA(h), "GitHub CLI", "hosts.yml"))


@probe(id="dev.git", level="L1", family=DEV, tier="T0", collect="core")
def dev_git(h, facts):
    """git on PATH + Git for Windows version from registry / file version (no spawn)."""
    exe = _which("git")
    k = _open(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\GitForWindows")
    reg_ver = _rv(k, "CurrentVersion") if k else None
    inst = _rv(k, "InstallPath") if k else None
    if not (exe or reg_ver):
        return None
    base = inst or os.path.join(_PF(), "Git")
    flavor = next((d for d in ("clangarm64", "mingw64", "mingw32") if _isdir(os.path.join(base, d))), None)
    return {"present": True, "on_path": bool(exe), "git_for_windows_version": reg_ver,
            "file_version": _file_version(os.path.join(base, "cmd", "git.exe")), "flavor": flavor}


@probe(id="dev.npm_globals", level="L2", family=DEV, tier="T1", collect="core", gate="apps.pkg_managers")
def dev_npm_globals(h, facts):
    """Global npm packages from %APPDATA%\\npm\\node_modules listing (no `npm root -g` spawn)."""
    root = os.path.join(_RA(h), "npm", "node_modules")
    names = _ls(root, 500)
    if names is None:
        return {"present": False}
    pk = []
    for n in names:
        if n.startswith("@"):
            pk += [f"{n}/{s}" for s in (_ls(os.path.join(root, n), 100) or [])]
        elif not n.startswith("."):
            pk.append(n)
    return {"present": True, "count": len(pk), "packages": sorted(pk)[:80]}


@probe(id="dev.npmrc_presence", level="L1", family=DEV, tier="T3", collect="core")
def dev_npmrc_presence(h, facts):
    """~/.npmrc presence/size (may hold registry tokens; never opened)."""
    return h.meta(os.path.join(_U(h), ".npmrc"))


_PATH_TOOLS = [("winget_links", r"\\WinGet\\Links"), ("winget_packages", r"\\WinGet\\Packages"),
               ("windowsapps", r"\\WindowsApps$"), ("git", r"\\Git\\(cmd|bin)"), ("nodejs", r"\\nodejs"),
               ("npm_global", r"\\npm$"), ("python", r"\\Python\d+"), ("uv_or_local_bin", r"\\\.local\\bin|\\uv\\"),
               ("hermes", r"\\hermes\\bin"), ("cargo", r"\\\.cargo\\bin"), ("go", r"\\go\\bin|\\Go\\bin"),
               ("vscode", r"Microsoft VS Code"), ("ollama", r"\\Ollama"), ("cua", r"\\Cua\\"),
               ("docker", r"\\Docker\\"), ("powershell7", r"\\PowerShell\\7"), ("dotnet", r"\\dotnet"),
               ("java", r"\\(jdk|Java|Adoptium)"), ("cuda", r"CUDA"), ("tailscale", r"Tailscale"),
               ("nvidia", r"NVIDIA"), ("gh", r"GitHub CLI"), ("obs", r"obs-studio"), ("ffmpeg", r"ffmpeg")]


@probe(id="dev.path_entries", level="L1", family=DEV, tier="T0", collect="core")
def dev_path_entries(h, facts):
    """User/machine PATH entry counts and known-tool matches (paths themselves not emitted)."""
    uk = _open(winreg.HKEY_CURRENT_USER, "Environment")
    mk = _open(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
    up = [x for x in ((_rv(uk, "Path") if uk else "") or "").split(";") if x]
    mp = [x for x in ((_rv(mk, "Path") if mk else "") or "").split(";") if x]
    tools = {"user": sorted({t for t, rx in _PATH_TOOLS for e in up if re.search(rx, e, re.I)}),
             "machine": sorted({t for t, rx in _PATH_TOOLS for e in mp if re.search(rx, e, re.I)})}
    return {"present": True, "user": len(up), "machine": len(mp), "tools": tools}


@probe(id="dev.pwsh7", level="L1", family=DEV, tier="T0", collect="core")
def dev_pwsh7(h, facts):
    """PowerShell 7 presence and file version (never spawned)."""
    exe = _which("pwsh")
    cand = os.path.join(_PF(), "PowerShell", "7", "pwsh.exe")
    path = exe if exe and "windowsapps" not in exe.lower() else (cand if os.path.exists(cand) else exe)
    if not path:
        return None
    return {"present": True, "on_path": bool(exe), "version": _file_version(path) if os.path.exists(path) else None,
            "store_alias": bool(exe and "windowsapps" in exe.lower())}


@probe(id="dev.ssh_config", level="L2", family=DEV, tier="T1", collect="core", gate="dev.path_entries")
def dev_ssh_config(h, facts):
    """~/.ssh/config Host alias count and Include count (hostnames not emitted); known_hosts line count."""
    d = os.path.join(_U(h), ".ssh")
    if not _isdir(d):
        return {"present": False}
    cfg = os.path.join(d, "config")
    aliases = includes = lines = 0
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f):
                if n > 5000:
                    break
                lines += 1
                m = re.match(r"^\s*Host\s+(.+)$", line, re.I)
                if m:
                    aliases += len([x for x in m.group(1).split() if x != "*"])
                if re.match(r"^\s*Include\s", line, re.I):
                    includes += 1
    kh = os.path.join(d, "known_hosts")
    khl = None
    if os.path.exists(kh):
        with open(kh, "rb") as f:
            khl = sum(1 for _ in zip(range(100000), f))
    return {"present": True, "config": os.path.exists(cfg), "config_lines": lines, "host_aliases": aliases,
            "includes": includes, "known_hosts_lines": khl}


_TOOLCHAIN = ["node", "npm", "pnpm", "yarn", "bun", "deno", "nvm", "fnm", "volta", "uv", "py", "pipx", "conda", "poetry",
              "rustc", "cargo", "rustup", "go", "dotnet", "java", "javac", "mvn", "gradle", "cmake", "ninja", "clang",
              "gcc", "zig", "nvcc", "gh", "pwsh", "docker", "kubectl", "terraform", "tailscale", "claude", "codex",
              "hermes", "ollama", "code", "cursor", "windsurf", "zed", "nvim", "vim", "python", "python3"]


@probe(id="dev.toolchain_presence", level="L1", family=DEV, tier="T0", collect="core")
def dev_toolchain_presence(h, facts):
    """which() for node/python/rust/go/java/native/AI CLIs + off-PATH install dirs. Git Bash by path only."""
    found, stubs = [], []
    for t in _TOOLCHAIN:
        p = _which(t)
        if not p:
            continue
        low = p.lower()
        if "\\windowsapps\\" in low and os.path.basename(low) in ("python.exe", "python3.exe"):
            try:
                if os.path.getsize(p) == 0:
                    stubs.append(t); continue
            except OSError:
                pass
        found.append(t)
    U, LA, RA, PF = _U(h), _LA(h), _RA(h), _PF()
    off = {k: _isdir(p) for k, p in [
        ("cargo_home", os.path.join(U, ".cargo")), ("rustup", os.path.join(U, ".rustup")), ("go_root", os.path.join(PF, "Go")),
        ("uv_pythons", os.path.join(RA, "uv", "python")), ("python_org", os.path.join(LA, "Programs", "Python")),
        ("dotnet_sdk", os.path.join(PF, "dotnet", "sdk")), ("java_adoptium", os.path.join(PF, "Eclipse Adoptium")),
        ("java_oracle", os.path.join(PF, "Java")), ("cuda", os.path.join(PF, "NVIDIA GPU Computing Toolkit", "CUDA")),
        ("llvm", os.path.join(PF, "LLVM")), ("android_sdk", os.path.join(LA, "Android", "Sdk")),
        ("miniconda", os.path.join(U, "miniconda3")), ("anaconda", os.path.join(U, "anaconda3")),
        ("pyenv", os.path.join(U, ".pyenv")), ("nvm_windows", os.path.join(RA, "nvm")),
        ("visual_studio", os.path.join(PF, "Microsoft Visual Studio")),
        ("vs_installer", os.path.join(_PF86(), "Microsoft Visual Studio", "Installer"))]}
    return {"present": True, "on_path": found, "store_stubs": stubs,
            "git_bash": os.path.exists(os.path.join(PF, "Git", "bin", "bash.exe")),
            "off_path": sorted(k for k, v in off.items() if v)}


@probe(id="dev.uv_tools", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_uv_tools(h, facts):
    """uv tools and uv-managed Pythons (dir listing), ~/.local/bin entries; no uv spawn."""
    RA, U = _RA(h), _U(h)
    tools = _ls(os.path.join(RA, "uv", "tools"), 200)
    py = _ls(os.path.join(RA, "uv", "python"), 200)
    lb = _ls(os.path.join(U, ".local", "bin"), 300)
    if tools is None and py is None and lb is None:
        return {"present": False}
    return {"present": True, "tools": sorted(t for t in (tools or []) if not t.startswith("."))[:60],
            "pythons": sorted(p for p in (py or []) if p.startswith(("cpython", "pypy")))[:30],
            "local_bin_count": len(lb or []), "local_bin": sorted(lb or [])[:40]}


@probe(id="dev.wsl", level="L1", family=DEV, tier="T0", collect="core")
def dev_wsl(h, facts):
    """WSL distros from the HKCU Lxss key (no wsl.exe), vhdx size, WSL package version, .wslconfig presence."""
    base = r"Software\Microsoft\Windows\CurrentVersion\Lxss"
    k = _open(winreg.HKEY_CURRENT_USER, base)
    distros = []
    if k is not None:
        default = _rv(k, "DefaultDistribution")
        for sub in _subkeys(k, 100):
            dk = _open(winreg.HKEY_CURRENT_USER, base + "\\" + sub)
            if dk is None:
                continue
            bp = (_rv(dk, "BasePath") or "").replace("\\\\?\\", "")
            vhd = None
            for fn in ("ext4.vhdx", "ext4.VHDX"):
                try:
                    vhd = round(os.path.getsize(os.path.join(bp, fn)) / 2**30, 2); break
                except OSError:
                    pass
            distros.append({"name": _rv(dk, "DistributionName"), "version": _rv(dk, "Version"),
                            "default": sub == default, "vhdx_gib": vhd})
    un = _uninst_match(h, r"^Windows Subsystem for Linux")
    appx = _appx_match(h, r"^MicrosoftCorporationII\.WindowsSubsystemForLinux$")
    pkg = (un[0]["version"] if un else None) or (appx[0]["version"] if appx else None) or \
        _file_version(os.path.join(_PF(), "WSL", "wsl.exe"), kind="file")
    inbox = os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wsl.exe"))
    if not (distros or pkg or k is not None):
        return {"present": False, "inbox_launcher": inbox}
    return {"present": True, "distros": distros, "distro_count": len(distros), "wsl_package_version": pkg,
            "inbox_launcher": inbox, "wslconfig": os.path.exists(os.path.join(_U(h), ".wslconfig"))}


@probe(id="dev.wt_profiles", level="L2", family=DEV, tier="T1", collect="core", gate="dev.toolchain_presence")
def dev_wt_profiles(h, facts):
    """Windows Terminal settings.json: profile count, source kinds, default kind, customisation counts."""
    LA = _LA(h)
    out = {}
    for label, p in [("stable", os.path.join(LA, "Packages", "Microsoft.WindowsTerminal_8wekyb3d8bbwe", "LocalState", "settings.json")),
                     ("preview", os.path.join(LA, "Packages", "Microsoft.WindowsTerminalPreview_8wekyb3d8bbwe", "LocalState", "settings.json")),
                     ("unpackaged", os.path.join(LA, "Microsoft", "Windows Terminal", "settings.json"))]:
        j = _read_json(p, jsonc=True)
        if not isinstance(j, dict):
            continue
        profs = j.get("profiles", {})
        plist = profs.get("list", []) if isinstance(profs, dict) else (profs or [])
        dflt = j.get("defaultProfile")
        kinds = collections.Counter()
        dkind = None
        for pr in plist:
            if not isinstance(pr, dict):
                continue
            src = pr.get("source") or ""
            if src:
                kind = src.split(".")[-1]
            else:
                exe = os.path.basename((pr.get("commandline") or "").split(" ")[0]).lower()
                kind = {"powershell.exe": "WindowsPowerShell", "cmd.exe": "CommandPrompt", "pwsh.exe": "PowerShell",
                        "wsl.exe": "WSL", "ssh.exe": "ssh"}.get(exe, "custom" if exe else "unknown")
            kinds[kind] += 1
            if pr.get("guid") == dflt:
                dkind = kind
        out[label] = {"profiles": len(plist), "visible": sum(1 for x in plist if isinstance(x, dict) and not x.get("hidden")),
                      "kinds": dict(kinds), "default_kind": dkind, "mtime": _mtime(p),
                      "actions": len(j.get("actions", []) or []) + len(j.get("keybindings", []) or []),
                      "schemes": len(j.get("schemes", []) or []), "theme_set": bool(j.get("theme"))}
    return {"present": bool(out), **out}


_AI_EXT = re.compile(r"copilot|continue|cline|claude|anthropic|openai|chatgpt|codeium|windsurf|tabnine|amazonq|gemini|cody|sourcegraph|supermaven|roo|kilo|ollama|aider|hermes", re.I)


def _ext_ids(d):
    j = _read_json(os.path.join(d, "extensions.json"))
    ids = []
    if isinstance(j, list):
        ids = [((e.get("identifier") or {}).get("id") or "").lower() for e in j if isinstance(e, dict)]
    if not any(ids):
        ids = [m.group(1).lower() for x in (_ls(d) or []) for m in [re.match(r"^([\w\-]+\.[\w\-]+)-\d", x)] if m]
    return sorted({i for i in ids if i})


@probe(id="editor.vscode", level="L2", family=DEV, tier="T2", collect="core", gate="dev.toolchain_presence")
def editor_vscode(h, facts):
    """VS Code-family editors: install, extension ids (AI ones flagged), workspaces opened, MCP/chat state; other editors."""
    U, LA, RA = _U(h), _LA(h), _RA(h)
    eds = {"vscode": ([os.path.join(LA, "Programs", "Microsoft VS Code"), os.path.join(_PF(), "Microsoft VS Code")],
                      os.path.join(U, ".vscode", "extensions"), os.path.join(RA, "Code", "User")),
           "vscode_insiders": ([os.path.join(LA, "Programs", "Microsoft VS Code Insiders")],
                               os.path.join(U, ".vscode-insiders", "extensions"), os.path.join(RA, "Code - Insiders", "User")),
           "cursor": ([os.path.join(LA, "Programs", "cursor")], os.path.join(U, ".cursor", "extensions"), os.path.join(RA, "Cursor", "User")),
           "windsurf": ([os.path.join(LA, "Programs", "Windsurf")], os.path.join(U, ".windsurf", "extensions"), os.path.join(RA, "Windsurf", "User")),
           "vscodium": ([os.path.join(LA, "Programs", "VSCodium")], os.path.join(U, ".vscode-oss", "extensions"), os.path.join(RA, "VSCodium", "User")),
           "kiro": ([os.path.join(LA, "Programs", "Kiro")], os.path.join(U, ".kiro", "extensions"), os.path.join(RA, "Kiro", "User"))}
    out = {}
    for name, (inst, extd, user) in eds.items():
        installed = any(_isdir(p) for p in inst)
        if not (installed or _isdir(extd) or _isdir(user)):
            continue
        ids = _ext_ids(extd) if _isdir(extd) else []
        e = {"installed": installed, "extensions": len(ids), "extension_ids": ids[:80],
             "ai_extensions": [i for i in ids if _AI_EXT.search(i)]}
        if _isdir(user):
            ws = _ls(os.path.join(user, "workspaceStorage"), 5000)
            e["workspaces_opened"] = len(ws) if ws is not None else 0
            e["settings_json"] = os.path.exists(os.path.join(user, "settings.json"))
            e["keybindings_json"] = os.path.exists(os.path.join(user, "keybindings.json"))
            e["profiles"] = len(_ls(os.path.join(user, "profiles")) or [])
            mj = _read_json(os.path.join(user, "mcp.json"), jsonc=True)
            e["mcp_servers"] = len((mj.get("servers") or mj.get("mcpServers") or {})) if isinstance(mj, dict) else 0
            e["chat_sessions"] = len(glob.glob(os.path.join(user, "globalStorage", "emptyWindowChatSessions", "*"))[:5000]) + \
                len(glob.glob(os.path.join(user, "workspaceStorage", "*", "chatSessions", "*"))[:5000])
            e["user_mtime"] = _mtime(user)
        out[name] = e
    other = {"jetbrains": [x for x in (_ls(os.path.join(RA, "JetBrains")) or []) if re.match(r"^[A-Za-z]+\d{4}\.\d", x)],
             "zed": _isdir(os.path.join(LA, "Zed")) or _isdir(os.path.join(RA, "Zed")),
             "neovim_config": _isdir(os.path.join(LA, "nvim")),
             "vimrc": os.path.exists(os.path.join(U, "_vimrc")) or os.path.exists(os.path.join(U, ".vimrc")),
             "notepadpp": _isdir(os.path.join(_PF(), "Notepad++")),
             "sublime": _isdir(os.path.join(_PF(), "Sublime Text")) or _isdir(os.path.join(RA, "Sublime Text")),
             "visual_studio": [x for x in (_ls(os.path.join(_PF(), "Microsoft Visual Studio")) or []) if x[:2].isdigit()]}
    other = {k: v for k, v in other.items() if v}
    customised = any(e["extensions"] or e.get("settings_json") or e.get("keybindings_json") for e in out.values())
    return {"present": bool(out or other), "extensions": sum(e["extensions"] for e in out.values()),
            "customised": customised, "editors": out, "other": other}


@probe(id="dev.git_global_config", level="L2", family=DEV, tier="T0", collect="extended", gate="dev.git")
def dev_git_global_config(h, facts):
    """~/.gitconfig parsed without spawn: booleans for identity/credential helper/editor + section counts only."""
    p = os.path.join(_U(h), ".gitconfig")
    if not os.path.exists(p):
        return {"present": False}
    sections, keys = collections.Counter(), set()
    cred_kind = None
    cur = None
    with open(p, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f):
            if n > 3000:
                break
            s = line.strip()
            m = re.match(r"^\[\s*([A-Za-z0-9.\-]+)", s)
            if m:
                cur = m.group(1).lower(); sections[cur] += 1; continue
            if "=" in s and cur and not s.startswith(("#", ";")):
                k, v = s.split("=", 1)
                k = k.strip().lower()
                keys.add(f"{cur}.{k}")
                if cur == "credential" and k == "helper" and v.strip():
                    lv = v.lower()
                    cred_kind = next((n for n, rx in (("gh", r"gh(\.exe)?['\"]?\s+auth"), ("manager", r"manager"),
                                                      ("wincred", r"wincred"), ("store", r"^\s*store"),
                                                      ("cache", r"^\s*cache")) if re.search(rx, lv)), "custom")
    return {"present": True, "bytes": os.path.getsize(p),
            "identity_set": "user.name" in keys and "user.email" in keys,
            "identity_name_set": "user.name" in keys, "identity_email_set": "user.email" in keys,
            "credential_helper": cred_kind, "editor_set": "core.editor" in keys,
            "signing_set": "user.signingkey" in keys or "commit.gpgsign" in keys,
            "sections": dict(sections)}


def _version_line(txt):
    for l in txt.splitlines():
        l = l.strip()
        if l:
            return _redact(l, 90)
    return None


@probe(id="dev.toolchain_versions", level="L2", family=DEV, tier="T0", collect="extended", gate="dev.toolchain_presence",
       timeout_ms=12000)
def dev_toolchain_versions(h, facts):
    """--version spawns (parallel, gated on which): node, npm, uv, dotnet, gh, AI CLIs, infra CLIs. Never bash/pwsh."""
    from concurrent.futures import ThreadPoolExecutor
    specs = [("node", ["--version"]), ("npm", ["--version"]), ("uv", ["--version"]), ("gh", ["--version"]),
             ("dotnet", ["--list-sdks"]), ("claude", ["--version"]), ("codex", ["--version"]),
             ("hermes", ["--version"]), ("ollama", ["--version"]), ("docker", ["--version"]),
             ("kubectl", ["version", "--client=true"]), ("tailscale", ["version"]), ("git", ["--version"]),
             ("rustc", ["--version"]), ("go", ["version"]), ("java", ["-version"]), ("deno", ["--version"]),
             ("bun", ["--version"]), ("pnpm", ["--version"])]
    todo = []
    for name, args in specs:
        exe = _which(name)
        if exe and "\\windowsapps\\" not in exe.lower() and os.path.basename(exe).lower() not in ("bash.exe", "pwsh.exe"):
            todo.append((name, [exe] + args))
    env = dict(os.environ, NO_COLOR="1", HERMES_NO_UPDATE_CHECK="1")

    def one(item):
        name, args = item
        rc, txt, ms = _spawn(args, timeout=8, env=env)
        if name == "dotnet":
            sdks = [l.split()[0] for l in txt.splitlines() if re.match(r"^\d", l.strip())]
            rc2, txt2, ms2 = _spawn([args[0], "--list-runtimes"], timeout=8, env=env)
            rts = sorted({" ".join(l.split()[:2]) for l in txt2.splitlines() if re.match(r"^Microsoft\.", l.strip())})
            return name, {"sdks": sdks[:10], "runtimes": rts[:10], "rc": rc, "ms": round(ms + ms2, 1)}
        return name, {"version": _version_line(txt) if rc == 0 or txt else None, "rc": rc, "ms": ms}
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, r in ex.map(one, todo):
            out[name] = r
    return {"present": bool(out), "tools": out}


# ---------------- repos (deep)

_SKIP_DIRS = {"node_modules", "appdata", "$recycle.bin", "system volume information", "windows", "program files",
              "program files (x86)", "programdata", "__pycache__", ".venv", "venv", "site-packages", ".cache",
              "cache", "temp", "tmp", "msocache", "recovery", "perflogs", "onedrivetemp"}


def _is_reparse(e):
    try:
        return bool(e.stat(follow_symlinks=False).st_file_attributes & 0x400) or e.is_symlink()
    except (OSError, AttributeError):
        return e.is_symlink()


def _repos(h):
    def build():
        deadline = time.perf_counter() + 8.0
        U, LA = _U(h), _LA(h)
        hits, seen = [], set()
        visited = [0]

        def walk(root, maxdepth, skip_appdata=True):
            stack = [(root, 0)]
            while stack and time.perf_counter() < deadline and visited[0] < 60000:
                d, depth = stack.pop()
                visited[0] += 1
                try:
                    with os.scandir(d) as it:
                        ents = list(zip(range(3000), it))
                except OSError:
                    continue
                names = {e.name.lower() for _, e in ents}
                if ".git" in names and os.path.exists(os.path.join(d, ".git", "HEAD")) or \
                        (".git" in names and os.path.isfile(os.path.join(d, ".git"))):
                    rp = os.path.normcase(os.path.realpath(d))
                    if rp not in seen:
                        seen.add(rp); hits.append(d)
                    continue
                if depth >= maxdepth:
                    continue
                for _, e in ents:
                    n = e.name.lower()
                    if n in _SKIP_DIRS and (skip_appdata or n != "appdata"):
                        continue
                    try:
                        if not e.is_dir(follow_symlinks=False) or _is_reparse(e):
                            continue
                    except OSError:
                        continue
                    stack.append((e.path, depth + 1))

        walk(U, 5)
        walk(LA, 3, skip_appdata=False)
        try:
            with os.scandir("C:\\") as it:
                tops = [e.path for e in it if e.is_dir(follow_symlinks=False) and e.name.lower() not in _SKIP_DIRS
                        and e.name.lower() != "users" and not _is_reparse(e)]
        except OSError:
            tops = []
        for t in tops[:60]:
            walk(t, 3)
        repos = []
        for d in hits[:200]:
            if _is_operator_path(h, d):
                cls = "operator"
            else:
                rel = os.path.relpath(d, U) if os.path.normcase(d).startswith(os.path.normcase(U)) else None
                first = rel.split(os.sep, 1)[0] if rel else ""
                if os.path.normcase(d).startswith(os.path.normcase(LA)) or first.lower() == "appdata" or first.startswith("."):
                    cls = "tool_managed"
                else:
                    cls = "user_area"
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
    return _memo(h, "repos", build)


def _remote_parse(url):
    m = re.match(r"^(?:\w+://)?(?:[^@/]+@)?([^/:]+)[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    return (m.group(1).lower(), m.group(2), m.group(3)) if m else (None, None, None)


@probe(id="dev.repos", level="L2", family=DEV, tier="T2", collect="deep", gate="dev.git", timeout_ms=15000)
def dev_repos(h, facts):
    """Bounded repo walk (home d5, LOCALAPPDATA d3, C:\\ tops d3): class, markers, freshness. Repo names are T2."""
    r = _repos(h)
    repos = r["repos"]
    by_cls = collections.Counter(x["class"] for x in repos)
    listed = [{"name": os.path.basename(x["path"]), "class": x["class"], "branch": x["branch"],
               "head_mtime": x["head_mtime"], "markers": x["markers"]} for x in repos if x["class"] != "operator"]
    return {"present": bool(repos), "count": len(repos), "by_class": dict(by_cls), "repos": listed[:50],
            "visited_dirs": r["visited_dirs"], "walk_timed_out": r["timed_out"]}


@probe(id="dev.repos.remotes", level="L2", family=DEV, tier="T2", collect="deep", gate="dev.git", timeout_ms=15000)
def dev_repos_remotes(h, facts):
    """Remote hosts (counts) and org/repo names for non-operator repos (T2)."""
    hosts, orgs = collections.Counter(), set()
    for x in _repos(h)["repos"]:
        if x["class"] == "operator":
            continue
        for u in x["remotes"]:
            host, org, name = _remote_parse(u)
            if host:
                hosts[host] += 1
                orgs.add(f"{host}/{org}/{name}")
    return {"present": bool(hosts), "hosts": dict(hosts), "repos": sorted(orgs)[:50]}


def _commits(h):
    def build():
        git = _which("git")
        if not git:
            return None
        ids = set()
        env_ids = os.environ.get("UIL_GIT_IDENTITY", "")
        ids |= {x.strip().lower() for x in env_ids.split(",") if x.strip()}
        gc = os.path.join(_U(h), ".gitconfig")
        try:
            with open(gc, encoding="utf-8", errors="replace") as f:
                ids |= {m.lower() for m in re.findall(r"^\s*email\s*=\s*(\S+)", f.read(100_000), re.M)}
        except OSError:
            pass
        res = {"identity_sources": {"env": bool(env_ids), "global": False, "repo_local": 0}, "by_class": {},
               "hours_local": collections.Counter(), "tz_offsets": collections.Counter(), "weekday": collections.Counter(),
               "month": collections.Counter(), "repos_with_own": {}}
        res["identity_sources"]["global"] = bool(ids - {x.strip().lower() for x in env_ids.split(",") if x.strip()})
        deadline = time.perf_counter() + 10.0
        for x in _repos(h)["repos"]:
            if x["class"] == "operator" or time.perf_counter() > deadline:
                continue
            rid = set(ids)
            if x["local_email"]:
                rid.add(x["local_email"]); res["identity_sources"]["repo_local"] += 1
            rc, txt, _ms = _spawn([git, "-C", x["path"], "log", "--all", "--no-merges", "--format=%ae%x09%an%x09%at%x09%ai",
                                   "-n", "100000"], timeout=10)
            if rc != 0:
                continue
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
            c[0] += mine; c[1] += total
            if mine:
                res["repos_with_own"][x["class"]] = res["repos_with_own"].get(x["class"], 0) + 1
        return res
    return _memo(h, "commits", build)


@probe(id="dev.repos.my_commits", level="L2", family=DEV, tier="T1", collect="deep", gate="dev.git", timeout_ms=20000)
def dev_repos_my_commits(h, facts):
    """Authored-commit counts [mine, all] by repo class, matched on-host to an identity set (never emitted)."""
    c = _commits(h)
    if not c:
        return None
    return {"present": bool(c["by_class"]), "by_class_mine_all": c["by_class"], "identity_sources": c["identity_sources"],
            "repos_with_own_commits": c["repos_with_own"].get("user_area", 0),
            "repos_with_own_commits_by_class": c["repos_with_own"],
            "note": "clone history includes upstream; tool_managed counts are not work on this box"}


@probe(id="dev.repos.commit_hours", level="L2", family=DEV, tier="T1", collect="deep", gate="dev.git", timeout_ms=20000)
def dev_repos_commit_hours(h, facts):
    """Host-timezone hour histogram, bands, weekday, month and author-TZ offsets of the user's commits."""
    c = _commits(h)
    if not c or not c["hours_local"]:
        return None
    hrs = c["hours_local"]
    bands = {f"{a:02d}-{a + 5:02d}": sum(hrs.get(x, 0) for x in range(a, a + 6)) for a in (0, 6, 12, 18)}
    return {"present": True, "commits": sum(hrs.values()), "hours": {str(k): v for k, v in sorted(hrs.items())},
            "bands": bands, "top_hours": [k for k, _ in hrs.most_common(4)], "weekday": dict(c["weekday"]),
            "month": dict(sorted(c["month"].items())), "tz_offsets": dict(c["tz_offsets"].most_common(6))}


@probe(id="dev.shell_history", level="L2", family=DEV, tier="T1", collect="deep", gate="dev.path_entries")
def dev_shell_history(h, facts):
    """Shell history files: line count and last-write only (text never emitted)."""
    out = {}
    psr = os.path.join(_RA(h), "Microsoft", "Windows", "PowerShell", "PSReadLine")
    files = [("psreadline:" + f, os.path.join(psr, f)) for f in (_ls(psr) or []) if f.lower().endswith(".txt")]
    files += [(f, os.path.join(_U(h), f)) for f in (".bash_history", ".zsh_history", ".python_history", ".node_repl_history")]
    for label, p in files:
        try:
            st = os.stat(p)
            if st.st_size > 50_000_000:
                out[label] = {"bytes": st.st_size, "lines": None, "mtime": _day(st.st_mtime)}
                continue
            with open(p, "rb") as f:
                n = sum(1 for l in f if l.strip())
            out[label] = {"lines": n, "bytes": st.st_size, "mtime": _day(st.st_mtime)}
        except OSError:
            continue
    return {"present": bool(out), **out}
