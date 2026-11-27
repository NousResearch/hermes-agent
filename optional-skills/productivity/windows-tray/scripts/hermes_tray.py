# -*- coding: utf-8 -*-
"""Hermes Windows tray icon (single resident process; no separate watchdog).

A system-tray icon that mirrors live agent state and offers minimize-to-tray.
Reads Hermes' own state files read-only; imports nothing from the Hermes core.

Dot state (polled every 2s; derivation lives in windows_tray_state.py):
  blue   a session is running a turn
  amber  the turn is parked waiting for you (clarify question / approval)
  grey   idle
  red    the last turn ended in error

The ICON (not the process) tracks the desktop session: it appears when a
Hermes.exe process shows up and hides when that process goes away. Keeping
one resident process instead of tray+watchdog halves the memory footprint
(~32 MB vs ~58 MB of pythonw processes) at the cost of no crash supervisor;
the Startup shortcut and single-instance lock keep it in place.

Desktop presence is probed with a kernel32 Toolhelp process snapshot, NOT
EnumWindows: EnumWindows messages every window thread and blocks forever
when a hung thread exists in the session (common around Electron relaunches)
— an earlier watchdog architecture froze exactly that way and silently
abandoned the tray. The snapshot touches no window thread and cannot hang.

Requires pystray + pillow in a DEDICATED venv (Hermes' own venv is stripped by
its dependency sync). Run under pythonw.exe so no console window exists.
"""
import ctypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import pystray
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windows_tray_state import (compute_state, hermes_home,  # noqa: E402
                                icon_visibility)

APP_NAME = "Hermes"
POLL_SECS = 2
TRAY_PORT = 45173

HERMES_HOME = hermes_home()
MY_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(MY_DIR, "tray.log")
FLAG = os.path.join(MY_DIR, ".quit_flag")  # "hide until next desktop session"

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
SW_HIDE, SW_RESTORE, SW_SHOW = 0, 9, 5
CREATE_NO_WINDOW = 0x08000000

_cfg = {"autohide": True}

SIZE = 32
BG = (24, 24, 26, 255)
FG = (232, 193, 96, 255)  # Hermes gold
STATES = {
    "needs_input": (255, 176, 32, 255),  # amber: waiting for you
    "active": (66, 133, 244, 255),       # blue: turn running
    "idle": (130, 130, 135, 255),        # grey
    "error": (225, 80, 80, 255),         # red: last turn failed
}
LABELS = {
    "needs_input": "Needs your input",
    "active": "Session running",
    "idle": "Idle",
    "error": "Last turn errored",
}


def draw_icon(state):
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, SIZE - 2, SIZE - 2], radius=8, fill=BG)
    try:
        font = ImageFont.truetype(
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"), 19)
    except OSError:
        font = ImageFont.load_default()
    try:
        d.text((SIZE // 2, SIZE // 2 - 1), "H", font=font, fill=FG, anchor="mm")
    except TypeError:  # older Pillow without anchor support
        d.text((10, 5), "H", font=font, fill=FG)
    d.ellipse([SIZE - 15, SIZE - 15, SIZE - 4, SIZE - 4], fill=STATES.get(state, STATES["idle"]))
    return img


# ---- win32 probes ------------------------------------------------------------
def _fn(mod, name, argtypes, restype=None):
    f = getattr(mod, name)
    f.argtypes = argtypes
    if restype:
        f.restype = restype
    return f


_IsWindow = _fn(user32, "IsWindow", [ctypes.c_void_p], ctypes.c_bool)
_IsWindowVisible = _fn(user32, "IsWindowVisible", [ctypes.c_void_p], ctypes.c_bool)
_IsIconic = _fn(user32, "IsIconic", [ctypes.c_void_p], ctypes.c_bool)
_ShowWindow = _fn(user32, "ShowWindow", [ctypes.c_void_p, ctypes.c_int], ctypes.c_bool)
_SetForegroundWindow = _fn(user32, "SetForegroundWindow", [ctypes.c_void_p], ctypes.c_bool)
_GetForegroundWindow = _fn(user32, "GetForegroundWindow", [], ctypes.c_void_p)
_BringWindowToTop = _fn(user32, "BringWindowToTop", [ctypes.c_void_p], ctypes.c_bool)
_GetWindowTextLengthW = _fn(user32, "GetWindowTextLengthW", [ctypes.c_void_p], ctypes.c_int)
_GetWindowTextW = _fn(user32, "GetWindowTextW", [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int], ctypes.c_int)
_GetWindowThreadProcessId = _fn(user32, "GetWindowThreadProcessId", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint)
_AttachThreadInput = _fn(user32, "AttachThreadInput", [ctypes.c_uint, ctypes.c_uint, ctypes.c_bool], ctypes.c_bool)
_GetCurrentThreadId = _fn(kernel32, "GetCurrentThreadId", [], ctypes.c_uint)

TH32CS_SNAPPROCESS = 0x2
_INVALID = ctypes.c_void_p(-1).value


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


# Without c_void_p restype the snapshot handle is truncated to int32 on x64
# and every lookup silently fails — the exact bug the first rewrite shipped.
kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
kernel32.Process32FirstW.restype = ctypes.c_bool
kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.restype = ctypes.c_bool
kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.CloseHandle.argtypes = [ctypes.c_void_p]


def desktop_pid():
    """PID owning a running Hermes.exe image, else None (never blocks)."""
    h = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not h or h == _INVALID:
        return None
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    pid = None
    ok = kernel32.Process32FirstW(h, ctypes.byref(entry))
    while ok:
        if entry.szExeFile.lower() == "hermes.exe":
            pid = entry.th32ProcessID or None
            break
        ok = kernel32.Process32NextW(h, ctypes.byref(entry))
    kernel32.CloseHandle(h)
    return pid


def _title(hwnd):
    n = _GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    _GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


_hwnd_cache = {"ts": 0.0, "hwnd": None, "kind": None}


def classify():
    """-> (hwnd, kind) kind in visible|minimized|hidden|None; prefer a visible one.

    EnumWindows is still needed here (find the window to hide/restore), but it
    runs behind a 5s cache and only while the desktop process is up, so a
    transiently hung third-party window thread costs at most one stale read
    instead of freezing the whole tray.
    """
    now = time.time()
    if now - _hwnd_cache["ts"] < 5 and (_hwnd_cache["hwnd"] is None
                                        or _IsWindow(_hwnd_cache["hwnd"])):
        return _hwnd_cache["hwnd"], _hwnd_cache["kind"]
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _lp):
        t = _title(hwnd)
        if t and (t == APP_NAME or t.startswith(APP_NAME + " ")):
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    _hwnd_cache["ts"] = now
    hidden = None
    for h in found:
        if _IsWindowVisible(h):
            r = (h, "minimized" if _IsIconic(h) else "visible")
            break
        if hidden is None:
            hidden = h
    else:
        r = (hidden, "hidden") if hidden else (None, None)
    _hwnd_cache["hwnd"], _hwnd_cache["kind"] = r
    return r


def bring_to_front(hwnd):
    _ShowWindow(hwnd, SW_SHOW)
    _ShowWindow(hwnd, SW_RESTORE)
    other = _GetWindowThreadProcessId(_GetForegroundWindow(), None)
    mine = _GetCurrentThreadId()
    _AttachThreadInput(mine, other, True)  # bare SetForegroundWindow is refused cross-process
    _SetForegroundWindow(hwnd)
    _BringWindowToTop(hwnd)
    _AttachThreadInput(mine, other, False)
    _hwnd_cache["ts"] = 0.0


def focus_or_launch():
    hwnd, _kind = classify()
    if hwnd:
        bring_to_front(hwnd)
        return
    exe = shutil.which("hermes")
    if exe:
        subprocess.Popen([exe, "desktop"],
                         creationflags=0x8 | 0x200 | 0x80000, close_fds=True)


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _log(msg):
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 200_000:
            os.replace(LOG_PATH, LOG_PATH + ".old")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except OSError:
        pass


# ---- tray loop -------------------------------------------------------------
_lbl = {"agent": "...", "win": "closed"}
# pystray 3.x menu text is read-only: dynamic labels must be a callable, the
# Win32 backend re-evaluates it every time the menu opens.
status_item = pystray.MenuItem(lambda item: "Status: " + _lbl["agent"], None, enabled=False)
win_item = pystray.MenuItem(lambda item: "Window: " + _lbl["win"], None, enabled=False)
_last = {"sig": None, "desktop": object()}  # sentinel: first poll sees "change"


def tick(icon):
    pid = desktop_pid()
    visible, clear_flag = icon_visibility(pid, _last["desktop"], os.path.exists(FLAG))
    if clear_flag:
        _rm(FLAG)  # new/ended session: a stale hide-request must not leak over
    if pid != _last["desktop"]:
        _last["desktop"] = pid
        _hwnd_cache["ts"] = 0.0
        _lbl["win"] = "closed"
        if pid is None:
            icon.visible = False
            _log("desktop gone -> icon hidden")
            return
        _log("desktop up (pid=%s)" % pid)
    if not visible:
        if icon.visible:
            icon.visible = False
        return
    try:
        icon.visible = True
    except Exception:
        _last["desktop"] = None  # pystray loop not ready yet: retry next beat
        return

    hwnd, kind = classify()
    if kind == "minimized" and _cfg["autohide"]:
        _ShowWindow(hwnd, SW_HIDE)  # leave the taskbar; dot still tracks agent state
        _log("hid minimized window %s" % hwnd)
        _hwnd_cache["ts"] = 0.0
        kind = "hidden"
    win = {"visible": "shown", "minimized": "in tray", "hidden": "in tray"}.get(kind, "in tray")
    state = compute_state(HERMES_HOME)
    sig = (state, win)
    if sig != _last["sig"]:
        _last["sig"] = sig
        _lbl["agent"], _lbl["win"] = LABELS[state], win
        icon.icon = draw_icon(state)
        icon.title = "Hermes - %s (%s)" % (LABELS[state], win)


def poll_loop(icon):
    beat = 0
    while True:
        try:
            tick(icon)
        except Exception as e:  # one failed probe must never kill the loop
            _log("poll error: %r" % e)
        beat += 1
        if beat % 300 == 0:  # ~10 min heartbeat: a frozen process stops logging
            _log("heartbeat beat=%d desktop=%s visible=%s"
                 % (beat, _last["desktop"], icon.visible))
        time.sleep(POLL_SECS)


def hide_until_new_session():
    """Hide the icon until the next desktop session (v1 quit_flag semantics)."""
    try:
        open(FLAG, "a").close()
    except OSError:
        pass
    icon.visible = False


def build_icon():
    menu = pystray.Menu(
        status_item,
        win_item,
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open / restore Hermes", lambda: focus_or_launch(), default=True),
        pystray.MenuItem("Hide minimized window to tray",
                         lambda: _cfg.update(autohide=not _cfg["autohide"]),
                         checked=lambda item: _cfg["autohide"]),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Hide icon until next Hermes launch", hide_until_new_session),
        pystray.MenuItem("Quit tray helper", lambda: icon.stop()),
    )
    return pystray.Icon("hermes-tray", draw_icon("idle"), "Hermes tray", menu)


icon = build_icon()


def main():
    single = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # lock inside main(): importing this module for diagnostics stays clean
        single.bind(("127.0.0.1", TRAY_PORT))
    except OSError:
        sys.exit(0)  # another tray instance is running
    threading.Thread(target=poll_loop, args=(icon,), daemon=True).start()
    _log("tray started (pid=%d home=%s)" % (os.getpid(), HERMES_HOME))
    icon.run()  # pystray starts invisible; first tick lights it by desktop state


if __name__ == "__main__":
    def _hook(ty, val, tb):
        import traceback
        _log("FATAL " + "".join(traceback.format_exception(ty, val, tb)))
    sys.excepthook = _hook
    main()
