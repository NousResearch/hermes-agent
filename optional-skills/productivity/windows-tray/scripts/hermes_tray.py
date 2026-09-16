# -*- coding: utf-8 -*-
"""Hermes Windows tray icon.

A resident system-tray icon showing live agent state, with minimize-to-tray.
Reads Hermes' own state files read-only; imports nothing from the Hermes core.

Dot state (polled every 2s; derivation lives in windows_tray_state.py):
  blue   a session is running a turn
  amber  the turn is parked waiting for you (clarify question / approval)
  grey   idle
  red    the last turn ended in error

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
from windows_tray_state import compute_state, hermes_home  # noqa: E402

APP_NAME = "Hermes"
POLL_SECS = 2

HERMES_HOME = hermes_home()
MY_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(MY_DIR, "tray.log")

_single = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    _single.bind(("127.0.0.1", 45173))
except OSError:
    sys.exit(0)  # another tray instance is running

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
SW_HIDE, SW_RESTORE, SW_SHOW = 0, 9, 5

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


# ---- window probes / actions --------------------------------------------------
def _fn(mod, name, argtypes, restype=None):
    f = getattr(mod, name)
    f.argtypes = argtypes
    if restype:
        f.restype = restype
    return f


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


def _title(hwnd):
    n = _GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    _GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def hermes_windows():
    """Top-level Hermes desktop windows, hidden/minimized included."""
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _lp):
        t = _title(hwnd)
        if t and (t == APP_NAME or t.startswith(APP_NAME + " ")):
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found


def classify():
    """-> (hwnd, kind) kind in visible|minimized|hidden|None; prefer a visible one."""
    hidden = None
    for h in hermes_windows():
        if _IsWindowVisible(h):
            return h, ("minimized" if _IsIconic(h) else "visible")
        if hidden is None:
            hidden = h
    return (hidden, "hidden") if hidden else (None, None)


def bring_to_front(hwnd):
    _ShowWindow(hwnd, SW_SHOW)
    _ShowWindow(hwnd, SW_RESTORE)
    other = _GetWindowThreadProcessId(_GetForegroundWindow(), None)
    mine = _GetCurrentThreadId()
    _AttachThreadInput(mine, other, True)  # bare SetForegroundWindow is refused cross-process
    _SetForegroundWindow(hwnd)
    _BringWindowToTop(hwnd)
    _AttachThreadInput(mine, other, False)


def focus_or_launch():
    hwnd, _kind = classify()
    if hwnd:
        bring_to_front(hwnd)
        return
    exe = shutil.which("hermes")
    if exe:
        subprocess.Popen([exe, "desktop"], creationflags=0x8 | 0x200, close_fds=True)


def _log(msg):
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 200_000:
            os.replace(LOG_PATH, LOG_PATH + ".old")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except OSError:
        pass


# ---- tray ---------------------------------------------------------------------
_lbl = {"agent": "..."}
# pystray 3.x menu text is read-only: dynamic labels must be a callable, the
# Win32 backend re-evaluates it every time the menu opens.
status_item = pystray.MenuItem(lambda item: "Status: " + _lbl["agent"], None, enabled=False)
_last = {"sig": None}


def tick(icon):
    hwnd, kind = classify()
    if kind == "minimized" and _cfg["autohide"]:
        _ShowWindow(hwnd, SW_HIDE)  # leave the taskbar; dot still tracks agent state
        _log("hid minimized window %s" % hwnd)
        kind = "hidden"
    win = {"visible": "shown", "minimized": "in tray", "hidden": "in tray", None: "closed"}[kind]
    state = compute_state(HERMES_HOME)
    sig = (state, win)
    if sig != _last["sig"]:
        _last["sig"] = sig
        _lbl["agent"] = LABELS[state]
        icon.icon = draw_icon(state)
        icon.title = "Hermes - %s (%s)" % (LABELS[state], win)


def poll_loop(icon):
    while True:
        try:
            tick(icon)
        except Exception as e:  # one failed probe must never kill the loop
            _log("poll error: %r" % e)
        time.sleep(POLL_SECS)


def quit_tray():
    """User quit: the watchdog honours this flag until the desktop session ends."""
    try:
        open(os.path.join(MY_DIR, ".quit_flag"), "a").close()
    except OSError:
        pass
    icon.stop()


def build_icon():
    menu = pystray.Menu(
        status_item,
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open / restore Hermes", lambda: focus_or_launch(), default=True),
        pystray.MenuItem("Hide minimized window to tray",
                         lambda: _cfg.update(autohide=not _cfg["autohide"]),
                         checked=lambda item: _cfg["autohide"]),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit tray helper", quit_tray),
    )
    return pystray.Icon("hermes-tray", draw_icon("idle"), "Hermes tray", menu)


icon = build_icon()


def main():
    threading.Thread(target=poll_loop, args=(icon,), daemon=True).start()
    _log("tray started (pid=%d home=%s)" % (os.getpid(), HERMES_HOME))
    icon.run()


if __name__ == "__main__":
    main()
