# -*- coding: utf-8 -*-
"""Hermes tray watchdog: keeps the tray icon in step with the desktop app.

Run as an orphan sentinel (via start_watchdog.js or the Startup shortcut —
never from a shell that is itself a child of the desktop app, or it dies with
the app tree on restart and the auto-start silently stops working).

Rules:
- a Hermes desktop window appears            -> start the tray (unless quit)
- the desktop window's PID changes           -> new session: clear the quit
  flag and re-raise (fast relaunch / update restart, whose no-window gap can
  be shorter than one poll)
- the desktop session ends (no window)       -> stop the tray, clear the flag
- "Quit tray helper" writes .quit_flag       -> honoured until the session ends

Tray liveness is probed by testing whether the tray's single-instance port
(45173) is bound — cheaper and shim-proof: under uv venvs pythonw.exe is a
launcher whose real interpreter is a child, so poll() alone can mislead.
Killing the tray uses taskkill /T to take the whole tree.
"""
import ctypes
import os
import socket
import subprocess
import sys
import time

MY_DIR = os.path.dirname(os.path.abspath(__file__))
TRAY = os.path.join(MY_DIR, "hermes_tray.py")
FLAG = os.path.join(MY_DIR, ".quit_flag")
TRAY_PORT = 45173
IS_WIN = sys.platform == "win32"
PYTHONW = os.path.join(sys.prefix, "Scripts", "pythonw.exe" if IS_WIN else "python")

SINGLE = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    SINGLE.bind(("127.0.0.1", 45174))
except OSError:
    sys.exit(0)  # a watchdog is already running

user32 = ctypes.windll.user32
_GetWindowTextLengthW = getattr(user32, "GetWindowTextLengthW")
_GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
_GetWindowTextLengthW.restype = ctypes.c_int
_GetWindowTextW = getattr(user32, "GetWindowTextW")
_GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
_GetWindowTextW.restype = ctypes.c_int
_GetWindowThreadProcessId = getattr(user32, "GetWindowThreadProcessId")
_GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_GetWindowThreadProcessId.restype = ctypes.c_uint


def desktop_pid():
    """PID owning a Hermes desktop window (hidden/minimized included), else None.
    Window-based on purpose: a CLI/gateway-only run must not light the tray."""
    found = {"pid": None}

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _lp):
        n = _GetWindowTextLengthW(hwnd)
        if 0 < n < 200:
            buf = ctypes.create_unicode_buffer(n + 1)
            _GetWindowTextW(hwnd, buf, n + 1)
            t = buf.value
            if t == "Hermes" or t.startswith("Hermes "):
                pid = ctypes.c_ulong()
                _GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                found["pid"] = pid.value or None
                return False
        return True

    user32.EnumWindows(cb, 0)
    return found["pid"]


def tray_alive():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", TRAY_PORT))
        return False
    except OSError:
        return True
    finally:
        s.close()


def spawn_tray():
    if not os.path.exists(PYTHONW):
        return None
    return subprocess.Popen([PYTHONW, TRAY], cwd=MY_DIR, close_fds=True,
                            creationflags=0x8 | 0x200)


def stop_tray(proc):
    if proc is None:
        return
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=10,
                       creationflags=0x08000000)  # CREATE_NO_WINDOW
    except Exception:
        proc.terminate()


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def main():
    tray = None
    last_pid = desktop_pid()
    while True:
        pid = desktop_pid()
        if pid is None:
            _rm(FLAG)  # session over: next launch may auto-raise again
            if tray_alive():
                stop_tray(tray)
            tray = None
        elif last_pid is not None and pid != last_pid:
            _rm(FLAG)  # desktop process swapped = new session
            if tray_alive():
                stop_tray(tray)  # old-session tray belongs to the old desktop
            tray = None
        if pid is not None and not tray_alive() and not os.path.exists(FLAG):
            tray = spawn_tray()
        last_pid = pid
        time.sleep(3)


if __name__ == "__main__":
    while True:  # the sentinel must survive transient probe failures
        try:
            main()
        except Exception:
            pass
        time.sleep(5)
