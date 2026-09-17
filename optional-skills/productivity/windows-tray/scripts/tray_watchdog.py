# -*- coding: utf-8 -*-
"""Hermes tray watchdog: keeps the tray icon in step with the desktop app.

Run as an orphan sentinel (via start_watchdog.js or the Startup shortcut —
never from a shell that is itself a child of the desktop app, or it dies with
the app tree on restart and the auto-start silently stops working).

Rules:
- a Hermes desktop process appears          -> start the tray (unless quit)
- the desktop process PID changes           -> new session: clear the quit
  flag and re-raise (fast relaunch / update restart, whose no-process gap can
  be shorter than one poll)
- the desktop session ends (no process)     -> stop the tray, clear the flag
- "Quit tray helper" writes .quit_flag      -> honoured until the session ends

Desktop detection uses a kernel32 Toolhelp process snapshot, NOT EnumWindows:
EnumWindows sends a message per window-thread and blocks forever when any hung
window thread exists in the session (common with Electron relaunches) — a
frozen watchdog silently abandons the tray. The snapshot touches no window
thread and cannot hang. Tracked by the PID of Hermes.exe so CLI/gateway-only
runs (no Hermes.exe image) still do not light the tray.

Tray liveness is probed by testing whether the tray's single-instance port
(45173) is bound — cheaper and shim-proof: under uv venvs pythonw.exe is a
launcher whose real interpreter is a child, so poll() alone can mislead.
Killing the tray uses taskkill /T to take the whole tree, with
CREATE_NO_WINDOW so no console flashes on a pythonw host.
Every action is logged to watchdog.log; a heartbeat line every ~10 min makes
a frozen sentinel diagnosable after the fact.
"""
import ctypes
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

MY_DIR = os.path.dirname(os.path.abspath(__file__))
TRAY = os.path.join(MY_DIR, "hermes_tray.py")
FLAG = os.path.join(MY_DIR, ".quit_flag")
LOG = os.path.join(MY_DIR, "watchdog.log")
TRAY_PORT = 45173
IS_WIN = sys.platform == "win32"
PYTHONW = os.path.join(sys.prefix, "Scripts", "pythonw.exe" if IS_WIN else "python")
CREATE_NO_WINDOW = 0x08000000

SINGLE = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    SINGLE.bind(("127.0.0.1", 45174))
except OSError:
    sys.exit(0)  # a watchdog is already running


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
    except OSError:
        pass


if IS_WIN:
    kernel32 = ctypes.windll.kernel32
    TH32CS_SNAPPROCESS = 0x2
    INVALID_HANDLE = ctypes.c_void_p(-1).value

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

    # Truncating the snapshot handle to int32 (default restype) invalidates it
    # on x64 — the first version of this file shipped that bug.
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.Process32FirstW.restype = ctypes.c_bool
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = ctypes.c_bool
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

    def desktop_pid():
        """PID owning a running Hermes.exe image, else None."""
        h = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not h or h == INVALID_HANDLE:
            return None  # snapshot failed: treat as not-found, retry next poll
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
else:
    def desktop_pid():
        return None


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
        log("spawn FAILED: no interpreter at %s" % PYTHONW)
        return None
    try:
        p = subprocess.Popen([PYTHONW, TRAY], cwd=MY_DIR, close_fds=True,
                             creationflags=0x8 | CREATE_NO_WINDOW)
    except Exception as e:
        log("spawn ERROR: %r" % (e,))
        return None
    log("spawned tray pid=%d" % p.pid)
    return p


def stop_tray(proc):
    if proc is None:
        return
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=10,
                       creationflags=CREATE_NO_WINDOW)
    except Exception:
        proc.terminate()


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def main():
    log("watchdog started (pid=%d, pythonw=%s)" % (os.getpid(), PYTHONW))
    tray = None
    last_pid = desktop_pid()
    beat = 0
    while True:
        pid = desktop_pid()
        if pid is None:
            _rm(FLAG)  # session over: next launch may auto-raise again
            if tray_alive():
                stop_tray(tray)
                log("desktop gone -> tray stopped")
            tray = None
        elif last_pid is not None and pid != last_pid:
            _rm(FLAG)  # desktop process swapped = new session
            if tray_alive():
                stop_tray(tray)  # old-session tray belongs to the old desktop
                log("desktop swapped -> old tray stopped")
            tray = None
        if pid is not None and not tray_alive() and not os.path.exists(FLAG):
            tray = spawn_tray()
        last_pid = pid
        beat += 1
        if beat % 200 == 0:  # ~10 min heartbeat; a frozen sentinel stops logging
            log("heartbeat beat=%d desktop=%s tray_port_up=%s"
                % (beat, last_pid, tray_alive()))
        time.sleep(3)


if __name__ == "__main__":
    while True:  # the sentinel must survive transient probe failures
        try:
            main()
        except Exception as e:
            log("main crashed: %r" % (e,))
        time.sleep(5)
