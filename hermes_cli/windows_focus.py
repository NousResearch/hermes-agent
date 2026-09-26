"""hermes:// protocol handler for Windows.

When a notification like `hermes://focus` is activated (e.g. by clicking
a toast), Windows launches this script. It then finds the Hermes terminal
window and brings it to the foreground.

Window-resolution strategy (most to least reliable):

1. ``AttachConsole`` to the Hermes process and grab its console window
   handle. This works for *real* consoles (conhost / ``powershell.exe``)
   where the Python process shares the console window with the terminal.
2. Walk up the process tree from the Hermes PID to the terminal host
   (``WindowsTerminal.exe`` / ``powershell.exe`` / ``conhost.exe`` / ...)
   and focus its main window. This covers Windows Terminal (ConPTY),
   which owns no real console window of its own.
3. Match any visible window whose title contains "Hermes" (last resort).
"""

import sys
import os
import argparse
import ctypes
from ctypes import wintypes

# pywin32 is Windows-only. Importing this module on a non-Windows host (or any
# machine without pywin32 installed) MUST degrade gracefully — it must never
# abort the parent process. The CLI imports this module to decide whether a
# toast should be suppressed, so a hard crash here would kill Hermes on launch
# or on every single notification event.
try:
    import win32gui
    import win32con
    import win32process
    import win32api
    _HAS_WIN32_GUI = True
except ImportError:
    _HAS_WIN32_GUI = False

try:
    import psutil
except ImportError:
    psutil = None


# --- Debug logging (this script runs hidden as a toast-click subprocess) ------

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes_focus.log")


def _log(msg):
    """Append a line to hermes_focus.log for diagnostics (subprocess has no UI)."""
    try:
        ts = __import__("datetime").datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write("[{}] {}\n".format(ts, msg))
            fh.flush()
    except Exception:
        pass


# --- Console window resolution (conhost / powershell.exe) -----------------

# Only bind the Win32 APIs when we are actually on Windows with pywin32. On
# every other platform these calls raise at import time and would break the
# import (and therefore the whole Hermes process that imported us).
if _HAS_WIN32_GUI:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SetFocus.argtypes = [wintypes.HWND]
    user32.SetFocus.restype = wintypes.HWND
else:
    kernel32 = None
    user32 = None


def _force_foreground(hwnd):
    """Bring *hwnd* to the foreground — safe, minimal approach.

    We do NOT use any of the classic "steal focus" tricks (AttachThreadInput,
    synthetic ALT key, SystemParametersInfoW foreground-lock timeout reset,
    LockSetForegroundWindow). Every single one of those has been shown to
    interfere with conhost / prompt_toolkit's input loop, causing the terminal
    to freeze ("can see it but can't type", or complete hang requiring
    task-kill).

    Instead we:
      1. Show/restore the window from minimized/taskbar state.
      2. Call SetForegroundWindow exactly once — if Windows allows it, great.
      3. If denied (returns 0), FlashWindow so the taskbar entry blinks and
         the user can click it manually.
    """
    if not _HAS_WIN32_GUI or not hwnd:
        _log("_force_foreground: unavailable or no hwnd, abort")
        return False

    _log("_force_foreground: hwnd={}".format(hwnd))

    # 1) Restore from minimized / show if hidden.
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        else:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    except Exception as exc:
        _log("  ShowWindow error: {}".format(exc))

    # 2) One clean attempt. No tricks.
    result = False
    try:
        result = bool(user32.SetForegroundWindow(hwnd))
        _log("  SetForegroundWindow -> {}".format(result))
    except Exception as exc:
        _log("  SetForegroundWindow error: {}".format(exc))
        result = False

    # 3) If Windows denied it, flash the taskbar so user notices.
    if not result:
        try:
            win32gui.FlashWindow(hwnd, True)
            _log("  flashed taskbar (focus denied)")
        except Exception as exc:
            _log("  FlashWindow error: {}".format(exc))

    return result


def _console_window_for_pid(pid):
    """DEPRECATED / unused.

    This used to call ``AttachConsole(pid)`` + ``GetConsoleWindow()`` +
    ``FreeConsole()``. That sequence attaches this (hidden) process to the
    target console and then detaches — which **resets the console's input
    mode** (clearing the raw/VT flags that prompt_toolkit sets). The result
    is exactly the reported symptom: after a toast click the terminal becomes
    unresponsive to keyboard/clicks yet still scrolls. We no longer use this
    path. Kept only as a documented tombstone.
    """
    _log("DEPRECATED _console_window_for_pid({}) called — returning None (no console touch)".format(pid))
    return None


# --- Process / window helpers --------------------------------------------

def _window_pid(hwnd):
    if not _HAS_WIN32_GUI:
        return None
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return pid
    except Exception:
        return None


def _main_window_for_pid(pid):
    """Return the first visible top-level window owned by *pid*."""
    if not _HAS_WIN32_GUI:
        return None
    result = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and not win32gui.GetParent(hwnd):
            if _window_pid(hwnd) == pid:
                result.append(hwnd)
        return True

    win32gui.EnumWindows(cb, None)
    return result[0] if result else None


_TERMINAL_NAMES = {
    "windowsterminal.exe", "wt.exe",
    "powershell.exe", "pwsh.exe", "cmd.exe",
    "conhost.exe", "bash.exe", "mintty.exe",
}


def _ancestor_terminal_window(pid):
    """Walk up from *pid* to a terminal host and focus its main window."""
    if not _HAS_WIN32_GUI or psutil is None:
        return None
    try:
        proc = psutil.Process(pid)
        chain = [proc]
        cur = proc
        for _ in range(12):
            parent = cur.parent()
            if parent is None:
                break
            chain.append(parent)
            cur = parent
        for p in chain:
            try:
                if p.name().lower() in _TERMINAL_NAMES:
                    hwnd = _main_window_for_pid(p.pid)
                    if hwnd:
                        return hwnd
            except Exception:
                continue
    except Exception:
        pass
    return None


def _window_by_title(substr):
    if not _HAS_WIN32_GUI:
        return None
    result = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and substr.lower() in title.lower():
                result.append(hwnd)
        return True

    win32gui.EnumWindows(cb, None)
    return result[0] if result else None


HERMES_FOCUS_PID_FILE = os.path.join(os.path.dirname(__file__), "hermes.pid")


def _read_pid_file():
    try:
        with open(HERMES_FOCUS_PID_FILE, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                return int(text)
    except Exception:
        pass
    return None


def find_hermes_window(target_pid=None):
    """Locate the Hermes terminal window by PID first, then fallbacks.

    IMPORTANT: never call ``_console_window_for_pid`` (AttachConsole-based) —
    it corrupts the console input mode. We only use process-tree ancestry and a
    title fallback, both of which are read-only window enumeration.
    """
    if target_pid is None:
        target_pid = _read_pid_file()
    _log("find_hermes_window target_pid={}".format(target_pid))
    if target_pid is not None:
        # 1. Terminal host window via process ancestry (Windows Terminal +
        #    bare PowerShell — walks up to powershell.exe / conhost.exe).
        hwnd = _ancestor_terminal_window(target_pid)
        if hwnd:
            _log("  found via ancestry: hwnd={}".format(hwnd))
            return hwnd
    # 2. Title-based fallback (works if the terminal title was set).
    hwnd = _window_by_title("Hermes")
    if hwnd:
        _log("  found via title: hwnd={}".format(hwnd))
        return hwnd
    _log("  NOT FOUND")
    return None


def is_hermes_foreground(target_pid=None):
    """Return True if the Hermes terminal window is currently the foreground
    window. Used by the CLI to suppress redundant toasts when the user is
    already looking at the terminal.

    Fails *open* (returns False) on any error so a focus-check failure never
    silently suppresses a notification that should have been shown. On a
    non-Windows host (where the Win32 APIs are unavailable) it simply returns
    False — the toast is shown unconditionally, which is the safe default.
    """
    if not _HAS_WIN32_GUI:
        return False
    try:
        fg = win32gui.GetForegroundWindow()
        if not fg:
            return False
        target = find_hermes_window(target_pid=target_pid)
        if target is None:
            return False
        return fg == target
    except Exception:
        return False


def _parse_pid_from_url(url):
    try:
        if "?" in url:
            query = url.split("?", 1)[1]
            for part in query.split("&"):
                if part.startswith("pid="):
                    return int(part.split("=", 1)[1])
    except Exception:
        pass
    return None


def bring_to_front(target_pid=None):
    # Support being launched directly from toast activation:
    #   pythonw.exe hermes_focus.py "hermes://focus?pid=12345"
    _log("bring_to_front target_pid={}".format(target_pid))
    if not _HAS_WIN32_GUI:
        print("hermes_focus: window focusing is only supported on Windows.", file=sys.stderr)
        return
    if target_pid is None and len(sys.argv) > 1:
        arg = sys.argv[1]
        if isinstance(arg, str) and arg.startswith("hermes://"):
            target_pid = _parse_pid_from_url(arg)

    hwnd = find_hermes_window(target_pid=target_pid)
    if hwnd:
        _force_foreground(hwnd)
        print("Brought Hermes window (hwnd={}) to foreground.".format(hwnd))
    else:
        _log("bring_to_front: Hermes window NOT FOUND")
        print("Hermes window not found. Launching hermes...")
        hermes_exe = os.path.join(sys.prefix, "Scripts", "hermes.exe")
        if not os.path.exists(hermes_exe):
            hermes_exe = "hermes"
        os.system('start "" "{}"'.format(hermes_exe))


if __name__ == "__main__":
    # Windows protocol activation passes the URL as a positional arg, e.g.:
    #   pythonw.exe hermes_focus.py "hermes://focus?pid=12345"
    # Strip it before argparse so the unknown-positional check doesn't fire.
    _log("=== hermes_focus.py launched, argv={} ===".format(sys.argv))
    url_arg = None
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith("hermes://"):
            url_arg = arg
            sys.argv.pop(i)
            break

    parser = argparse.ArgumentParser(description="Focus the Hermes window.")
    parser.add_argument("--pid", type=int, default=None, help="Hermes process PID")
    args = parser.parse_args()

    # If launched via protocol, treat the URL as the pid source.
    if args.pid is None and url_arg:
        args.pid = _parse_pid_from_url(url_arg)

    bring_to_front(target_pid=args.pid)
