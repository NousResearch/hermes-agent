"""Cross-platform desktop notification dispatcher for Hermes.

This is the single entry point Hermes calls to surface a desktop notification
(session-ended, approval-required, etc.). It dispatches to the best available
backend for the current platform and degrades gracefully when none is available
— it never raises, and never blocks the caller for more than a moment:

* Windows  -> :mod:`hermes_cli.windows_notify` (WinRT toast, PowerShell WinRT
              fallback) with AUMID + click-to-focus via the ``hermes://``
              protocol.  That module owns all Windows-specific logic.
* macOS    -> ``osascript display notification`` (Notification Center).
* Linux    -> ``notify-send`` (libnotify), with a raw D-Bus fallback
              (``gdbus`` / ``dbus-send``) when ``notify-send`` is missing.
* other    -> no-op (returns ``False``).

No third-party dependency is required on any platform. Everything is done with
the stdlib plus the OS-native notification binary, so this module imports
cleanly and safely on servers, CI, and any OS Hermes might run on.
"""

import os
import sys
import shutil
import subprocess

__all__ = ["show_notification"]

_APP_NAME = "Hermes"


def _show_linux(title, body, timeout_ms=3000):
    """Show a libnotify notification on Linux.

    Tries ``notify-send`` first, then falls back to a raw D-Bus call via
    ``gdbus`` / ``dbus-send``. Returns ``True`` if a backend binary was found
    and the call was attempted, ``False`` if no backend is available.
    """
    notify_send = shutil.which("notify-send")
    if notify_send:
        try:
            subprocess.run(
                [notify_send, "-a", _APP_NAME, "-t", str(timeout_ms), title, body],
                check=False,
                capture_output=True,
                text=True,
            )
            return True
        except Exception:
            pass

    # Fallback: D-Bus org.freedesktop.Notifications.Notify. Prefer gdbus (one
    # binary, structured args), then dbus-send (shell-quoted).
    try:
        if shutil.which("gdbus"):
            subprocess.run(
                [
                    "gdbus", "call", "--session",
                    "--dest", "org.freedesktop.Notifications",
                    "--object-path", "/org/freedesktop/Notifications",
                    "--method", "org.freedesktop.Notifications.Notify",
                    _APP_NAME, "0", "", title, body, "[]", "{}", str(timeout_ms),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            return True
        if shutil.which("dbus-send"):
            safe_title = title.replace('"', '\\"')
            safe_body = body.replace('"', '\\"')
            call = (
                'dbus-send --session --dest=org.freedesktop.Notifications '
                '--type=method_call /org/freedesktop/Notifications '
                'org.freedesktop.Notifications.Notify '
                'string:"{app}" int32:0 string:"" '
                'string:"{title}" string:"{body}" array:string:"" '
                'dict:string:{{}} int32:{timeout}'
            ).format(app=_APP_NAME, title=safe_title, body=safe_body,
                     timeout=timeout_ms)
            subprocess.run(["sh", "-c", call], check=False,
                           capture_output=True, text=True)
            return True
    except Exception:
        pass
    return False


def _show_darwin(title, body):
    """Show a macOS Notification Center notification via ``osascript``."""
    # Escape backslashes and double quotes for the AppleScript string literal.
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_body = body.replace("\\", "\\\\").replace('"', '\\"')
    script = 'display notification "{body}" with title "{title}"'.format(
        body=safe_body, title=safe_title)
    try:
        subprocess.run(["osascript", "-e", script], check=False,
                       capture_output=True, text=True)
        return True
    except Exception:
        return False


def _show_windows(title, body, pid=None):
    """Delegate to the Windows-specific WinRT/PowerShell toast backend."""
    try:
        from hermes_cli.windows_notify import show_notification as _win_notify

        return _win_notify(title, body, pid=pid)
    except Exception:
        return False


def show_notification(title, body, pid=None):
    """Show a desktop notification on the current platform.

    Returns ``True`` if a backend was available and the notification was fired
    (best-effort), ``False`` otherwise. Never raises.
    """
    plat = sys.platform
    try:
        if plat == "win32":
            return _show_windows(title, body, pid=pid)
        if plat == "darwin":
            return _show_darwin(title, body)
        if plat.startswith("linux"):
            return _show_linux(title, body)
    except Exception:
        return False
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show a Hermes desktop notification.")
    parser.add_argument("--title", default="Hermes", help="Notification title")
    parser.add_argument("--body", default="", help="Notification body text")
    parser.add_argument("--pid", type=int, default=None, help="Hermes process PID")
    args = parser.parse_args()
    ok = show_notification(args.title, args.body, pid=args.pid)
    print("OK" if ok else "NO_BACKEND")
