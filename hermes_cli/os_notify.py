"""OS-native desktop notification fallback.

This module provides a fire-and-forget OS notification mechanism for terminals that
ignore OSC 9 (the terminal-native notification protocol used by ``terminal_notify.py``).
It is intended as a fallback only — the primary notification path remains OSC 9 because
it integrates better with terminal emulators. On macOS the notifications are attributed
to Script Editor rather than Hermes (Apple removed the sender override), which is why
this module exists solely as a secondary path.

Windows delivery is not implemented — deferred until someone with a Windows host can
validate the mechanism end-to-end (documented follow-up). Previous iterations attempted
PowerShell-driven WinRT toasts, but were removed because:
1. PowerShell folds trailing arguments after ``-Command <script>`` into the parsed
   command string, creating a command-substitution injection surface with model-generated text.
2. The WinRT toast construction bypassed creating a ``Windows.UI.Notifications.ToastNotification``
   object (passing ``XmlDocument`` directly to ``Show()``), which cannot be validated without
   a Windows host.

The public API follows a frozen interface used elsewhere in the CLI:

* ``notifier_argv(kind: str, title: str, body: str) -> list[str] | None`` — returns the
  subprocess argv for the requested OS, or ``None`` for unsupported kinds. ``kind`` must
  be one of ``"darwin"`` or ``"linux"`` (``"win32"`` returns ``None``); the function never reads
  ``sys.platform`` internally.

* ``usable_kind(*, env=None, platform=None, which=None) -> str | None`` — determines
  which notifier to use on the current host. Returns ``None`` when no notifier should
  be used (e.g. under SSH, missing binaries, or unsupported platforms like win32). ``which`` is a
  callable compatible with ``shutil.which`` and is injectable for tests.

* ``notify(title: str, body: str) -> bool`` — spawns the notification process in a
  detached child. Returns ``True`` when a notifier was launched, ``False`` otherwise.
  All exceptions are caught so the clarify prompt is never delayed.

The implementation is deliberately minimal and fail-graceful — any failure to launch
a notification must not break the CLI flow.
"""

import os
import shutil
import subprocess
import sys
from typing import Callable, List, Optional

# Export the public symbols for ``from hermes_cli.os_notify import *``
__all__ = ["notifier_argv", "usable_kind", "notify"]

# ``hermes_cli._subprocess_compat`` provides the correct detach flags for the
# current Python version and platform.  It is part of the shipped CLI helpers.
try:
    from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
except Exception:  # pragma: no cover — fallback if the compat module moves
    # ``start_new_session`` works on POSIX; on Windows we try to detach the process.
    def windows_detach_popen_kwargs() -> dict:
        return {"start_new_session": True}


def notifier_argv(kind: str, title: str, body: str) -> Optional[List[str]]:
    """Return the subprocess argv for an OS notification.

    Parameters
    ----------
    kind: str
        One of ``"darwin"`` or ``"linux"``. The function never reads
        ``sys.platform`` — the caller must provide the correct kind.
        ``"win32"`` is not implemented and returns ``None``.
    title: str
        Notification title.
    body: str
        Notification body text.

    Returns
    -------
    list[str] | None
        The argv ready for ``subprocess.Popen``. ``None`` for unsupported ``kind``.
    """
    if kind == "darwin":
        # AppleScript via osascript.  The title and body are passed as run arguments
        # after ``--`` so they are not interpolated into the script source.  This avoids
        # any quoting or injection surface and matches the verified working shape on
        # macOS.
        return [
            "osascript",
            "-e",
            "on run {t, b}",
            "-e",
            "display notification b with title t",
            "-e",
            "end run",
            "--",
            title,
            body,
        ]
    if kind == "linux":
        # ``notify-send`` respects the freedesktop.org desktop notification spec.
        return ["notify-send", "--app-name=Hermes", title, body]
    return None


def usable_kind(
    *,
    env: Optional[dict] = None,
    platform: Optional[str] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[str]:
    """Determine which notifier to use on this host.

    Parameters
    ----------
    env: dict, optional
        Environment mapping (defaults to ``os.environ``).  ``SSH_CONNECTION``,
        ``SSH_TTY`` or ``SSH_CLIENT`` in this mapping cause an immediate ``None``
        return because notifications on a remote desktop are not intended for the
        local user.
    platform: str, optional
        Value to treat as ``sys.platform`` (defaults to ``sys.platform``).
    which: Callable[[str], Optional[str]], optional
        A function compatible with ``shutil.which`` used to probe for binaries.
        Injectable for tests; defaults to ``shutil.which``.

    Returns
    -------
    str | None
        The ``kind`` to use, or ``None`` when no notifier should be launched.
    """
    if env is None:
        env = os.environ
    # SSH check — notifications should never fire over SSH.
    if any(var in env for var in ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT")):
        return None
    if platform is None:
        platform = sys.platform
    if which is None:
        which = shutil.which
    # Probe for required binaries.
    if platform == "darwin":
        if not which("osascript"):
            return None
        return "darwin"
    if platform == "linux":
        if not which("notify-send"):
            return None
        return "linux"
    return None


_ACTIVE: list = []   # spawned notifier children, reaped on the next spawn


def _reap() -> None:
    """Drop notifier children that have already exited (Popen.poll() reaps them)."""
    still_active = []
    for proc in _ACTIVE:
        try:
            if proc.poll() is None:
                still_active.append(proc)
        except Exception:
            pass
    _ACTIVE[:] = still_active


def notify(title: str, body: str) -> bool:
    """Fire a desktop notification in a detached child process.

    All failures are caught so the clarify prompt is never delayed.  The function
    returns ``True`` when a notifier was successfully launched, ``False`` otherwise.
    """
    _reap()
    try:
        kind = usable_kind()
        if kind is None:
            return False
        argv = notifier_argv(kind, title, body)
        if argv is None:
            return False
        # ``windows_detach_popen_kwargs`` is the repo's cross-platform detach helper
        # (``start_new_session=True`` on POSIX, Win32 creation flags on Windows).
        kwargs = windows_detach_popen_kwargs()
        # ``DEVNULL`` suppresses all output from the notification process.
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        if proc is not None:
            _ACTIVE.append(proc)
        return True
    except Exception:
        return False
