"""Hermes Windows notification helper.

Primary path: ``winrt`` Python package (Python <=3.12).
Fallback:  PowerShell + WinRT COM interop (works on any Windows 10/11,
          zero pip deps).  Automatically selected at import time.

Behavior: show a Windows toast with Hermes AUMID; clicking or auto-dismiss
brings the correct Hermes window to front by matching PID.
"""

import os
import sys
import subprocess
import argparse

# winreg is Windows-only. Guard the import so this module can be imported (and
# smoke-tested) on any platform without crashing; the registry calls below are
# only ever reached on Windows at toast-show time.
try:
    import winreg
except ImportError:
    winreg = None

# --- Debug logging (this script may run hidden; log to a file for diagnosis) --

_NOTIFY_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes_notify.log")


def _log(msg):
    try:
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(_NOTIFY_LOG, "a", encoding="utf-8") as fh:
            fh.write("[{}] {}\n".format(ts, msg))
            fh.flush()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Import strategy: try winrt first, fall back to power-shell bridge
# ---------------------------------------------------------------------------
_HAS_WINRT = False
try:
    from winrt.windows.ui.notifications import (
        ToastNotificationManager,
        ToastNotification,
        ToastActivatedEventArgs,
        ToastDismissedEventArgs,
        ToastFailedEventArgs,
    )
    from winrt.windows.data.xml.dom import XmlDocument
    from winrt.windows.foundation import IPropertyValue

    _HAS_WINRT = True
except ImportError:
    pass  # will use PowerShell fallback


HERMES_AUMID = "Hermes"
HERMES_TOAST_TAG = "hermes_notify"
HERMES_TOAST_GROUP = "default"


# ---------------------------------------------------------------------------
# Registry helpers (shared by both paths)
# ---------------------------------------------------------------------------

def _ensure_aumid():
    """Register Hermes AUMID under HKCU so the toast shows 'Hermes' as app name."""
    if winreg is None:
        return
    try:
        key_path = rf"Software\Classes\AppUserModelId\{HERMES_AUMID}"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path):
                return
        except FileNotFoundError:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Hermes")
    except Exception:
        pass


def _ensure_protocol_registered():
    """Register the ``hermes://`` URL protocol (HKCU) so clicking a toast
    launches ``windows_focus.py`` (this package's focus helper) with the activation URL.
    """
    if winreg is None:
        return
    try:
        focus_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windows_focus.py")
        if not os.path.exists(focus_script):
            return
        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        launcher = pythonw if os.path.exists(pythonw) else sys.executable
        command = '"{launcher}" "{script}" "%1"'.format(
            launcher=launcher, script=focus_script
        )

        base = r"Software\Classes\hermes"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base) as key:
            winreg.SetValueEx(key, None, 0, winreg.REG_SZ, "URL:Hermes Protocol")
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\shell\open\command") as key:
            winreg.SetValueEx(key, None, 0, winreg.REG_SZ, command)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Focus helper (shared)
# ---------------------------------------------------------------------------

def bring_hermes_to_front(pid=None):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windows_focus.py")
    if not os.path.exists(script):
        print("windows_focus.py not found at: {}".format(script), file=sys.stderr)
        return False
    try:
        cmd = [sys.executable, script]
        if pid is not None:
            cmd.extend(["--pid", str(pid)])
        subprocess.run(cmd, check=False)
        return True
    except Exception as exc:
        print("Failed to run hermes_focus.py: {}".format(exc), file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Path A – winrt (native Python)
# ---------------------------------------------------------------------------

def _activated_args(event):
    e = ToastActivatedEventArgs._from(event)
    user_input = dict([
        (name, IPropertyValue._from(e.user_input[name]).get_string())
        for name in e.user_input
    ])
    return {"arguments": e.arguments, "user_input": user_input}


def _escape_xml(s):
    """Escape XML-special characters so embedded text (commands, paths, etc.)
    that contains &, <, >, \" or ' cannot corrupt the toast XML and silently
    suppress the notification."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


async def _show_via_winrt_async(title, body, duration_seconds=3, pid=None):
    import asyncio

    _ensure_aumid()
    _ensure_protocol_registered()

    launch = f"hermes://focus?pid={pid}" if pid is not None else "hermes://focus"
    xml = (
        f"<toast activationType='protocol' launch='{launch}' duration='short'>"
        "<visual><binding template='ToastGeneric'>"
        f"<text>{_escape_xml(title)}</text>"
        f"<text>{_escape_xml(body)}</text>"
        "</binding></visual></toast>"
    )

    document = XmlDocument()
    document.load_xml(xml)

    notifier = ToastNotificationManager.create_toast_notifier_with_id(HERMES_AUMID)
    notification = ToastNotification(document)
    notification.tag = HERMES_TOAST_TAG
    notification.group = HERMES_TOAST_GROUP
    notifier.show(notification)

    loop = asyncio.get_running_loop()
    activated_future = loop.create_future()
    dismissed_future = loop.create_future()
    failed_future = loop.create_future()

    def on_activated(*args):
        # Only focus the window when the user actually clicks the toast —
        # never on auto-dismiss/timeout, to avoid yanking focus away from
        # whatever the user is doing.
        _log("winrt on_activated -> bring_hermes_to_front(pid={})".format(pid))
        bring_hermes_to_front(pid=pid)
        loop.call_soon_threadsafe(activated_future.set_result, _activated_args(args[0]))

    def on_dismissed(_, event_args):
        loop.call_soon_threadsafe(
            dismissed_future.set_result,
            ToastDismissedEventArgs._from(event_args).reason,
        )

    def on_failed(_, event_args):
        loop.call_soon_threadsafe(
            failed_future.set_result,
            ToastFailedEventArgs._from(event_args).error_code,
        )

    notification.add_activated(on_activated)
    notification.add_dismissed(on_dismissed)
    notification.add_failed(on_failed)

    timer = loop.create_task(asyncio.sleep(max(1, duration_seconds)))

    done, pending = await asyncio.wait(
        [activated_future, dismissed_future, failed_future, timer],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for p in pending:
        p.cancel()

    try:
        ToastNotificationManager.history.remove_grouped_tag_with_id(
            HERMES_TOAST_TAG, HERMES_TOAST_GROUP, HERMES_AUMID
        )
    except Exception:
        pass


def _show_via_winrt(title, body, duration_seconds=3, pid=None):
    import asyncio

    try:
        asyncio.run(_show_via_winrt_async(title, body, duration_seconds, pid))
    except RuntimeError:
        # Event loop already running — fire-and-forget
        _ensure_aumid()
        _ensure_protocol_registered()
        launch = f"hermes://focus?pid={pid}" if pid is not None else "hermes://focus"
        xml = (
            f"<toast activationType='protocol' launch='{launch}' duration='short'>"
            "<visual><binding template='ToastGeneric'>"
            f"<text>{title}</text>"
            f"<text>{body}</text>"
            "</binding></visual></toast>"
        )
        document = XmlDocument()
        document.load_xml(xml)
        notifier = ToastNotificationManager.create_toast_notifier_with_id(HERMES_AUMID)
        notification = ToastNotification(document)
        notification.tag = HERMES_TOAST_TAG
        notification.group = HERMES_TOAST_GROUP
        notifier.show(notification)


# ---------------------------------------------------------------------------
# Path B – PowerShell fallback (no pip deps needed)
# ---------------------------------------------------------------------------

_POWERSHELL_TOAST_SCRIPT = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'

try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
} catch {
    Write-Host 'WINRT_UNAVAILABLE'
    exit 2
}

# Escape XML-special characters so command text (which may contain &, <, >, ")
# does not corrupt the toast XML and silently suppress the notification.
function Escape-Xml($s) {
    if ($null -eq $s) { return '' }
    return ([Security.SecurityElement]::Escape($s))
}

$title = Escape-Xml $args[0]
$body  = Escape-Xml $args[1]
$pidArg = $args[2]

# Register the Hermes AUMID (DisplayName) so the toast renders its real
# title/body instead of a generic placeholder. A bare AUMID without at least a
# DisplayName makes Windows collapse it into a "1 new notification" stub.
$aumid = 'Hermes'
$regPath = 'HKCU:\Software\Classes\AppUserModelId\' + $aumid
if (-not (Test-Path $regPath)) {
    New-Item -Path $regPath -Force | Out-Null
    Set-ItemProperty -Path $regPath -Name 'DisplayName' -Value 'Hermes' -Force
}

$launch = if ($pidArg) { "hermes://focus?pid=$pidArg" } else { 'hermes://focus' }
$xml = "<toast activationType='protocol' launch='" + $launch + "' duration='short'><visual><binding template='ToastGeneric'><text>" + $title + "</text><text>" + $body + "</text></binding></visual></toast>"

$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml($xml)

$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($aumid)
$toast = New-Object Windows.UI.Notifications.ToastNotification $doc
$notifier.Show($toast)

Write-Host 'OK'
exit 0
"""


def _show_via_powershell(title, body, duration_seconds=3, pid=None):
    """Fire-and-forget toast via PowerShell WinRT interop.

    The script is written to a temp .ps1 and launched with ``-File`` so the
    title/body arguments are passed as literal strings. (Passing them on the
    ``-Command`` line makes PowerShell treat ``&`` and other chars in the body
    as operators, which breaks e.g. approval command text.)
    """
    pid_str = str(pid) if pid is not None else ""
    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".ps1", delete=False, encoding="utf-8-sig"
        ) as fh:
            fh.write(_POWERSHELL_TOAST_SCRIPT)
            tmp_path = fh.name
        # Run the PowerShell host fully hidden so firing a toast never flashes a
        # visible console/terminal window in front of the user.
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-File",
                tmp_path,
                title,
                body,
                pid_str,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(
                "PowerShell toast failed (rc={}): {}".format(result.returncode, err),
                file=sys.stderr,
            )
            return False
        # NOTE: Do NOT auto-focus here. Focus only happens when the user
        # *clicks* the toast, which routes through the hermes:// protocol ->
        # hermes_focus.py. Auto-focusing on show/timeout would yank the
        # terminal to the foreground even when the user just let it time out.
        return True
    except FileNotFoundError:
        print("powershell.exe not found — cannot show toast", file=sys.stderr)
        return False
    except Exception as exc:
        print("PowerShell toast error: {}".format(exc), file=sys.stderr)
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def show_notification(title, body, duration_seconds=3, pid=None):
    _log("show_notification path={} pid={} title={!r}".format(
        "winrt" if _HAS_WINRT else "powershell", pid, title))
    if _HAS_WINRT:
        return _show_via_winrt(title, body, duration_seconds, pid)
    else:
        return _show_via_powershell(title, body, duration_seconds, pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show a Hermes Windows toast notification.")
    parser.add_argument("--title", default="Hermes", help="Notification title")
    parser.add_argument("--body", default="点击回到 Hermes", help="Notification body text")
    parser.add_argument("--duration", type=int, default=3, help="Auto-dismiss after N seconds")
    parser.add_argument("--pid", type=int, default=None, help="Hermes process PID to focus")
    args = parser.parse_args()

    show_notification(args.title, args.body, args.duration, args.pid)
