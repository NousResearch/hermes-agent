#!/usr/bin/env python3
"""Launch (or detect) the operator's local Chrome/Chromium over the DevTools Protocol (CDP).

Phase-2 work -- sending opt-out/CCPA email through the operator's logged-in webmail, and driving
session-bound multi-step opt-out gates (e.g. PeopleConnect guided-mode) -- must run in the
operator's OWN browser: real fingerprint, residential IP, and the operator's signed-in sessions.
A headless cloud browser (Browserbase) is the wrong tool there (it has no webmail session and is
itself anti-bot-gated on those exact flows). This module launches the operator's real Chrome with
remote debugging on a DEDICATED profile so Hermes's browser tools can attach at 127.0.0.1:<port>.

Stdlib only; cross-platform (macOS / Linux / Windows). Nothing here touches a password or PII.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import paths

DEFAULT_PORT = 9222

# Chromium-family binaries we know how to drive, in preference order. Names first (works on any OS
# where one is on PATH), then per-OS absolute-path fallbacks below.
_PATH_NAMES = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "brave-browser", "microsoft-edge", "microsoft-edge-stable", "chrome",
)


def default_profile() -> Path:
    """Dedicated debug profile dir, NOT the operator's Default Chrome profile.

    Chrome refuses remote-debugging on a profile that is already open in another Chrome instance,
    so we isolate the debug session in its own user-data-dir under HERMES_HOME.
    """
    return paths.hermes_home() / "chrome-debug"


def _mac_candidates() -> list[str]:
    return [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    ]


def _windows_candidates() -> list[str]:
    bases = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    rels = [
        r"Google\Chrome\Application\chrome.exe",
        r"Chromium\Application\chrome.exe",
        r"BraveSoftware\Brave-Browser\Application\brave.exe",
        r"Microsoft\Edge\Application\msedge.exe",
    ]
    out: list[str] = []
    for base in bases:
        if not base:
            continue
        for rel in rels:
            out.append(str(Path(base) / rel))
    return out


def find_browser(override: str | None = None) -> str | None:
    """Return the first usable Chromium-family browser path/command, or None.

    `override` (an explicit path, or a command on PATH) wins when it resolves.
    """
    if override:
        if Path(override).exists():
            return override
        return shutil.which(override)  # may be None -> caller reports "not found"
    for name in _PATH_NAMES:
        found = shutil.which(name)
        if found:
            return found
    if sys.platform == "darwin":
        candidates = _mac_candidates()
    elif sys.platform == "win32":
        candidates = _windows_candidates()
    else:
        candidates = []
    for cand in candidates:
        if Path(cand).exists():
            return cand
    return None


def launch_command(browser: str, port: int = DEFAULT_PORT, profile: Path | None = None) -> list[str]:
    """The exact argv used to start the debug browser (also handy for `--print`)."""
    profile = profile or default_profile()
    return [
        browser,
        f"--remote-debugging-port={int(port)}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def _http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "unbroker-cdp/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost only)
        return resp.read()


def endpoint_status(port: int = DEFAULT_PORT, host: str = "127.0.0.1",
                    timeout: float = 1.0) -> dict | None:
    """Return the CDP `/json/version` dict if a debuggable browser is live at host:port, else None.

    (Chrome restricts this endpoint to localhost/IP Host headers, so we always hit 127.0.0.1.)
    """
    url = f"http://{host}:{int(port)}/json/version"
    try:
        raw = _http_get(url, timeout)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError):
        return None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, AttributeError):
        return None
    return data if isinstance(data, dict) else None


def launch(browser: str, port: int = DEFAULT_PORT, profile: Path | None = None) -> int:
    """Start the browser detached with remote debugging; return the child PID.

    Detach so the browser outlives this short-lived CLI call. POSIX uses start_new_session (which
    avoids referencing os.setsid, so there is no Windows import-time footgun); Windows uses
    DETACHED_PROCESS + a new process group.
    """
    profile = profile or default_profile()
    profile.mkdir(parents=True, exist_ok=True)
    cmd = launch_command(browser, port, profile)
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # windows-footgun: ok
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    return proc.pid
