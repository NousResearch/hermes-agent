"""Attach Hermes browser sessions to running Electron/Chromium desktop apps.

Driving a desktop Electron app (Obsidian, Slack, VS Code, Discord, ...) over
CDP is far more reliable than OS-level input synthesis: Chromium drops
synthetic pointer events into occluded, unfocused renderers, while a CDP
attach gives exact DOM-level control with zero focus steal.

This module owns the three pieces that make that a one-command flow:

* **Electron discovery** — scan running processes for Electron *main*
  processes (``resources/app.asar`` next to the executable, no ``--type=``
  child marker) and read any ``--remote-debugging-port`` they were launched
  with.
* **The named-session CDP registry** — ``$HERMES_HOME/browser-sessions.json``
  maps a browser_exec session name to the app's CDP endpoint, so ONE named
  session drives the app while the default session keeps browsing the web.
* **The relaunch flow** — for a detected app that does not expose a CDP
  port, offer to quit and relaunch it with ``--remote-debugging-port``.

Consent model: CDP into a live desktop app exposes everything the app can
see (Slack DMs, vault contents, editor buffers) — the same power class as
cua-driver's gated ``existing_profile`` attach. The agent therefore never
opens a debug port itself: ``hermes browser attach`` is user-invoked, and
the relaunch confirmation is the consent moment.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import time
from typing import Any, Dict, List, Optional

from hermes_cli.browser_connect import discover_local_cdp_url, find_free_debug_port
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "browser-sessions.json"

_DEBUG_PORT_RE = re.compile(r"--remote-debugging-port=(\d+)")

# Chromium child processes carry --type=renderer/gpu-process/utility/...;
# the main process never does. This is the cheapest main-vs-child split.
_CHILD_TYPE_RE = re.compile(r"--type=\w")


# ── Electron detection ─────────────────────────────────────────────


def electron_resource_roots(exe_path: str) -> List[str]:
    """Return the candidate ``resources/`` directories for an executable.

    Electron ships the packaged app as ``resources/app.asar`` (or an
    unpacked ``resources/app/`` tree) next to the binary on Windows/Linux,
    and under ``Contents/Resources/`` on macOS (the binary lives in
    ``Contents/MacOS/``).
    """
    exe_dir = os.path.dirname(exe_path)
    roots = [os.path.join(exe_dir, "resources")]
    if os.path.basename(exe_dir) == "MacOS":
        roots.append(os.path.join(os.path.dirname(exe_dir), "Resources"))
    return roots


def is_electron_executable(exe_path: str) -> bool:
    """True when ``exe_path`` looks like a packaged Electron app binary.

    Signature: ``app.asar`` (packaged) or ``app/package.json`` (unpacked)
    under the executable's resources root. Real browsers (Chrome, Brave,
    Edge) ship neither, so they are excluded automatically.
    """
    if not exe_path:
        return False
    for root in electron_resource_roots(exe_path):
        if os.path.isfile(os.path.join(root, "app.asar")):
            return True
        if os.path.isfile(os.path.join(root, "app", "package.json")):
            return True
    return False


def debug_port_from_cmdline(cmdline: List[str]) -> Optional[int]:
    """Extract ``--remote-debugging-port=N`` from a process command line."""
    for arg in cmdline or []:
        m = _DEBUG_PORT_RE.search(str(arg))
        if m:
            try:
                port = int(m.group(1))
            except ValueError:
                continue
            # Port 0 means "OS-assigned" and is not probeable from here.
            if port > 0:
                return port
    return None


def is_electron_child(cmdline: List[str]) -> bool:
    """True for renderer/gpu/utility children (``--type=...`` marker)."""
    return any(_CHILD_TYPE_RE.search(str(arg)) for arg in cmdline or [])


def app_display_name(exe_path: str, fallback: str) -> str:
    """Human name for the app: the ``.app`` bundle on macOS, else the binary."""
    bundle = _bundle_path(exe_path)
    return os.path.basename(bundle)[: -len(".app")] if bundle else fallback


def session_slug(name: str) -> str:
    """Registry/session-safe slug (matches browser_exec's session grammar).

    browser_exec's ``_SESSION_RE`` requires an alphanumeric FIRST char, so
    strip leading/trailing underscores as well as dashes.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-_").lower()
    return slug[:64] or "app"


def scan_electron_apps() -> List[Dict[str, Any]]:
    """Enumerate running Electron *main* processes.

    Returns entries of shape::

        {"pid", "name", "exe", "debug_port" (int|None), "cdp_url" (str|None)}

    ``cdp_url`` is set only when the advertised port answered a CDP
    discovery probe. Best-effort by design: unreadable processes (access
    denied, zombies) are skipped silently.
    """
    import psutil

    found: List[Dict[str, Any]] = []
    seen_exes: set = set()
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            info = proc.info
            exe = info.get("exe") or ""
            cmdline = info.get("cmdline") or []
            if not exe or is_electron_child(cmdline):
                continue
            if not is_electron_executable(exe):
                continue
            # One entry per app binary: the first (lowest-pid) main wins.
            if exe in seen_exes:
                continue
            seen_exes.add(exe)
            port = debug_port_from_cmdline(cmdline)
            cdp_url: Optional[str] = None
            if port:
                # Dual-stack probe: an IPv4-loopback squatter can push the
                # relaunched app's debug listener onto [::1] only.
                cdp_url = discover_local_cdp_url(port, timeout=0.5)
            found.append(
                {
                    "pid": info["pid"],
                    "name": app_display_name(exe, info.get("name") or "app"),
                    "exe": exe,
                    "debug_port": port,
                    "cdp_url": cdp_url,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:  # pragma: no cover — defensive per-proc guard
            logger.debug("electron scan: skipping process: %s", e)
            continue
    return sorted(found, key=lambda a: a["name"].lower())


# ── Named-session CDP registry ─────────────────────────────────────


def registry_path() -> str:
    return str(get_hermes_home() / _REGISTRY_FILENAME)


def load_registry() -> Dict[str, Dict[str, Any]]:
    """Return ``{session_name: {cdp_url, app, attached_at}}`` (never raises)."""
    try:
        with open(registry_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        sessions = data.get("sessions")
        if isinstance(sessions, dict):
            return {
                str(k): v
                for k, v in sessions.items()
                if isinstance(v, dict) and v.get("cdp_url")
            }
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("browser session registry unreadable (%s): %s", registry_path(), e)
    return {}


def _write_registry(sessions: Dict[str, Dict[str, Any]]) -> None:
    from utils import atomic_json_write

    atomic_json_write(registry_path(), {"sessions": sessions})


def save_session_endpoint(name: str, cdp_url: str, app: str) -> None:
    sessions = load_registry()
    sessions[name] = {
        "cdp_url": cdp_url,
        "app": app,
        "attached_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_registry(sessions)


def remove_session_endpoint(name: str) -> bool:
    sessions = load_registry()
    if name not in sessions:
        return False
    del sessions[name]
    _write_registry(sessions)
    return True


def resolve_session_endpoint(name: str) -> Optional[str]:
    """CDP URL registered for a browser_exec session name, or None."""
    entry = load_registry().get(name)
    if not entry:
        return None
    url = str(entry.get("cdp_url") or "").strip()
    return url or None


# ── Relaunch with a debug port ─────────────────────────────────────


def _exe_of(proc) -> Optional[str]:
    """Best-effort executable path of a psutil process (None if denied)."""
    try:
        return proc.exe()
    except Exception:
        return None


def _terminate_app(pid: int, timeout: float = 10.0) -> bool:
    """Gracefully stop an app's main process (children follow). True on exit."""
    import psutil

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    if platform.system() == "Darwin":
        # Ask the app to quit properly first — Electron apps persist state
        # (open vaults, drafts) on graceful AppleEvent quit; SIGTERM alone
        # can skip their before-quit handlers.
        import subprocess

        exe_path = _exe_of(proc)
        app_name = app_display_name(exe_path, proc.name()) if exe_path else proc.name()
        quit_sent = False
        try:
            quit_sent = (
                subprocess.run(
                    ["osascript", "-e", f'quit app "{app_name}"'],
                    capture_output=True,
                    timeout=5,
                ).returncode
                == 0
            )
        except Exception as e:
            logger.debug("osascript quit failed (falling back to terminate): %s", e)
        if quit_sent:
            # Give the AppleEvent quit a grace window before SIGTERM —
            # terminating immediately defeats the graceful quit and can
            # skip the app's before-quit handlers (the point of osascript).
            try:
                proc.wait(timeout=5)
                return True
            except psutil.NoSuchProcess:
                return True
            except psutil.TimeoutExpired:
                pass
    try:
        if proc.is_running():
            proc.terminate()
        proc.wait(timeout=timeout)
        return True
    except psutil.NoSuchProcess:
        return True
    except psutil.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=3)
            return True
        except Exception:
            return False
    except Exception as e:
        logger.debug("terminate failed for pid %d: %s", pid, e)
        return not psutil.pid_exists(pid)


def _spawn_with_debug_port(exe: str, port: int) -> None:
    """Relaunch the app binary with the debug flag, detached from us."""
    import subprocess

    from hermes_cli.browser_connect import _detach_kwargs

    system = platform.system()
    if system == "Darwin":
        bundle = _bundle_path(exe)
        if bundle:
            subprocess.Popen(
                ["open", "-a", bundle, "--args", f"--remote-debugging-port={port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
    subprocess.Popen(
        [exe, f"--remote-debugging-port={port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_detach_kwargs(system),
    )


def _bundle_path(exe: str) -> Optional[str]:
    """``.../Foo.app`` prefix of a macOS bundle executable path, or None."""
    parts = exe.split(os.sep)
    for i, part in enumerate(parts):
        if part.endswith(".app"):
            return os.sep.join(parts[: i + 1])
    return None


def relaunch_with_debug_port(
    app: Dict[str, Any], port: Optional[int] = None, wait_s: float = 25.0
) -> Optional[str]:
    """Quit ``app`` and relaunch it exposing CDP. Returns the live URL or None.

    Chromium's single-instance lock means the running instance must fully
    exit before the relaunch, or the new invocation just forwards to the old
    process and exits without opening the port.
    """
    port = port or find_free_debug_port()
    if not _terminate_app(app["pid"]):
        logger.warning("could not stop %s (pid %d)", app["name"], app["pid"])
        return None
    _spawn_with_debug_port(app["exe"], port)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        # Dual-stack: the relaunched app may bind [::1] only (see
        # discover_local_cdp_url) — probe both loopbacks each pass.
        url = discover_local_cdp_url(port, timeout=0.5)
        if url:
            return url
        time.sleep(0.3)
    return None
