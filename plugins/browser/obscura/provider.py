"""Obscura local browser provider, plugin form.

Subclasses :class:`agent.browser_provider.BrowserProvider`. Unlike the cloud
backends (Browserbase, Browser Use, Firecrawl) this provider runs a *local*
browser: it spawns ``obscura serve`` as a subprocess and hands the agent the
process's CDP endpoint. Obscura (https://github.com/h4ckf0r0day/obscura) is a
Rust headless browser that speaks the Chrome DevTools Protocol with no Chrome
or Node.js dependency, a single ~70 MB binary.

Opt-in only. The registry never auto-selects Obscura; choose it explicitly::

    browser:
      cloud_provider: "obscura"

Env vars::

    OBSCURA_BIN=obscura          # binary path, or a name on PATH (default "obscura")
    OBSCURA_STEALTH=false        # pass --stealth (default false)
    OBSCURA_PORT=                # fixed CDP port (default: an ephemeral free port)
    OBSCURA_STARTUP_TIMEOUT=15   # seconds to wait for the CDP server (default 15)
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, Optional

import requests

from agent.browser_provider import BrowserProvider

logger = logging.getLogger(__name__)

_DEFAULT_BIN = "obscura"
_DEFAULT_STARTUP_TIMEOUT = 15.0


class ObscuraBrowserProvider(BrowserProvider):
    """Local Obscura (https://github.com/h4ckf0r0day/obscura) CDP browser.

    Spawns one ``obscura serve`` process per session and tears it down on
    close. Lives entirely on localhost; no credentials or network calls.
    """

    def __init__(self) -> None:
        # bb_session_id -> the obscura serve process that session owns.
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "obscura"

    @property
    def display_name(self) -> str:
        return "Obscura"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _resolve_binary(self) -> Optional[str]:
        """Resolve the obscura executable, or None if it cannot be found.

        ``shutil.which`` handles a path (absolute or relative) and a bare name
        on PATH alike, and appends the Windows executable suffix. Cheap and
        offline: it never spawns anything.
        """
        return shutil.which(os.environ.get("OBSCURA_BIN", _DEFAULT_BIN))

    def is_available(self) -> bool:
        return self._resolve_binary() is not None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, task_id: str) -> Dict[str, object]:
        binary = self._resolve_binary()
        if binary is None:
            raise ValueError(
                "Obscura binary not found. Install obscura "
                "(https://github.com/h4ckf0r0day/obscura), put it on PATH, or set "
                "OBSCURA_BIN to its path."
            )

        stealth = os.environ.get("OBSCURA_STEALTH", "false").lower() == "true"
        port = _resolve_port(os.environ.get("OBSCURA_PORT"))
        timeout = _resolve_timeout(os.environ.get("OBSCURA_STARTUP_TIMEOUT"))

        cmd = [binary, "serve", "--port", str(port)]
        if stealth:
            cmd.append("--stealth")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to launch obscura: {exc}") from exc

        cdp_url = _await_cdp(port, proc, timeout)
        if cdp_url is None:
            _terminate(proc)
            raise RuntimeError(
                f"Obscura CDP server did not become ready on port {port} within "
                f"{timeout:g}s."
            )

        session_id = uuid.uuid4().hex
        with self._lock:
            self._procs[session_id] = proc

        session_name = f"hermes_{task_id}_{session_id[:8]}"
        features = {"stealth": stealth, "local": True}
        logger.info(
            "Started Obscura session %s (pid %s, port %s, stealth=%s)",
            session_name,
            proc.pid,
            port,
            stealth,
        )
        return {
            "session_name": session_name,
            "bb_session_id": session_id,
            "cdp_url": cdp_url,
            "features": features,
        }

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            proc = self._procs.pop(session_id, None)
        if proc is None:
            logger.debug("No Obscura process tracked for session %s", session_id)
            return False
        try:
            _terminate(proc)
            logger.debug("Closed Obscura session %s", session_id)
            return True
        except Exception as exc:  # never raise out of cleanup
            logger.error("Failed to close Obscura session %s: %s", session_id, exc)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        with self._lock:
            proc = self._procs.pop(session_id, None)
        if proc is None:
            return
        try:
            _terminate(proc)
        except Exception as exc:
            logger.debug(
                "Emergency cleanup failed for Obscura session %s: %s", session_id, exc
            )

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Obscura",
            "badge": "local",
            "tag": "Local Rust headless browser over CDP (no Chrome or Node needed)",
            "env_vars": [
                {
                    "key": "OBSCURA_BIN",
                    "prompt": "Path to the obscura binary (optional if 'obscura' is on PATH)",
                    "url": "https://github.com/h4ckf0r0day/obscura",
                },
            ],
            "post_setup": "agent_browser",
        }


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _resolve_port(configured: Optional[str]) -> int:
    """Use OBSCURA_PORT when valid, else ask the OS for a free ephemeral port."""
    if configured:
        try:
            value = int(configured)
            if 1 <= value <= 65535:
                return value
        except ValueError:
            pass
        logger.warning(
            "Invalid OBSCURA_PORT %r; picking a free port instead", configured
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _resolve_timeout(configured: Optional[str]) -> float:
    if configured:
        try:
            value = float(configured)
            if value > 0:
                return value
        except ValueError:
            pass
        logger.warning(
            "Invalid OBSCURA_STARTUP_TIMEOUT %r; using %.0fs",
            configured,
            _DEFAULT_STARTUP_TIMEOUT,
        )
    return _DEFAULT_STARTUP_TIMEOUT


def _await_cdp(port: int, proc: subprocess.Popen, timeout: float) -> Optional[str]:
    """Poll ``/json/version`` until obscura is ready.

    Returns the ``webSocketDebuggerUrl`` (the CDP endpoint the agent connects
    to), or None if the process dies or the deadline passes.
    """
    deadline = time.monotonic() + timeout
    version_url = f"http://127.0.0.1:{port}/json/version"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None  # process exited before serving
        try:
            resp = requests.get(version_url, timeout=1.0)
            if resp.ok:
                ws_url = resp.json().get("webSocketDebuggerUrl")
                if ws_url:
                    return ws_url
        except requests.RequestException:
            pass
        time.sleep(0.1)
    return None


def _terminate(proc: subprocess.Popen) -> None:
    """Terminate a process, escalating to kill after a short grace period."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
