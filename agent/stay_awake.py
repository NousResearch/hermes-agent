"""Stay-awake inhibitor — prevents the OS from sleeping while Hermes is working.

Provides a context manager that, when enabled, prevents system/idle sleep
across macOS, Linux, and Windows.  Display sleep is intentionally NOT
prevented — the goal is to keep API calls and tool execution alive, not
keep the screen lit.

Usage::

    with StayAwake(enabled=True):
        # Agent loop runs here — machine won't sleep
        ...
    # Machine can sleep again

Config key: ``agent.stay_awake`` (bool, default False).
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class StayAwake:
    """Prevent OS sleep/idle for the duration of the context block.

    Safe to use when the inhibitor isn't available (containers, headless
    servers) — logs a warning and degrades to a no-op.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._process: Optional[subprocess.Popen] = None
        self._original_state = None  # Windows: previous ES_* flags

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "StayAwake":
        if not self._enabled:
            return self
        try:
            system = platform.system()
            if system == "Darwin":
                self._start_macos()
            elif system == "Linux":
                self._start_linux()
            elif system == "Windows":
                self._start_windows()
            if self._process is not None or self._original_state is not None:
                logger.info("Stay-awake inhibitor started (os=%s)", system)
        except Exception as exc:
            logger.warning("Failed to start stay-awake inhibitor: %s", exc)
        return self

    def __exit__(self, *args: object) -> None:
        if not self._enabled:
            return
        try:
            if self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None
                logger.info("Stay-awake inhibitor stopped")
            if self._original_state is not None:
                self._stop_windows()
        except Exception as exc:
            logger.warning("Failed to stop stay-awake inhibitor: %s", exc)

    # ── OS-specific backends ─────────────────────────────────────────────

    def _start_macos(self) -> None:
        """caffeinate -i: prevent idle sleep only (display and disk free)."""
        self._process = subprocess.Popen(
            ["caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _start_linux(self) -> None:
        """systemd-inhibit: block sleep + idle while the agent runs.

        Falls back gracefully if systemd is not available (containers,
        WSL1, non-systemd distros) — the Popen will raise FileNotFoundError
        which is caught by __enter__.
        """
        self._process = subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=sleep:idle",
                "--why=Hermes agent working",
                "--who=hermes",
                "sleep",
                "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _start_windows(self) -> None:
        """SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED).

        Stores the previous state so __exit__ can restore it.
        """
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        self._original_state = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )

    def _stop_windows(self) -> None:
        """Restore the previous thread execution state (clears our flags)."""
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        self._original_state = None


# ── Process-wide refcounted scope ────────────────────────────────────────────
# One OS inhibitor per process, shared by concurrent turns (gateway sessions,
# subagent children): the first active turn starts it, the last one stops it.
# Refcounting matters because Windows SetThreadExecutionState is not nestable —
# a second StayAwake exiting would clear the first one's flags.

_lock = threading.Lock()
_count = 0
_inhibitor: Optional[StayAwake] = None


def _config_enabled() -> bool:
    """``agent.stay_awake`` from config.yaml (default False)."""
    try:
        from hermes_cli.config import load_config_readonly

        return bool((load_config_readonly().get("agent", {}) or {}).get("stay_awake", False))
    except Exception:
        return False


@contextmanager
def turn_scope(enabled: Optional[bool] = None):
    """Hold the shared stay-awake inhibitor for the duration of one agent turn.

    ``enabled=None`` reads ``agent.stay_awake`` from config; the whole scope is a
    no-op when disabled. Re-entrant across threads via refcount.
    """
    global _count, _inhibitor
    active = _config_enabled() if enabled is None else enabled
    if not active:
        yield
        return
    with _lock:
        _count += 1
        if _count == 1:
            _inhibitor = StayAwake(enabled=True)
            _inhibitor.__enter__()
    try:
        yield
    finally:
        with _lock:
            _count -= 1
            if _count == 0 and _inhibitor is not None:
                _inhibitor.__exit__(None, None, None)
                _inhibitor = None
