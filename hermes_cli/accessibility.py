"""Reduced-motion resolution for the CLI and TUI (port of openai/codex#46040 / #46832).

Screen readers re-announce every repaint, so a braille spinner or a rotating verb turns into a
stream of noise (#26689). ``display.reduced_motion`` is the master switch; when the user has not
set it, a bounded one-shot probe for an active screen reader decides. Explicit settings always win:

1. ``HERMES_REDUCED_MOTION=1|0`` (one launch, no config edit);
2. ``display.reduced_motion: true|false`` in config.yaml;
3. an active screen reader (``HERMES_SCREEN_READER=1|0`` overrides the OS probe).

The probe is bounded (``_PROBE_TIMEOUT_S``) and never raises: an unanswered or unavailable probe
means "no reader", never a stalled startup.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 0.45
_SPI_GETSCREENREADER = 0x0046


def _env_bool(name: str) -> bool | None:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _run_probe(argv: list[str]) -> str:
    """Stdout of a short-lived probe command, ``""`` on any failure or timeout."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no user input
            argv, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
            stdin=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _windows_screen_reader() -> bool:
    try:
        import ctypes

        flag = ctypes.c_int(0)
        ok = ctypes.windll.user32.SystemParametersInfoW(  # type: ignore[attr-defined]
            _SPI_GETSCREENREADER, 0, ctypes.byref(flag), 0)
        return bool(ok) and bool(flag.value)
    except Exception:
        return False


def _macos_screen_reader() -> bool:
    # VoiceOver mirrors its on/off state into this preference key.
    out = _run_probe(["defaults", "read", "com.apple.universalaccess", "voiceOverOnOffKey"])
    return out.strip() == "1"


def _linux_screen_reader() -> bool:
    # AT-SPI publishes the reader flag on the session bus; without a bus there is no reader.
    if not (os.environ.get("DBUS_SESSION_BUS_ADDRESS") or os.environ.get("XDG_RUNTIME_DIR")):
        return False
    out = _run_probe(["busctl", "--user", f"--timeout={_PROBE_TIMEOUT_S}s", "get-property",
                      "org.a11y.Bus", "/org/a11y/bus", "org.a11y.Status", "ScreenReaderEnabled"])
    if out:
        return out.strip().endswith("true")
    out = _run_probe(["gdbus", "call", "--session", "--dest", "org.a11y.Bus",
                      "--object-path", "/org/a11y/bus", "--method",
                      "org.freedesktop.DBus.Properties.Get", "org.a11y.Status", "ScreenReaderEnabled"])
    return "true" in out


@lru_cache(maxsize=1)
def screen_reader_active() -> bool:
    """True when the host reports an active screen reader (cached per process; never raises)."""
    forced = _env_bool("HERMES_SCREEN_READER")
    if forced is not None:
        return forced
    try:
        if sys.platform == "win32":
            return _windows_screen_reader()
        if sys.platform == "darwin":
            return _macos_screen_reader()
        if sys.platform.startswith("linux"):
            return _linux_screen_reader()
    except Exception:
        logger.debug("screen reader probe failed", exc_info=True)
    return False


def reduced_motion_enabled(display_config: dict | None = None) -> bool:
    """Resolve the reduced-motion switch: env override → ``display.reduced_motion`` → screen reader.

    Without ``display_config`` the answer is cached per process (spinner repaints ask ten times a
    second); the switch is a launch-time decision, like the TUI's.
    """
    if display_config is None:
        return _resolve_from_process()
    return _resolve(display_config)


@lru_cache(maxsize=1)
def _resolve_from_process() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        display_config = (load_config_readonly() or {}).get("display") or {}
    except Exception:
        display_config = {}
    return _resolve(display_config)


def _resolve(display_config: dict) -> bool:
    forced = _env_bool("HERMES_REDUCED_MOTION")
    if forced is not None:
        return forced
    configured = display_config.get("reduced_motion") if isinstance(display_config, dict) else None
    if isinstance(configured, bool):
        return configured
    if screen_reader_active():
        logger.info("Screen reader detected: animations off (display.reduced_motion: false overrides).")
        return True
    return False


def reset_caches() -> None:
    """Forget the per-process probe results (tests, or a config edit the caller wants honoured)."""
    screen_reader_active.cache_clear()
    _resolve_from_process.cache_clear()
