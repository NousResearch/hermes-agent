"""Thread-owned host sleep prevention for long-running Hermes surfaces.

User-facing configuration lives under ``power.prevent_sleep`` in config.yaml.
``system`` prevents idle system sleep while allowing the display to turn off;
``display`` also keeps the display awake.
"""

from __future__ import annotations

import atexit
import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

_DEFAULT_SURFACES = ("desktop", "gateway")
_VALID_MODES = {"system", "display"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool(value: Any, *, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _normalize_surface(surface: str) -> str:
    return str(surface or "").strip().lower().replace("_", "-")


def _normalize_surfaces(value: Any) -> list[str]:
    if value is None:
        return list(_DEFAULT_SURFACES)
    if isinstance(value, str):
        raw = value.strip().strip("[]")
        if not raw:
            return []
        items = raw.split(",") if "," in raw else raw.split()
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    surfaces: list[str] = []
    for item in items:
        normalized = _normalize_surface(str(item).strip().strip("'\""))
        if normalized and normalized not in surfaces:
            surfaces.append(normalized)
    return surfaces


def prevent_sleep_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return normalized ``power.prevent_sleep`` config."""
    power = (config or {}).get("power", {}) if isinstance(config, Mapping) else {}
    raw = power.get("prevent_sleep", {}) if isinstance(power, Mapping) else {}
    if isinstance(raw, bool):
        return {
            "enabled": raw,
            "surfaces": list(_DEFAULT_SURFACES),
            "mode": "system",
        }
    if not isinstance(raw, Mapping):
        return {
            "enabled": False,
            "surfaces": list(_DEFAULT_SURFACES),
            "mode": "system",
        }

    enabled = _parse_bool(raw.get("enabled"), default=False)
    mode = str(raw.get("mode") or "system").strip().lower()
    if mode not in _VALID_MODES:
        mode = "system"
    return {
        "enabled": bool(enabled),
        "surfaces": _normalize_surfaces(raw.get("surfaces")),
        "mode": mode,
    }


def resolve_prevent_sleep_config(
    surface: str,
    *,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the normalized, surface-specific prevent-sleep decision."""
    block = prevent_sleep_config(config)
    return {
        **block,
        "enabled": bool(block["enabled"])
        and _normalize_surface(surface) in block["surfaces"],
    }


def prevent_sleep_mode(*, config: Mapping[str, Any] | None = None) -> str:
    """Return ``system`` or ``display`` for the current prevent-sleep config."""
    return str(prevent_sleep_config(config).get("mode") or "system")


def should_prevent_sleep(surface: str, *, config: Mapping[str, Any] | None = None) -> bool:
    """Return whether ``surface`` should hold a sleep assertion."""
    return bool(resolve_prevent_sleep_config(surface, config=config)["enabled"])


def _execution_state_flags(mode: str) -> int:
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if mode == "display":
        flags |= ES_DISPLAY_REQUIRED
    return flags


def _set_thread_execution_state(flags: int) -> int:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint
    return int(kernel32.SetThreadExecutionState(flags))


@dataclass
class SleepPreventionHandle:
    """A started or no-op sleep-prevention assertion."""

    surface: str
    mode: str = "system"
    started: bool = False
    reason: str = "disabled"
    _set_thread_execution_state: Callable[[int], int] | None = None
    _owner_thread_id: int | None = None
    _previous_execution_state: int = ES_CONTINUOUS
    _stopped: bool = False

    def stop(self) -> bool:
        """Restore the owner thread's prior execution state."""
        if not self.started or self._stopped:
            return False
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            logger.debug(
                "Refusing to clear sleep prevention for %s from a non-owner thread",
                self.surface,
            )
            return False
        setter = self._set_thread_execution_state or _set_thread_execution_state
        try:
            result = setter(self._previous_execution_state)
        except Exception:
            logger.debug("Failed to clear sleep prevention for %s", self.surface, exc_info=True)
            return False
        if not result:
            logger.debug(
                "Failed to clear sleep prevention for %s: SetThreadExecutionState returned 0",
                self.surface,
            )
            return False
        self._stopped = True
        self.started = False
        self.reason = "stopped"
        return True


def start_prevent_sleep(
    surface: str,
    *,
    config: Mapping[str, Any] | None = None,
    platform: str | None = None,
    set_thread_execution_state: Callable[[int], int] | None = None,
    log: logging.Logger | None = None,
) -> SleepPreventionHandle:
    """Start thread-owned host sleep prevention when configured.

    Windows uses ``SetThreadExecutionState``. Non-Windows platforms return an
    inactive handle so gateway callers can remain platform-neutral; Desktop
    uses Electron's cross-platform ``powerSaveBlocker`` instead.
    """
    normalized_surface = _normalize_surface(surface)
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    resolved = resolve_prevent_sleep_config(normalized_surface, config=config)
    mode = str(resolved["mode"])
    if not resolved["enabled"]:
        return SleepPreventionHandle(surface=normalized_surface, mode=mode, reason="disabled")

    current_platform = platform if platform is not None else sys.platform
    if current_platform != "win32":
        return SleepPreventionHandle(
            surface=normalized_surface,
            mode=mode,
            reason=f"unsupported-platform:{current_platform}",
        )

    setter = set_thread_execution_state or _set_thread_execution_state
    flags = _execution_state_flags(mode)
    sink = log or logger
    try:
        result = setter(flags)
    except Exception as exc:
        sink.warning("Failed to enable sleep prevention for %s: %s", normalized_surface, exc)
        return SleepPreventionHandle(surface=normalized_surface, mode=mode, reason="api-error")

    if not result:
        sink.warning(
            "Failed to enable sleep prevention for %s: SetThreadExecutionState returned 0",
            normalized_surface,
        )
        return SleepPreventionHandle(surface=normalized_surface, mode=mode, reason="api-failed")

    handle = SleepPreventionHandle(
        surface=normalized_surface,
        mode=mode,
        started=True,
        reason="started",
        _set_thread_execution_state=setter,
        _owner_thread_id=threading.get_ident(),
        _previous_execution_state=result,
    )
    atexit.register(handle.stop)
    sink.info(
        "Sleep prevention enabled for %s (mode=%s; display sleep %s)",
        normalized_surface,
        mode,
        "blocked" if mode == "display" else "allowed",
    )
    return handle
