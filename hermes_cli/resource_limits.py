"""Best-effort process resource-limit adjustments for long-running services."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from hermes_cli.config_defaults import DEFAULT_CONFIG

try:  # ``resource`` is POSIX-only (and unavailable on Windows).
    import resource as _resource
except (ImportError, ModuleNotFoundError):  # pragma: no cover - Windows only
    _resource = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_NOFILE_SOFT_LIMIT = int(DEFAULT_CONFIG["runtime"]["nofile_soft_limit"])
DEFAULT_PROCESS_HARDENING = str(DEFAULT_CONFIG["security"]["process_hardening"])
_MISSING = object()

# ``security.process_hardening`` modes. ``core-only`` keeps the cheapest, least observable
# guarantee: a crash can never spill the credentials the process holds in memory onto disk.
# ``full`` adds debugger refusal, which is opt-in because it breaks gdb/lldb attach.
_HARDENING_MODES = ("core-only", "full")
_DISABLED_ALIASES = frozenset({"", "0", "false", "no", "none", "off", "disabled"})
_ENABLED_ALIASES = frozenset({"1", "true", "yes", "on", "enabled"})
# Linux: clear the process's dumpable attribute (man 2 prctl) — blocks ptrace/PROC_MEM reads by
# unprivileged peers and core dumps. Darwin: refuse external debugger attachment (sys/ptrace.h).
_PR_SET_DUMPABLE = 4
_PT_DENY_ATTACH = 31


def configured_process_hardening(config: Mapping[str, Any] | None = None) -> str | None:
    """``security.process_hardening`` from a loaded config, or ``None`` when disabled/unresolvable.

    Missing key → :data:`DEFAULT_PROCESS_HARDENING`. Explicit ``"off"``/``false``/``0``/``null``
    disable; unrecognized values are ignored (caller fails open). Accepts the boolean and string
    spellings of both, so `hermes config set security.process_hardening off` and `false` agree.
    """
    if config is None:
        try:
            # Profile-aware loader (applies managed-scope overlays and defaults).
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            logger.debug("Could not load config for process hardening", exc_info=True)
            return None
    if not isinstance(config, Mapping):
        return None
    security = config.get("security", _MISSING)
    if security is _MISSING:
        return DEFAULT_PROCESS_HARDENING
    if not isinstance(security, Mapping):
        return None
    raw_value = security.get("process_hardening", _MISSING)
    if raw_value is _MISSING:
        return DEFAULT_PROCESS_HARDENING
    if isinstance(raw_value, bool):
        return DEFAULT_PROCESS_HARDENING if raw_value else None
    if not isinstance(raw_value, str):
        return None
    mode = raw_value.strip().lower()
    if mode in _DISABLED_ALIASES:
        return None
    if mode in _ENABLED_ALIASES:
        return DEFAULT_PROCESS_HARDENING
    if mode in _HARDENING_MODES:
        return mode
    return None


def _disable_core_dumps() -> bool:
    """Drop this process's core-dump limit; ``True`` when the limit was lowered. Never raises."""
    try:
        _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
        return True
    except Exception:
        logger.debug("Could not disable core dumps", exc_info=True)
        return False


def _disable_debugger_attach() -> bool:
    """Refuse debugger attach where the kernel supports it; ``False`` elsewhere. Never raises.

    Linux ``PR_SET_DUMPABLE=0`` and Darwin ``PT_DENY_ATTACH`` are the two portable-ish levers;
    Windows has no equivalent, and denied/unsupported calls are a no-op rather than an error.
    """
    try:
        import ctypes
        import platform

        system = platform.system()
        libc = ctypes.CDLL(None, use_errno=True)
        if system == "Linux":
            return libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) == 0
        if system == "Darwin":
            return libc.ptrace(_PT_DENY_ATTACH, 0, None, 0) == 0
    except Exception:
        logger.debug("Could not refuse debugger attach", exc_info=True)
    return False


def apply_process_hardening(config: Mapping[str, Any] | None = None) -> str:
    """Best-effort hardening of this process's own kernel surface; returns the mode requested.

    ``"core-only"`` (the default) lowers ``RLIMIT_CORE`` to zero so a crash cannot write the
    provider credentials held in memory to disk; ``"full"`` additionally refuses debugger attach.
    Returns ``"off"`` when disabled or unsupported (Windows). Every step is best-effort: an
    unsupported platform, a malformed setting, or a denied ``setrlimit``/``prctl`` must never
    prevent a service from starting (fail open, never raises). Also inherited by children, so
    workers spawned after this call can neither dump core nor be attached to unprivileged.
    """
    if _resource is None:
        return "off"
    mode = configured_process_hardening(config)
    if mode is None:
        return "off"
    try:
        _disable_core_dumps()
        if mode == "full":
            _disable_debugger_attach()
    except Exception:  # a service must still start even if a lever misbehaves
        logger.debug("Could not apply process hardening", exc_info=True)
    return mode


def apply_nofile_soft_limit(config: Mapping[str, Any] | None = None) -> bool:
    """Best-effort raise of this process's ``RLIMIT_NOFILE`` soft limit; ``True`` iff changed.

    Target = ``runtime.nofile_soft_limit`` (default :data:`DEFAULT_NOFILE_SOFT_LIMIT`), clamped
    to a finite hard limit; never lowers a higher soft limit. Unsupported platforms, malformed
    settings, and denied ``setrlimit`` must never prevent a server from starting.
    """
    if _resource is None:
        return False
    target = configured_nofile_soft_limit(config)
    if target is None:
        return False
    try:
        nofile = _resource.RLIMIT_NOFILE
        current_soft, current_hard = _resource.getrlimit(nofile)
        # RLIM_INFINITY may be -1, which ordinary ordering would treat as "lower than any
        # target"; never replace infinity with a finite limit.
        infinity = getattr(_resource, "RLIM_INFINITY", object())
        if current_soft == infinity or current_soft >= target:
            return False
        new_soft = target if current_hard == infinity else min(target, current_hard)
        if new_soft <= current_soft:
            return False
        _resource.setrlimit(nofile, (new_soft, current_hard))
        return True
    except Exception:
        logger.debug("Could not raise RLIMIT_NOFILE soft limit", exc_info=True)
        return False


def configured_nofile_soft_limit(config: Mapping[str, Any] | None = None) -> int | None:
    """``runtime.nofile_soft_limit`` from a loaded config, or ``None`` when disabled/unresolvable.

    Missing key → default. Explicit ``0``/``false``/``null`` disable; other non-int or negative
    values are ignored (caller fails open). Shared by the in-process floor and service-definition
    generators (launchd plist) so both use one knob.
    """
    if config is None:
        try:
            # Profile-aware loader (applies managed-scope overlays and defaults).
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            logger.debug("Could not load config for RLIMIT_NOFILE", exc_info=True)
            return None
    if not isinstance(config, Mapping):
        return None
    runtime = config.get("runtime", _MISSING)
    if runtime is _MISSING:
        return DEFAULT_NOFILE_SOFT_LIMIT
    if not isinstance(runtime, Mapping):
        return None
    raw_value = runtime.get("nofile_soft_limit", _MISSING)
    if raw_value is _MISSING:
        return DEFAULT_NOFILE_SOFT_LIMIT
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
        return None
    return raw_value


__all__ = [
    "DEFAULT_NOFILE_SOFT_LIMIT",
    "DEFAULT_PROCESS_HARDENING",
    "apply_nofile_soft_limit",
    "apply_process_hardening",
    "configured_nofile_soft_limit",
    "configured_process_hardening",
]
