"""Best-effort process resource-limit adjustments for long-running services."""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Mapping
from typing import Any

from hermes_agent.hermes_cli.config_defaults import DEFAULT_CONFIG

try:  # ``resource`` is POSIX-only (and unavailable on Windows).
    import resource as _resource
except (ImportError, ModuleNotFoundError):  # pragma: no cover - Windows only
    _resource = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_NOFILE_SOFT_LIMIT = int(DEFAULT_CONFIG["runtime"]["nofile_soft_limit"])
DEFAULT_MALLOC_ARENA_MAX = int(DEFAULT_CONFIG["runtime"]["malloc_arena_max"])
_M_ARENA_MAX = -8  # glibc <malloc.h> mallopt parameter
_MISSING = object()


def _runtime_int(config: Mapping[str, Any] | None, key: str, default: int) -> int | None:
    """Positive int ``runtime.<key>`` from a loaded config, or ``None`` when disabled/unresolvable.

    Missing key → *default*. Explicit ``0``/``false``/``null`` disable; other non-int or negative
    values are ignored (caller fails open).
    """
    if config is None:
        try:
            # Profile-aware loader (applies managed-scope overlays and defaults).
            from hermes_agent.hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            logger.debug("Could not load config for runtime.%s", key, exc_info=True)
            return None
    if not isinstance(config, Mapping):
        return None
    runtime = config.get("runtime", _MISSING)
    if runtime is _MISSING:
        return default
    if not isinstance(runtime, Mapping):
        return None
    raw_value = runtime.get(key, _MISSING)
    if raw_value is _MISSING:
        return default
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
        return None
    return raw_value


def configured_nofile_soft_limit(config: Mapping[str, Any] | None = None) -> int | None:
    """``runtime.nofile_soft_limit`` (see :func:`_runtime_int`). Shared by the in-process floor and
    service-definition generators (launchd plist) so both use one knob."""
    return _runtime_int(config, "nofile_soft_limit", DEFAULT_NOFILE_SOFT_LIMIT)


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


def glibc_malloc() -> Any | None:
    """This process's libc handle when it is glibc with ``mallopt`` (else ``None``: musl, macOS, Windows)."""
    try:
        libc = ctypes.CDLL(None)
    except (OSError, TypeError):  # Windows: no process-global symbol namespace
        return None
    return libc if hasattr(libc, "gnu_get_libc_version") and hasattr(libc, "mallopt") else None


def _arena_max_setting(env: Mapping[str, str]) -> str | None:
    """The arena cap an environment asks glibc for (exact ``glibc.malloc.arena_max`` tunable, else
    ``MALLOC_ARENA_MAX``), or ``None``."""
    for item in env.get("GLIBC_TUNABLES", "").split(":"):
        name, sep, value = item.partition("=")
        if sep and name == "glibc.malloc.arena_max":
            return value
    return env.get("MALLOC_ARENA_MAX")


def _launch_environ() -> dict[str, str] | None:
    """The environment this process was exec'd with (what glibc read at startup), or ``None``.

    ``/proc/self/environ`` keeps the exec-time block; ``os.environ`` also holds values Hermes loaded
    later from the profile ``.env``, which glibc never saw.
    """
    try:
        raw = open("/proc/self/environ", "rb").read()
    except OSError:
        return None
    entries = (os.fsdecode(entry).partition("=") for entry in raw.split(b"\0") if entry)
    return {name: value for name, sep, value in entries if sep}


def _positive_c_int(value: str) -> int | None:
    try:
        number = int(value.strip())
    except ValueError:
        return None
    return number if 0 < number <= 2**31 - 1 else None


def apply_malloc_arena_max(config: Mapping[str, Any] | None = None) -> bool:
    """Cap glibc's malloc arena count for this process (``mallopt(M_ARENA_MAX)``); ``True`` iff set.

    glibc gives every thread that contends on an arena lock a new arena, up to 8 x cores, and never
    unmaps them. A long-lived server with hundreds of short-lived Python threads plus native pools
    (tokio, rayon, sqlite) fills every slot (157 of 160 on a 20-core host), and each arena keeps its
    own free pages. The GIL already serialises Python's allocations, so a small cap costs no
    measurable throughput. Applied in-process rather than via an exported ``MALLOC_ARENA_MAX`` so
    exec'd children (terminal commands, MCP servers, builds) keep glibc's default.

    Call before the process starts any thread: arenas that exist never go away, and glibc latches its
    limit once a thread asks for an arena past ``glibc.malloc.arena_test``. (The gateway's startup
    watchdog thread arms before config loads, so a gateway may hold one arena beyond a cap of 1.)

    An arena setting in the exec-time environment was already applied by glibc and wins. One that
    is only in ``os.environ`` because Hermes loaded it from the profile ``.env`` is applied here with
    the operator's number (glibc never read it). glibc only; never raises.
    """
    libc = glibc_malloc()
    if libc is None:
        return False
    current = _arena_max_setting(os.environ)
    launch_env = _launch_environ()
    launched = current if launch_env is None else _arena_max_setting(launch_env)
    if launched is not None and current in (None, launched):
        return False
    if current is not None:
        target = _positive_c_int(current)
    else:
        target = _runtime_int(config, "malloc_arena_max", DEFAULT_MALLOC_ARENA_MAX)
        target = target if target is not None and target <= 2**31 - 1 else None
    if target is None:
        return False
    try:
        return bool(libc.mallopt(_M_ARENA_MAX, target))
    except Exception:
        logger.debug("Could not cap malloc arenas", exc_info=True)
        return False


def server_process_argv(argv: list[str]) -> bool:
    """True when *argv* (``sys.argv[1:]``) launches a long-lived server: ``serve``, ``dashboard`` or a
    foreground ``gateway`` (``gateway`` / ``gateway run``)."""
    from hermes_agent.hermes_cli._parser import command_argv

    args = command_argv(argv)
    if not args:
        return False
    if args[0] in ("serve", "dashboard"):
        return True
    return args[0] == "gateway" and (len(args) == 1 or args[1] == "run" or args[1].startswith("-"))


__all__ = [
    "DEFAULT_MALLOC_ARENA_MAX", "DEFAULT_NOFILE_SOFT_LIMIT", "apply_malloc_arena_max",
    "apply_nofile_soft_limit", "configured_nofile_soft_limit", "glibc_malloc", "server_process_argv"]
