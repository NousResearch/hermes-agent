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
_MISSING = object()


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


__all__ = ["DEFAULT_NOFILE_SOFT_LIMIT", "apply_nofile_soft_limit", "configured_nofile_soft_limit"]
