"""Best-effort process resource-limit adjustments for long-running services.

The public helper in this module is shared by the gateway and the dashboard/
serve entrypoints. It deliberately has no user-facing environment-variable
control: the target comes from the profile's canonical ``config.yaml`` loader.
"""

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


def _configured_nofile_soft_limit(
    config: Mapping[str, Any] | None,
) -> int | None:
    """Resolve ``runtime.nofile_soft_limit`` from a loaded config.

    A missing key uses the default. Explicit ``0``, ``false``, and ``null``
    disable the adjustment. Other non-integer or negative values are invalid
    and are ignored (the caller fails open without changing the process limit).
    """
    if config is None:
        try:
            # Use Hermes's real, profile-aware loader rather than reading YAML
            # here. This also applies managed-scope overlays and defaults.
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
    if raw_value is None or raw_value is False:
        return None
    if raw_value is True or not isinstance(raw_value, int):
        return None
    if raw_value <= 0:
        return None
    return raw_value


def configured_nofile_soft_limit(
    config: Mapping[str, Any] | None = None,
) -> int | None:
    """Public accessor for the resolved ``runtime.nofile_soft_limit`` target.

    Used by service-definition generators (e.g. the launchd plist) so the
    persisted service limits and the in-process floor share one config knob.
    Returns ``None`` when the adjustment is disabled or unresolvable.
    """
    return _configured_nofile_soft_limit(config)


def apply_nofile_soft_limit(
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Raise this process's ``RLIMIT_NOFILE`` soft limit when possible.

    The target defaults to :data:`DEFAULT_NOFILE_SOFT_LIMIT` and can be set with
    ``runtime.nofile_soft_limit``. The target is clamped to a finite hard limit,
    never lowers an existing higher soft limit, and returns ``False`` for an
    explicit opt-out or when the platform/sandbox refuses the operation.

    This is intentionally best-effort. Unsupported platforms, malformed
    settings, and denied ``setrlimit`` calls must never prevent a server from
    starting.
    """
    if _resource is None:
        return False

    target = _configured_nofile_soft_limit(config)
    if target is None:
        return False

    try:
        nofile = _resource.RLIMIT_NOFILE
        current_soft, current_hard = _resource.getrlimit(nofile)
        # On platforms where RLIM_INFINITY is represented as -1, ordinary
        # integer ordering would make an unlimited soft limit look lower than
        # every positive target. Never replace infinity with a finite limit.
        if current_soft == getattr(_resource, "RLIM_INFINITY", object()):
            return False
        if current_soft >= target:
            return False

        if current_hard == getattr(_resource, "RLIM_INFINITY", object()):
            new_soft = target
        else:
            new_soft = min(target, current_hard)
        if new_soft <= current_soft:
            return False

        _resource.setrlimit(nofile, (new_soft, current_hard))
        return True
    except Exception:
        # This helper runs before server startup and must fail open for
        # unsupported/sandboxed environments and denied resource changes.
        logger.debug("Could not raise RLIMIT_NOFILE soft limit", exc_info=True)
        return False


__all__ = [
    "DEFAULT_NOFILE_SOFT_LIMIT",
    "apply_nofile_soft_limit",
    "configured_nofile_soft_limit",
]
