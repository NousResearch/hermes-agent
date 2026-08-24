"""Timezone-aware clock for Hermes.

``now()`` returns a tz-aware datetime in the user's configured IANA timezone. Resolution order:
``HERMES_TIMEZONE`` env var, then ``timezone`` in ``~/.hermes/config.yaml``, else server-local
time. Invalid timezone values log a warning and fall back — never crash.
"""

import logging
import os
import threading
from datetime import datetime
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from hermes_constants import get_config_path

logger = logging.getLogger(__name__)

# Cache keyed by timezone *source* identity. This process can multiplex profiles by switching
# HERMES_HOME, so one unkeyed global would leak the first profile's timezone into later
# profile-scoped work (e.g. the desktop multiplex cron ticker persisting another profile's
# ``next_run_at``). Entries are published atomically under ``_cache_lock`` as one
# ``identity -> (name, ZoneInfo | None)`` value, so racing resolvers can never publish a mixed
# identity/value pair. Call reset_cache() after in-place config changes.
_cache_lock = threading.Lock()
_tz_cache: Dict[Tuple[str, str], Tuple[str, Optional[ZoneInfo]]] = {}


def _timezone_cache_identity() -> Tuple[str, str]:
    tz_env = os.getenv("HERMES_TIMEZONE", "").strip()
    return ("environment", tz_env) if tz_env else ("config", str(get_config_path()))


def _resolve_timezone_name() -> str:
    """Read the configured IANA timezone string (or ``""``). Does file I/O — callers cache."""
    tz_env = os.getenv("HERMES_TIMEZONE", "").strip()
    if tz_env:
        return tz_env
    try:
        # Prefer the shared cached raw-config reader (mtime-keyed + libyaml): a direct safe_load of
        # a large config.yaml costs ~100 ms and this ran inside the FIRST system prompt build.
        try:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config() or {}
        except Exception:
            import yaml
            config_path = get_config_path()
            cfg = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}) if config_path.exists() else {}
        if cfg:
            # Managed scope: an administrator can pin ``timezone`` too (fail-open overlay).
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
            tz_cfg = cfg.get("timezone", "")
            if isinstance(tz_cfg, str) and tz_cfg.strip():
                return tz_cfg.strip()
    except Exception:
        pass
    return ""


def get_timezone() -> Optional[ZoneInfo]:
    """Return the active profile's configured ZoneInfo, or None (server-local)."""
    cache_identity = _timezone_cache_identity()
    with _cache_lock:
        entry = _tz_cache.get(cache_identity)
        if entry is not None:
            return entry[1]
    # Resolve outside the lock (config file I/O); first writer wins so concurrent resolvers of the
    # same identity converge on one ZoneInfo object.
    name = _resolve_timezone_name()
    tz = None
    if name:
        try:
            tz = ZoneInfo(name)
        except Exception as exc:
            logger.warning("Invalid timezone '%s': %s. Falling back to server local time.", name, exc)
    with _cache_lock:
        return _tz_cache.setdefault(cache_identity, (name, tz))[1]


def reset_cache() -> None:
    """Clear the cached timezone so the next call re-resolves it (after config/env changes)."""
    with _cache_lock:
        _tz_cache.clear()


def now() -> datetime:
    """Current time as a tz-aware datetime: configured zone, else server-local."""
    tz = get_timezone()
    return datetime.now(tz) if tz is not None else datetime.now().astimezone()
