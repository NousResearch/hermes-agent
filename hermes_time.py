"""
Timezone-aware clock for Hermes.

Provides a single ``now()`` helper that returns a timezone-aware datetime
based on the user's configured IANA timezone (e.g. ``Asia/Kolkata``).

Resolution order:
  1. ``HERMES_TIMEZONE`` environment variable
  2. ``timezone`` key in ``~/.hermes/config.yaml``
  3. Falls back to the server's local time (``datetime.now().astimezone()``)

Invalid timezone values log a warning and fall back safely — Hermes never
crashes due to a bad timezone string.
"""

import logging
import os
import re
import threading
from datetime import datetime
from typing import Dict, Optional, Tuple

from agent.message_sanitization import _sanitize_surrogates
from hermes_constants import get_config_path

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8 fallback (shouldn't be needed — Hermes requires 3.9+)
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

# Cached state, keyed to the active timezone source. This process can multiplex
# profiles by switching HERMES_HOME (context override or env), so a single
# unkeyed process-global value would leak the first profile's timezone into
# later profile-scoped work (e.g. the desktop multiplex cron ticker persisting
# another profile's ``next_run_at`` under the backend's own timezone).
#
# Entries are published atomically under ``_cache_lock`` as one
# ``identity -> (name, ZoneInfo | None)`` mapping, so two profile-scoped
# threads racing through resolution can never publish a mixed
# identity/value pair. Each profile's resolved zone stays hot across
# multiplex switches. Call reset_cache() after in-place config changes.
_cache_lock = threading.Lock()
_tz_cache: Dict[Tuple[str, str], Tuple[str, Optional[ZoneInfo]]] = {}

_WEEKDAY_NAMES = (
    ("Monday", "Mon"),
    ("Tuesday", "Tue"),
    ("Wednesday", "Wed"),
    ("Thursday", "Thu"),
    ("Friday", "Fri"),
    ("Saturday", "Sat"),
    ("Sunday", "Sun"),
)
_MONTH_NAMES = (
    ("January", "Jan"),
    ("February", "Feb"),
    ("March", "Mar"),
    ("April", "Apr"),
    ("May", "May"),
    ("June", "Jun"),
    ("July", "Jul"),
    ("August", "Aug"),
    ("September", "Sep"),
    ("October", "Oct"),
    ("November", "Nov"),
    ("December", "Dec"),
)
_LOCALE_DIRECTIVE_RE = re.compile(r"(?<!%)%(?:[EO])?([aAbBchpXxZz])")


def _numeric_utc_offset(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None:
        return ""
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    suffix = f"{seconds:02d}" if seconds else ""
    return f"{sign}{hours:02d}{minutes:02d}{suffix}"


def _portable_directive(value: datetime, directive: str) -> str:
    weekday_long, weekday_short = _WEEKDAY_NAMES[value.weekday()]
    month_long, month_short = _MONTH_NAMES[value.month - 1]
    hour = getattr(value, "hour", 0)
    minute = getattr(value, "minute", 0)
    second = getattr(value, "second", 0)
    replacements = {
        "a": weekday_short,
        "A": weekday_long,
        "b": month_short,
        "h": month_short,
        "B": month_long,
        "c": (
            f"{weekday_short} {month_short} {value.day:02d} "
            f"{hour:02d}:{minute:02d}:{second:02d} {value.year:04d}"
        ),
        "p": "AM" if hour < 12 else "PM",
        "X": f"{hour:02d}:{minute:02d}:{second:02d}",
        "x": f"{value.year:04d}-{value.month:02d}-{value.day:02d}",
        "z": _numeric_utc_offset(value),
    }
    if directive == "Z":
        try:
            return value.tzname() or ""
        except UnicodeEncodeError:
            return ""
    return replacements[directive]


def safe_strftime(value: datetime, fmt: str) -> str:
    """Format a datetime without leaking invalid locale surrogates.

    Some Windows locale/code-page combinations raise ``UnicodeEncodeError``
    inside ``strftime`` before Python receives a string. Retry with portable
    replacements for locale-sensitive directives, then scrub any surrogate
    code points returned by the platform or ``tzname()``.
    """
    try:
        rendered = value.strftime(fmt)
    except UnicodeEncodeError:
        replacements: Dict[str, str] = {}

        def replace_directive(match: re.Match[str]) -> str:
            token = f"__HERMES_TIME_{len(replacements)}__"
            replacements[token] = _sanitize_surrogates(
                _portable_directive(value, match.group(1))
            )
            return token

        rendered = value.strftime(_LOCALE_DIRECTIVE_RE.sub(replace_directive, fmt))
        for token, replacement in replacements.items():
            rendered = rendered.replace(token, replacement)
    return _sanitize_surrogates(rendered)


def _timezone_cache_identity() -> Tuple[str, str]:
    """Return the active source identity for the timezone cache."""
    tz_env = os.getenv("HERMES_TIMEZONE", "").strip()
    if tz_env:
        return ("environment", tz_env)
    return ("config", str(get_config_path()))


def _resolve_timezone_name() -> str:
    """Read the configured IANA timezone string (or empty string).

    This does file I/O when falling through to config.yaml, so callers
    should cache the result rather than calling on every ``now()``.
    """
    # 1. Environment variable (highest priority — set by Supervisor, etc.)
    tz_env = os.getenv("HERMES_TIMEZONE", "").strip()
    if tz_env:
        return tz_env

    # 2. config.yaml ``timezone`` key
    try:
        # Prefer the shared cached raw-config reader (mtime/size-keyed cache +
        # libyaml C loader) — a direct yaml.safe_load of a large config.yaml
        # costs ~100ms+ and this used to run inside the FIRST system prompt
        # build, on the time-to-first-token critical path.
        try:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config() or {}
        except Exception:
            import yaml
            config_path = get_config_path()
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                cfg = {}
        if cfg:
            # Managed scope: an administrator can pin ``timezone`` too. Overlay
            # via the shared helper (fail-open) since this reads config.yaml directly.
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


def _get_zoneinfo(name: str) -> Optional[ZoneInfo]:
    """Validate and return a ZoneInfo, or None if invalid."""
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (KeyError, Exception) as exc:
        logger.warning(
            "Invalid timezone '%s': %s. Falling back to server local time.",
            name, exc,
        )
        return None


def get_timezone() -> Optional[ZoneInfo]:
    """Return the active profile's configured ZoneInfo, or None (server-local).

    The cache is isolated by the active timezone source — the explicit
    ``HERMES_TIMEZONE`` override or the active profile's config path — so a
    process that multiplexes profiles (desktop cron ticker, multiplex
    gateway) never reuses another profile's timezone. Call ``reset_cache()``
    after editing the active config in place.
    """
    cache_identity = _timezone_cache_identity()
    with _cache_lock:
        entry = _tz_cache.get(cache_identity)
        if entry is not None:
            return entry[1]
    # Resolve outside the lock (config file I/O); publish atomically below.
    name = _resolve_timezone_name()
    tz = _get_zoneinfo(name)
    with _cache_lock:
        # First writer wins so concurrent resolvers of the SAME identity
        # converge on one ZoneInfo object; a different identity's write can
        # never be mixed into this one — the (name, tz) pair is one value.
        return _tz_cache.setdefault(cache_identity, (name, tz))[1]


def reset_cache() -> None:
    """Clear the cached timezone so the next call re-resolves it.

    Call this after the configured timezone may have changed (e.g. after a
    config edit or ``HERMES_TIMEZONE`` update) to force ``get_timezone()`` /
    ``now()`` to read the new value instead of the value cached at first use.
    """
    with _cache_lock:
        _tz_cache.clear()


def now() -> datetime:
    """
    Return the current time as a timezone-aware datetime.

    If a valid timezone is configured, returns wall-clock time in that zone.
    Otherwise returns the server's local time (via ``astimezone()``).
    """
    tz = get_timezone()
    if tz is not None:
        return datetime.now(tz)
    # No timezone configured — use server-local (still tz-aware)
    return datetime.now().astimezone()
