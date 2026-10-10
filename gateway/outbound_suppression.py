"""Operator-configured outbound suppression (``suppress_outbound`` in config.yaml).

Opt-in regex strings, set globally and/or per platform (a platform's list extends the global
one), matched with ``re.search`` against NON-STREAMED outbound chat sends: final replies, status
updates, platform notices and both shutdown-notice rails. A match drops the message before send.
Streamed replies are delivered as progressive edits, so a whole-message regex over partial text
is unsound there and mid-stream delivery is deliberately not filtered. Programmatic surfaces
(``gateway.run._GATEWAY_RAW_TEXT_PLATFORMS``) are always exempt. Empty by default.

``gateway.run`` internals are imported lazily inside function bodies (import cycle).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Compiled-pattern cache keyed by pattern string; None marks a pattern that failed to compile
# (warned once, then skipped without retrying).
_COMPILED: Dict[str, Optional["re.Pattern[str]"]] = {}

# Resolved-config cache so the per-message hot path does not re-parse config.yaml. Keyed by the
# resolved config path -> (mtime stamp, GatewayConfig): context-local profile homes route different
# config files through one process and two files can share an mtime, so the path is part of the
# identity. Bounded by the number of profile homes, so no eviction is needed.
_CONFIG_CACHE: Dict[str, tuple[Any, Any]] = {}


def normalize_suppress_outbound(value: Any) -> List[str]:
    """Coerce a ``suppress_outbound`` value to a list of pattern strings.

    A list of strings is the documented shape; a single string is a one-element list. Non-string
    entries and other shapes are skipped with a warning (a config mistake never crashes the
    gateway). Regex validity is checked where patterns are compiled, not here.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        logger.warning(
            "Ignoring invalid suppress_outbound value (expected list of regex strings, got %s)",
            type(value).__name__,
        )
        return []
    patterns: List[str] = []
    for item in value:
        if isinstance(item, str) and item:
            patterns.append(item)
        elif item is not None:
            logger.warning("Ignoring non-string suppress_outbound entry: %r", item)
    return patterns


def _compile(pattern: str) -> Optional["re.Pattern[str]"]:
    """Compile one pattern as written (case-sensitive; ``(?i)`` is the operator's opt-in)."""
    if pattern not in _COMPILED:
        try:
            _COMPILED[pattern] = re.compile(pattern)
        except re.error as exc:
            logger.warning("Ignoring invalid suppress_outbound pattern %r: %s", pattern, exc)
            _COMPILED[pattern] = None
    return _COMPILED[pattern]


def _patterns_for(platform: Any) -> List[str]:
    """Effective pattern list for *platform* from the active profile's config (fail-open)."""
    from gateway.config import Platform, load_gateway_config
    from gateway.run import _gateway_config_home, _gateway_platform_value

    try:
        config_path = _gateway_config_home() / "config.yaml"
        try:
            stamp: Any = config_path.stat().st_mtime_ns
        except OSError:
            stamp = None
        cached = _CONFIG_CACHE.get(str(config_path))
        if cached is None or cached[0] != stamp:
            # load_gateway_config() honors the same context-local home override as
            # _gateway_config_home(), so the loaded config matches this cache key.
            cached = (stamp, load_gateway_config())
            _CONFIG_CACHE[str(config_path)] = cached
        try:
            platform_enum: Optional[Platform] = Platform(_gateway_platform_value(platform))
        except (ValueError, KeyError):
            platform_enum = None
        return cached[1].get_suppress_outbound(platform_enum)
    except Exception:
        logger.debug("suppress_outbound config resolution failed", exc_info=True)
        return []


def outbound_suppressed(platform: Any, text: Any) -> bool:
    """True when an operator ``suppress_outbound`` pattern matches *text* on *platform*.

    A drop logs one info line with a redacted, truncated preview.
    """
    from gateway.run import (
        _gateway_platform_value,
        _gateway_surface_passes_raw_text,
        _redact_gateway_user_facing_secrets,
    )

    if not text or _gateway_surface_passes_raw_text(platform):
        return False
    body = str(text)
    for pattern in _patterns_for(platform):
        compiled = _compile(pattern)
        if compiled is not None and compiled.search(body):
            logger.info(
                "Dropped outbound %s message matching suppress_outbound %r: %s",
                _gateway_platform_value(platform) or "unknown", pattern,
                _redact_gateway_user_facing_secrets(body)[:120],
            )
            return True
    return False
