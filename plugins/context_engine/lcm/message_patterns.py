"""Message-content pattern helpers for LCM ingest filtering.

Patterns are Python regex strings. Compilation is tolerant: an invalid
pattern emits a warning and is skipped, leaving valid patterns in the
same list still active. Matching uses the optional ``regex`` package so
per-pattern timeouts can guard synchronous ingest against catastrophic
backtracking. User-supplied anchors (``^``, ``\b``) and inline flags
(``(?is)``) work as written.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

try:  # pragma: no cover - exercised when the optional dependency is absent
    import regex as _regex_engine
except Exception:  # pragma: no cover - keep the plugin importable in minimal installs
    _regex_engine = None

logger = logging.getLogger(__name__)

MESSAGE_PATTERN_MATCH_TIMEOUT_SECONDS = 0.05
_TIMEOUT_WARNED_PATTERNS: set[str] = set()
_TIMEOUT_UNSUPPORTED_WARNED_PATTERNS: set[str] = set()
_MISSING_REGEX_WARNING_EMITTED = False


def _pattern_label(pattern: Any) -> str:
    return str(getattr(pattern, "pattern", repr(pattern)))


def compile_message_patterns(patterns: Iterable[str]) -> list[Any]:
    """Compile configured message patterns once at startup.

    The optional ``regex`` package is required for active message-level regex
    filtering because stdlib ``re`` cannot enforce match timeouts. Minimal
    installs remain importable, but configured patterns are disabled with a
    warning instead of running unbounded matches in the ingest path.
    """
    global _MISSING_REGEX_WARNING_EMITTED

    pattern_list = list(patterns)
    if not pattern_list:
        return []
    if _regex_engine is None:
        if not _MISSING_REGEX_WARNING_EMITTED:
            _MISSING_REGEX_WARNING_EMITTED = True
            logger.warning(
                "LCM ignore_message_patterns configured but optional dependency 'regex' is not installed; "
                "message-level regex filtering is disabled to avoid unbounded stdlib re matching"
            )
        return []

    compiled: list[Any] = []
    for pattern in pattern_list:
        try:
            compiled.append(_regex_engine.compile(pattern))
        except _regex_engine.error as exc:
            logger.warning(
                "LCM ignore_message_patterns: skipping invalid regex %r: %s",
                pattern,
                exc,
            )
    return compiled


def _search_with_timeout(pattern: Any, text: str) -> Any:
    return pattern.search(text, timeout=MESSAGE_PATTERN_MATCH_TIMEOUT_SECONDS)


def _warn_timeout_once(pattern: Any) -> None:
    label = _pattern_label(pattern)
    if label in _TIMEOUT_WARNED_PATTERNS:
        return
    _TIMEOUT_WARNED_PATTERNS.add(label)
    logger.warning(
        "LCM ignore_message_patterns: regex %r timed out after %.3gs; treating as no match",
        label,
        MESSAGE_PATTERN_MATCH_TIMEOUT_SECONDS,
    )


def _warn_timeout_unsupported_once(pattern: Any) -> None:
    label = _pattern_label(pattern)
    if label in _TIMEOUT_UNSUPPORTED_WARNED_PATTERNS:
        return
    _TIMEOUT_UNSUPPORTED_WARNED_PATTERNS.add(label)
    logger.warning(
        "LCM ignore_message_patterns: regex %r does not support timeout matching; treating as no match",
        label,
    )


def matches_message_pattern(text: str, patterns: Iterable[Any]) -> bool:
    """Return True when ``text`` matches any compiled pattern.

    A pattern timeout is treated as a non-match and logged once per process for
    that pattern. Patterns that do not accept a timeout are also skipped rather
    than retried unsafely without a timeout.
    """
    if not text:
        return False
    for pattern in patterns:
        try:
            if _search_with_timeout(pattern, text):
                return True
        except TimeoutError:
            _warn_timeout_once(pattern)
            continue
        except TypeError as exc:
            if "timeout" not in str(exc):
                raise
            _warn_timeout_unsupported_once(pattern)
            continue
    return False
