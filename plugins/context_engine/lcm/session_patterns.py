"""Session-pattern helpers for LCM session filtering.

`*` matches within a single colon-delimited segment.
`**` can span across colons.
"""

from __future__ import annotations

import re
from typing import Iterable, List


def compile_session_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a session glob into a regex.

    `*` matches any non-colon characters, while `**` can span colons.
    """
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\*\*", "\x00")
    escaped = escaped.replace(r"\*", "[^:]*")
    escaped = escaped.replace("\x00", ".*")
    return re.compile(rf"^{escaped}$")


def compile_session_patterns(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    """Compile configured session patterns once at startup."""
    return [compile_session_pattern(pattern) for pattern in patterns]


def build_session_match_keys(session_id: str, platform: str = "") -> list[str]:
    """Build candidate keys that session filters may match against.

    Hermes currently gives the engine a random-ish session_id plus some session
    metadata like platform. Matching against multiple shapes keeps the filter
    useful now while leaving room for richer host-provided keys later.
    """
    keys: list[str] = []
    if session_id:
        keys.append(session_id)
    if platform:
        keys.append(platform)
    if session_id and platform:
        keys.append(f"{platform}:{session_id}")
    return keys


def matches_session_pattern(session_keys: Iterable[str], patterns: Iterable[re.Pattern[str]]) -> bool:
    """Check whether any session key matches any compiled pattern."""
    keys = [key for key in session_keys if key]
    if not keys:
        return False
    return any(pattern.match(key) for pattern in patterns for key in keys)
