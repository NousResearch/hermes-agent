"""``agent.max_turns`` spellings -> an int iteration cap (re-exported by ``hermes_cli.config``)."""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for an unlimited turn budget. ``sys.maxsize`` survives the str->int round-trip through
# the HERMES_MAX_ITERATIONS env bridge, works in every ``<``/``>=``/``max - used`` comparison in
# the iteration budget without an "unlimited" special case, and is unreachable in practice.
TURN_LIMIT_UNLIMITED = sys.maxsize

# Spellings that mean "no limit" (compared lowercased, whitespace-stripped).
_UNLIMITED_SPELLINGS = frozenset({
    "none", "null", "unlimited", "infinite", "infinity", "inf", "∞", "-1", "0"})


def resolve_turn_limit(raw: Any, default: int = TURN_LIMIT_UNLIMITED) -> int:
    """Normalize a raw ``agent.max_turns`` value into an int iteration cap (always >= 1)."""
    # bool is a subclass of int; reject explicitly so True/False don't become 1/0.
    if raw is None or isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        n = int(raw)
    elif isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return default
        if s in _UNLIMITED_SPELLINGS:
            return TURN_LIMIT_UNLIMITED
        try:
            n = int(s)
        except ValueError:
            try:
                n = int(float(s))
            except ValueError:
                logger.debug("resolve_turn_limit: unparseable value %r → default %d", raw, default)
                return default
    else:
        # Unknown type (list, dict, …) — don't crash the agent over a bad config.
        logger.debug("resolve_turn_limit: unsupported type %s (%r) → default %d", type(raw).__name__, raw, default)
        return default
    return TURN_LIMIT_UNLIMITED if n <= 0 else n
