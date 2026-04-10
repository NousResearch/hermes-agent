import math
from typing import Any

UNLIMITED_ITERATION_TOKENS = frozenset({
    "unlimited",
    "infinite",
    "infinity",
    "inf",
    "none",
    "no-limit",
    "nolimit",
})


def is_unlimited_iteration_limit(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        try:
            return math.isinf(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(value, str):
        return value.strip().lower() in UNLIMITED_ITERATION_TOKENS
    return False


def parse_iteration_limit(value: Any, *, default: Any = 90) -> float | int:
    """Parse an iteration-limit value.

    Accepts integers or string sentinels like ``unlimited`` / ``inf``.
    Returns ``math.inf`` for unlimited, or an ``int`` for numeric inputs.
    Empty values fall back to ``default``.
    """
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        if is_unlimited_iteration_limit(stripped):
            return math.inf
        return int(stripped)
    if is_unlimited_iteration_limit(value):
        return math.inf
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid iteration limits")
    return int(value)


def format_iteration_limit(value: Any) -> str:
    if is_unlimited_iteration_limit(value):
        return "unlimited"
    try:
        parsed = parse_iteration_limit(value, default=value)
    except (TypeError, ValueError):
        return str(value)
    if is_unlimited_iteration_limit(parsed):
        return "unlimited"
    return str(parsed)
