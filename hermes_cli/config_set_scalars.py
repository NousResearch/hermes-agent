"""Scalar coercion for ``hermes config set`` string arguments (bool/None/int/float words)."""

from decimal import Decimal, InvalidOperation
from typing import Optional


def _coerce_int(value: str):
    """int(value) for a clean integer literal (signs/whitespace/underscores OK), else None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: str):
    """``float(value)`` only when the conversion preserves its decimal value; NaN/inf rejected.
    Decimal-looking identifiers more precise than a binary float must stay strings."""
    try:
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")) or Decimal(value) != Decimal(str(f)):
            return None
    except (TypeError, ValueError, InvalidOperation):
        return None
    return f


_SCALAR_WORDS = {
    'true': True, 'yes': True, 'on': True,
    'false': False, 'no': False, 'off': False,
    # YAML null. Many DEFAULT_CONFIG leaves are "null/absent = off"; without this,
    # ``config set X null`` stored the truthy string "null" and the feature could never be cleared.
    'null': None, 'none': None, '~': None}


def _quoted_bool(stripped: str) -> Optional[bool]:
    """The bool a quoted boolean word spells (``'"true"'`` -> True), else None. Shell-quoted
    YAML scalars (``hermes config set gateway.strict '"true"'``) arrive with the quotes as literal
    characters; stored verbatim on a bool-schema key, readers saw an unknown string and fell back
    to the knob's default (#143009: strict media delivery silently stayed off)."""
    if len(stripped) < 2 or stripped[0] != stripped[-1] or stripped[0] not in ("'", '"'):
        return None
    word = _SCALAR_WORDS.get(stripped[1:-1].strip().lower())
    return word if isinstance(word, bool) else None


def _is_unseeded_effort_key(key: str) -> bool:
    """A reasoning-effort slot with no ``DEFAULT_CONFIG`` leaf (``agent.reasoning_effort``,
    ``agent.reasoning_overrides.<model>``, MoA slots). There ``none`` is the level that turns
    thinking off, while null means "use the default effort"; a leaf the schema declares nullable
    keeps the null reading."""
    from hermes_cli.config import _MISSING, DEFAULT_CONFIG, _split_key_path, cfg_get

    parts = _split_key_path(key)
    is_effort = parts[-1] == "reasoning_effort" or (len(parts) > 1 and parts[-2] == "reasoning_overrides")
    return is_effort and cfg_get(DEFAULT_CONFIG, *parts, default=_MISSING) is _MISSING
