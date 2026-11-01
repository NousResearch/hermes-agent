"""Normalization for per-turn loop limits (KAN-319).

``max_steps`` and ``max_iterations`` reach the agent from constructor args,
``hermes-routing.config.yaml`` and the delegation config. YAML can yield
``None``, strings, booleans or non-positive numbers; comparing those against
an ``int`` counter raised ``TypeError: '>=' not supported between 'int' and
'NoneType'`` in the turn finalizer. Every consumer goes through these helpers
so the loop and the finalizer agree on what "no cap" means.
"""

from __future__ import annotations

import sys
from typing import Any, Optional


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def normalize_max_steps(value: Any) -> Optional[int]:
    """Positive ``int`` cap, or ``None`` meaning "no step cap".

    ``None``, booleans, unparsable values and values ``<= 0`` all mean no cap.
    """
    return _coerce_positive_int(value)


def normalize_max_iterations(value: Any, default: int = sys.maxsize) -> int:
    """Positive ``int`` iteration cap; invalid/missing values fall back to *default*."""
    number = _coerce_positive_int(value)
    return default if number is None else number


def agent_max_steps(agent: Any) -> Optional[int]:
    """Normalized ``max_steps`` of *agent* (tolerates agents without the attribute)."""
    return normalize_max_steps(getattr(agent, "max_steps", None))
