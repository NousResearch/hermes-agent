"""QQBot config coercion shared by the adapter."""

from __future__ import annotations

from typing import Any


def coerce_list(value: Any) -> list[str]:
    """Coerce a comma-separated string / list / tuple / set / scalar into a trimmed string list."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else value if isinstance(value, (list, tuple, set)) else [value]
    return [s for s in (str(item).strip() for item in items) if s]
