"""Timestamp parsing shared by credential-pool status, refresh, and cooldown bookkeeping."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse of epoch seconds / epoch ms / ISO-8601 into epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except OverflowError:  # a JSON integer past float range
            return None
        if numeric <= 0:
            return None
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw).timestamp()
        except (ValueError, OverflowError, OSError):
            # OSError: Windows' local-time conversion of a naive out-of-range value ("0001-01-01T00:00:00").
            return None
    return None
