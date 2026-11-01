"""Timestamp parsing shared by credential-pool status, refresh, and cooldown bookkeeping."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse of epoch seconds / epoch ms / ISO-8601 into epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
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
        except ValueError:
            return None
    return None


def _singleton_predates_entry(state: Any, entry: PooledCredential) -> bool:
    """True only when the auth.json singleton is PROVABLY older than *entry*.

    Both sides stamp ``last_refresh`` on every successful rotation. When
    either side lacks a parseable stamp this returns False (cannot prove),
    which keeps the historical adopt-on-difference behavior (#70111) intact
    for legacy writers.
    """
    entry_ts = _parse_absolute_timestamp(entry.last_refresh)
    if entry_ts is None:
        return False
    state_ts = _parse_absolute_timestamp(
        state.get("last_refresh") if isinstance(state, dict) else None
    )
    if state_ts is None:
        return False
    return state_ts < entry_ts
