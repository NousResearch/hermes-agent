"""Data types and validation helpers for trustworthy anticipation.

This module is deliberately inert: it defines typed config structures and
validation, but it does not schedule jobs, send messages, or call tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class AnticipationPermission(str, Enum):
    """Permission ceiling for proactive anticipation candidates."""

    SILENT_LOG = "silent_log"
    SUGGEST = "suggest"
    DRAFT = "draft"
    ASK_TO_EXECUTE = "ask_to_execute"
    EXECUTE_SAFE = "execute_safe"


_PERMISSION_RANK = {
    AnticipationPermission.SILENT_LOG: 0,
    AnticipationPermission.SUGGEST: 1,
    AnticipationPermission.DRAFT: 2,
    AnticipationPermission.ASK_TO_EXECUTE: 3,
    AnticipationPermission.EXECUTE_SAFE: 4,
}


@dataclass(frozen=True)
class NotificationBudget:
    max_per_day: int = 3
    min_minutes_between: int = 120


@dataclass(frozen=True)
class QuietHours:
    enabled: bool = False
    start: str = "22:00"
    end: str = "08:00"


@dataclass(frozen=True)
class AnticipationLoopConfig:
    name: str
    enabled: bool
    schedule: str
    permission: AnticipationPermission
    min_confidence: float = 0.70
    lookback_days: int = 14


@dataclass(frozen=True)
class AnticipationRuntimeConfig:
    """Flattened policy inputs for one loop decision.

    Keeping this flat makes the policy gate pure and easy to test. The code
    that later wires config.yaml into loops can translate nested config into
    this shape without dragging YAML-specific concerns into policy decisions.
    """

    enabled: bool
    loop_enabled: bool
    loop_permission: AnticipationPermission
    min_confidence: float
    quiet_hours_enabled: bool
    quiet_hours_start: str
    quiet_hours_end: str
    max_per_day: int
    min_minutes_between: int

    def __post_init__(self) -> None:
        if not isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("Anticipation runtime min_confidence must be between 0 and 1")
        if self.max_per_day < 0:
            raise ValueError("Anticipation runtime max_per_day must be >= 0")
        if self.min_minutes_between < 0:
            raise ValueError("Anticipation runtime min_minutes_between must be >= 0")


def parse_permission(value: str | AnticipationPermission) -> AnticipationPermission:
    """Parse an anticipation permission string into its enum value."""

    if isinstance(value, AnticipationPermission):
        return value
    try:
        return AnticipationPermission(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(permission.value for permission in AnticipationPermission)
        raise ValueError(
            f"Invalid anticipation permission {value!r}; expected one of: {allowed}"
        ) from exc


def permission_allows(ceiling: AnticipationPermission, requested: AnticipationPermission) -> bool:
    """Return whether *ceiling* permits *requested* proactive action."""

    return _PERMISSION_RANK[requested] <= _PERMISSION_RANK[ceiling]


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse config booleans without treating every non-empty string as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def parse_loop_config(name: str, raw: Mapping[str, Any] | None) -> AnticipationLoopConfig:
    """Parse one loop config mapping into a typed config object."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"Anticipation loop {name!r} must be a mapping")

    min_confidence = float(raw.get("min_confidence", 0.70))
    if not isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError(f"Anticipation loop {name!r} min_confidence must be between 0 and 1")

    lookback_days = int(raw.get("lookback_days", 14))
    if lookback_days < 1:
        raise ValueError(f"Anticipation loop {name!r} lookback_days must be >= 1")

    return AnticipationLoopConfig(
        name=name,
        enabled=parse_bool(raw.get("enabled", False)),
        schedule=str(raw.get("schedule", "")),
        permission=parse_permission(raw.get("permission", AnticipationPermission.SUGGEST.value)),
        min_confidence=min_confidence,
        lookback_days=lookback_days,
    )
