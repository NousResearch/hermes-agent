"""Pure policy gate for trustworthy anticipation candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from math import isfinite
from typing import Literal

from agent.anticipation import (
    AnticipationPermission,
    AnticipationRuntimeConfig,
    permission_allows,
)

DecisionAction = Literal["skip", "silent_log", "suggest", "draft", "ask_to_execute", "execute_safe"]


@dataclass(frozen=True)
class AnticipationCandidate:
    loop_id: str
    title: str
    body: str
    confidence: float
    proposed_permission: AnticipationPermission
    dedupe_key: str
    created_at: datetime


@dataclass(frozen=True)
class AnticipationDecision:
    action: DecisionAction
    reason: str
    candidate: AnticipationCandidate


@dataclass(frozen=True)
class AnticipationDecisionHistory:
    """Minimal recent history needed for dedupe and notification budgets."""

    recent_dedupe_keys: dict[str, datetime] = field(default_factory=dict)
    notifications_today: int = 0
    last_notification_at: datetime | None = None


def decide_anticipation_action(
    candidate: AnticipationCandidate,
    config: AnticipationRuntimeConfig,
    history: AnticipationDecisionHistory,
    *,
    now: datetime,
) -> AnticipationDecision:
    """Decide whether a proactive candidate may proceed.

    This function is intentionally side-effect-free. It does not log, send,
    schedule, query memory, or mutate state. That keeps proactive behavior
    auditable and lets callers choose how to handle skip reasons.
    """

    if not config.enabled:
        return _skip(candidate, "anticipation_disabled")
    if not config.loop_enabled:
        return _skip(candidate, "loop_disabled")
    if not isfinite(candidate.confidence) or candidate.confidence < config.min_confidence:
        return _skip(candidate, "below_confidence_threshold")
    if not candidate.body.strip():
        return _skip(candidate, "empty_candidate_body")
    if _is_duplicate(candidate, history, config, now):
        return _skip(candidate, "duplicate_dedupe_key")
    is_notification = candidate.proposed_permission is not AnticipationPermission.SILENT_LOG
    if is_notification and history.notifications_today >= config.max_per_day:
        return _skip(candidate, "notification_budget_exhausted")
    if is_notification and _too_soon_since_last_notification(history, config, now):
        return _skip(candidate, "notification_budget_exhausted")
    if is_notification and config.quiet_hours_enabled and _inside_quiet_hours(
        now, config.quiet_hours_start, config.quiet_hours_end
    ):
        return _skip(candidate, "inside_quiet_hours")
    if not permission_allows(config.loop_permission, candidate.proposed_permission):
        return _skip(candidate, "permission_exceeds_ceiling")

    return AnticipationDecision(
        action=candidate.proposed_permission.value,  # type: ignore[return-value]
        reason="passed",
        candidate=candidate,
    )


def _skip(candidate: AnticipationCandidate, reason: str) -> AnticipationDecision:
    return AnticipationDecision(action="skip", reason=reason, candidate=candidate)


def _is_duplicate(
    candidate: AnticipationCandidate,
    history: AnticipationDecisionHistory,
    config: AnticipationRuntimeConfig,
    now: datetime,
) -> bool:
    last_seen = history.recent_dedupe_keys.get(candidate.dedupe_key)
    if last_seen is None:
        return False
    return now - last_seen < timedelta(minutes=config.min_minutes_between)


def _too_soon_since_last_notification(
    history: AnticipationDecisionHistory,
    config: AnticipationRuntimeConfig,
    now: datetime,
) -> bool:
    if history.last_notification_at is None:
        return False
    return now - history.last_notification_at < timedelta(minutes=config.min_minutes_between)


def _inside_quiet_hours(now: datetime, start_text: str, end_text: str) -> bool:
    start = _parse_hhmm(start_text)
    end = _parse_hhmm(end_text)
    current = now.timetz().replace(tzinfo=None)

    if start == end:
        return True
    if start < end:
        return start <= current < end
    return current >= start or current < end


def _parse_hhmm(value: str) -> time:
    try:
        hour_text, minute_text = value.split(":", 1)
        return time(hour=int(hour_text), minute=int(minute_text))
    except Exception as exc:
        raise ValueError(f"Invalid quiet-hours time {value!r}; expected HH:MM") from exc
