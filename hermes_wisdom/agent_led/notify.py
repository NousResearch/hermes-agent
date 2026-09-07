"""Legacy recommendation fixtures and local JSON snapshots.

Runtime delivery uses WisdomMediation and its transactional receipt outbox.
These old event targets are not consent authority; compatibility handlers only
open fresh controls. The legacy sender is disabled to prevent duplicate paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from .schemas import CandidateRecommendation, RecipientRecommendation
from .templates import RenderedNotice, render_share, render_teammate, render_update

logger = logging.getLogger(__name__)

EventType = Literal[
    "wisdom.agent.share_candidate",
    "wisdom.agent.teammate_published",
    "wisdom.agent.update_available",
]

STALE_ACTION_MESSAGE = (
    "This suggestion has expired or was already handled. "
    "Use /wisdom to browse current skills."
)


@dataclass(frozen=True)
class AllowedAction:
    id: str
    label: str
    target: str  # opaque; the adapter/CLI resolves it, never the agent
    primary: bool = False


@dataclass
class RecommendationEvent:
    event_type: EventType
    organization_id: str
    recipient_id: str
    skill_id: str
    skill_version: int | None
    title: str
    explanation: str
    evidence_summary: str
    safety_summary: str
    allowed_actions: list[AllowedAction]
    expires_at: str
    dedup_key: str
    rendering_hints: dict[str, Any] = field(default_factory=dict)
    notice: RenderedNotice | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "organization_id": self.organization_id,
            "recipient_id": self.recipient_id,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "title": self.title,
            "explanation": self.explanation,
            "evidence_summary": self.evidence_summary,
            "safety_summary": self.safety_summary,
            "allowed_actions": [
                {"id": a.id, "label": a.label, "target": a.target, "primary": a.primary}
                for a in self.allowed_actions
            ],
            "expires_at": self.expires_at,
            "dedup_key": self.dedup_key,
            "rendering_hints": dict(self.rendering_hints),
            "notice": self.notice.as_dict() if self.notice else None,
        }

    def is_expired(self, at: datetime | None = None) -> bool:
        now = at or datetime.now(timezone.utc)
        try:
            return now >= datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True


def make_dedup_key(*parts: Any) -> str:
    digest = hashlib.sha256(
        json.dumps([str(p) for p in parts], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:32]


def action_target(dedup_key: str, action_id: str) -> str:
    """Opaque target bound to one event; ``wa:`` marks agent-led callbacks."""
    return f"wa:{action_id}:{dedup_key}"


def parse_action_target(target: str) -> tuple[str, str] | None:
    parts = target.split(":", 2)
    if len(parts) != 3 or parts[0] != "wa":
        return None
    return parts[1], parts[2]


def _expiry(ttl_hours: int, at: datetime | None) -> str:
    now = at or datetime.now(timezone.utc)
    return (now + timedelta(hours=ttl_hours)).isoformat()


def share_candidate_event(
    rec: CandidateRecommendation,
    *,
    organization_id: str,
    recipient_id: str,
    checks_passed: bool | None = None,
    failed_checks: list[str] | None = None,
    ttl_hours: int = 24 * 7,
    at: datetime | None = None,
) -> RecommendationEvent:
    dedup = make_dedup_key("share", organization_id, recipient_id, rec.skill_name, rec.content_hash)
    targets = {aid: action_target(dedup, aid) for aid in rec.allowed_actions}
    notice = render_share(
        editorial_name=rec.editorial_name,
        description=rec.one_line_description,
        count_7d=rec.evidence.invocation_count,
        specific_work=rec.how_user_relied_on_it,
        audience=rec.audience,
        reason=rec.why_coworkers_benefit,
        checks_passed=checks_passed,
        failed_checks=failed_checks,
        window_days=rec.evidence.window_days,
        targets=targets,
    )
    return RecommendationEvent(
        event_type="wisdom.agent.share_candidate",
        organization_id=organization_id,
        recipient_id=recipient_id,
        skill_id=rec.skill_name,
        skill_version=None,
        title=notice.title,
        explanation=rec.what_it_does,
        evidence_summary=(
            f"{rec.evidence.invocation_count} uses on {rec.evidence.days_used} days "
            f"in the last {rec.evidence.window_days} days"
        ),
        safety_summary="Local checks complete" if checks_passed is True else
        "Local checks found issues" if checks_passed is False else "Local checks unavailable",
        allowed_actions=[
            AllowedAction(a.id, a.label, a.target or "", a.primary)
            for a in notice.actions
            if a.id in rec.allowed_actions
        ],
        expires_at=_expiry(ttl_hours, at),
        dedup_key=dedup,
        rendering_hints={**notice.hints, "content_hash": rec.content_hash},
        notice=notice,
    )


def teammate_event(
    rec: RecipientRecommendation,
    *,
    organization_id: str,
    recipient_id: str,
    popular_threshold: int = 10,
    ttl_hours: int = 24 * 7,
    at: datetime | None = None,
) -> RecommendationEvent:
    if not rec.relevant:
        raise ValueError("cannot build a teammate event for a non-relevant recommendation")
    dedup = make_dedup_key("teammate", organization_id, recipient_id, rec.skill_id, rec.version)
    targets = {aid: action_target(dedup, aid) for aid in rec.allowed_actions}
    count = rec.evidence.invocation_count if rec.evidence else 0
    notice = render_teammate(
        editorial_name=rec.editorial_name or "",
        outcome_one_liner=rec.outcome_one_liner or "",
        publisher_name=rec.publisher_name or "",
        specific_work=rec.publisher_uses_it_for or "",
        count_7d=count,
        recipient_reason=rec.why_it_helps_recipient or "",
        installation_count=rec.installation_count,
        popular_threshold=popular_threshold,
        window_days=rec.evidence.window_days if rec.evidence else 7,
        targets=targets,
    )
    return RecommendationEvent(
        event_type="wisdom.agent.teammate_published",
        organization_id=organization_id,
        recipient_id=recipient_id,
        skill_id=rec.skill_id,
        skill_version=rec.version,
        title=notice.title,
        explanation=rec.outcome_one_liner or "",
        evidence_summary=f"{count} uses by publisher in the last 7 days",
        safety_summary=rec.safety_summary or "Security check complete",
        allowed_actions=[
            AllowedAction(a.id, a.label, a.target or "", a.primary)
            for a in notice.actions
            if a.id in rec.allowed_actions
        ],
        expires_at=_expiry(ttl_hours, at),
        dedup_key=dedup,
        rendering_hints=dict(notice.hints),
        notice=notice,
    )


def update_event(
    *,
    organization_id: str,
    recipient_id: str,
    skill_id: str,
    skill_name: str,
    version: int,
    summary: str,
    reason: str,
    setup_impact: str | None,
    ttl_hours: int = 24 * 7,
    at: datetime | None = None,
) -> RecommendationEvent:
    dedup = make_dedup_key("update", organization_id, recipient_id, skill_id, version)
    targets = {aid: action_target(dedup, aid) for aid in ("update", "view_changes", "mute")}
    notice = render_update(
        skill_name=skill_name, summary=summary, reason=reason, setup_impact=setup_impact, targets=targets
    )
    return RecommendationEvent(
        event_type="wisdom.agent.update_available",
        organization_id=organization_id,
        recipient_id=recipient_id,
        skill_id=skill_id,
        skill_version=version,
        title=notice.title,
        explanation=summary,
        evidence_summary="",
        safety_summary=f"Setup impact: {setup_impact or 'None'}",
        allowed_actions=[AllowedAction(a.id, a.label, a.target or "", a.primary) for a in notice.actions],
        expires_at=_expiry(ttl_hours, at),
        dedup_key=dedup,
        rendering_hints=dict(notice.hints),
        notice=notice,
    )


# ---------------------------------------------------------------------------
# Delivery ledger: idempotent, retried, resolvable for stale buttons
# ---------------------------------------------------------------------------

class DeliveryError(RuntimeError):
    pass


class DeliveryLedger:
    """JSON ledger keyed by dedup key; safe across retries and restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        return raw if isinstance(raw, dict) else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, dedup_key: str) -> dict[str, Any] | None:
        value = self._data.get(dedup_key)
        return dict(value) if isinstance(value, dict) else None

    def delivered(self, dedup_key: str) -> bool:
        record = self.get(dedup_key)
        return bool(record and record.get("state") == "delivered")

    def record(self, event: RecommendationEvent, *, state: str, detail: str | None = None) -> None:
        with self._lock:
            existing = self._data.get(event.dedup_key) or {}
            self._data[event.dedup_key] = {
                **existing,
                "event": event.as_dict(),
                "state": state,
                "detail": detail,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "attempts": int(existing.get("attempts", 0)) + (1 if state != "delivered" else 0),
            }
            self._save()

    def resolve_action(self, target: str, *, at: datetime | None = None) -> dict[str, Any]:
        """Resolve an opaque action target; stale/unknown targets get a clear message."""
        parsed = parse_action_target(target)
        if parsed is None:
            return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
        action_id, dedup_key = parsed
        record = self.get(dedup_key)
        if not record:
            return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
        event = record.get("event") or {}
        actions = {a.get("id"): a for a in event.get("allowed_actions", [])}
        try:
            expired = (at or datetime.now(timezone.utc)) >= datetime.fromisoformat(event.get("expires_at", ""))
        except ValueError:
            expired = True
        if record.get("state") == "acted" or expired or action_id not in actions:
            return {"ok": False, "stale": True, "message": STALE_ACTION_MESSAGE}
        return {"ok": True, "stale": False, "action": action_id, "event": event}

    def mark_acted(self, dedup_key: str, action_id: str) -> bool:
        """Idempotent: returns False when the event was already acted on."""
        with self._lock:
            record = self._data.get(dedup_key)
            if not record or record.get("state") == "acted":
                return False
            record["state"] = "acted"
            record["acted_with"] = action_id
            record["acted_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            return True


Sender = Callable[[RecommendationEvent], Any]


def deliver(
    event: RecommendationEvent,
    *,
    sender: Sender,
    ledger: DeliveryLedger,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compatibility failure, never a second send path around native ownership."""
    raise DeliveryError("Legacy Wisdom delivery is retired; use the profile-owned mediation scheduler")
