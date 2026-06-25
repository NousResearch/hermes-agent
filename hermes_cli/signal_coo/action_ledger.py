"""Durable action memory for the Torben Signal COO operator."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

HANDLE_RE = re.compile(r"\b(?P<handle>[A-Z]{2,8}-\d{8}-\d{3,5})\b")
ACTION_ALIAS_RE = re.compile(
    r"\b(?P<action>draft|source|hold|post|monitor|learn)\s+(?P<rank>[1-9][0-9]*)\b",
    re.IGNORECASE,
)
OPEN_STATUSES = {"drafted", "staged", "approval_required", "approved", "executing"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ActionRecord:
    handle: str
    scope: str
    summary: str
    evidence_ids: list[str]
    allowed_next_actions: list[str]
    status: str
    risk_class: str = "low"
    outbound_message_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    user_visible_summary: str | None = None
    executor_state: dict[str, Any] = field(default_factory=dict)
    resolution_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionRecord":
        return cls(
            handle=str(payload["handle"]),
            scope=str(payload["scope"]),
            summary=str(payload["summary"]),
            evidence_ids=list(payload.get("evidence_ids") or []),
            allowed_next_actions=list(payload.get("allowed_next_actions") or []),
            status=str(payload.get("status") or "staged"),
            risk_class=str(payload.get("risk_class") or "low"),
            outbound_message_id=payload.get("outbound_message_id"),
            created_at=parse_time(payload.get("created_at")) or utc_now(),
            expires_at=parse_time(payload.get("expires_at")),
            user_visible_summary=payload.get("user_visible_summary"),
            executor_state=dict(payload.get("executor_state") or {}),
            resolution_history=list(payload.get("resolution_history") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "scope": self.scope,
            "summary": self.summary,
            "evidence_ids": self.evidence_ids,
            "allowed_next_actions": self.allowed_next_actions,
            "status": self.status,
            "risk_class": self.risk_class,
            "outbound_message_id": self.outbound_message_id,
            "created_at": format_time(self.created_at),
            "expires_at": format_time(self.expires_at),
            "user_visible_summary": self.user_visible_summary,
            "executor_state": self.executor_state,
            "resolution_history": self.resolution_history,
        }

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or utc_now()).astimezone(timezone.utc) >= self.expires_at

    def is_open(self, now: datetime | None = None) -> bool:
        return self.status in OPEN_STATUSES and not self.is_expired(now)


@dataclass
class ReplyResolution:
    status: str
    reply_text: str
    record: ActionRecord | None = None
    matched_handle: str | None = None
    candidates: list[ActionRecord] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reply_text": self.reply_text,
            "matched_handle": self.matched_handle,
            "record": self.record.to_dict() if self.record else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason": self.reason,
        }


class ActionLedger:
    """JSON-backed action ledger with stable handle resolution."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def load(self) -> list[ActionRecord]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        if not isinstance(payload, list):
            raise ValueError(f"Action ledger must contain a JSON list: {self.path}")
        return [ActionRecord.from_dict(item) for item in payload]

    def save(self, records: Iterable[ActionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in records]
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def next_handle(self, scope: str, now: datetime | None = None) -> str:
        now = (now or utc_now()).astimezone(timezone.utc)
        prefix = f"{scope.upper()}-{now:%Y%m%d}-"
        highest = 0
        for record in self.load():
            if record.handle.startswith(prefix):
                try:
                    highest = max(highest, int(record.handle.rsplit("-", 1)[1]))
                except ValueError:
                    continue
        return f"{prefix}{highest + 1:03d}"

    def add_action(
        self,
        *,
        scope: str,
        summary: str,
        evidence_ids: Iterable[str] = (),
        allowed_next_actions: Iterable[str] = ("revise", "approve", "discard"),
        status: str = "staged",
        risk_class: str = "low",
        outbound_message_id: str | None = None,
        user_visible_summary: str | None = None,
        ttl_hours: int = 24,
        now: datetime | None = None,
        executor_state: dict[str, Any] | None = None,
    ) -> ActionRecord:
        now = (now or utc_now()).astimezone(timezone.utc)
        records = self.load()
        record = ActionRecord(
            handle=self.next_handle(scope, now),
            scope=scope.lower(),
            summary=summary,
            evidence_ids=list(evidence_ids),
            allowed_next_actions=list(allowed_next_actions),
            status=status,
            risk_class=risk_class,
            outbound_message_id=outbound_message_id,
            created_at=now,
            expires_at=now + timedelta(hours=ttl_hours) if ttl_hours else None,
            user_visible_summary=user_visible_summary or summary,
            executor_state=dict(executor_state or {}),
        )
        records.append(record)
        self.save(records)
        return record

    def get(self, handle: str) -> ActionRecord | None:
        normalized = handle.upper()
        for record in self.load():
            if record.handle.upper() == normalized:
                return record
        return None

    def recent_open(self, *, scope: str | None = None, now: datetime | None = None) -> list[ActionRecord]:
        records = [
            record
            for record in self.load()
            if record.is_open(now) and (scope is None or record.scope == scope.lower())
        ]
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def resolve_reply(self, reply_text: str, *, now: datetime | None = None) -> ReplyResolution:
        now = now or utc_now()
        match = HANDLE_RE.search(reply_text.upper())
        if match:
            handle = match.group("handle")
            record = self.get(handle)
            if record is None:
                return ReplyResolution(
                    status="not_found",
                    reply_text=reply_text,
                    matched_handle=handle,
                    reason="No action with that handle exists.",
                )
            if record.is_expired(now):
                return ReplyResolution(
                    status="expired",
                    reply_text=reply_text,
                    matched_handle=handle,
                    record=record,
                    reason="The action exists but its source context expired.",
                )
            if not record.is_open(now):
                return ReplyResolution(
                    status="closed",
                    reply_text=reply_text,
                    matched_handle=handle,
                    record=record,
                    reason=f"The action exists but is not open: {record.status}.",
                )
            return ReplyResolution(
                status="resolved",
                reply_text=reply_text,
                matched_handle=handle,
                record=record,
            )

        alias_match = ACTION_ALIAS_RE.search(reply_text)
        if alias_match:
            action = alias_match.group("action").lower()
            rank = int(alias_match.group("rank"))
            alias = f"{action} {rank}"
            candidates = [
                record
                for record in self.recent_open(scope="gtm", now=now)
                if _matches_action_alias(record, alias=alias, action=action, rank=rank)
            ]
            if not candidates:
                return ReplyResolution(
                    status="not_found",
                    reply_text=reply_text,
                    reason=f"No open GTM action matches '{alias}'.",
                )
            record = candidates[0]
            return ReplyResolution(
                status="resolved_alias",
                reply_text=reply_text,
                matched_handle=record.handle,
                record=record,
                reason=f"Resolved '{alias}' to the latest matching GTM action.",
            )

        candidates = self.recent_open(now=now)
        if not candidates:
            return ReplyResolution(
                status="not_found",
                reply_text=reply_text,
                reason="No open action is available for this reply.",
            )
        if len(candidates) == 1:
            return ReplyResolution(
                status="resolved_recent",
                reply_text=reply_text,
                record=candidates[0],
                reason="Resolved to the only open recent action.",
            )
        return ReplyResolution(
            status="ambiguous",
            reply_text=reply_text,
            candidates=candidates[:5],
            reason="More than one open action could match this reply.",
        )


def _matches_action_alias(record: ActionRecord, *, alias: str, action: str, rank: int) -> bool:
    state = record.executor_state or {}
    aliases = {str(value).strip().lower() for value in state.get("reply_aliases") or []}
    if alias in aliases:
        return True
    try:
        record_rank = int(state.get("radar_rank") or 0)
    except (TypeError, ValueError):
        return False
    if record_rank != rank:
        return False
    allowed = {str(value).strip().lower() for value in state.get("reply_actions") or []}
    return action in allowed
