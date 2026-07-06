"""Durable action memory for the Torben Signal COO operator."""

from __future__ import annotations

import json
import os
import re
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

HANDLE_RE = re.compile(r"\b(?P<handle>[A-Z]{2,8}-\d{8}-\d{3,5})\b")
ACTION_ALIAS_RE = re.compile(
    r"\b(?P<action>draft|source|hold|post|monitor|learn)\s+(?P<rank>[1-9][0-9]*)\b",
    re.IGNORECASE,
)
OPEN_STATUSES = {"drafted", "staged", "approval_required", "approved", "executing"}
TORBEN_LEDGER_STEM = "torben-action-ledger"
TORBEN_LEDGER_JSON = f"{TORBEN_LEDGER_STEM}.json"
TORBEN_LEDGER_JOURNAL = f"{TORBEN_LEDGER_STEM}.jsonl"
TORBEN_LEDGER_SNAPSHOT = f"{TORBEN_LEDGER_STEM}.snapshot.json"


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


class LedgerMigrationHoldError(RuntimeError):
    """Raised when a journal write is attempted during a migration hold."""


def default_torben_ledger_path(state_dir: str | os.PathLike[str]) -> Path:
    return Path(state_dir) / TORBEN_LEDGER_JOURNAL


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
    """Durable action ledger with stable handle resolution.

    Generic ``.json`` paths keep the historical array rewrite behavior for
    tests and non-Torben callers. ``.jsonl`` paths use an append-only journal
    plus a compacted snapshot; the journal is always the source of truth.
    """

    def __init__(self, path: str | os.PathLike[str]):
        requested = Path(path)
        if requested.name == TORBEN_LEDGER_JSON:
            sibling_journal = requested.with_name(TORBEN_LEDGER_JOURNAL)
            if sibling_journal.exists():
                requested = sibling_journal
        self.path = requested
        self._journal_mode = self.path.suffix == ".jsonl"

    @property
    def snapshot_path(self) -> Path:
        if self._journal_mode:
            return self.path.with_name(TORBEN_LEDGER_SNAPSHOT)
        return self.path

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    @property
    def hold_path(self) -> Path:
        return self.path.with_name("torben-action-ledger.migration-hold")

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    def _assert_writable(self) -> None:
        if self._journal_mode and self.hold_path.exists():
            raise LedgerMigrationHoldError(
                f"Action ledger writes are held during migration: {self.hold_path}"
            )

    def load(self) -> list[ActionRecord]:
        if not self.path.exists():
            return []
        if self._journal_mode:
            return self._load_journal()
        payload = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        if not isinstance(payload, list):
            raise ValueError(f"Action ledger must contain a JSON list: {self.path}")
        return [ActionRecord.from_dict(item) for item in payload]

    def save(self, records: Iterable[ActionRecord]) -> None:
        if self._journal_mode:
            with self._file_lock():
                self._assert_writable()
                current = {record.handle: record.to_dict() for record in self._load_journal()}
                changed: list[ActionRecord] = []
                for record in records:
                    payload = record.to_dict()
                    if current.get(record.handle) != payload:
                        changed.append(record)
                if changed:
                    self._append_journal_records(changed)
                    self._write_snapshot(self._load_journal())
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in records]
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def next_handle(self, scope: str, now: datetime | None = None) -> str:
        if self._journal_mode:
            with self._file_lock():
                return self._next_handle(scope, now=now)
        return self._next_handle(scope, now=now)

    def _next_handle(self, scope: str, now: datetime | None = None) -> str:
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
        if self._journal_mode:
            with self._file_lock():
                self._assert_writable()
                record = ActionRecord(
                    handle=self._next_handle(scope, now),
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
                self._append_journal_records([record])
                self._write_snapshot(self._load_journal())
                return record

        record = ActionRecord(
            handle=self._next_handle(scope, now),
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
        records = self.load()
        records.append(record)
        self.save(records)
        return record

    def _load_journal(self) -> list[ActionRecord]:
        if not self.path.exists():
            return []
        by_handle: dict[str, ActionRecord] = {}
        order: list[str] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Journal line {line_number} must be a JSON object: {self.path}")
            record = ActionRecord.from_dict(payload)
            if record.handle not in by_handle:
                order.append(record.handle)
            by_handle[record.handle] = record
        return [by_handle[handle] for handle in order if handle in by_handle]

    def _append_journal_records(self, records: Iterable[ActionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_snapshot(self, records: Iterable[ActionRecord] | None = None) -> None:
        if not self._journal_mode:
            return
        payload = [record.to_dict() for record in (records if records is not None else self._load_journal())]
        tmp = self.snapshot_path.with_name(f".{self.snapshot_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.snapshot_path)

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
