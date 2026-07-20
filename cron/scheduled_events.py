"""Durable, host-neutral one-shot events for plugin wakeups."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cron.jobs import _jobs_lock
from hermes_constants import get_hermes_home
from utils import atomic_replace


def _default_path() -> Path:
    return get_hermes_home() / "cron" / "scheduled_events.json"


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("due_at must include a timezone")
    return parsed.astimezone(timezone.utc)


class ScheduledEventStore:
    """Atomic snapshot store for typed deadlines without prompts or domain policy."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()

    def upsert(
        self,
        *,
        event_id: str,
        subject_id: str,
        due_at: str,
        event_type: str,
        correlation_id: str = "",
    ) -> dict[str, Any]:
        identity = {
            "event_id": self._required(event_id, "event_id"),
            "subject_id": self._required(subject_id, "subject_id"),
            "due_at": _parse_timestamp(due_at).isoformat(),
            "event_type": self._required(event_type, "event_type"),
            "correlation_id": str(correlation_id or ""),
        }
        with _jobs_lock():
            events = self._load()
            current = next(
                (event for event in events if event.get("event_id") == identity["event_id"]),
                None,
            )
            comparable = {
                key: current.get(key) for key in identity
            } if current is not None else None
            if current is not None and comparable == identity and current.get("status") == "pending":
                return dict(current)
            generation = int(current.get("generation", 0)) + 1 if current else 1
            saved = {**identity, "generation": generation, "status": "pending"}
            events = [
                saved if event.get("event_id") == identity["event_id"] else event
                for event in events
            ]
            if current is None:
                events.append(saved)
            self._save(events)
            return dict(saved)

    def cancel(self, event_id: str) -> bool:
        with _jobs_lock():
            events = self._load()
            for event in events:
                if event.get("event_id") != event_id or event.get("status") != "pending":
                    continue
                event["status"] = "cancelled"
                self._save(events)
                return True
            return False

    def snapshot(self, subject_id: str | None = None) -> list[dict[str, Any]]:
        with _jobs_lock():
            events = self._load()
        if subject_id is not None:
            events = [event for event in events if event.get("subject_id") == subject_id]
        return sorted(events, key=lambda event: (event.get("due_at", ""), event["event_id"]))

    def claim_due(
        self, *, now: datetime | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with _jobs_lock():
            events = self._load()
            due = [
                event
                for event in events
                if event.get("status") == "pending"
                and _parse_timestamp(event["due_at"]) <= current_time
            ]
            due.sort(key=lambda event: (event["due_at"], event["event_id"]))
            if limit is not None:
                due = due[: max(0, limit)]
            claimed_at = current_time.isoformat()
            for event in due:
                event["status"] = "claimed"
                event["claimed_at"] = claimed_at
                event["claim_id"] = uuid.uuid4().hex
            if due:
                self._save(events)
        return [dict(event) for event in due]

    @staticmethod
    def _required(value: str, name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{name} must not be blank")
        return normalized

    def _load(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
            raise RuntimeError("scheduled event store is corrupt")
        return [dict(event) for event in payload["events"] if isinstance(event, dict)]

    def _save(self, events: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".scheduled-events-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "events": events}, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(temporary, self.path)
        finally:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass


def dispatch_due_scheduled_events(
    *,
    store: ScheduledEventStore | None = None,
    deliver: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    adapters: Any = None,
    loop: Any = None,
) -> int:
    """Claim and notify due events only when a plugin consumer is ready."""
    from hermes_cli.plugins import has_hook, invoke_hook

    if not has_hook("scheduled_event_due"):
        return 0
    claimed = (store or ScheduledEventStore()).claim_due()
    for event in claimed:
        results = invoke_hook(
            "scheduled_event_due",
            event_id=event["event_id"],
            subject_id=event["subject_id"],
            due_at=event["due_at"],
            event_type=event["event_type"],
            correlation_id=event["correlation_id"],
            generation=event["generation"],
            claim_id=event["claim_id"],
        )
        request = next(
            (
                result
                for result in results
                if _valid_outreach_request(result, event["subject_id"])
            ),
            None,
        )
        if request is None:
            continue
        receipt = (deliver or _deliver_outreach)(event, request, adapters=adapters, loop=loop) if deliver is None else deliver(event, request)
        invoke_hook(
            "scheduled_outreach_delivery",
            event_id=event["event_id"],
            subject_id=event["subject_id"],
            generation=event["generation"],
            claim_id=event["claim_id"],
            action_id=request["action_id"],
            delivery_attempt_id=request["delivery_attempt_id"],
            status=str(receipt.get("status") or "failed"),
            targets=list(receipt.get("targets") or []),
            error=receipt.get("error"),
        )
    return len(claimed)


def _valid_outreach_request(value: object, subject_id: str) -> bool:
    if not isinstance(value, dict) or value.get("subject_id") != subject_id:
        return False
    required = ("action_id", "delivery_attempt_id", "content")
    if not all(isinstance(value.get(key), str) and value[key].strip() for key in required):
        return False
    return len(value["content"]) <= 4000


def _deliver_outreach(
    event: dict[str, Any],
    request: dict[str, Any],
    *,
    adapters: Any = None,
    loop: Any = None,
) -> dict[str, Any]:
    from cron.scheduler import _deliver_result_with_receipt

    job = {
        "id": event["event_id"],
        "name": event["event_type"],
        "deliver": "origin",
        "origin": None,
        "response_mode": "text_only",
        "metadata": {
            "action_id": request["action_id"],
            "delivery_attempt_id": request["delivery_attempt_id"],
            "subject_id": request["subject_id"],
        },
    }
    return _deliver_result_with_receipt(
        job, request["content"], adapters=adapters, loop=loop
    ).as_dict()


__all__ = ["ScheduledEventStore", "dispatch_due_scheduled_events"]
