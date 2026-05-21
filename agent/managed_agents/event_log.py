"""Managed Agents event log helpers.

This module keeps the Managed Agents audit surface on top of the existing
Hermes SQLite EventLog, so delegation events share one replay chain.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.session_event_log import EventLog, SessionEvent

EVENT_POLICY_EVALUATED = "policy_evaluated"
EVENT_TASK_DELEGATED = "task_delegated"
EVENT_TASK_RESULT_RECEIVED = "task_result_received"
EVENT_TOOL_PERMISSION_DENIED = "tool_permission_denied"
EVENT_REVIEW_REQUESTED = "review_requested"
EVENT_REVIEW_COMPLETED = "review_completed"

MANAGED_EVENT_TYPES = frozenset(
    {
        EVENT_POLICY_EVALUATED,
        EVENT_TASK_DELEGATED,
        EVENT_TASK_RESULT_RECEIVED,
        EVENT_TOOL_PERMISSION_DENIED,
        EVENT_REVIEW_REQUESTED,
        EVENT_REVIEW_COMPLETED,
    }
)

SECRET_KEY_MARKERS = ("key", "token", "secret", "password", "credential")
REDACTED = "[REDACTED]"


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in SECRET_KEY_MARKERS):
                redacted[key_text] = REDACTED
            else:
                redacted[key_text] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secrets(item) for item in value]
    return value


@dataclass(slots=True)
class ManagedAgentEventLog:
    """Thin Managed Agents wrapper around the shared Hermes EventLog."""

    event_log: EventLog

    @classmethod
    def from_db_path(cls, db_path: str | Path) -> "ManagedAgentEventLog":
        return cls(EventLog(db_path=Path(db_path)))

    def close(self) -> None:
        self.event_log.close()

    def log_policy_evaluated(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_id: str,
        risk_level: str,
        decision: str,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionEvent:
        return self._log_managed_event(
            task_id,
            session_id,
            EVENT_POLICY_EVALUATED,
            {
                "agent_id": agent_id,
                "risk_level": risk_level,
                "decision": decision,
                "reason": reason,
                "metadata": dict(metadata or {}),
            },
        )

    def log_task_delegated(
        self,
        *,
        task_id: str,
        session_id: str,
        from_agent: str,
        to_agent: str,
        risk_level: str = "",
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionEvent:
        return self._log_managed_event(
            task_id,
            session_id,
            EVENT_TASK_DELEGATED,
            {
                "from": from_agent,
                "to": to_agent,
                "risk_level": risk_level,
                "reason": reason,
                "metadata": dict(metadata or {}),
            },
        )

    def log_task_result_received(
        self,
        *,
        task_id: str,
        session_id: str,
        from_agent: str,
        status: str,
        summary: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionEvent:
        return self._log_managed_event(
            task_id,
            session_id,
            EVENT_TASK_RESULT_RECEIVED,
            {
                "from": from_agent,
                "status": status,
                "summary": summary[:300],
                "metadata": dict(metadata or {}),
            },
        )

    def log_tool_permission_denied(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_id: str,
        tool_name: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionEvent:
        return self._log_managed_event(
            task_id,
            session_id,
            EVENT_TOOL_PERMISSION_DENIED,
            {
                "agent_id": agent_id,
                "tool_name": tool_name,
                "reason": reason,
                "metadata": dict(metadata or {}),
            },
        )

    def log_review_requested(
        self,
        *,
        task_id: str,
        session_id: str,
        reviewer: str,
        subject_agent: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionEvent:
        return self._log_managed_event(
            task_id,
            session_id,
            EVENT_REVIEW_REQUESTED,
            {
                "reviewer": reviewer,
                "subject_agent": subject_agent,
                "metadata": dict(metadata or {}),
            },
        )

    def log_review_completed(
        self,
        *,
        task_id: str,
        session_id: str,
        reviewer: str,
        decision: str,
        summary: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionEvent:
        return self._log_managed_event(
            task_id,
            session_id,
            EVENT_REVIEW_COMPLETED,
            {
                "reviewer": reviewer,
                "decision": decision,
                "summary": summary[:300],
                "metadata": dict(metadata or {}),
            },
        )

    def get_timeline(self, task_id: str) -> list[dict[str, Any]]:
        return [
            event
            for event in self.event_log.get_events_for_task(task_id)
            if event["type"] in MANAGED_EVENT_TYPES
        ]

    def replay_task(self, task_id: str) -> list[dict[str, Any]]:
        return self.get_timeline(task_id)

    def export_audit_report(self, task_id: str) -> dict[str, Any]:
        events = self.get_timeline(task_id)
        final_status = "unknown"
        for event in events:
            if event["type"] == EVENT_TASK_RESULT_RECEIVED:
                final_status = event.get("payload", {}).get("status", final_status)
            elif event["type"] == EVENT_REVIEW_COMPLETED:
                final_status = event.get("payload", {}).get("decision", final_status)

        return {
            "task_id": task_id,
            "event_count": len(events),
            "events": events,
            "delegations": self._events_of_type(events, EVENT_TASK_DELEGATED),
            "reviews": [
                event
                for event in events
                if event["type"] in {EVENT_REVIEW_REQUESTED, EVENT_REVIEW_COMPLETED}
            ],
            "permission_denials": self._events_of_type(
                events,
                EVENT_TOOL_PERMISSION_DENIED,
            ),
            "final_status": final_status,
        }

    def _log_managed_event(
        self,
        task_id: str,
        session_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=event_type,
            timestamp=time.time(),
            source="system",
            payload=_redact_secrets(dict(payload)),
        )
        self.event_log.write_event(event)
        return event

    @staticmethod
    def _events_of_type(
        events: list[dict[str, Any]],
        event_type: str,
    ) -> list[dict[str, Any]]:
        return [event for event in events if event["type"] == event_type]
