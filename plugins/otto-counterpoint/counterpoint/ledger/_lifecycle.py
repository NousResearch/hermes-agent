# -*- coding: utf-8 -*-
"""Deterministic execution lifecycle built on the Phase 2 ledger."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ._contract import (
    EventEnvelope,
    EventType,
    LedgerConflictError,
    LedgerContractError,
)
from ._ledger import Ledger


class RunLifecycle:
    """Create and advance one correlated run without an external queue."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def create(
        self,
        *,
        project_id: str,
        linear_issue_id: Optional[str],
        run_id: str,
        agent_id: str,
        evidence_refs: Sequence[str] = (),
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        event = EventEnvelope.create(
            event_type=EventType.CREATED,
            project_id=project_id,
            linear_issue_id=linear_issue_id,
            run_id=run_id,
            agent_id=agent_id,
            occurred_at=_now(),
            idempotency_key=f"run:{run_id}:created",
            evidence_refs=evidence_refs,
            payload=payload,
            sequence=1,
        )
        return self.ledger.append(event)

    def start(self, run_id: str) -> dict[str, Any]:
        latest = self._latest(run_id)
        if latest["event_type"] == EventType.STARTED:
            return latest
        if latest["event_type"] not in {EventType.CREATED, EventType.BLOCKED}:
            raise LedgerContractError("run can only start from created or blocked")
        return self._append_next(latest, EventType.STARTED)

    def resume(self, run_id: str) -> dict[str, Any]:
        latest = self._latest(run_id)
        if latest["event_type"] == EventType.STARTED:
            return latest
        if latest["event_type"] != EventType.BLOCKED:
            raise LedgerContractError("only a blocked run can be resumed")
        return self._append_next(latest, EventType.STARTED)

    def block(
        self,
        run_id: str,
        reason: str,
        *,
        evidence_refs: Sequence[str] = (),
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        latest = self._latest(run_id)
        if latest["event_type"] == EventType.BLOCKED:
            if latest["blocked_reason"] == reason:
                return latest
            raise LedgerConflictError("blocked run was reused with a different reason")
        if latest["event_type"] not in {EventType.CREATED, EventType.STARTED}:
            raise LedgerContractError("run can only be blocked before it is terminal")
        return self._append_next(
            latest,
            EventType.BLOCKED,
            blocked_reason=reason,
            evidence_refs=evidence_refs,
            payload=payload,
        )

    def finish(
        self,
        run_id: str,
        terminal_state: str,
        *,
        evidence_refs: Sequence[str] = (),
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        latest = self._latest(run_id)
        if latest["event_type"] == EventType.FINISHED:
            if latest["terminal_state"] == terminal_state:
                return latest
            raise LedgerConflictError("terminal run was reused with a different result")
        if latest["event_type"] not in {EventType.STARTED, EventType.BLOCKED}:
            raise LedgerContractError("run must be started or blocked before finishing")
        return self._append_next(
            latest,
            EventType.FINISHED,
            terminal_state=terminal_state,
            evidence_refs=evidence_refs,
            payload=payload,
        )

    def retry(
        self,
        *,
        previous_run_id: str,
        new_run_id: str,
        agent_id: str,
        evidence_refs: Sequence[str] = (),
    ) -> dict[str, Any]:
        previous = self._latest(previous_run_id)
        if previous["event_type"] != EventType.FINISHED:
            raise LedgerContractError("only a terminal run can be retried")
        if new_run_id == previous_run_id:
            raise LedgerContractError("retry must use a new run_id")
        return self.create(
            project_id=previous["project_id"],
            linear_issue_id=previous["linear_issue_id"],
            run_id=new_run_id,
            agent_id=agent_id,
            evidence_refs=evidence_refs,
            payload={"retry_of": previous_run_id},
        )

    def _latest(self, run_id: str) -> dict[str, Any]:
        events = self.ledger.list_run(run_id)
        if not events:
            raise LedgerContractError(f"run does not exist: {run_id}")
        return events[-1]

    def _append_next(
        self,
        latest: dict[str, Any],
        next_type: str,
        *,
        blocked_reason: Optional[str] = None,
        terminal_state: Optional[str] = None,
        evidence_refs: Sequence[str] = (),
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        event = EventEnvelope.create(
            event_type=next_type,
            project_id=latest["project_id"],
            linear_issue_id=latest["linear_issue_id"],
            run_id=latest["run_id"],
            agent_id=latest["agent_id"],
            occurred_at=_now(),
            idempotency_key=(
                f"run:{latest['run_id']}:{next_type}:{latest['event_id']}"
            ),
            evidence_refs=evidence_refs,
            blocked_reason=blocked_reason,
            terminal_state=terminal_state,
            payload=payload,
            previous_event_id=latest["event_id"],
            sequence=latest["sequence"] + 1,
        )
        return self.ledger.append(event)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
