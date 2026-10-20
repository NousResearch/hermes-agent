# -*- coding: utf-8 -*-
"""Deterministic, read-only projections over the Phase 2 event ledger."""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional

from ._contract import EventType, VALID_EVENT_TYPES, VALID_TERMINAL_STATES, VALID_TRANSITIONS
from ._ledger import Ledger

_TERMINAL_STATUSES = frozenset(VALID_TERMINAL_STATES)
_OBSERVED_STATUS = {
    EventType.CREATED: "created",
    EventType.STARTED: "running",
    EventType.BLOCKED: "blocked",
}


class ExecutionProjection:
    """Replay ledger events into deterministic executive query snapshots.

    The projection never mutates the ledger and intentionally omits raw payload
    values from query results. Payload keys remain visible as provenance without
    turning the executive surface into a secret or diagnostic dump.
    """

    def __init__(self, events: Iterable[Mapping[str, Any]]) -> None:
        self._runs = self._build_runs(events)

    @classmethod
    def from_ledger(cls, ledger: Ledger) -> "ExecutionProjection":
        return cls(ledger.list_events())

    @classmethod
    def from_events(cls, events: Iterable[Mapping[str, Any]]) -> "ExecutionProjection":
        return cls(events)

    def runs(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        records = []
        for record in self._runs.values():
            if project_id is not None and record["project_id"] != project_id:
                continue
            if status is not None and record["status"] != status:
                continue
            records.append(copy.deepcopy(record))
        records.sort(key=lambda item: (item["project_id"], item["run_id"]))
        return records

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        record = self._runs.get(run_id)
        return None if record is None else copy.deepcopy(record)

    def active_runs(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        return [
            record
            for record in self.runs(project_id)
            if record["status"] in {"created", "running"}
        ]

    def blocked_runs(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        return self.runs(project_id, status="blocked")

    def finished_runs(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        return [
            record
            for record in self.runs(project_id)
            if record["status"] in _TERMINAL_STATUSES
        ]

    def failed_runs(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        return self.runs(project_id, status="failed")

    def project_summary(self, project_id: str) -> dict[str, Any]:
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        records = self.runs(project_id)
        task_ids = {
            record["linear_issue_id"]
            for record in records
            if record["linear_issue_id"] is not None
        }
        registered_refs = sorted(
            {
                ref
                for record in records
                for ref in record["evidence_refs"]
            }
        )
        with_evidence = sum(bool(record["evidence_refs"]) for record in records)
        with_issue = sum(record["linear_issue_id"] is not None for record in records)
        return {
            "project_id": project_id,
            "task_count": len(task_ids),
            "run_count": len(records),
            "attempt_count": len(records),
            "in_progress_count": sum(
                record["status"] in {"created", "running"} for record in records
            ),
            "blocked_count": sum(record["status"] == "blocked" for record in records),
            "finished_count": sum(
                record["status"] in _TERMINAL_STATUSES for record in records
            ),
            "success_count": sum(record["status"] == "succeeded" for record in records),
            "failure_count": sum(record["status"] == "failed" for record in records),
            "cancelled_count": sum(record["status"] == "cancelled" for record in records),
            "inconsistent_count": sum(
                record["status"] == "inconsistent" for record in records
            ),
            "task_completion": "not_derived_from_run_status",
            "evidence": {
                "runs_with_evidence": with_evidence,
                "runs_without_evidence": len(records) - with_evidence,
                "registered_refs": registered_refs,
            },
            "coverage": {
                "runs_with_linear_issue": with_issue,
                "runs_without_linear_issue": len(records) - with_issue,
                "runs_with_agent": sum(bool(record["agent_id"]) for record in records),
                "runs_without_agent": sum(
                    not bool(record["agent_id"]) for record in records
                ),
                "inconsistencies": [
                    {
                        "run_id": record["run_id"],
                        "items": copy.deepcopy(record["inconsistencies"]),
                    }
                    for record in records
                    if record["inconsistencies"]
                ],
            },
        }

    def evidence_for_run(self, run_id: str) -> Optional[dict[str, Any]]:
        record = self._runs.get(run_id)
        if record is None:
            return None
        registrations = []
        for event in record["timeline"]:
            for ref in event["evidence_refs"]:
                registrations.append(
                    {
                        "ref": ref,
                        "event_id": event["event_id"],
                        "event_type": event["event_type"],
                        "sequence": event["sequence"],
                    }
                )
        return {
            "run_id": run_id,
            "refs": list(record["evidence_refs"]),
            "registrations": registrations,
            "consistent": record["consistent"],
        }

    @classmethod
    def _build_runs(
        cls, events: Iterable[Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        issues_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: dict[str, tuple[str, dict[str, Any]]] = {}

        for index, source in enumerate(events):
            event = copy.deepcopy(dict(source))
            raw_run_id = event.get("run_id")
            run_id = raw_run_id if isinstance(raw_run_id, str) and raw_run_id else f"__invalid-{index}"
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                event_id = f"__missing-event-id-{index}"
                event["event_id"] = event_id
                issues_by_run[run_id].append(
                    cls._issue("missing_event_id", "event has no event_id")
                )
            if event_id in seen:
                first_run_id, first_event = seen[event_id]
                if cls._event_fingerprint(first_event) != cls._event_fingerprint(event):
                    issue = cls._issue(
                        "duplicate_event_id_conflict",
                        f"event_id {event_id!r} has divergent records",
                        event_id=event_id,
                    )
                    issues_by_run[first_run_id].append(issue)
                    issues_by_run[run_id].append(copy.deepcopy(issue))
                continue
            seen[event_id] = (run_id, event)
            grouped[run_id].append(event)

        records: dict[str, dict[str, Any]] = {}
        for run_id, run_events in grouped.items():
            run_events.sort(key=cls._event_sort_key)
            issues = list(issues_by_run.get(run_id, []))
            cls._validate_run(run_id, run_events, issues)
            records[run_id] = cls._snapshot(run_id, run_events, issues)
        return records

    @classmethod
    def _validate_run(
        cls,
        run_id: str,
        events: list[dict[str, Any]],
        issues: list[dict[str, Any]],
    ) -> None:
        first = events[0]
        if first.get("event_type") != EventType.CREATED:
            issues.append(
                cls._issue(
                    "missing_created",
                    "run history must start with run.created",
                )
            )
        if first.get("sequence") != 1:
            issues.append(
                cls._issue(
                    "invalid_initial_sequence",
                    "run history must start at sequence 1",
                    sequence=first.get("sequence"),
                )
            )
        if first.get("previous_event_id") is not None:
            issues.append(
                cls._issue(
                    "unexpected_initial_previous_event",
                    "run.created must not have a previous_event_id",
                )
            )

        for previous, current in zip(events, events[1:]):
            if current.get("sequence") != previous.get("sequence", 0) + 1:
                issues.append(
                    cls._issue(
                        "sequence_gap",
                        "event sequence is not contiguous",
                        previous_sequence=previous.get("sequence"),
                        sequence=current.get("sequence"),
                    )
                )
            if current.get("previous_event_id") != previous.get("event_id"):
                issues.append(
                    cls._issue(
                        "previous_event_mismatch",
                        "previous_event_id does not point to the prior event",
                        event_id=current.get("event_id"),
                    )
                )
            if current.get("run_id") != run_id:
                issues.append(
                    cls._issue(
                        "run_identity_changed",
                        "event run_id differs from its projection group",
                        event_id=current.get("event_id"),
                    )
                )
            if current.get("project_id") != previous.get("project_id"):
                issues.append(
                    cls._issue(
                        "project_identity_changed",
                        "project_id changed inside one run",
                        event_id=current.get("event_id"),
                    )
                )
            if current.get("linear_issue_id") != previous.get("linear_issue_id"):
                issues.append(
                    cls._issue(
                        "task_identity_changed",
                        "linear_issue_id changed inside one run",
                        event_id=current.get("event_id"),
                    )
                )
            if current.get("agent_id") != previous.get("agent_id"):
                issues.append(
                    cls._issue(
                        "agent_identity_changed",
                        "agent_id changed inside one run",
                        event_id=current.get("event_id"),
                    )
                )
            allowed = VALID_TRANSITIONS.get(previous.get("event_type"), frozenset())
            if current.get("event_type") not in allowed:
                issues.append(
                    cls._issue(
                        "invalid_transition",
                        "event transition is not permitted by the contract",
                        previous_type=previous.get("event_type"),
                        event_type=current.get("event_type"),
                    )
                )

        for event in events:
            if event.get("event_type") not in VALID_EVENT_TYPES:
                issues.append(
                    cls._issue(
                        "unknown_event_type",
                        "event_type is outside the versioned contract",
                        event_type=event.get("event_type"),
                    )
                )
            if event.get("schema_version") != "v1":
                issues.append(
                    cls._issue(
                        "unknown_schema_version",
                        "event schema_version is not v1",
                        schema_version=event.get("schema_version"),
                    )
                )

    @classmethod
    def _snapshot(
        cls,
        run_id: str,
        events: list[dict[str, Any]],
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        first = events[0]
        last = events[-1]
        observed_status = cls._observed_status(last)
        if last.get("event_type") == EventType.FINISHED and last.get("terminal_state") not in VALID_TERMINAL_STATES:
            issues.append(
                cls._issue(
                    "invalid_terminal_state",
                    "run.finished has no valid terminal_state",
                    terminal_state=last.get("terminal_state"),
                )
            )
        normalized_issues = sorted(
            (copy.deepcopy(item) for item in issues),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
        evidence_refs = sorted(
            {
                ref
                for event in events
                for ref in event.get("evidence_refs", [])
                if isinstance(ref, str) and ref
            }
        )
        retry_of = None
        first_payload = first.get("payload")
        if isinstance(first_payload, dict) and isinstance(first_payload.get("retry_of"), str):
            retry_of = first_payload["retry_of"]
        timeline = [cls._timeline_event(event) for event in events]
        consistent = not normalized_issues
        return {
            "run_id": run_id,
            "project_id": first.get("project_id"),
            "linear_issue_id": first.get("linear_issue_id"),
            "agent_id": first.get("agent_id"),
            "status": observed_status if consistent else "inconsistent",
            "observed_status": observed_status,
            "observed_terminal_state": last.get("terminal_state"),
            "is_terminal": consistent and observed_status in _TERMINAL_STATUSES,
            "consistent": consistent,
            "event_count": len(events),
            "last_sequence": last.get("sequence"),
            "last_event_id": last.get("event_id"),
            "last_event_type": last.get("event_type"),
            "updated_at": last.get("recorded_at"),
            "last_occurred_at": last.get("occurred_at"),
            "retry_of": retry_of,
            "evidence_refs": evidence_refs,
            "timeline": timeline,
            "inconsistencies": normalized_issues,
        }

    @staticmethod
    def _timeline_event(event: Mapping[str, Any]) -> dict[str, Any]:
        payload = event.get("payload")
        payload_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
        evidence_refs = event.get("evidence_refs", [])
        return {
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "schema_version": event.get("schema_version"),
            "sequence": event.get("sequence"),
            "occurred_at": event.get("occurred_at"),
            "recorded_at": event.get("recorded_at"),
            "previous_event_id": event.get("previous_event_id"),
            "blocked_reason": event.get("blocked_reason"),
            "terminal_state": event.get("terminal_state"),
            "evidence_refs": sorted(
                ref for ref in evidence_refs if isinstance(ref, str) and ref
            ),
            "payload_keys": payload_keys,
        }

    @staticmethod
    def _observed_status(event: Mapping[str, Any]) -> str:
        if event.get("event_type") == EventType.FINISHED:
            terminal_state = event.get("terminal_state")
            return terminal_state if isinstance(terminal_state, str) else "finished_unknown"
        return _OBSERVED_STATUS.get(event.get("event_type"), "unknown")

    @staticmethod
    def _event_sort_key(event: Mapping[str, Any]) -> tuple[Any, str]:
        sequence = event.get("sequence")
        if not isinstance(sequence, int):
            sequence = 2**63 - 1
        return sequence, str(event.get("event_id", ""))

    @staticmethod
    def _event_fingerprint(event: Mapping[str, Any]) -> str:
        body = copy.deepcopy(dict(event))
        body.pop("recorded_at", None)
        return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
        issue = {"code": code, "message": message}
        issue.update(details)
        return issue
