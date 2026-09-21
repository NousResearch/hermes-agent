"""Validated governance finding events and their pure current-state reducer.

The profile activity ledger is deliberately generic.  This module supplies the
narrow, append-only contract for governance findings so callers cannot
accidentally persist prompts, summaries, credentials, or other free-form
telemetry as governance evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from hermes_cli import profile_activity_ledger as ledger

FINDING_EVENT_TYPES = frozenset(
    {
        "governance.finding.opened",
        "governance.finding.updated",
        "governance.finding.resolved",
        "governance.finding.dismissed",
        "governance.risk.accepted",
    }
)

_FINDING_STATES = {
    "governance.finding.opened": "open",
    "governance.finding.updated": "open",
    "governance.finding.resolved": "resolved",
    "governance.finding.dismissed": "dismissed",
    "governance.risk.accepted": "risk_accepted",
}
_REQUIRED_FIELDS = (
    "finding_id",
    "detector",
    "subject_type",
    "subject_id",
    "severity",
    "source_observed_at",
    "evidence_refs",
    "owner",
    "task_id",
    "resolution_ref",
    "dedupe_key",
)
_ALLOWED_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})
_REF_RE = re.compile(
    r"^(?:test|report|commit|readback|accepted-risk|task):[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,511}$")
_SECRET_RE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|api[_-]?key|password|authorization|bearer\s|secret)"
)


class FindingValidationError(ValueError):
    """Raised when a governance finding event violates the metadata contract."""


def _required_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FindingValidationError(f"{name} must be a non-empty string")
    value = value.strip()
    if len(value) > 512 or not _IDENTIFIER_RE.fullmatch(value):
        raise FindingValidationError(f"{name} must be a bounded structured token")
    if _SECRET_RE.search(value):
        raise FindingValidationError(f"{name} contains sensitive content")
    return value


def _optional_identifier(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _required_identifier(name, value)


def _optional_reference(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _REF_RE.fullmatch(value.strip()):
        raise FindingValidationError(f"{name} must be a typed evidence reference")
    value = value.strip()
    if _SECRET_RE.search(value):
        raise FindingValidationError(f"{name} contains sensitive content")
    return value


def _source_observed_at(value: Any) -> int:
    if isinstance(value, bool):
        raise FindingValidationError("source_observed_at must be an integer timestamp")
    if isinstance(value, (int, float)):
        if int(value) != value or int(value) < 0:
            raise FindingValidationError("source_observed_at must be an integer timestamp")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise FindingValidationError("source_observed_at must be an integer timestamp or ISO-8601 value") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    raise FindingValidationError("source_observed_at must be an integer timestamp or ISO-8601 value")


def _evidence_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise FindingValidationError("evidence_refs must be a list of typed references")
    refs: list[str] = []
    for ref in value:
        if not isinstance(ref, str) or not _REF_RE.fullmatch(ref):
            raise FindingValidationError("evidence_refs contains a malformed reference")
        if _SECRET_RE.search(ref) or any(char.isspace() for char in ref):
            raise FindingValidationError("evidence_refs contains sensitive content")
        refs.append(ref)
    return refs


def validate_finding_event(event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise one finding event payload.

    The returned mapping contains exactly the contract fields in a stable
    order.  Unknown fields are rejected rather than persisted, which is the
    boundary preventing raw prompts and free-form summaries entering the
    governance ledger.
    """
    if event_type not in FINDING_EVENT_TYPES:
        raise FindingValidationError(f"unknown finding event type: {event_type}")
    if not isinstance(payload, Mapping):
        raise FindingValidationError("finding payload must be a mapping")
    keys = set(payload)
    missing = [field for field in _REQUIRED_FIELDS if field not in keys]
    if missing:
        raise FindingValidationError(f"missing required payload fields: {', '.join(missing)}")
    unknown = sorted(keys - set(_REQUIRED_FIELDS))
    if unknown:
        raise FindingValidationError(f"unknown payload field: {unknown[0]}")

    severity = _required_identifier("severity", payload["severity"]).lower()
    if severity not in _ALLOWED_SEVERITIES:
        raise FindingValidationError(f"severity must be one of {sorted(_ALLOWED_SEVERITIES)}")
    return {
        "finding_id": _required_identifier("finding_id", payload["finding_id"]),
        "detector": _required_identifier("detector", payload["detector"]),
        "subject_type": _required_identifier("subject_type", payload["subject_type"]),
        "subject_id": _required_identifier("subject_id", payload["subject_id"]),
        "severity": severity,
        "source_observed_at": _source_observed_at(payload["source_observed_at"]),
        "evidence_refs": _evidence_refs(payload["evidence_refs"]),
        "owner": _optional_identifier("owner", payload["owner"]),
        "task_id": _optional_identifier("task_id", payload["task_id"]),
        "resolution_ref": _optional_reference("resolution_ref", payload["resolution_ref"]),
        "dedupe_key": _required_identifier("dedupe_key", payload["dedupe_key"]),
    }


def _event_id(payload: Mapping[str, Any], event_type: str) -> str:
    basis = json.dumps(
        {"event_type": event_type, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "finding:" + hashlib.sha256(basis).hexdigest()[:32]


def append_finding_event(
    *,
    event_type: str,
    payload: Mapping[str, Any],
    event_id: str | None = None,
    occurred_at: int | None = None,
) -> str:
    """Validate and append a finding event through the canonical ledger."""
    canonical = validate_finding_event(event_type, payload)
    resolved_event_id = event_id or _event_id(canonical, event_type)
    return ledger.append_event(
        event_id=resolved_event_id,
        source="governance.findings",
        event_type=event_type,
        object_type="governance_finding",
        object_id=canonical["finding_id"],
        summary="Governance finding lifecycle event",
        occurred_at=canonical["source_observed_at"] if occurred_at is None else occurred_at,
        payload=canonical,
    )


def _logical_event_key(event: Mapping[str, Any]) -> tuple[int, int, int, int, str, str]:
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    try:
        source_time = _source_observed_at(payload.get("source_observed_at"))
    except FindingValidationError:
        source_time = int(event.get("occurred_at") or 0)
    return (
        source_time,
        int(event.get("occurred_at") or 0),
        int(event.get("created_at") or 0),
        int(event.get("id") or 0),
        str(event.get("event_id") or ""),
        json.dumps(payload, sort_keys=True, default=str),
    )


def reduce_findings(events: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reduce lifecycle events into one deterministic current finding per dedupe key.

    Events are sorted by source observation time, not ingestion order.  Duplicate
    event IDs are ignored and late updates cannot move a finding backwards in
    time.  The input mappings and historical ledger rows are never mutated.
    """
    unique: dict[str, Mapping[str, Any]] = {}
    for event in sorted(events, key=_logical_event_key):
        event_id = str(event.get("event_id") or "")
        if not event_id:
            event_id = _event_id(event.get("payload") or {}, str(event.get("event_type") or ""))
        unique.setdefault(event_id, event)

    current: dict[str, dict[str, Any]] = {}
    for event in sorted(unique.values(), key=_logical_event_key):
        event_type = str(event.get("event_type") or "")
        if event_type not in FINDING_EVENT_TYPES:
            continue
        canonical = validate_finding_event(event_type, event.get("payload") or {})
        key = canonical["dedupe_key"]
        prior = current.get(key)
        if event_type == "governance.finding.updated" and prior is not None:
            item = dict(prior)
            for field, value in canonical.items():
                if field == "resolution_ref":
                    item[field] = value
                elif value is not None:
                    item[field] = value
            item["state"] = "open"
            item["event_id"] = str(event.get("event_id") or "")
            item["event_type"] = event_type
            item["occurred_at"] = int(event.get("occurred_at") or canonical["source_observed_at"])
            item["source_observed_at"] = canonical["source_observed_at"]
            current[key] = item
            continue

        item = dict(canonical)
        item["state"] = _FINDING_STATES[event_type]
        item["event_type"] = event_type
        item["event_id"] = str(event.get("event_id") or "")
        item["occurred_at"] = int(event.get("occurred_at") or canonical["source_observed_at"])
        current[key] = item
    return current


def _finding_contract_payload(payload: Mapping[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
    canonical = validate_finding_event("governance.finding.opened", payload)
    if task_id is not None:
        canonical["task_id"] = _optional_identifier("task_id", task_id)
    return canonical


def link_finding_to_task(
    conn: Any,
    payload: Mapping[str, Any],
    *,
    created_by: str = "denji",
    board: str | None = None,
) -> str:
    """Create or reuse one Kanban action task for a finding.

    The task is keyed by the finding dedupe key.  A task completion only writes
    a normal Kanban completion event; callers must separately call
    :func:`resolve_finding` with explicit resolution evidence.
    """
    canonical = _finding_contract_payload(payload)
    from hermes_cli import kanban_db as kanban

    idempotency_key = f"governance:finding:{canonical['dedupe_key']}"
    task_id = kanban.create_task(
        conn,
        title=(
            f"Governance finding: {canonical['detector']} "
            f"({canonical['subject_type']}/{canonical['subject_id']})"
        ),
        body=json.dumps(
            {
                "finding_id": canonical["finding_id"],
                "dedupe_key": canonical["dedupe_key"],
                "severity": canonical["severity"],
                "evidence_refs": canonical["evidence_refs"],
            },
            sort_keys=True,
        ),
        assignee=canonical["owner"] or None,
        created_by=created_by,
        priority=_SEVERITY_PRIORITY[canonical["severity"]],
        initial_status="running",
        idempotency_key=idempotency_key,
        board=board,
    )

    linked = conn.execute(
        """SELECT 1 FROM task_events
           WHERE task_id = ? AND kind = 'finding_linked'
             AND json_extract(payload, '$.finding_id') = ?
           LIMIT 1""",
        (task_id, canonical["finding_id"]),
    ).fetchone()
    if linked is None:
        with kanban.write_txn(conn):
            kanban._append_event(
                conn,
                task_id,
                "finding_linked",
                {
                    "finding_id": canonical["finding_id"],
                    "dedupe_key": canonical["dedupe_key"],
                    "severity": canonical["severity"],
                },
            )

    existing = reduce_findings(
        ledger.query_events(event_types=sorted(FINDING_EVENT_TYPES))
    )
    prior = existing.get(canonical["dedupe_key"])
    if prior is None or prior.get("state") in {"resolved", "dismissed", "risk_accepted"}:
        append_finding_event(
            event_type="governance.finding.opened",
            payload=canonical,
            event_id=(
                f"finding:{canonical['dedupe_key']}:opened:"
                f"{canonical['source_observed_at']}"
            ),
        )
    linked_payload = dict(canonical)
    linked_payload["task_id"] = task_id
    linked_payload["source_observed_at"] = int(time.time())
    append_finding_event(
        event_type="governance.finding.updated",
        payload=linked_payload,
        event_id=(
            f"finding:{canonical['dedupe_key']}:task-link:"
            f"{canonical['source_observed_at']}"
        ),
    )
    return task_id


def resolve_finding(
    payload: Mapping[str, Any],
    resolution_ref: str,
    *,
    task_id: str | None = None,
    conn: Any = None,
) -> str:
    """Resolve a finding only with an explicit typed proof reference."""
    if not isinstance(resolution_ref, str) or not _REF_RE.fullmatch(resolution_ref):
        raise FindingValidationError("resolution_ref must be a typed evidence reference")
    resolved_payload = dict(payload)
    resolved_payload["resolution_ref"] = resolution_ref
    resolved_payload["task_id"] = task_id if task_id is not None else payload.get("task_id")
    resolved_payload["source_observed_at"] = int(time.time())
    canonical = validate_finding_event("governance.finding.resolved", resolved_payload)
    event_id = f"finding:{canonical['dedupe_key']}:resolved:{resolution_ref}"
    result = append_finding_event(
        event_type="governance.finding.resolved",
        payload=canonical,
        event_id=event_id,
    )
    if conn is not None and canonical["task_id"]:
        from hermes_cli import kanban_db as kanban

        with kanban.write_txn(conn):
            kanban._append_event(
                conn,
                canonical["task_id"],
                "finding_resolution_recorded",
                {
                    "finding_id": canonical["finding_id"],
                    "dedupe_key": canonical["dedupe_key"],
                    "resolution_ref": canonical["resolution_ref"],
                },
            )
    return result


# Severity is intentionally an ordering for Kanban priority only; it does not
# replace the finding severity in the event contract.
_SEVERITY_PRIORITY = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


# Explicit aliases make the contract discoverable to scripts and dashboard adapters.
current_findings = reduce_findings
reduce_finding_events = reduce_findings


__all__ = [
    "FINDING_EVENT_TYPES",
    "FindingValidationError",
    "append_finding_event",
    "current_findings",
    "link_finding_to_task",
    "reduce_finding_events",
    "reduce_findings",
    "resolve_finding",
    "validate_finding_event",
]
