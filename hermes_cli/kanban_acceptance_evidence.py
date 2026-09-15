"""Pure validation for the versioned Kanban acceptance-evidence contract.

Persistence and lifecycle enforcement stay in ``kanban_db``; this module deliberately
owns only parsing and structural/relational validation so every existing surface can
share one interpretation without adding a model tool.
"""
from __future__ import annotations

from typing import Any


class EvidenceValidationError(ValueError):
    """The persisted acceptance-evidence value is malformed or unsatisfied."""


_KINDS = {"check", "canary", "pr", "review", "artifact", "readback", "custom"}
_STATUSES = {"passed", "failed", "pending", "not_applicable"}
_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}


def normalize_contract(value: Any) -> dict[str, Any] | None:
    """Return validated v1 evidence, preserving ``None`` for legacy cards.

    This is intentionally strict: declarations are durable policy, so unknown shapes
    fail instead of being silently ignored. Completion callers additionally verify
    that passed canary PID/run values match the live run readback.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"version", "required", "observed", "blocker"}:
        raise EvidenceValidationError("acceptance_evidence must be a v1 object with known fields")
    if value.get("version") != 1:
        raise EvidenceValidationError("acceptance_evidence.version must be 1")
    required = value.get("required")
    observed = value.get("observed")
    if not isinstance(required, list) or not required or not isinstance(observed, list):
        raise EvidenceValidationError("acceptance_evidence requires non-empty required and list observed")

    declared: dict[str, bool] = {}
    for item in required:
        if not isinstance(item, dict) or set(item) - {"id", "kind", "description", "applicable"}:
            raise EvidenceValidationError("invalid required evidence item")
        ident, kind, description = item.get("id"), item.get("kind"), item.get("description")
        if (not _identifier(ident) or not isinstance(ident, str) or ident in declared or kind not in _KINDS
                or not isinstance(description, str) or not description.strip()):
            raise EvidenceValidationError("required evidence items need unique id, known kind, and description")
        applicable = item.get("applicable", True)
        if not isinstance(applicable, bool):
            raise EvidenceValidationError("required evidence applicable must be boolean")
        declared[ident] = applicable

    seen: set[str] = set()
    for item in observed:
        if not isinstance(item, dict) or set(item) - {"id", "status", "observed_at", "source", "payload"}:
            raise EvidenceValidationError("invalid observed evidence item")
        ident, status = item.get("id"), item.get("status")
        if ident not in declared or ident in seen or status not in _STATUSES:
            raise EvidenceValidationError("observations must uniquely reference declared evidence")
        if not isinstance(item.get("observed_at"), int) or item["observed_at"] < 1 or not isinstance(item.get("source"), str) or not item["source"].strip():
            raise EvidenceValidationError("observations require timestamp and source")
        if status == "not_applicable" and declared[ident]:
            raise EvidenceValidationError("applicable evidence cannot be marked not_applicable")
        payload = item.get("payload")
        if payload is not None and not isinstance(payload, dict):
            raise EvidenceValidationError("evidence payload must be an object")
        if _required_kind(required, ident) == "canary" and status == "passed":
            _validate_canary(payload)
        seen.add(ident)
    return value


def unsatisfied_requirement_ids(contract: dict[str, Any]) -> list[str]:
    """Return applicable requirements without one passing final observation."""
    observed = {item["id"]: item["status"] for item in contract["observed"]}
    return [
        item["id"] for item in contract["required"]
        if item.get("applicable", True) and observed.get(item["id"]) != "passed"
    ]


def _identifier(value: Any) -> bool:
    import re
    return isinstance(value, str) and bool(re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value))


def _required_kind(required: list[dict[str, Any]], ident: str) -> str:
    return next(item["kind"] for item in required if item["id"] == ident)


def _validate_canary(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise EvidenceValidationError("passed canary evidence requires a payload")
    pid, run_id = payload.get("pid"), payload.get("run_id")
    spawned_at, heartbeat_at = payload.get("spawned_at"), payload.get("heartbeat_at")
    if not all(isinstance(value, int) and value > 0 for value in (pid, run_id, spawned_at, heartbeat_at)):
        raise EvidenceValidationError("passed canary evidence requires positive pid, run_id, spawned_at, and heartbeat_at")
    assert isinstance(spawned_at, int) and isinstance(heartbeat_at, int)
    if heartbeat_at <= spawned_at:
        raise EvidenceValidationError("canary heartbeat_at must be later than spawned_at")


def validate_blocker(value: Any) -> dict[str, Any]:
    """Validate the explicit irreducible blocker record used instead of completion."""
    if not isinstance(value, dict) or set(value) - {"kind", "reason", "irreducible", "recorded_at", "owner"}:
        raise EvidenceValidationError("invalid irreducible blocker")
    if value.get("kind") not in _BLOCK_KINDS or value.get("irreducible") is not True:
        raise EvidenceValidationError("blocker needs known kind and irreducible=true")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip() or not isinstance(value.get("recorded_at"), int) or value["recorded_at"] < 1:
        raise EvidenceValidationError("blocker needs reason and recorded_at")
    return value
