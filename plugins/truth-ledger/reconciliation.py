from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

LEDGER_OPERATIONS = {"assert", "confirm", "supersede", "retract"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _logical_key(fact: Mapping[str, Any]) -> str:
    return f"{fact['scope']}|{fact['subject']}|{fact['key']}"


def _source_identity(observation: Mapping[str, Any]) -> str:
    # Deterministic, first-observation source identity. No message text.
    basis = "|".join(
        str(observation.get(name, ""))
        for name in (
            "profile",
            "platform",
            "session_id",
            "turn_id",
            "task_id",
            "speaker_id",
            "conversation_id",
            "thread_id",
        )
    )
    return f"src_{_sha256(basis)[:20]}"


def _build_active_index(history: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    active: dict[str, Mapping[str, Any]] = {}
    for event in history:
        fact = event.get("fact") or {}
        if not isinstance(fact, Mapping):
            continue
        if not all(k in fact for k in ("scope", "subject", "key")):
            continue
        lkey = _logical_key(fact)
        op = event.get("operation")

        if op == "retract":
            # Retraction removes currently active entry for the logical key.
            active.pop(lkey, None)
            continue

        if op in {"assert", "confirm", "supersede"}:
            active[lkey] = event

    return active


def _find_same_source_duplicate(
    history: Sequence[Mapping[str, Any]],
    source_identity: str,
    logical_key: str,
    canonical_value_hash: str,
) -> Mapping[str, Any] | None:
    for event in reversed(history):
        fact = event.get("fact") or {}
        if not isinstance(fact, Mapping):
            continue
        if _logical_key(fact) != logical_key:
            continue
        if event.get("source_identity") != source_identity:
            continue
        if event.get("canonical_value_hash") != canonical_value_hash:
            continue
        return event
    return None


def _normalize_operation(
    proposed_operation: str,
    active_event: Mapping[str, Any] | None,
    canonical_value: str,
) -> tuple[str | None, str | None]:
    if proposed_operation == "NONE":
        return None, "none_operation"

    if active_event is None:
        if proposed_operation in {"confirm", "supersede", "retract"}:
            return None, f"out_of_order_{proposed_operation}"
        return "assert", None

    if proposed_operation == "retract":
        return "retract", None

    active_hash = active_event.get("canonical_value_hash")
    if active_hash == _sha256(canonical_value):
        return "confirm", None
    return "supersede", None


def reconcile_candidate(
    *,
    history: Sequence[Mapping[str, Any]],
    observation: Mapping[str, Any],
    candidate: Mapping[str, Any],
    occurred_at: str,
) -> dict[str, Any]:
    """Deterministic idempotent reconciliation for a single candidate fact.

    Returns one of:
      - {decision: "append", event: <event>, reason: None}
      - {decision: "duplicate", event: <existing event>, reason: "idempotent_replay"}
      - {decision: "none", event: None, reason: <why>}
    """

    # Never mutate caller-owned history/candidate payloads.
    history_view = tuple(copy.deepcopy(history))
    fact = {
        "scope": candidate.get("scope"),
        "kind": candidate.get("kind"),
        "subject": candidate.get("subject"),
        "key": candidate.get("key"),
        "value": candidate.get("value"),
    }

    if not all(fact.get(k) is not None for k in ("scope", "kind", "subject", "key")):
        return {"decision": "none", "event": None, "reason": "invalid_candidate"}

    proposed_operation = str(candidate.get("proposed_operation") or "assert")
    if proposed_operation not in LEDGER_OPERATIONS | {"NONE"}:
        return {"decision": "none", "event": None, "reason": "invalid_operation"}

    source_identity = _source_identity(observation)
    logical_key = _logical_key(fact)
    canonical_value = canonical_json(fact.get("value"))
    canonical_value_hash = _sha256(canonical_value)

    duplicate = _find_same_source_duplicate(
        history=history_view,
        source_identity=source_identity,
        logical_key=logical_key,
        canonical_value_hash=canonical_value_hash,
    )
    if duplicate is not None:
        return {
            "decision": "duplicate",
            "event": copy.deepcopy(duplicate),
            "reason": "idempotent_replay",
        }

    active = _build_active_index(history_view)
    active_event = active.get(logical_key)

    operation, reason = _normalize_operation(
        proposed_operation=proposed_operation,
        active_event=active_event,
        canonical_value=canonical_value,
    )
    if operation is None:
        return {"decision": "none", "event": None, "reason": reason}

    if operation == "confirm" and active_event is not None:
        fact_id = str(active_event["fact_id"])
        supersedes = None
    elif operation == "retract" and active_event is not None:
        fact_id = str(active_event["fact_id"])
        supersedes = str(active_event["fact_id"])
    else:
        fact_id_basis = f"{logical_key}|{canonical_value_hash}|{source_identity}"
        fact_id = f"fact_{_sha256(fact_id_basis)[:24]}"
        supersedes = str(active_event["fact_id"]) if active_event is not None else None

    event_key = "|".join(
        [
            source_identity,
            operation,
            logical_key,
            canonical_value_hash,
            fact_id,
            supersedes or "",
        ]
    )
    event_id = f"evt_{_sha256(event_key)[:24]}"

    event = {
        "schema_version": 1,
        "event_id": event_id,
        "event_key": event_key,
        "occurred_at": occurred_at,
        "operation": operation,
        "fact_id": fact_id,
        "supersedes": supersedes,
        "fact": fact,
        "source_identity": source_identity,
        "canonical_value_hash": canonical_value_hash,
        "evidence": {
            "profile": observation.get("profile"),
            "platform": observation.get("platform"),
            "session_id": observation.get("session_id"),
            "turn_id": observation.get("turn_id"),
            "task_id": observation.get("task_id"),
            "speaker_id": observation.get("speaker_id"),
            "conversation_id": observation.get("conversation_id"),
            "thread_id": observation.get("thread_id"),
        },
    }

    return {"decision": "append", "event": event, "reason": None}
