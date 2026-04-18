from __future__ import annotations

from typing import Any

from agent.memory_policy import ConflictDecision
from agent.memory_records import MemoryRecord


def _record_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "topic_key": record.topic_key,
        "scope": record.scope.value,
        "status": record.status.value,
        "trust_tier": record.trust_tier.value,
        "salience_tier": record.salience_tier.value,
        "source_kind": record.source_kind,
    }



def explain_write(record: MemoryRecord, reason: str) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["reason"] = reason
    return payload



def explain_archive(record: MemoryRecord, reason: str) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["reason"] = reason
    return payload



def explain_retrieval(record: MemoryRecord, reason: str, *, score: float | None = None) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["reason"] = reason
    if score is not None:
        payload["score"] = score
    return payload



def explain_conflict(decision: ConflictDecision) -> dict[str, Any]:
    return {
        "winner_record_id": decision.winner.record_id,
        "loser_record_id": decision.loser.record_id,
        "winner_status": decision.winner.status.value,
        "loser_status": decision.loser_status.value,
        "winner_scope": decision.winner.scope.value,
        "loser_scope": decision.loser.scope.value,
        "winner_trust_tier": decision.winner.trust_tier.value,
        "loser_trust_tier": decision.loser.trust_tier.value,
        "winner_salience_tier": decision.winner.salience_tier.value,
        "loser_salience_tier": decision.loser.salience_tier.value,
        "winner_source_kind": decision.winner.source_kind,
        "loser_source_kind": decision.loser.source_kind,
        "reason": decision.reason,
        "topic_key": decision.winner.topic_key,
    }
