from agent.memory_inspection import explain_archive, explain_conflict, explain_retrieval, explain_write
from agent.memory_policy import ConflictDecision
from agent.memory_records import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RecordStatus,
    SalienceTier,
    TrustTier,
)


def _make_record(record_id: str, *, status: RecordStatus = RecordStatus.ACTIVE) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        memory_type=MemoryType.PROFILE,
        scope=MemoryScope.OPERATOR,
        topic_key="preference:spelling",
        content="User prefers British spelling.",
        source="memory_tool:add",
        source_kind="explicit_user_statement",
        trust_tier=TrustTier.USER_ASSERTED,
        salience_tier=SalienceTier.HIGH,
        status=status,
    )


def test_explain_write_includes_reason_topic_scope_and_source_kind():
    record = _make_record("rec-1")

    payload = explain_write(record, "explicit_operator_signal")

    assert payload["record_id"] == "rec-1"
    assert payload["topic_key"] == "preference:spelling"
    assert payload["scope"] == "operator"
    assert payload["source_kind"] == "explicit_user_statement"
    assert payload["reason"] == "explicit_operator_signal"


def test_explain_conflict_reports_winner_loser_and_reason():
    winner = _make_record("rec-2")
    loser = _make_record("rec-1", status=RecordStatus.SUPERSEDED)
    decision = ConflictDecision(
        winner=winner,
        loser=loser,
        loser_status=RecordStatus.SUPERSEDED,
        reason="higher_trust_new_record",
    )

    payload = explain_conflict(decision)

    assert payload == {
        "winner_record_id": "rec-2",
        "loser_record_id": "rec-1",
        "winner_status": "active",
        "loser_status": "superseded",
        "winner_scope": "operator",
        "loser_scope": "operator",
        "winner_trust_tier": "user_asserted",
        "loser_trust_tier": "user_asserted",
        "winner_salience_tier": "high",
        "loser_salience_tier": "high",
        "winner_source_kind": "explicit_user_statement",
        "loser_source_kind": "explicit_user_statement",
        "reason": "higher_trust_new_record",
        "topic_key": "preference:spelling",
    }


def test_explain_archive_reports_reason_and_source_kind():
    record = _make_record("rec-3", status=RecordStatus.ARCHIVED)

    payload = explain_archive(record, "operator_requested_archive")

    assert payload == {
        "record_id": "rec-3",
        "topic_key": "preference:spelling",
        "scope": "operator",
        "status": "archived",
        "trust_tier": "user_asserted",
        "salience_tier": "high",
        "source_kind": "explicit_user_statement",
        "reason": "operator_requested_archive",
    }


def test_explain_retrieval_reports_reason_and_source_kind():
    record = _make_record("rec-4")

    payload = explain_retrieval(record, "matched_topic_and_scope")

    assert payload == {
        "record_id": "rec-4",
        "topic_key": "preference:spelling",
        "scope": "operator",
        "status": "active",
        "trust_tier": "user_asserted",
        "salience_tier": "high",
        "source_kind": "explicit_user_statement",
        "reason": "matched_topic_and_scope",
    }
