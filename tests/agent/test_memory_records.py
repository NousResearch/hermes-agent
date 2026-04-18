"""Tests for agent.memory_records."""

import json

from agent.memory_records import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RecordStatus,
    SalienceTier,
    TrustTier,
    normalize_legacy_entry,
)


def test_memory_enums_expose_stable_wire_values():
    assert MemoryType.PROFILE.value == "profile"
    assert MemoryScope.OPERATOR.value == "operator"
    assert TrustTier.USER_ASSERTED.value == "user_asserted"
    assert SalienceTier.HIGH.value == "high"
    assert RecordStatus.ACTIVE.value == "active"


def test_memory_record_round_trip_dict():
    record = MemoryRecord(
        record_id="rec-1",
        memory_type=MemoryType.PROFILE,
        scope=MemoryScope.OPERATOR,
        topic_key="preference:response-detail",
        content="User prefers concise responses for routine tasks.",
        source="memory_tool:add",
        source_kind="explicit_user_statement",
        trust_tier=TrustTier.USER_ASSERTED,
        salience_tier=SalienceTier.HIGH,
        status=RecordStatus.ACTIVE,
        revision=1,
    )

    payload = record.to_dict()
    restored = MemoryRecord.from_dict(json.loads(json.dumps(payload)))

    assert restored.record_id == "rec-1"
    assert restored.topic_key == "preference:response-detail"
    assert restored.trust_tier is TrustTier.USER_ASSERTED
    assert restored.status is RecordStatus.ACTIVE


def test_normalize_legacy_entry_defaults_to_profile_record():
    record = normalize_legacy_entry(
        target="user",
        content="User prefers British spelling.",
        created_at="2026-04-18T00:00:00Z",
    )

    assert record.memory_type is MemoryType.PROFILE
    assert record.scope is MemoryScope.OPERATOR
    assert record.status is RecordStatus.ACTIVE
    assert record.content == "User prefers British spelling."


def test_topic_key_required_for_auto_conflict_resolution_eligibility():
    record = normalize_legacy_entry(
        target="memory",
        content="Project deploys with make ship.",
        created_at="2026-04-18T00:00:00Z",
    )

    assert record.topic_key is not None
    assert record.topic_key.startswith(("workspace:", "env:", "preference:"))
