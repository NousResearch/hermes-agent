"""Tests for agent.memory_records."""

import json

import pytest

from agent.memory_records import (
    EpisodeRecord,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RecordStatus,
    SalienceTier,
    TrustTier,
    normalize_legacy_entry,
    records_from_sidecar_payload,
    records_to_sidecar_payload,
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


def test_episode_record_round_trip_dict():
    record = EpisodeRecord(
        record_id="ep-1",
        memory_type=MemoryType.EPISODIC,
        scope=MemoryScope.WORKSPACE,
        topic_key="workspace:memory-records",
        content="Fixed the memory record serializers and verified targeted tests.",
        source="memory_tool:add",
        source_kind="tool_observation",
        trust_tier=TrustTier.OBSERVED,
        salience_tier=SalienceTier.MEDIUM,
        status=RecordStatus.ACTIVE,
        task_signature="fix-memory-record-serialization",
        problem_summary="Task 1 review findings in memory_records.py",
        approach_summary="Add validation, detach serialized payloads, preserve episodic subtype.",
        key_actions=["update tests", "patch serializers"],
        tools_used=["pytest", "git"],
        outcome="passed",
        outcome_evidence="tests/agent/test_memory_records.py",
        validation_status="verified",
        reuse_count=2,
        source_session_id="sess-123",
        metadata={"result": "ok"},
    )

    payload = record.to_dict()
    restored = EpisodeRecord.from_dict(json.loads(json.dumps(payload)))

    assert isinstance(restored, EpisodeRecord)
    assert restored.memory_type is MemoryType.EPISODIC
    assert restored.task_signature == "fix-memory-record-serialization"
    assert restored.key_actions == ["update tests", "patch serializers"]
    assert restored.tools_used == ["pytest", "git"]


def test_sidecar_payload_round_trip_serializes_and_restores_record_types():
    records = [
        MemoryRecord(
            record_id="rec-1",
            memory_type=MemoryType.PROFILE,
            scope=MemoryScope.OPERATOR,
            topic_key="preference:response-detail",
            content="User prefers concise responses.",
            source="memory_tool:add",
            source_kind="explicit_user_statement",
            trust_tier=TrustTier.USER_ASSERTED,
            salience_tier=SalienceTier.HIGH,
            status=RecordStatus.ACTIVE,
        ),
        EpisodeRecord(
            record_id="ep-1",
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.WORKSPACE,
            topic_key="workspace:memory-records",
            content="Fixed review findings for Task 1.",
            source="memory_tool:add",
            source_kind="tool_observation",
            trust_tier=TrustTier.OBSERVED,
            salience_tier=SalienceTier.MEDIUM,
            status=RecordStatus.ACTIVE,
            task_signature="fix-task-1-review-findings",
        ),
    ]

    payload = records_to_sidecar_payload(records)
    restored = records_from_sidecar_payload(json.loads(json.dumps(payload)))

    assert payload["version"] == 1
    assert len(restored) == 2
    assert isinstance(restored[0], MemoryRecord)
    assert not isinstance(restored[0], EpisodeRecord)
    assert isinstance(restored[1], EpisodeRecord)
    assert restored[1].task_signature == "fix-task-1-review-findings"


def test_records_from_sidecar_payload_preserves_episodic_subtype_without_task_signature():
    episode_payload = EpisodeRecord(
        record_id="ep-1",
        memory_type=MemoryType.EPISODIC,
        scope=MemoryScope.WORKSPACE,
        topic_key="workspace:memory-records",
        content="Captured an episodic record without a task signature.",
        source="memory_tool:add",
        source_kind="tool_observation",
        trust_tier=TrustTier.OBSERVED,
        salience_tier=SalienceTier.MEDIUM,
        status=RecordStatus.ACTIVE,
        task_signature="temporary-signature",
    ).to_dict()
    episode_payload.pop("task_signature")

    restored = records_from_sidecar_payload({"version": 1, "records": [episode_payload]})

    assert len(restored) == 1
    assert isinstance(restored[0], EpisodeRecord)
    assert restored[0].memory_type is MemoryType.EPISODIC
    assert restored[0].task_signature == ""


def test_to_dict_returns_detached_payloads():
    record = EpisodeRecord(
        record_id="ep-1",
        memory_type=MemoryType.EPISODIC,
        scope=MemoryScope.WORKSPACE,
        topic_key="workspace:memory-records",
        content="Validate detached serialization payloads.",
        source="memory_tool:add",
        source_kind="tool_observation",
        trust_tier=TrustTier.OBSERVED,
        salience_tier=SalienceTier.MEDIUM,
        status=RecordStatus.ACTIVE,
        conflicts_with=["rec-0"],
        tags=["memory-v2"],
        metadata={"evidence": {"steps": ["before"]}},
        key_actions=["update tests"],
        tools_used=["pytest"],
    )

    payload = record.to_dict()
    payload["conflicts_with"].append("rec-9")
    payload["tags"].append("mutated")
    payload["metadata"]["evidence"]["steps"].append("after")
    payload["key_actions"].append("commit")
    payload["tools_used"].append("git")

    assert record.conflicts_with == ["rec-0"]
    assert record.tags == ["memory-v2"]
    assert record.metadata == {"evidence": {"steps": ["before"]}}
    assert record.key_actions == ["update tests"]
    assert record.tools_used == ["pytest"]


@pytest.mark.parametrize(
    "record_factory",
    [
        lambda metadata: MemoryRecord(
            record_id="rec-1",
            memory_type=MemoryType.PROFILE,
            scope=MemoryScope.OPERATOR,
            topic_key="preference:response-detail",
            content="User prefers concise responses.",
            source="memory_tool:add",
            source_kind="explicit_user_statement",
            trust_tier=TrustTier.USER_ASSERTED,
            salience_tier=SalienceTier.HIGH,
            status=RecordStatus.ACTIVE,
            metadata=metadata,
        ),
        lambda metadata: EpisodeRecord(
            record_id="ep-1",
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.WORKSPACE,
            topic_key="workspace:memory-records",
            content="Captured nested metadata for an episodic record.",
            source="memory_tool:add",
            source_kind="tool_observation",
            trust_tier=TrustTier.OBSERVED,
            salience_tier=SalienceTier.MEDIUM,
            status=RecordStatus.ACTIVE,
            metadata=metadata,
        ),
    ],
)
def test_record_construction_deep_copies_nested_metadata(record_factory):
    metadata = {"evidence": {"steps": ["before"]}}

    record = record_factory(metadata)
    metadata["evidence"]["steps"].append("after")

    assert record.metadata == {"evidence": {"steps": ["before"]}}


def test_episode_record_rejects_non_episodic_memory_type_on_construction():
    with pytest.raises(ValueError, match="episodic"):
        EpisodeRecord(
            record_id="ep-1",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.WORKSPACE,
            topic_key="workspace:memory-records",
            content="Attempted to create an episode with the wrong type.",
            source="memory_tool:add",
            source_kind="tool_observation",
            trust_tier=TrustTier.OBSERVED,
            salience_tier=SalienceTier.MEDIUM,
            status=RecordStatus.ACTIVE,
        )


def test_episode_record_from_dict_rejects_non_episodic_memory_type():
    payload = {
        "record_id": "ep-1",
        "memory_type": MemoryType.SEMANTIC.value,
        "scope": MemoryScope.WORKSPACE.value,
        "topic_key": "workspace:memory-records",
        "content": "Attempted to deserialize an episode with the wrong type.",
        "source": "memory_tool:add",
        "source_kind": "tool_observation",
        "trust_tier": TrustTier.OBSERVED.value,
        "salience_tier": SalienceTier.MEDIUM.value,
        "status": RecordStatus.ACTIVE.value,
    }

    with pytest.raises(ValueError, match="episodic"):
        EpisodeRecord.from_dict(payload)


@pytest.mark.parametrize("field_name", ["record_id", "content", "source", "source_kind"])
def test_memory_record_from_dict_rejects_none_for_required_string_fields(field_name):
    payload = {
        "record_id": "rec-1",
        "memory_type": MemoryType.PROFILE.value,
        "scope": MemoryScope.OPERATOR.value,
        "topic_key": "preference:response-detail",
        "content": "User prefers concise responses.",
        "source": "memory_tool:add",
        "source_kind": "explicit_user_statement",
        "trust_tier": TrustTier.USER_ASSERTED.value,
        "salience_tier": SalienceTier.HIGH.value,
        "status": RecordStatus.ACTIVE.value,
    }
    payload[field_name] = None

    with pytest.raises((TypeError, ValueError)):
        MemoryRecord.from_dict(payload)


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
