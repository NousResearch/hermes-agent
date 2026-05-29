"""Schema tests for Memory v2 canonical record types."""

from __future__ import annotations

import pytest

from plugins.memory.memory_v2.schemas import (
    CandidateMemory,
    GateDecision,
    MemoryItem,
    MemoryPacket,
    MemoryStatus,
    MemoryType,
    ProjectCard,
    SourceRef,
    SourceType,
    ValidationError,
    WorkingMemory,
    normalize_project_id,
)


def test_source_ref_round_trips_to_dict():
    source = SourceRef(
        id="source_memory_v2_thread",
        type=SourceType.SESSION,
        uri="discord://thread/1508915896054452264",
        title="Memory v2 design thread",
        observed_at="2026-05-26T00:00:00Z",
        quote="Dylan asked for robust memory.",
    )

    payload = source.to_dict()
    restored = SourceRef.from_dict(payload)

    assert payload == {
        "id": "source_memory_v2_thread",
        "type": "session",
        "uri": "discord://thread/1508915896054452264",
        "title": "Memory v2 design thread",
        "observed_at": "2026-05-26T00:00:00Z",
        "quote": "Dylan asked for robust memory.",
    }
    assert restored == source


def test_source_ref_rejects_blank_required_fields():
    with pytest.raises(ValidationError, match="id"):
        SourceRef(id="", type="session", uri="session://abc")

    with pytest.raises(ValidationError, match="uri"):
        SourceRef(id="source_1", type="session", uri="")


def test_memory_item_accepts_string_enums_and_round_trips():
    item = MemoryItem(
        id="pref_response_style",
        type="preference",
        subject="Dylan",
        predicate="prefers_response_style",
        value="direct, no-BS, tool-grounded help",
        summary="Dylan prefers direct tool-grounded help.",
        status="active",
        confidence=0.98,
        importance=0.95,
        source_refs=["source_user_profile"],
        tags=["user_preference", "style"],
    )

    payload = item.to_dict()
    restored = MemoryItem.from_dict(payload)

    assert payload["type"] == "preference"
    assert payload["status"] == "active"
    assert payload["source_refs"] == ["source_user_profile"]
    assert restored == item


def test_memory_item_rejects_invalid_confidence_and_importance():
    with pytest.raises(ValidationError, match="confidence"):
        MemoryItem(id="mem_bad", type=MemoryType.FACT, subject="Dylan", confidence=1.1)

    with pytest.raises(ValidationError, match="importance"):
        MemoryItem(id="mem_bad", type=MemoryType.FACT, subject="Dylan", importance=-0.01)


def test_memory_item_requires_superseded_by_when_status_superseded():
    with pytest.raises(ValidationError, match="superseded_by"):
        MemoryItem(
            id="pref_old_voice",
            type=MemoryType.PREFERENCE,
            subject="Dylan",
            status=MemoryStatus.SUPERSEDED,
        )

    item = MemoryItem(
        id="pref_old_voice",
        type=MemoryType.PREFERENCE,
        subject="Dylan",
        status=MemoryStatus.SUPERSEDED,
        superseded_by="pref_current_voice",
    )
    assert item.superseded_by == "pref_current_voice"


def test_memory_item_rejects_active_item_with_superseded_by():
    with pytest.raises(ValidationError, match="active"):
        MemoryItem(
            id="pref_current_voice",
            type=MemoryType.PREFERENCE,
            subject="Dylan",
            status=MemoryStatus.ACTIVE,
            superseded_by="pref_future_voice",
        )


def test_normalize_project_id_and_project_card_round_trip():
    assert normalize_project_id("Hermes Memory v2") == "project:hermes-memory-v2"
    assert normalize_project_id("project:qwen-reasoning-loop") == "project:qwen-reasoning-loop"

    card = ProjectCard(
        id="Hermes Memory v2",
        name="Hermes Memory v2",
        goal="Build robust source-grounded memory for Hermes.",
        current_state="Schema layer under implementation.",
        decisions=["Use an external MemoryProvider first."],
        open_questions=["How strict should auto-promotion be?"],
        next_actions=["Implement file store."],
        source_refs=["source_memory_v2_thread"],
        related_entities=["Dylan", "Hermes"],
        injection_policy={"inject_when": ["user asks about memory_v2"], "default_budget_tokens": 500},
    )

    assert card.id == "project:hermes-memory-v2"
    assert ProjectCard.from_dict(card.to_dict()) == card


def test_candidate_defaults_to_pending_gate_decision_and_round_trips():
    candidate = CandidateMemory(
        id="cand_memory_v2_goal",
        type=MemoryType.PROJECT_STATE,
        claim="Dylan wants robust low-compute Memory v2.",
        proposed_destination="semantic/projects/hermes-memory-v2.yaml",
        confidence=0.9,
        importance=0.8,
        source_refs=["source_memory_v2_thread"],
    )

    assert candidate.gate_decision == GateDecision.PENDING
    assert candidate.to_dict()["gate_decision"] == "pending"
    assert CandidateMemory.from_dict(candidate.to_dict()) == candidate


def test_candidate_rejects_non_pending_promotion_without_decision_reason():
    with pytest.raises(ValidationError, match="decision_reason"):
        CandidateMemory(
            id="cand_rejected",
            type=MemoryType.FACT,
            claim="Temporary thing",
            gate_decision=GateDecision.REJECTED,
        )


def test_working_memory_round_trips_with_defaults():
    working = WorkingMemory(
        session_id="session-1",
        focus={"task": "Design Memory v2", "active_entities": ["Dylan", "Hermes"]},
    )

    payload = working.to_dict()
    restored = WorkingMemory.from_dict(payload)

    assert payload["session_id"] == "session-1"
    assert payload["scratchpad"] == {
        "relevant_paths": [],
        "relevant_commands": [],
        "retrieved_memory_ids": [],
    }
    assert restored == working


def test_memory_packet_tracks_route_confidence_budget_and_items():
    packet = MemoryPacket(
        route="project_continuity",
        confidence="high",
        token_budget=1200,
        items=[
            {
                "id": "project:hermes-memory-v2",
                "type": "project_state",
                "summary": "Memory v2 uses layered memory and gated writes.",
                "source_refs": ["source_memory_v2_thread"],
            }
        ],
        warnings=["Some items are old."],
    )

    assert packet.to_dict()["route"] == "project_continuity"
    assert packet.to_dict()["token_budget"] == 1200
    assert MemoryPacket.from_dict(packet.to_dict()) == packet


def test_memory_packet_rejects_negative_budget():
    with pytest.raises(ValidationError, match="token_budget"):
        MemoryPacket(route="project_continuity", confidence="high", token_budget=-1)
