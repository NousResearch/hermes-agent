"""File-store tests for Memory v2 canonical records."""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.memory.memory_v2.schemas import (
    CandidateMemory,
    CoreMemoryRecord,
    GateDecision,
    MemoryItem,
    MemoryType,
    ProjectCard,
    SourceRef,
    ValidationError,
)
from plugins.memory.memory_v2.store import MemoryV2Store


EXPECTED_STORE_DIRS = [
    "working",
    "core",
    "sources",
    "inbox",
    "semantic/projects",
    "semantic/environment",
    "episodic/daily",
    "episodic/sessions",
    "graph",
    "indexes/vector",
    "evals",
    "reports/daily_consolidation",
    "reports/weekly_reflection",
]


def test_store_initialize_creates_profile_scoped_layout(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")

    store.initialize()

    for rel in EXPECTED_STORE_DIRS:
        assert (tmp_path / "memory_v2" / rel).is_dir(), rel
    assert (tmp_path / "memory_v2" / "README.md").is_file()
    assert (tmp_path / "memory_v2" / "config.yaml").is_file()
    assert (tmp_path / "memory_v2" / "inbox" / "raw_events.jsonl").is_file()
    assert (tmp_path / "memory_v2" / "inbox" / "candidates.jsonl").is_file()
    assert (tmp_path / "memory_v2" / "inbox" / "rejected.jsonl").is_file()


def test_append_and_read_raw_events_jsonl(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    first = store.append_raw_event({"type": "turn", "session_id": "session-1", "content": "hello"})
    second = store.append_raw_event({"type": "tool", "session_id": "session-1", "tool": "memory_v2_status"})

    events = store.read_raw_events()

    assert first["id"].startswith("event_")
    assert first["created_at"]
    assert second["id"] != first["id"]
    assert [event["type"] for event in events] == ["turn", "tool"]
    assert events[0]["session_id"] == "session-1"


def test_append_raw_event_materializes_canonical_source_ref(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    event = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Remember that source refs should be canonical.",
            "assistant_content": "Queued.",
        }
    )

    source = store.read_source_ref(event["id"])
    assert source is not None
    assert source.id == event["id"]
    assert source.type.value == "message"
    assert source.uri == f"raw_event:{event['id']}"
    assert source.title == "Raw turn evidence from session session-1"
    assert source.observed_at == event["created_at"]
    assert source.quote == "Remember that source refs should be canonical."
    assert (tmp_path / "memory_v2" / "sources" / f"{event['id']}.yaml").is_file()


def test_append_raw_event_rejects_non_object_payload(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    with pytest.raises(ValidationError, match="raw event"):
        store.append_raw_event(["not", "a", "dict"])  # type: ignore[arg-type]


def test_append_and_list_candidates(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    candidate = CandidateMemory(
        id="cand_memory_v2_goal",
        type=MemoryType.PROJECT_STATE,
        claim="Dylan wants robust low-compute Memory v2.",
        proposed_destination="semantic/projects/hermes-memory-v2.yaml",
        confidence=0.9,
        importance=0.8,
        source_refs=["source_memory_v2_thread"],
    )

    store.append_candidate(candidate)

    candidates = store.list_candidates()
    assert candidates == [candidate]
    assert store.count_pending_candidates() == 1


def test_append_rejected_candidate_records_decision(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    candidate = CandidateMemory(
        id="cand_rejected",
        type="fact",
        claim="Temporary detail",
        gate_decision=GateDecision.REJECTED,
        decision_reason="Too ephemeral for durable memory.",
    )

    store.append_rejected_candidate(candidate)

    assert store.list_rejected_candidates() == [candidate]
    assert store.count_pending_candidates() == 0


def test_write_read_and_list_core_memory_records_by_category(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    user_record = CoreMemoryRecord(
        id="core_user_style",
        category="user",
        statement="Dylan prefers direct, grounded answers.",
        priority=0.95,
        source_refs=["source_user_profile"],
    )
    identity_record = CoreMemoryRecord(
        id="core_assistant_identity",
        category="assistant_identity",
        statement="Hermes should be intellectually honest and tool-grounded.",
        priority=0.9,
        source_refs=["source_soul"],
    )

    user_path = store.write_core_memory_record(user_record)
    identity_path = store.write_core_memory_record(identity_record)

    assert user_path == tmp_path / "memory_v2" / "core" / "user.yaml"
    assert identity_path == tmp_path / "memory_v2" / "core" / "assistant_identity.yaml"
    assert store.read_core_memory_record("core_user_style") == user_record
    assert store.list_core_memory_records(category="user") == [user_record]
    assert store.list_core_memory_records() == [user_record, identity_record]


def test_write_read_and_list_project_cards(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    card = ProjectCard(
        id="Hermes Memory v2",
        name="Hermes Memory v2",
        goal="Build robust source-grounded memory for Hermes.",
        current_state="File store layer under implementation.",
        decisions=["Use human-readable canonical files."],
        source_refs=["source_memory_v2_thread"],
    )

    path = store.write_project_card(card)
    loaded_by_id = store.read_project_card("project:hermes-memory-v2")
    loaded_by_name = store.read_project_card("Hermes Memory v2")
    listed = store.list_project_cards()

    assert path == tmp_path / "memory_v2" / "semantic" / "projects" / "hermes-memory-v2.yaml"
    assert path.is_file()
    assert loaded_by_id == card
    assert loaded_by_name == card
    assert listed == [card]


def test_write_read_and_list_memory_items_as_canonical_yaml(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    item = MemoryItem(
        id="pref_response_style",
        type="preference",
        subject="Dylan",
        predicate="prefers_response_style",
        value="direct no-BS tool-grounded help",
        summary="Dylan prefers direct, no-BS, tool-grounded help.",
        confidence=0.98,
        importance=0.95,
        source_refs=["source_user_profile"],
        tags=["user_preference", "style"],
    )

    path = store.write_memory_item(item)
    loaded = store.read_memory_item("pref_response_style")
    listed = store.list_memory_items()

    assert path == tmp_path / "memory_v2" / "semantic" / "items" / "pref_response_style.yaml"
    assert path.is_file()
    assert loaded == item
    assert listed == [item]


def test_memory_item_paths_do_not_collide_for_distinct_unsafe_ids(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    slash_id = MemoryItem(id="a/b", type="fact", subject="first", value="first")
    colon_id = MemoryItem(id="a:b", type="fact", subject="second", value="second")

    slash_path = store.write_memory_item(slash_id)
    colon_path = store.write_memory_item(colon_id)

    assert slash_path != colon_path
    assert store.read_memory_item("a/b") == slash_id
    assert store.read_memory_item("a:b") == colon_id
    assert sorted(item.id for item in store.list_memory_items()) == ["a/b", "a:b"]


def test_source_ref_paths_do_not_collide_for_distinct_unsafe_ids(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    slash_source = SourceRef(id="source/a", type="manual", uri="memory://source/a")
    colon_source = SourceRef(id="source:a", type="manual", uri="memory://source:a")

    slash_path = store.write_source_ref(slash_source)
    colon_path = store.write_source_ref(colon_source)

    assert slash_path != colon_path
    assert store.read_source_ref("source/a") == slash_source
    assert store.read_source_ref("source:a") == colon_source


def test_list_memory_items_can_filter_by_type_and_status(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    active_pref = MemoryItem(id="pref_active", type="preference", subject="Dylan", status="active")
    superseded_pref = MemoryItem(
        id="pref_old",
        type="preference",
        subject="Dylan",
        status="superseded",
        superseded_by="pref_active",
    )
    env = MemoryItem(id="env_host", type="environment", subject="Hermes runtime")
    for item in (active_pref, superseded_pref, env):
        store.write_memory_item(item)

    assert store.list_memory_items(memory_type="preference", status="active") == [active_pref]
    assert store.list_memory_items(memory_type="preference") == [active_pref, superseded_pref]
    assert store.list_memory_items(status="active") == [env, active_pref]


def test_write_read_and_list_source_refs_as_canonical_yaml(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    source = SourceRef(
        id="source_memory_v2_thread",
        type="session",
        uri="discord://thread/1508915896054452264",
        title="Memory v2 design thread",
        observed_at="2026-05-26T00:00:00Z",
        quote="Dylan asked for robust low-compute memory.",
    )

    path = store.write_source_ref(source)
    loaded = store.read_source_ref("source_memory_v2_thread")
    listed = store.list_source_refs()

    assert path == tmp_path / "memory_v2" / "sources" / "source_memory_v2_thread.yaml"
    assert path.is_file()
    assert loaded == source
    assert listed == [source]


def test_read_missing_project_card_returns_none(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    assert store.read_project_card("project:missing") is None


def test_store_counts_raw_event_lines_without_blank_lines(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    raw_path = tmp_path / "memory_v2" / "inbox" / "raw_events.jsonl"
    raw_path.write_text("\n{\"id\": \"event_1\"}\n\n{\"id\": \"event_2\"}\n", encoding="utf-8")

    assert store.count_raw_events() == 2


def test_read_raw_events_limit_zero_returns_empty_list(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    for i in range(3):
        store.append_raw_event({"id": f"event_{i}", "content": str(i)})

    assert store.read_raw_events(limit=0) == []


def test_read_raw_events_rejects_negative_limit(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    try:
        store.read_raw_events(limit=-1)
    except ValidationError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("negative limit should raise ValidationError")
