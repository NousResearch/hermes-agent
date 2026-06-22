"""Promotion/consolidation tests for Memory v2 candidates."""

from __future__ import annotations

from plugins.memory.memory_v2.consolidation import RuleBasedConsolidator
from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.schemas import CandidateMemory, GateDecision, MemoryItem, MemoryStatus, ProjectCard, ProjectStatus
from plugins.memory.memory_v2.store import MemoryV2Store


def _store_and_index(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    return store, index


def _seed_sources(store: MemoryV2Store, *event_ids: str) -> None:
    for event_id in event_ids:
        store.append_raw_event({"id": event_id, "type": "turn", "user_content": f"source evidence {event_id}"})


def test_consolidation_promotes_pending_preference_to_canonical_memory_item(tmp_path):
    store, index = _store_and_index(tmp_path)
    _seed_sources(store, "event_123")
    candidate = CandidateMemory(
        id="cand_response_style",
        type="preference",
        claim="Alex prefers concise direct answers for simple tasks.",
        proposed_destination="semantic/items",
        confidence=0.92,
        importance=0.86,
        promotion_reason="core_update: explicit stable preference",
        source_refs=["event_123"],
    )
    store.append_candidate(candidate)
    index.index_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 1
    assert report.rejected == 0
    promoted = store.list_memory_items(memory_type="preference", status="active")
    assert len(promoted) == 1
    item = promoted[0]
    assert item.id.startswith("mem_preference_")
    assert item.subject == "Alex"
    assert item.predicate == "prefers"
    assert item.value == candidate.claim
    assert item.summary == candidate.claim
    assert item.confidence == candidate.confidence
    assert item.importance == candidate.importance
    assert item.source_refs == ["event_123"]

    updated_candidate = store.list_candidates()[0]
    assert updated_candidate.gate_decision == GateDecision.PROMOTED
    assert item.id in updated_candidate.decision_reason
    assert index.search("concise direct answers", limit=5)[0]["id"] == item.id


def test_consolidation_rejects_procedure_candidates_without_semantic_promotion(tmp_path):
    store, index = _store_and_index(tmp_path)
    candidate = CandidateMemory(
        id="cand_procedure",
        type="procedure_ref",
        claim="when modifying Hermes providers, load the hermes-agent skill first.",
        proposed_destination="skills",
        promotion_reason="skill_candidate: procedure should become a skill",
        source_refs=["event_proc"],
    )
    store.append_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 0
    assert report.rejected == 1
    assert store.list_memory_items() == []
    assert store.list_candidates()[0].gate_decision == GateDecision.REJECTED
    assert store.list_rejected_candidates()[0].id == "cand_procedure"


def test_consolidation_archives_open_loop_candidates_without_semantic_memory(tmp_path):
    store, index = _store_and_index(tmp_path)
    candidate = CandidateMemory(
        id="cand_open_loop",
        type="episode",
        claim="follow up on the memory eval dashboard tomorrow.",
        proposed_destination="working/open_loops.yaml",
        promotion_reason="open_loop: pending task state",
        source_refs=["event_loop"],
    )
    store.append_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 0
    assert report.archived_only == 1
    assert store.list_memory_items() == []
    updated_candidate = store.list_candidates()[0]
    assert updated_candidate.gate_decision == GateDecision.ARCHIVED_ONLY
    assert "working/open_loops" in updated_candidate.decision_reason


def test_consolidation_supersedes_existing_same_type_subject_and_predicate(tmp_path):
    store, index = _store_and_index(tmp_path)
    _seed_sources(store, "event_new")
    old = MemoryItem(
        id="mem_preference_old_voice",
        type="preference",
        subject="Alex",
        predicate="prefers",
        value="Alex prefers en-US-ChristopherNeural for TTS.",
        summary="Alex prefers en-US-ChristopherNeural for TTS.",
        source_refs=["event_old"],
        tags=["preference"],
    )
    store.write_memory_item(old)
    index.index_memory_item(old)
    candidate = CandidateMemory(
        id="cand_new_voice",
        type="preference",
        claim="Alex prefers en-US-AndrewNeural for TTS now, not en-US-ChristopherNeural.",
        proposed_destination="semantic/items",
        promotion_reason="supersede_existing: explicit conflict",
        confidence=0.88,
        importance=0.9,
        source_refs=["event_new"],
    )
    store.append_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 1
    assert report.superseded == 1
    current = [item for item in store.list_memory_items(memory_type="preference") if item.status == MemoryStatus.ACTIVE][0]
    superseded = store.read_memory_item("mem_preference_old_voice")
    assert superseded is not None
    assert superseded.status == MemoryStatus.SUPERSEDED
    assert superseded.superseded_by == current.id
    assert current.supersedes == ["mem_preference_old_voice"]
    search_results = index.search("ChristopherNeural", limit=5)
    by_id = {result["id"]: result for result in search_results}
    assert by_id["mem_preference_old_voice"]["status"] == "superseded"


def test_consolidation_does_not_supersede_unrelated_preference_with_broad_predicate(tmp_path):
    store, index = _store_and_index(tmp_path)
    _seed_sources(store, "event_new_voice")
    response_style = MemoryItem(
        id="pref_response_style",
        type="preference",
        subject="Alex",
        predicate="prefers",
        value="Alex prefers concise direct answers.",
        summary="Alex prefers concise direct answers.",
        source_refs=["event_style"],
        tags=["preference", "response_style"],
    )
    old_voice = MemoryItem(
        id="pref_old_voice",
        type="preference",
        subject="Alex",
        predicate="prefers",
        value="Alex prefers en-US-ChristopherNeural for TTS.",
        summary="Alex prefers en-US-ChristopherNeural for TTS.",
        source_refs=["event_old_voice"],
        tags=["preference", "tts_voice"],
    )
    for item in (response_style, old_voice):
        store.write_memory_item(item)
        index.index_memory_item(item)
    candidate = CandidateMemory(
        id="cand_new_voice",
        type="preference",
        claim="Alex prefers en-US-AndrewNeural for TTS now, not en-US-ChristopherNeural.",
        proposed_destination="semantic/items",
        promotion_reason="supersede_existing: explicit conflict",
        source_refs=["event_new_voice"],
    )
    store.append_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.superseded_ids == ["pref_old_voice"]
    assert store.read_memory_item("pref_response_style").status == MemoryStatus.ACTIVE
    assert store.read_memory_item("pref_old_voice").status == MemoryStatus.SUPERSEDED


def test_consolidation_rejects_source_less_semantic_candidate(tmp_path):
    store, index = _store_and_index(tmp_path)
    candidate = CandidateMemory(
        id="cand_no_source",
        type="fact",
        claim="A detached fact with no evidence should not become active memory.",
        proposed_destination="semantic/items",
    )
    store.append_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 0
    assert report.rejected == 1
    assert store.list_memory_items() == []
    updated = store.list_candidates()[0]
    assert updated.gate_decision == GateDecision.REJECTED
    assert "source_refs" in updated.decision_reason


def test_consolidation_merges_project_state_candidate_into_project_card(tmp_path):
    store, index = _store_and_index(tmp_path)
    _seed_sources(store, "event_new")
    store.write_project_card(
        ProjectCard(
            id="Memory v2",
            name="Memory v2",
            goal="Build robust source-grounded memory.",
            current_state="Initial scaffold exists.",
            decisions=["Use local files as durable truth."],
            next_actions=["Add candidate promotion."],
            source_refs=["event_old"],
        )
    )
    candidate = CandidateMemory(
        id="cand_project_card_consolidation",
        type="project_state",
        claim="Project Memory v2 current state: project-card consolidation is the next architectural unlock.",
        proposed_destination="semantic/projects/memory-v2.yaml",
        promotion_reason="project_update: current_state",
        confidence=0.91,
        importance=0.87,
        source_refs=["event_new"],
    )
    store.append_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 1
    assert report.promoted_ids == ["project:memory-v2"]
    assert store.list_memory_items(memory_type="project_state") == []
    card = store.read_project_card("Memory v2")
    assert card is not None
    assert card.id == "project:memory-v2"
    assert card.name == "Memory v2"
    assert card.goal == "Build robust source-grounded memory."
    assert card.current_state == "project-card consolidation is the next architectural unlock."
    assert card.decisions == ["Use local files as durable truth."]
    assert card.next_actions == ["Add candidate promotion."]
    assert card.source_refs == ["event_old", "event_new"]
    updated = store.list_candidates()[0]
    assert updated.gate_decision == GateDecision.PROMOTED
    assert "ProjectCard project:memory-v2" in updated.decision_reason
    assert index.search("project card consolidation", route="project_continuity", limit=5)[0]["id"] == "project:memory-v2"


def test_consolidation_project_updates_merge_lists_status_and_dedupe_sources(tmp_path):
    store, index = _store_and_index(tmp_path)
    _seed_sources(store, "event_decision", "event_question", "event_pause")
    candidates = [
        CandidateMemory(
            id="cand_decision",
            type="project_state",
            claim="Project Memory v2 decision: consolidate project updates into semantic/projects cards.",
            proposed_destination="semantic/projects/memory-v2.yaml",
            promotion_reason="project_update: decision",
            source_refs=["event_decision"],
        ),
        CandidateMemory(
            id="cand_question",
            type="project_state",
            claim="Project Memory v2 open question: how strict should source-ref validation be?",
            proposed_destination="semantic/projects/memory-v2.yaml",
            promotion_reason="project_update: open_question",
            source_refs=["event_question"],
        ),
        CandidateMemory(
            id="cand_next_action",
            type="project_state",
            claim="Project Memory v2 next action: add manual promote and reject tools.",
            proposed_destination="semantic/projects/memory-v2.yaml",
            promotion_reason="project_update: next_action",
            source_refs=["event_decision"],
        ),
        CandidateMemory(
            id="cand_paused",
            type="project_state",
            claim="Project Memory v2 status: paused.",
            proposed_destination="semantic/projects/memory-v2.yaml",
            promotion_reason="project_update: status",
            source_refs=["event_pause"],
        ),
    ]
    for candidate in candidates:
        store.append_candidate(candidate)

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 4
    assert report.promoted_ids == ["project:memory-v2"] * 4
    card = store.read_project_card("memory-v2")
    assert card is not None
    assert card.status == ProjectStatus.PAUSED
    assert card.decisions == ["consolidate project updates into semantic/projects cards."]
    assert card.open_questions == ["how strict should source-ref validation be?"]
    assert card.next_actions == ["add manual promote and reject tools."]
    assert card.source_refs == ["event_decision", "event_question", "event_pause"]
    assert {candidate.gate_decision for candidate in store.list_candidates()} == {GateDecision.PROMOTED}


def test_consolidation_rejects_project_update_without_source_refs(tmp_path):
    store, index = _store_and_index(tmp_path)
    store.append_candidate(
        CandidateMemory(
            id="cand_project_no_source",
            type="project_state",
            claim="Project Memory v2 current state: source-less update should not land.",
            proposed_destination="semantic/projects/memory-v2.yaml",
            promotion_reason="project_update: current_state",
        )
    )

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 0
    assert report.rejected == 1
    assert store.list_project_cards() == []
    assert store.list_candidates()[0].gate_decision == GateDecision.REJECTED


def test_consolidation_rejects_dangling_source_refs(tmp_path):
    store, index = _store_and_index(tmp_path)
    store.append_candidate(
        CandidateMemory(
            id="cand_dangling_source",
            type="preference",
            claim="Alex prefers source-backed automatic consolidation.",
            proposed_destination="semantic/items",
            source_refs=["event_missing"],
        )
    )

    report = RuleBasedConsolidator().consolidate(store, index)

    assert report.promoted == 0
    assert report.rejected == 1
    assert store.list_memory_items() == []
    assert store.list_candidates()[0].gate_decision == GateDecision.REJECTED
    assert "dangling" in store.list_candidates()[0].decision_reason.lower()


def test_provider_project_updates_land_in_project_card_and_prefetch_renders_fields(tmp_path):
    from plugins.memory.memory_v2 import MemoryV2Provider

    provider = MemoryV2Provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    provider.sync_turn(
        "Remember that Project Memory v2 next action: improve project-card continuity recall.",
        "Queued.",
        session_id="session-1",
    )

    report = provider.handle_tool_call("memory_v2_consolidate", {})
    card = provider.store.read_project_card("Memory v2")
    packet = provider.prefetch("Where did we leave Memory v2?", session_id="session-1")

    assert '"promoted": 1' in report
    assert provider.store.list_memory_items(memory_type="project_state") == []
    assert card is not None
    assert card.next_actions == ["improve project-card continuity recall."]
    assert "next_actions" in packet
    assert "improve project-card continuity recall" in packet
    assert "project_update: project_update" not in provider.store.list_candidates()[0].promotion_reason


def test_provider_consolidation_tool_reports_counts_and_updates_status(tmp_path):
    from plugins.memory.memory_v2 import MemoryV2Provider

    provider = MemoryV2Provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.sync_turn(
        "Remember that Alex prefers Memory v2 promotion to preserve source refs.",
        "Queued.",
        session_id="session-1",
    )

    result = provider.handle_tool_call("memory_v2_consolidate", {})
    status = provider._status_payload()

    assert '"promoted": 1' in result
    assert status["counts"]["pending_candidates"] == 0
    assert status["counts"]["memory_items"] == 1
