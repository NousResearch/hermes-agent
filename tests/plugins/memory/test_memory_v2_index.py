"""SQLite FTS index/search tests for Memory v2."""

from __future__ import annotations

import sqlite3

from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem, MemoryType, ProjectCard, SourceRef
from plugins.memory.memory_v2.store import MemoryV2Store


def _store(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    return store


def test_index_initialize_creates_sqlite_schema(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")

    index.initialize()

    assert index.db_path.is_file()
    tables = index.table_names()
    assert "memories" in tables
    assert "memories_fts" in tables
    assert "retrieval_log" in tables
    assert "source_refs" in tables


def test_initialize_rebuilds_legacy_fts_table_with_missing_columns(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(index.db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE memories (
              id TEXT PRIMARY KEY,
              type TEXT NOT NULL,
              title TEXT,
              body TEXT,
              summary TEXT,
              status TEXT,
              updated_at TEXT NOT NULL,
              source_refs TEXT,
              tags TEXT,
              file_path TEXT
            );
            CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, title, body, summary);
            """
        )

    index.initialize()
    item = MemoryItem(
        id="pref_response_style",
        type="preference",
        subject="Alex",
        predicate="prefers_response_style",
        value="direct answers",
        summary="Alex prefers direct answers.",
        source_refs=["event_1"],
    )
    index.index_memory_item(item)

    assert index.search("direct answers", limit=5)[0]["id"] == "pref_response_style"


def test_index_project_card_is_searchable(tmp_path):
    store = _store(tmp_path)
    card = ProjectCard(
        id="Hermes Memory v2",
        name="Hermes Memory v2",
        goal="Build robust source-grounded low-compute memory for Hermes.",
        current_state="SQLite FTS search layer under implementation.",
        decisions=["Use project-card block retrieval before chunk retrieval."],
        source_refs=["source_memory_v2_thread"],
    )
    store.write_project_card(card)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_project_card(card, file_path=store.projects_dir / "hermes-memory-v2.yaml")
    results = index.search("source grounded memory", limit=5)

    assert len(results) == 1
    assert results[0]["id"] == "project:hermes-memory-v2"
    assert results[0]["type"] == "project_state"
    assert "source-grounded" in results[0]["body"]
    assert results[0]["source_refs"] == ["source_memory_v2_thread"]


def test_index_candidate_is_searchable(tmp_path):
    store = _store(tmp_path)
    candidate = CandidateMemory(
        id="cand_memory_v2_goal",
        type=MemoryType.PROJECT_STATE,
        claim="Alex wants Memory v2 to stay low-compute and source-grounded.",
        source_refs=["event_123"],
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_candidate(candidate)
    results = index.search("low compute source grounded", limit=5)

    assert [result["id"] for result in results] == ["cand_memory_v2_goal"]
    assert results[0]["type"] == "candidate"
    assert results[0]["status"] == "pending"


def test_index_raw_event_is_searchable(tmp_path):
    store = _store(tmp_path)
    event = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Remember that Memory v2 needs gated writes.",
            "assistant_content": "Queued as a candidate.",
        }
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_raw_event(event)
    results = index.search("gated writes", limit=5)

    assert len(results) == 1
    assert results[0]["id"] == event["id"]
    assert results[0]["type"] == "raw_event"
    assert results[0]["source_refs"] == [event["id"]]


def test_rebuild_from_store_indexes_project_cards_candidates_and_events(tmp_path):
    store = _store(tmp_path)
    store.write_project_card(
        ProjectCard(
            id="Hermes Memory v2",
            name="Hermes Memory v2",
            goal="Build robust memory search.",
            current_state="Rebuild indexes existing files.",
        )
    )
    store.append_candidate(
        CandidateMemory(
            id="cand_eval_contract",
            type="fact",
            claim="Memory v2 should be benchmark-first and source-grounded.",
        )
    )
    store.append_raw_event(
        {"type": "turn", "session_id": "session-1", "user_content": "Need retrieval logs", "assistant_content": "Next."}
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)

    assert counts == {"project_cards": 1, "candidates": 1, "raw_events": 1, "source_refs": 1, "memory_items": 0, "open_loops": 0}
    assert index.count_memories() == 3
    source = store.list_source_refs()[0]
    indexed_source = index.source_ref(source.id)
    assert indexed_source is not None
    assert indexed_source["uri"] == f"raw_event:{source.id}"
    assert index.search("benchmark first", limit=5)[0]["id"] == "cand_eval_contract"
    assert index.search("retrieval logs", limit=5)[0]["type"] == "raw_event"


def test_rebuild_from_store_indexes_memory_items_with_structured_fields(tmp_path):
    store = _store(tmp_path)
    item = MemoryItem(
        id="pref_response_style",
        type="preference",
        subject="Alex",
        predicate="prefers_response_style",
        value="direct no-BS tool-grounded help",
        summary="Alex prefers direct, no-BS, tool-grounded help.",
        confidence=0.98,
        importance=0.95,
        source_refs=["source_user_profile"],
        tags=["user_preference", "style"],
    )
    store.write_memory_item(item)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)
    results = index.search("response style direct no BS", route="preference_recall", limit=5)

    assert counts == {"project_cards": 0, "candidates": 0, "raw_events": 0, "source_refs": 0, "memory_items": 1, "open_loops": 0}
    assert len(results) == 1
    result = results[0]
    assert result["id"] == "pref_response_style"
    assert result["type"] == "preference"
    assert result["subject"] == "Alex"
    assert result["predicate"] == "prefers_response_style"
    assert result["value"] == "direct no-BS tool-grounded help"
    assert result["confidence"] == 0.98
    assert result["importance"] == 0.95
    assert result["source_refs"] == ["source_user_profile"]


def test_memory_item_supersession_fields_survive_index_search(tmp_path):
    store = _store(tmp_path)
    old = MemoryItem(
        id="pref_tts_voice_old",
        type="preference",
        subject="Alex",
        predicate="preferred_tts_voice",
        value="en-US-ChristopherNeural",
        summary="Old voice preference.",
        status="superseded",
        superseded_by="pref_tts_voice_current",
        source_refs=["source_old_voice"],
        tags=["voice"],
    )
    current = MemoryItem(
        id="pref_tts_voice_current",
        type="preference",
        subject="Alex",
        predicate="preferred_tts_voice",
        value="en-US-AndrewNeural",
        summary="Current voice preference.",
        supersedes=["pref_tts_voice_old"],
        source_refs=["source_new_voice"],
        tags=["voice"],
    )
    store.write_memory_item(old)
    store.write_memory_item(current)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.rebuild_from_store(store)

    results = index.search("preferred tts voice", route="preference_recall", limit=5)

    by_id = {result["id"]: result for result in results}
    assert list(by_id)[0] == "pref_tts_voice_current"
    assert by_id["pref_tts_voice_current"]["supersedes"] == ["pref_tts_voice_old"]
    assert by_id["pref_tts_voice_old"]["superseded_by"] == "pref_tts_voice_current"


def test_rebuild_from_store_indexes_source_refs_for_packet_expansion(tmp_path):
    store = _store(tmp_path)
    source = SourceRef(
        id="source_memory_v2_thread",
        type="session",
        uri="discord://thread/memory-v2-thread",
        title="Memory v2 design thread",
        observed_at="2026-05-26T00:00:00Z",
        quote="Alex asked for robust low-compute memory.",
    )
    store.write_source_ref(source)
    store.write_project_card(
        ProjectCard(
            id="Hermes Memory v2",
            name="Hermes Memory v2",
            goal="Build source-backed memory packets.",
            current_state="SourceRef store and index rebuild are next.",
            source_refs=["source_memory_v2_thread"],
        )
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)

    assert counts == {"project_cards": 1, "candidates": 0, "raw_events": 0, "source_refs": 1, "memory_items": 0, "open_loops": 0}
    assert index.source_ref("source_memory_v2_thread") == source.to_dict()
    result = index.search("source backed memory packets", route="project_continuity", limit=5)[0]
    assert result["id"] == "project:hermes-memory-v2"
    assert result["source_refs"] == ["source_memory_v2_thread"]


def test_search_limit_and_empty_query(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    for i in range(3):
        index.index_candidate(
            CandidateMemory(
                id=f"cand_{i}",
                type="fact",
                claim=f"Memory v2 search limit shared term {i}",
            )
        )

    assert len(index.search("shared term", limit=2)) == 2
    assert index.search("   ") == []


def test_search_logs_retrieval_decision(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_candidate(CandidateMemory(id="cand_log", type="fact", claim="Retrieval log smoke test."))

    results = index.search("retrieval log", route="project_continuity", limit=5)

    logs = index.retrieval_logs()
    assert results
    assert len(logs) == 1
    assert logs[0]["query"] == "retrieval log"
    assert logs[0]["route"] == "project_continuity"
    assert logs[0]["retrieved_ids"] == ["cand_log"]


def test_index_search_rejects_invalid_limit(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    for bad_limit in (None, "bad"):
        try:
            index.search("memory", limit=bad_limit)
        except ValueError as exc:
            assert "limit" in str(exc)
        else:
            raise AssertionError("invalid limit should raise ValueError")


def test_search_excludes_expired_and_not_yet_valid_active_memories(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_memory_item(
        MemoryItem(
            id="pref_expired",
            type="preference",
            subject="Alex",
            value="Alex prefers expired retrieval behavior.",
            summary="Expired retrieval behavior.",
            expires_at="2000-01-01T00:00:00Z",
            source_refs=["event_expired"],
        )
    )
    index.index_memory_item(
        MemoryItem(
            id="pref_future",
            type="preference",
            subject="Alex",
            value="Alex prefers future retrieval behavior.",
            summary="Future retrieval behavior.",
            valid_from="2999-01-01T00:00:00Z",
            source_refs=["event_future"],
        )
    )
    index.index_memory_item(
        MemoryItem(
            id="pref_current",
            type="preference",
            subject="Alex",
            value="Alex prefers current retrieval behavior.",
            summary="Current retrieval behavior.",
            source_refs=["event_current"],
        )
    )

    results = index.search("retrieval behavior", route="preference_recall", limit=10)

    assert [result["id"] for result in results] == ["pref_current"]


def test_search_orders_candidate_gate_decisions_explicitly(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    for candidate in (
        CandidateMemory(id="cand_rejected", type="fact", claim="shared candidate ordering", gate_decision="rejected", decision_reason="no"),
        CandidateMemory(id="cand_archived", type="fact", claim="shared candidate ordering", gate_decision="archived_only", decision_reason="archive"),
        CandidateMemory(id="cand_promoted", type="fact", claim="shared candidate ordering", gate_decision="promoted", decision_reason="promoted"),
        CandidateMemory(id="cand_pending", type="fact", claim="shared candidate ordering"),
    ):
        index.index_candidate(candidate)

    results = index.search("shared candidate ordering", limit=10)

    assert [result["id"] for result in results] == ["cand_pending", "cand_promoted", "cand_archived", "cand_rejected"]


def test_index_raw_event_preserves_event_timestamps(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_raw_event(
        {
            "id": "event_old",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Memory v2 historical timestamp evidence.",
            "created_at": "2001-01-01T00:00:00Z",
            "updated_at": "2001-01-02T00:00:00Z",
        }
    )

    result = index.search("historical timestamp evidence", route="past_conversation_exact", limit=1)[0]

    assert result["created_at"] == "2001-01-01T00:00:00Z"
    assert result["updated_at"] == "2001-01-02T00:00:00Z"


def test_hybrid_search_uses_field_overlap_to_rerank_relaxed_fts_matches(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_raw_event(
        {
            "id": "event_noise",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Memory retrieval routing mentioned a generic benchmark note.",
        }
    )
    index.index_raw_event(
        {
            "id": "event_target",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "LoCoMo evidence retrieval recall benchmark improved hybrid Memory v2 ranking.",
        }
    )

    results = index.search("LoCoMo evidence retrieval benchmark missing", route="research_recall", limit=2)

    assert [result["id"] for result in results] == ["event_target", "event_noise"]
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]
    assert results[0]["score_components"]["token_overlap"] > results[1]["score_components"]["token_overlap"]


def test_hybrid_search_boosts_project_cards_for_project_continuity_route(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_raw_event(
        {
            "id": "event_raw_project",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Project Memory v2 status raw note: old exploratory benchmark chatter.",
        }
    )
    index.index_project_card(
        ProjectCard(
            id="Memory v2",
            name="Memory v2",
            goal="Build robust memory.",
            current_state="Project Memory v2 status: hybrid retrieval implementation is current.",
            source_refs=["source_project_current"],
        )
    )

    results = index.search("Project Memory v2 status", route="project_continuity", limit=2)

    assert results[0]["id"] == "project:memory-v2"
    assert results[0]["type"] == "project_state"
    assert results[0]["score_components"]["route_type_boost"] > 0


def test_hybrid_search_boosts_preferences_for_preference_recall_route(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_raw_event(
        {
            "id": "event_pref_raw",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Alex prefers concise answers was mentioned in raw chat.",
        }
    )
    index.index_memory_item(
        MemoryItem(
            id="pref_concise_answers",
            type="preference",
            subject="Alex",
            predicate="prefers_response_style",
            value="concise answers",
            summary="Alex prefers concise answers.",
            source_refs=["event_pref_raw"],
        )
    )

    results = index.search("what does Alex prefer for answers", route="preference_recall", limit=2)

    assert results[0]["id"] == "pref_concise_answers"
    assert results[0]["type"] == "preference"
    assert results[0]["score_components"]["route_type_boost"] > 0


def test_rebuild_from_store_skips_raw_events_missing_ids(tmp_path):
    store = _store(tmp_path)
    store._append_jsonl(store.raw_events_path, {"type": "turn", "user_content": "legacy missing id"})
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)

    assert counts["raw_events"] == 0
    assert index.count_memories() == 0


def test_rebuild_from_store_indexes_open_loops_and_stopword_queries_do_not_match(tmp_path):
    store = _store(tmp_path)
    store.upsert_open_loop({"id": "loop_dragonfruit", "text": "finish dragonfruit migration", "source_refs": ["event_loop"]})
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)

    assert counts["open_loops"] == 1
    assert index.search("dragonfruit migration", limit=5)[0]["id"] == "loop_dragonfruit"
    assert index.search("the and or", limit=5) == []


def test_retrieval_log_redacts_query_and_hashes_redacted_sentinel(tmp_path):
    import hashlib

    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    query = "Authorization:\nBearer fakebearertoken123456789"
    index.log_retrieval(query, route="current_task", retrieved_ids=[])
    log = index.retrieval_logs()[0]

    assert log["query"] == "[REDACTED sensitive query]"
    assert log["query_hash"] != hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert log["query_hash"] == hashlib.sha256(b"[REDACTED sensitive query]").hexdigest()
