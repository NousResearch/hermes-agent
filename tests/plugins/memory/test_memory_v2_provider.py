"""Tests for the local Memory v2 provider skeleton."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from agent.memory_provider import MemoryProvider
from plugins.memory import load_memory_provider


EXPECTED_MEMORY_V2_DIRS = [
    "working",
    "core",
    "inbox",
    "semantic",
    "semantic/projects",
    "semantic/environment",
    "episodic",
    "episodic/daily",
    "episodic/sessions",
    "graph",
    "indexes",
    "indexes/vector",
    "evals",
    "reports",
    "reports/daily_consolidation",
    "reports/weekly_reflection",
]


def _new_provider():
    module = importlib.import_module("plugins.memory.memory_v2")
    return module.MemoryV2Provider()


def test_memory_v2_provider_loads_through_plugin_loader():
    provider = load_memory_provider("memory_v2")

    assert isinstance(provider, MemoryProvider)
    assert provider.name == "memory_v2"
    assert provider.is_available() is True


def test_initialize_creates_profile_scoped_memory_tree(tmp_path):
    provider = _new_provider()

    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    base_dir = tmp_path / "memory_v2"
    assert provider.base_dir == base_dir
    assert provider.session_id == "session-1"
    assert provider.platform == "cli"
    assert base_dir.is_dir()
    for rel in EXPECTED_MEMORY_V2_DIRS:
        assert (base_dir / rel).is_dir(), rel

    assert (base_dir / "README.md").is_file()
    assert (base_dir / "config.yaml").is_file()


def test_system_prompt_block_is_small_stable_and_not_path_specific(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")

    first = provider.system_prompt_block()
    second = provider.system_prompt_block()

    assert first == second
    assert "Memory v2" in first
    assert "session-1" not in first
    assert str(tmp_path) not in first
    assert len(first) <= 500


def test_prefetch_returns_empty_context_when_no_memories_exist(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    assert provider.prefetch("Where did we leave Memory v2?", session_id="session-1") == ""


def test_status_tool_reports_initialized_profile_scoped_paths(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    schemas = {schema["name"]: schema for schema in provider.get_tool_schemas()}
    assert "memory_v2_status" in schemas
    assert "memory_v2_search" in schemas

    result = json.loads(provider.handle_tool_call("memory_v2_status", {}))

    assert result["success"] is True
    assert result["provider"] == "memory_v2"
    assert result["session_id"] == "session-1"
    assert Path(result["base_dir"]) == tmp_path / "memory_v2"
    assert result["counts"]["pending_candidates"] == 0
    assert result["counts"]["raw_events"] == 0
    assert result["counts"]["indexed_memories"] == 0


def test_sync_turn_appends_raw_event_without_candidate_for_ordinary_turn(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")

    provider.sync_turn("hello", "Hey Dylan — I’m here.", session_id="session-override")

    events = provider.store.read_raw_events()
    candidates = provider.store.list_candidates()

    assert len(events) == 1
    assert events[0]["type"] == "turn"
    assert events[0]["session_id"] == "session-override"
    assert events[0]["provider_session_id"] == "session-1"
    assert events[0]["platform"] == "discord"
    assert events[0]["user_content"] == "hello"
    assert events[0]["assistant_content"] == "Hey Dylan — I’m here."
    assert candidates == []


def test_sync_turn_uses_provider_session_id_when_not_provided(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn("What next?", "File store layer.")

    events = provider.store.read_raw_events()
    assert events[0]["session_id"] == "session-1"


def test_sync_turn_creates_pending_candidate_for_explicit_remember_request(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")

    provider.sync_turn(
        "Remember that Dylan prefers Memory v2 to be source-grounded and low-compute.",
        "Got it — I’ll treat that as a candidate memory.",
        session_id="session-1",
    )

    candidates = provider.store.list_candidates()
    status = json.loads(provider.handle_tool_call("memory_v2_status", {}))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.id.startswith("cand_")
    assert candidate.type.value == "preference"
    assert candidate.gate_decision.value == "pending"
    assert "source-grounded and low-compute" in candidate.claim
    assert candidate.proposed_destination == "semantic/items"
    assert "core_update" in candidate.promotion_reason
    assert candidate.source_refs == [provider.store.read_raw_events()[0]["id"]]
    assert status["counts"]["raw_events"] == 1
    assert status["counts"]["pending_candidates"] == 1
    assert status["counts"]["indexed_memories"] == 2


def test_sync_turn_write_gate_routes_procedures_to_procedure_ref_candidate(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")

    provider.sync_turn(
        "Remember that when modifying Hermes providers, load the hermes-agent skill first.",
        "Queued for review.",
        session_id="session-1",
    )

    candidate = provider.store.list_candidates()[0]
    assert candidate.type.value == "procedure_ref"
    assert candidate.proposed_destination == "skills"
    assert "skill_candidate" in candidate.promotion_reason


def test_sync_turn_write_gate_archives_ephemeral_remember_request_without_candidate(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")

    provider.sync_turn(
        "Remember that today I parked by the blue sign.",
        "Noted for this turn only.",
        session_id="session-1",
    )

    assert len(provider.store.read_raw_events()) == 1
    assert provider.store.list_candidates() == []


def test_memory_v2_search_tool_searches_indexed_sync_turn_content(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    provider.sync_turn(
        "Remember that Memory v2 needs gated writes and source-grounded recall.",
        "Queued as a candidate.",
        session_id="session-1",
    )

    result = json.loads(
        provider.handle_tool_call("memory_v2_search", {"query": "gated writes source grounded", "limit": 5})
    )

    assert result["success"] is True
    assert result["count"] >= 1
    assert result["results"][0]["id"].startswith(("cand_", "event_"))
    assert "source" in result["results"][0]["body"].lower()


def test_sync_turn_skips_empty_turns(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn("   ", "assistant response", session_id="session-1")
    provider.sync_turn("user request", "   ", session_id="session-1")

    assert provider.store.read_raw_events() == []
    assert provider.store.list_candidates() == []


def test_session_switch_updates_provider_session_for_future_writes(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.on_session_switch("session-2", parent_session_id="session-1", reset=True)
    provider.sync_turn("Remember that session switching should be respected.", "Queued.")

    event = provider.store.read_raw_events()[0]
    assert provider.session_id == "session-2"
    assert event["session_id"] == "session-2"
    assert event["provider_session_id"] == "session-2"


def test_sync_turn_redacts_obvious_sensitive_values_in_raw_archive(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn("temporary password is hunter2", "The token abc123secret is not safe to save.")

    raw_json = json.dumps(provider.store.read_raw_events())
    assert "hunter2" not in raw_json
    assert "abc123secret" not in raw_json
    assert "[REDACTED]" in raw_json


def test_sync_turn_redacts_common_credential_formats_in_raw_archive_and_index(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn(
        'password=hunter2 api_key=sk-test Authorization: Bearer *** secret: dontsave token=tok_123',
        'ok',
    )

    raw_json = json.dumps(provider.store.read_raw_events())
    indexed_json = json.dumps(provider.index.search("REDACTED", limit=10))
    for leaked in ("hunter2", "sk-test", "dontsave", "tok_123"):
        assert leaked not in raw_json
        assert leaked not in indexed_json
    assert raw_json.count("[REDACTED]") >= 5


def test_sync_turn_redacts_natural_language_api_key_secret_and_env_key_formats(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.sync_turn("api key sk-test-123 secret dontsave OPENAI_API_KEY sk-live-456", "ok")

    raw_json = json.dumps(provider.store.read_raw_events())
    indexed_json = json.dumps(provider.index.search("REDACTED", limit=10))
    for leaked in ("sk-test-123", "dontsave", "sk-live-456"):
        assert leaked not in raw_json
        assert leaked not in indexed_json
    assert raw_json.count("[REDACTED]") >= 3


def test_working_memory_tracks_current_focus_and_last_exchange(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")

    provider.on_turn_start(3, "Now make the working memory+open loops complete", model="test-model")
    current = provider.store.read_current_working_memory()

    assert current is not None
    assert current.session_id == "session-1"
    assert current.focus["turn_number"] == 3
    assert current.focus["current_user_message"] == "Now make the working memory+open loops complete"
    assert current.focus["platform"] == "discord"
    assert current.focus["model"] == "test-model"

    provider.sync_turn("Remember that Memory v2 current plan is working memory.", "Queued.", session_id="session-1")
    updated = provider.store.read_current_working_memory()
    assert updated.focus["last_user_message"] == "Remember that Memory v2 current plan is working memory."
    assert updated.focus["last_assistant_message"] == "Queued."
    assert len(updated.scratchpad["retrieved_memory_ids"]) == 1


def test_consolidation_persists_open_loop_candidates_and_prefetch_recalls_them(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    provider.sync_turn(
        "Remember to follow up on the memory eval dashboard tomorrow.",
        "I’ll track that as an open loop.",
        session_id="session-1",
    )

    result = json.loads(provider.handle_tool_call("memory_v2_consolidate", {}))
    loops = provider.store.list_open_loops()
    prefetch = provider.prefetch("what open loops are pending?", session_id="session-1")

    assert result["archived_only"] == 1
    assert len(loops) == 1
    assert loops[0]["status"] == "open"
    assert "follow up on the memory eval dashboard tomorrow" in loops[0]["text"]
    assert loops[0]["source_refs"] == [provider.store.read_raw_events()[0]["id"]]
    assert "working_open_loops" in prefetch
    assert "follow up on the memory eval dashboard tomorrow" in prefetch


def test_session_end_archives_working_memory_snapshot_without_dropping_open_loops(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.on_turn_start(1, "Continue Memory v2")
    provider.sync_turn("Remember to follow up on dangling source refs tomorrow.", "Tracked.")
    provider.handle_tool_call("memory_v2_consolidate", {})

    provider.on_session_end([
        {"role": "user", "content": "Continue Memory v2"},
        {"role": "assistant", "content": "Working on it."},
    ])

    archives = provider.store.list_session_archives()
    assert len(archives) == 1
    assert archives[0]["session_id"] == "session-1"
    assert archives[0]["working_memory"]["focus"]["current_user_message"] == "Continue Memory v2"
    assert archives[0]["open_loops"][0]["status"] == "open"
    assert provider.store.read_current_working_memory() is None
    assert len(provider.store.list_open_loops()) == 1


def test_manual_control_tool_schemas_are_exposed(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    schemas = {schema["name"] for schema in provider.get_tool_schemas()}

    assert {
        "memory_v2_candidates",
        "memory_v2_promote",
        "memory_v2_reject",
        "memory_v2_show_source",
        "memory_v2_resolve_open_loop",
    }.issubset(schemas)


def test_candidates_tool_lists_and_filters_candidates(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    provider.sync_turn("Remember that Dylan prefers source backed memory.", "Queued.")
    provider.sync_turn("Remember to follow up on memory evals tomorrow.", "Tracked.")

    all_result = json.loads(provider.handle_tool_call("memory_v2_candidates", {}))
    filtered = json.loads(provider.handle_tool_call("memory_v2_candidates", {"type": "episode", "status": "pending"}))

    assert all_result["success"] is True
    assert all_result["count"] == 2
    assert {candidate["type"] for candidate in all_result["candidates"]} == {"preference", "episode"}
    assert filtered["count"] == 1
    assert filtered["candidates"][0]["type"] == "episode"
    assert "follow up on memory evals" in filtered["candidates"][0]["claim"]


def test_manual_reject_updates_candidate_status_and_index(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    provider.sync_turn("Remember that Dylan prefers noisy landfill memory.", "Queued.")
    candidate_id = provider.store.list_candidates()[0].id

    result = json.loads(
        provider.handle_tool_call("memory_v2_reject", {"candidate_id": candidate_id, "reason": "bad candidate"})
    )

    candidate = provider.store.list_candidates()[0]
    indexed = provider.index.search("bad candidate", limit=5)[0]
    assert result["success"] is True
    assert result["candidate"]["gate_decision"] == "rejected"
    assert candidate.gate_decision.value == "rejected"
    assert candidate.decision_reason == "bad candidate"
    assert provider.store.list_rejected_candidates()[0].id == candidate_id
    assert indexed["id"] == candidate_id
    assert indexed["status"] == "rejected"


def test_manual_promote_promotes_specific_candidate_and_show_source_uses_canonical_source_ref(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    provider.sync_turn("Remember that Dylan prefers manual memory controls.", "Queued.", session_id="session-1")
    candidate = provider.store.list_candidates()[0]
    event_id = provider.store.read_raw_events()[0]["id"]

    promoted = json.loads(provider.handle_tool_call("memory_v2_promote", {"candidate_id": candidate.id}))
    source = json.loads(provider.handle_tool_call("memory_v2_show_source", {"id": promoted["promoted_ids"][0]}))
    direct_source = json.loads(provider.handle_tool_call("memory_v2_show_source", {"id": event_id}))

    assert promoted["success"] is True
    assert promoted["promoted"] == 1
    assert promoted["promoted_ids"][0].startswith("mem_preference_")
    assert provider.store.list_candidates()[0].gate_decision.value == "promoted"
    assert source["success"] is True
    assert source["record"]["id"] == promoted["promoted_ids"][0]
    assert source["sources"][0]["id"] == event_id
    assert source["sources"][0]["type"] == "message"
    assert source["sources"][0]["uri"] == f"raw_event:{event_id}"
    assert "manual memory controls" in source["sources"][0]["quote"]
    assert direct_source["record"]["type"] == "source_ref"
    assert direct_source["sources"][0] == source["sources"][0]


def test_manual_promote_refuses_missing_or_dangling_sources_unless_forced(tmp_path):
    from plugins.memory.memory_v2.schemas import CandidateMemory

    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    provider.store.append_candidate(
        CandidateMemory(id="cand_no_source", type="fact", claim="Detached fact", proposed_destination="semantic/items")
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_dangling",
            type="fact",
            claim="Dangling source fact",
            proposed_destination="semantic/items",
            source_refs=["source_missing"],
        )
    )

    no_source = json.loads(provider.handle_tool_call("memory_v2_promote", {"candidate_id": "cand_no_source"}))
    dangling = json.loads(provider.handle_tool_call("memory_v2_promote", {"candidate_id": "cand_dangling"}))
    forced = json.loads(provider.handle_tool_call("memory_v2_promote", {"candidate_id": "cand_dangling", "force": True}))

    assert no_source["success"] is False
    assert "source_refs" in no_source["error"]
    assert dangling["success"] is False
    assert "dangling" in dangling["error"]
    assert forced["success"] is True
    assert forced["promoted"] == 1


def test_resolve_open_loop_tool_updates_status_and_preserves_history(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="discord")
    provider.sync_turn("Remember to follow up on memory evals tomorrow.", "Tracked.")
    provider.handle_tool_call("memory_v2_consolidate", {})
    loop_id = provider.store.list_open_loops()[0]["id"]

    result = json.loads(
        provider.handle_tool_call(
            "memory_v2_resolve_open_loop",
            {"loop_id": loop_id, "status": "resolved", "resolution": "eval follow-up completed"},
        )
    )

    loops = provider.store.list_open_loops()
    assert result["success"] is True
    assert result["loop"]["status"] == "resolved"
    assert loops[0]["status"] == "resolved"
    assert loops[0]["resolution"] == "eval follow-up completed"
    assert loops[0]["resolved_at"]
    assert provider.store.list_open_loops(status="open") == []


def test_prefetch_redacts_common_credential_formats_in_retrieval_log(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.prefetch('api_key=sk-test Authorization: Bearer *** token tok_123 password hunter2 bearer bearer_123 session_id=session-1')

    logs_json = json.dumps(provider.index.retrieval_logs())
    for leaked in ("sk-test", "tok_123", "hunter2", "bearer_123"):
        assert leaked not in logs_json
    assert "query_hash" in logs_json


def test_prefetch_redacts_natural_language_credential_formats_in_retrieval_log(tmp_path):
    provider = _new_provider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.prefetch('token tok_123 password hunter2 bearer bearer_123', session_id="session-1")

    logs_json = json.dumps(provider.index.retrieval_logs())
    for leaked in ("tok_123", "hunter2", "bearer_123"):
        assert leaked not in logs_json
