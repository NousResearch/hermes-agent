import json
import time
import uuid

from plugins.memory.ferrosa import FerrosaMemoryProvider, FerrosaSkillProvider
from agent.skill_providers import SkillMetadata, SkillPayload, clear_skill_providers, register_skill_provider


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        if tool_name == "retrieve_skills_for_context":
            return {
                "results": [
                    {
                        "skill_name": "blueprint",
                        "description": "Generates a complete project blueprint.",
                        "category": "task-level",
                        "version": "2026050185",
                    }
                ]
            }
        if tool_name == "invoke_skill":
            assert arguments == {"skill_name": "blueprint"}
            return {
                "skill_name": "blueprint",
                "description": "Generates a complete project blueprint.",
                "category": "task-level",
                "version": "2026050185",
                "first_step_prompt": "Follow the Spec Root Resolution guidance.",
                "steps": [
                    {"phase": "Spec Root Resolution", "instruction": "Resolve spec root."},
                    {"phase": "Phase 0", "instruction": "Run reconnaissance."},
                ],
                "completion_criteria": "Blueprint artifacts are complete.",
                "output_artifacts": ["specs/architecture.md"],
            }
        raise AssertionError(tool_name)


def test_ferrosa_skill_provider_lists_blueprint_from_fmem():
    provider = FerrosaSkillProvider(client=FakeClient())

    skills = provider.list_skills()

    assert len(skills) == 1
    assert skills[0].name == "blueprint"
    assert skills[0].description == "Generates a complete project blueprint."
    assert skills[0].category == "task-level"


def test_ferrosa_skill_provider_resolves_blueprint_as_virtual_skill_payload():
    provider = FerrosaSkillProvider(client=FakeClient())

    payload = provider.resolve_skill("blueprint")

    assert isinstance(payload, SkillPayload)
    assert payload.name == "blueprint"
    assert payload.description == "Generates a complete project blueprint."
    assert "# blueprint" in payload.content
    assert "Resolve spec root." in payload.content
    assert "Blueprint artifacts are complete." in payload.content


class FakeRegisteredProvider:
    def list_skills(self):
        return [
            SkillMetadata(
                name="blueprint",
                description="Generates a complete project blueprint.",
                category="task-level",
            )
        ]

    def resolve_skill(self, name):
        if name != "blueprint":
            return None
        return SkillPayload(
            name="blueprint",
            description="Generates a complete project blueprint.",
            content="---\nname: blueprint\ndescription: Generates a complete project blueprint.\n---\n\n# blueprint\n\nFrom fmem.\n",
        )


def test_skills_list_and_view_load_active_memory_skill_provider(monkeypatch):
    from tools import skills_tool
    import plugins.memory

    clear_skill_providers()

    def fake_register_active_memory_skill_providers():
        register_skill_provider(FakeRegisteredProvider(), namespace="fmem")

    monkeypatch.setattr(
        plugins.memory,
        "register_active_memory_skill_providers",
        fake_register_active_memory_skill_providers,
    )

    listed = json.loads(skills_tool.skills_list())
    assert any(skill["name"] == "fmem:blueprint" for skill in listed["skills"])

    viewed = json.loads(skills_tool.skill_view("fmem:blueprint"))
    assert viewed["success"] is True
    assert viewed["name"] == "fmem:blueprint"
    assert viewed["provider"] == "fmem"
    assert "From fmem." in viewed["content"]


class FakeMemoryClient:
    def __init__(self):
        self.calls = []
        self._next_id = 0
        self.fail_tools = set()

    def call(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        if tool_name in self.fail_tools:
            raise RuntimeError(f"forced failure for {tool_name}")
        if tool_name == "ingest_context_segments":
            return {
                "segments_created": 2,
                "segments_skipped": 0,
                "segments": [
                    {"segment_id": "segment-1", "segment_index": 0},
                    {"segment_id": "segment-2", "segment_index": 1},
                ],
                "edges_created": 2,
                "warnings": [],
            }
        if tool_name == "search_context_segments":
            return {
                "results": [
                    {
                        "segment": {
                            "segment_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                            "segment_text": "raw temporal page about gateway reset",
                            "segment_index": 4,
                        },
                        "score": 0.42,
                        "sources": ["bm25"],
                        "expanded_context": [
                            {"direction": "previous", "segment": {"segment_text": "previous page"}},
                            {"direction": "hit", "segment": {"segment_text": "hit page"}},
                            {"direction": "next", "segment": {"segment_text": "next page"}},
                        ],
                    }
                ]
            }
        if tool_name == "get_context_window":
            return {
                "segments": [
                    {"direction": "previous", "segment": {"segment_text": "window previous page"}},
                    {"direction": "hit", "segment": {"segment_text": "window hit page"}},
                    {"direction": "next", "segment": {"segment_text": "window next page"}},
                ],
                "token_count": 123,
            }
        if tool_name == "hybrid_search":
            return {"results": []}
        if tool_name == "smart_ingest":
            self._next_id += 1
            return {"entity_id": f"entity-{self._next_id}"}
        if tool_name == "write_temporal_fact":
            return {"event_id": "event-1"}
        if tool_name == "get_temporal_chain":
            return {"event_id": "event-1", "fact_text": "ok"}
        if tool_name in {"run_consolidation", "create_edge"}:
            return {"ok": True}
        raise AssertionError(tool_name)


def _provider_with_fake_client(session_id="session-123"):
    provider = FerrosaMemoryProvider()
    provider._client = FakeMemoryClient()
    provider._session_id = session_id
    return provider


def test_ferrosa_memory_provider_records_session_temporal_fact_and_consolidates_same_session():
    provider = _provider_with_fake_client()

    provider.on_session_end([
        {"role": "user", "content": "debug why Discord gateway failed"},
        {"role": "assistant", "content": "Fixed gateway mention-role routing"},
    ])

    calls = provider._client.calls
    assert ("run_consolidation", {"session_id": "session-123"}) in calls

    temporal_calls = [args for name, args in calls if name == "write_temporal_fact"]
    assert temporal_calls
    assert temporal_calls[0]["entity_id"] == "entity-1"
    assert temporal_calls[0]["session_id"] == "session-123"
    assert "Session completed" in temporal_calls[0]["fact_text"]

    verification_calls = [args for name, args in calls if name == "get_temporal_chain"]
    assert verification_calls == [{"entity_id": "entity-1", "session_id": "session-123"}]


def test_ferrosa_memory_provider_prefers_context_segment_ingest_for_pre_compress():
    provider = _provider_with_fake_client()

    note = provider.on_pre_compress([
        {"role": "user", "content": "first debugging page"},
        {"role": "assistant", "content": "second debugging page"},
        {"role": "user", "content": "third debugging page"},
    ])

    assert "Persisted 2" in note
    ingest_calls = [args for name, args in provider._client.calls if name == "ingest_context_segments"]
    assert len(ingest_calls) == 1
    assert uuid.UUID(ingest_calls[0]["session_id"])
    assert ingest_calls[0]["session_id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "hermes-context-segments:session-123"))
    assert ingest_calls[0]["source_session_id"] == "session-123"
    assert ingest_calls[0]["conversation_id"] == "hermes-session-123"
    assert ingest_calls[0]["embed_missing"] is True
    assert [msg["turn_index"] for msg in ingest_calls[0]["messages"]] == [0, 1, 2]
    assert ingest_calls[0]["messages"][0]["role"] == "user"
    assert ingest_calls[0]["messages"][0]["content"] == "first debugging page"

    assert not [name for name, _ in provider._client.calls if name == "create_edge"]
    temporal_calls = [args for name, args in provider._client.calls if name == "write_temporal_fact"]
    assert temporal_calls
    assert "context segments" in temporal_calls[0]["fact_text"]


def test_ferrosa_memory_provider_retries_context_segment_ingest_without_embeddings_before_fallback():
    provider = _provider_with_fake_client()
    original_call = provider._client.call
    attempts = {"ingest_context_segments": 0}

    def fail_embedding_once(tool_name, arguments):
        if tool_name == "ingest_context_segments":
            attempts["ingest_context_segments"] += 1
            if attempts["ingest_context_segments"] == 1:
                provider._client.calls.append((tool_name, arguments))
                raise RuntimeError("embedding endpoint unavailable")
        return original_call(tool_name, arguments)

    provider._client.call = fail_embedding_once

    note = provider.on_pre_compress([
        {"role": "user", "content": "needle-index-term first page"},
        {"role": "assistant", "content": "needle-index-term second page"},
    ])

    assert "context segments" in note
    ingest_calls = [args for name, args in provider._client.calls if name == "ingest_context_segments"]
    assert len(ingest_calls) == 2
    assert ingest_calls[0]["embed_missing"] is True
    assert ingest_calls[1]["embed_missing"] is False
    section_fallbacks = [
        args for name, args in provider._client.calls
        if name == "smart_ingest" and args.get("entity_type") == "section"
    ]
    assert not section_fallbacks


def test_ferrosa_memory_provider_falls_back_to_linked_smart_ingest_chunks_when_segment_tool_unavailable():
    provider = _provider_with_fake_client()
    provider._client.fail_tools.add("ingest_context_segments")

    note = provider.on_pre_compress([
        {"role": "user", "content": "first debugging page"},
        {"role": "assistant", "content": "second debugging page"},
        {"role": "user", "content": "third debugging page"},
    ])

    assert "Persisted" in note
    smart_ingests = [args for name, args in provider._client.calls if name == "smart_ingest"]
    chunk_ingests = [args for args in smart_ingests if args.get("entity_type") == "section"]
    assert len(chunk_ingests) >= 2
    assert all("session_id" not in args for args in chunk_ingests)
    assert any("first debugging page" in args["content"] for args in chunk_ingests)

    edge_calls = [args for name, args in provider._client.calls if name == "create_edge"]
    assert edge_calls
    assert edge_calls[0]["src_entity_id"] == "entity-2"
    assert edge_calls[0]["dst_entity_id"] == "entity-3"
    assert edge_calls[0]["edge_type"] == "related_to"
    assert "next_context_chunk" in edge_calls[0]["metadata"]
    assert edge_calls[1]["src_entity_id"] == "entity-3"
    assert edge_calls[1]["dst_entity_id"] == "entity-2"
    assert edge_calls[1]["edge_type"] == "related_to"
    assert "previous_context_chunk" in edge_calls[1]["metadata"]


def test_ferrosa_memory_provider_prefetch_expands_context_segments_before_entity_fallback():
    provider = _provider_with_fake_client()

    block = provider.prefetch("gateway reset temporal fixes", session_id="session-123")

    search_calls = [args for name, args in provider._client.calls if name == "search_context_segments"]
    expected_segment_session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "hermes-context-segments:session-123"))
    assert search_calls == [
        {
            "query": "gateway reset temporal fixes",
            "session_id": expected_segment_session_id,
            "source_session_id": "session-123",
            "limit": 5,
            "expand": {"prev": 1, "next": 2, "max_tokens": 4000},
        }
    ]
    assert "## Ferrosa Memory Context Segments" in block
    assert "previous page" in block
    assert "hit page" in block
    assert "next page" in block



def test_ferrosa_memory_provider_prefetch_uses_context_window_when_search_hit_is_not_expanded():
    provider = _provider_with_fake_client()

    def call_without_expansion(tool_name, arguments):
        provider._client.calls.append((tool_name, arguments))
        if tool_name == "search_context_segments":
            return {
                "results": [
                    {
                        "segment": {
                            "segment_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                            "segment_text": "hit without embedded expansion",
                        },
                        "score": 0.5,
                        "sources": ["bm25"],
                    }
                ]
            }
        if tool_name == "get_context_window":
            return {
                "segments": [
                    {"direction": "previous", "segment": {"segment_text": "window previous page"}},
                    {"direction": "hit", "segment": {"segment_text": "window hit page"}},
                    {"direction": "next", "segment": {"segment_text": "window next page"}},
                ],
                "token_count": 123,
            }
        raise AssertionError(tool_name)

    provider._client.call = call_without_expansion

    block = provider.prefetch("gateway reset temporal fixes", session_id="session-123")

    expected_segment_session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "hermes-context-segments:session-123"))
    window_calls = [args for name, args in provider._client.calls if name == "get_context_window"]
    assert window_calls == [
        {
            "session_id": expected_segment_session_id,
            "source_session_id": "session-123",
            "segment_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "prev": 1,
            "next": 2,
            "max_tokens": 4000,
        }
    ]
    assert "window previous page" in block
    assert "window hit page" in block
    assert "window next page" in block


def test_ferrosa_memory_provider_prefetch_retries_context_segments_without_session_partition_before_entity_fallback():
    provider = _provider_with_fake_client()

    def call_partition_then_broad(tool_name, arguments):
        provider._client.calls.append((tool_name, arguments))
        if tool_name == "search_context_segments":
            if "session_id" in arguments:
                return {"results": []}
            return {
                "results": [
                    {
                        "segment": {"segment_text": "broad prior session page"},
                        "expanded_context": [
                            {"direction": "hit", "segment": {"segment_text": "broad prior session page"}}
                        ],
                    }
                ]
            }
        if tool_name == "hybrid_search":
            raise AssertionError("entity fallback should not run after broad context hit")
        raise AssertionError(tool_name)

    provider._client.call = call_partition_then_broad

    block = provider.prefetch("prior session memory routing", session_id="session-123")

    search_calls = [args for name, args in provider._client.calls if name == "search_context_segments"]
    assert len(search_calls) == 2
    assert "session_id" in search_calls[0]
    assert "source_session_id" in search_calls[0]
    assert "session_id" not in search_calls[1]
    assert "source_session_id" not in search_calls[1]
    assert "broad prior session page" in block


def test_ferrosa_memory_provider_session_end_does_not_block_on_slow_consolidation():
    provider = _provider_with_fake_client()
    original_call = provider._client.call

    def slow_consolidation(tool_name, arguments):
        if tool_name == "run_consolidation":
            provider._client.calls.append((tool_name, arguments))
            time.sleep(0.25)
            return {"ok": True}
        return original_call(tool_name, arguments)

    provider._client.call = slow_consolidation

    start = time.monotonic()
    provider.on_session_end([
        {"role": "user", "content": "debug why Discord gateway failed"},
        {"role": "assistant", "content": "Fixed gateway mention-role routing"},
    ])
    elapsed = time.monotonic() - start

    assert elapsed < 0.10
    assert any(name == "write_temporal_fact" for name, _ in provider._client.calls)
    assert ("run_consolidation", {"session_id": "session-123"}) in provider._client.calls

