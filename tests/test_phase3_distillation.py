"""Tests for Phase 3 — Distillation & Recall pipeline.

Covers:
  - extraction.py: entity ID generation, turn formatting
  - merge.py: entity upsert, episode creation from facts
  - compress.py: D0/D1/D2 compression (with mocked LLM calls)
  - context_injector.py: token-budgeted context assembly
  - Provider recall tools: memory_grep, memory_expand, memory_describe
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memory.episodic_store import EpisodicStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh episodic store for testing."""
    db_path = tmp_path / "test_index.db"
    s = EpisodicStore(db_path=db_path)
    s.ensure_session("test-session-1", source="test")
    yield s
    s.close()


# ── extraction.py ────────────────────────────────────────────────────────


class TestExtraction:
    def test_entity_id_from_name(self):
        from memory.extraction import entity_id_from_name

        assert entity_id_from_name("Aaron", "person") == "person-aaron"
        assert entity_id_from_name("Hermes Agent", "project") == "project-hermes-agent"
        assert entity_id_from_name("GitHub CLI", "tool") == "tool-github-cli"
        assert entity_id_from_name("  Spaces  ", "concept") == "concept-spaces"

    def test_entity_id_from_name_uses_canonical_person_name_over_email(self):
        from memory.extraction import entity_id_from_name

        assert entity_id_from_name("darren@apex-z.com", "person", {"full_name": "Darren Aion"}) == "person-darren-aion"

    def test_entity_id_from_name_treats_url_as_resource(self):
        from memory.extraction import entity_id_from_name

        assert entity_id_from_name(
            "https://docs.google.com/document/d/abc123/edit",
            "location",
            {"title": "Episode Memory Release Notes", "role": "shared document URL"},
        ) == "resource-episode-memory-release-notes"

    def test_turns_to_extraction_input(self):
        from memory.extraction import turns_to_extraction_input

        turns = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        batch = turns_to_extraction_input(turns, start_idx=5, batch_size=10)
        assert len(batch) == 10
        assert batch[0]["content"] == "msg 5"

        # Edge: beyond end
        batch = turns_to_extraction_input(turns, start_idx=15, batch_size=10)
        assert len(batch) == 5

    def test_format_turns(self):
        from memory.extraction import _format_turns

        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "result", "tool_name": "search"},
        ]
        formatted = _format_turns(turns)
        assert "[user] Hello" in formatted
        assert "[assistant] Hi there" in formatted
        assert "[Tool: search] result" in formatted

    @patch("memory.extraction.call_llm")
    def test_extract_from_turns_success(self, mock_call_llm):
        from memory.extraction import extract_from_turns

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "entities": [{"name": "Hermes", "type": "project", "attributes": {"status": "active"}}],
            "facts": [{"subject": "user", "predicate": "prefers", "object": "Python", "confidence": "high"}],
            "events": [],
        })
        mock_call_llm.return_value = mock_response

        turns = [{"role": "user", "content": "I prefer Python for this project"}]
        result = extract_from_turns(turns)

        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Hermes"
        assert len(result["facts"]) == 1
        assert result["facts"][0]["object"] == "Python"
        mock_call_llm.assert_called_once()

    @patch("memory.extraction.call_llm")
    def test_extract_from_turns_with_markdown_fences(self, mock_call_llm):
        from memory.extraction import extract_from_turns

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"entities": [], "facts": [], "events": []}\n```'
        mock_call_llm.return_value = mock_response

        result = extract_from_turns([{"role": "user", "content": "test"}])
        assert result == {"entities": [], "facts": [], "events": []}

    @patch("memory.extraction.call_llm")
    def test_extract_from_turns_invalid_json(self, mock_call_llm):
        from memory.extraction import extract_from_turns

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not json at all"
        mock_call_llm.return_value = mock_response

        result = extract_from_turns([{"role": "user", "content": "test"}])
        assert result == {"entities": [], "facts": [], "events": []}

    @patch("memory.extraction.call_llm")
    def test_extract_from_turns_llm_error(self, mock_call_llm):
        from memory.extraction import extract_from_turns

        mock_call_llm.side_effect = Exception("API error")

        result = extract_from_turns([{"role": "user", "content": "test"}])
        assert result == {"entities": [], "facts": [], "events": []}

    def test_extract_from_turns_empty(self):
        from memory.extraction import extract_from_turns

        result = extract_from_turns([])
        assert result == {"entities": [], "facts": [], "events": []}


# ── merge.py ─────────────────────────────────────────────────────────────


class TestMerge:
    def test_merge_entities_only(self, store):
        from memory.merge import merge_extracted_facts

        extracted = {
            "entities": [
                {"name": "Aaron", "type": "person", "attributes": {"role": "business partner"}},
                {"name": "Hermes", "type": "project", "attributes": {"language": "Python"}},
            ],
            "facts": [],
            "events": [],
        }

        stats = merge_extracted_facts(store, extracted, "test-session-1")
        assert stats["added"] == 2
        assert stats["errors"] == 0

        # Verify entities were created
        aaron = store.get_entity("person-aaron")
        assert aaron is not None
        assert aaron["name"] == "Aaron"
        assert aaron["type"] == "person"
        assert aaron["profile_json"]["role"] == "business partner"

    def test_merge_entity_update(self, store):
        from memory.merge import merge_extracted_facts

        # First merge
        extracted1 = {
            "entities": [{"name": "Hermes", "type": "project", "attributes": {"language": "Python"}}],
            "facts": [],
            "events": [],
        }
        merge_extracted_facts(store, extracted1, "test-session-1")

        # Second merge with new attributes
        extracted2 = {
            "entities": [{"name": "Hermes", "type": "project", "attributes": {"version": "2.0"}}],
            "facts": [],
            "events": [],
        }
        stats = merge_extracted_facts(store, extracted2, "test-session-2")
        assert stats["added"] == 1

        # Profile should be merged
        hermes = store.get_entity("project-hermes")
        assert hermes is not None
        assert hermes["profile_json"]["language"] == "Python"  # Old
        assert hermes["profile_json"]["version"] == "2.0"  # New

    def test_merge_creates_episode_from_facts(self, store):
        from memory.merge import merge_extracted_facts

        extracted = {
            "entities": [],
            "facts": [
                {"subject": "user", "predicate": "uses", "object": "Ollama", "confidence": "high"},
                {"subject": "project", "predicate": "status", "object": "active", "confidence": "high"},
            ],
            "events": [],
        }

        stats = merge_extracted_facts(store, extracted, "test-session-1")
        assert stats["added"] >= 1  # At least one episode created

        # Check episodes
        episodes = store.get_recent_episodes(limit=5)
        assert len(episodes) >= 1
        assert "Ollama" in episodes[0]["summary"]

    def test_merge_creates_episode_from_events(self, store):
        from memory.merge import merge_extracted_facts

        extracted = {
            "entities": [],
            "facts": [],
            "events": [
                {"description": "Deployed v2.0 to production", "participants": ["Hermes", "Jefe"]},
            ],
        }

        stats = merge_extracted_facts(store, extracted, "test-session-1")
        assert stats["added"] >= 1

        episodes = store.get_recent_episodes(limit=5)
        assert any("Deployed" in ep["summary"] for ep in episodes)

    def test_merge_empty(self, store):
        from memory.merge import merge_session

        stats = merge_session(store, "test-session-1", {"entities": [], "facts": [], "events": []})
        assert stats == {"added": 0, "updated": 0, "deleted": 0, "noop": 0, "errors": 0}

    def test_merge_skips_invalid_entities(self, store):
        from memory.merge import merge_extracted_facts

        extracted = {
            "entities": [
                {"name": "", "type": "person"},  # Empty name
                {"type": "project"},  # No name key
                {"name": "Valid", "type": "tool", "attributes": {}},
            ],
            "facts": [],
            "events": [],
        }

        stats = merge_extracted_facts(store, extracted, "test-session-1")
        assert stats["added"] == 1  # Only the valid one

    def test_merge_canonicalizes_person_email_to_full_name(self, store):
        from memory.merge import merge_extracted_facts

        extracted = {
            "entities": [
                {
                    "name": "darren@apex-z.com",
                    "type": "person",
                    "attributes": {"full_name": "Darren Aion", "preferred_name": "Jefe", "access": "writer"},
                }
            ],
            "facts": [],
            "events": [],
        }

        stats = merge_extracted_facts(store, extracted, "test-session-1")
        assert stats["added"] == 1

        entity = store.get_entity("person-darren-aion")
        assert entity is not None
        assert entity["name"] == "Darren Aion"
        assert "darren@apex-z.com" in entity["profile_json"]["aliases"]
        assert entity["profile_json"]["preferred_name"] == "Jefe"

    def test_merge_canonicalizes_doc_url_to_resource(self, store):
        from memory.merge import merge_extracted_facts

        extracted = {
            "entities": [
                {
                    "name": "https://docs.google.com/document/d/abc123/edit",
                    "type": "location",
                    "attributes": {"title": "Episode Memory Release Notes", "role": "shared document URL", "platform": "Google Docs"},
                }
            ],
            "facts": [],
            "events": [],
        }

        merge_extracted_facts(store, extracted, "test-session-1")

        entity = store.get_entity("resource-episode-memory-release-notes")
        assert entity is not None
        assert entity["type"] == "resource"
        assert entity["name"] == "Episode Memory Release Notes"
        assert "https://docs.google.com/document/d/abc123/edit" in entity["profile_json"]["aliases"]

    def test_merge_reuses_existing_canonical_person_entity(self, store):
        from memory.merge import merge_extracted_facts

        store.upsert_entity(
            entity_id="person-darren-aion",
            entity_type="person",
            name="Darren Aion",
            profile_json={"preferred_name": "Jefe", "aliases": ["Jefe"]},
        )

        extracted = {
            "entities": [
                {
                    "name": "darren@apex-z.com",
                    "type": "person",
                    "attributes": {"full_name": "Darren Aion", "access": "writer"},
                }
            ],
            "facts": [],
            "events": [],
        }

        merge_extracted_facts(store, extracted, "test-session-1")

        entity = store.get_entity("person-darren-aion")
        assert entity is not None
        assert "darren@apex-z.com" in entity["profile_json"]["aliases"]
        assert entity["profile_json"]["access"] == "writer"


# ── compress.py ──────────────────────────────────────────────────────────


class TestCompress:
    @patch("memory.compress.call_llm")
    def test_compress_session_to_d0(self, mock_call_llm, store):
        from memory.compress import compress_session_to_d0

        # Add some turns
        for i in range(5):
            store.append_turn("test-session-1", "user", f"User message {i}")
            store.append_turn("test-session-1", "assistant", f"Assistant reply {i}")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "topic": "Test conversation",
            "summary": "A test conversation about testing.",
            "key_decisions": ["Use pytest"],
            "unresolved": [],
            "participants": ["user"],
        })
        mock_call_llm.return_value = mock_response

        turns = store.get_turns_for_session("test-session-1")
        node_id = compress_session_to_d0(store, "test-session-1", turns)

        assert node_id is not None
        assert node_id.startswith("d0-")

        # Verify DAG node
        node = store.get_dag_node(node_id)
        assert node is not None
        assert node["depth"] == 0
        content = json.loads(node["content"])
        assert content["topic"] == "Test conversation"

        # Verify episode was also created
        episodes = store.get_recent_episodes(limit=5)
        assert len(episodes) >= 1

    def test_compress_session_empty(self, store):
        from memory.compress import compress_session_to_d0

        result = compress_session_to_d0(store, "test-session-1", [])
        assert result is None

    @patch("memory.compress.call_llm")
    def test_compress_d0_to_d1(self, mock_call_llm, store):
        from memory.compress import compress_d0_to_d1

        # Create D0 nodes first
        store.create_dag_node("d0-s1-e1", [], 0, json.dumps({"topic": "Setup", "summary": "Setup work"}))
        store.create_dag_node("d0-s2-e1", [], 0, json.dumps({"topic": "Testing", "summary": "Testing work"}))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "date": "2026-04-21",
            "key_facts": ["Completed setup", "Ran tests"],
            "decisions": ["Use pytest"],
            "active_entities": ["Hermes"],
            "summary": "A productive day of setup and testing.",
        })
        mock_call_llm.return_value = mock_response

        node_id = compress_d0_to_d1(store, "2026-04-21", ["d0-s1-e1", "d0-s2-e1"])
        assert node_id == "d1-2026-04-21"

        node = store.get_dag_node(node_id)
        assert node is not None
        assert node["depth"] == 1
        assert node["parent_ids"] == ["d0-s1-e1", "d0-s2-e1"]

    @patch("memory.compress.call_llm")
    def test_compress_d1_to_d2(self, mock_call_llm, store):
        from memory.compress import compress_d1_to_d2

        store.create_dag_node("d1-2026-04-21", ["d0-s1"], 1,
                              json.dumps({"summary": "Day 1 summary"}))
        store.create_dag_node("d1-2026-04-22", ["d0-s2"], 1,
                              json.dumps({"summary": "Day 2 summary"}))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "week": "2026-W17",
            "topics": [{"topic": "Setup", "one_line_summary": "Initial project setup", "entities": ["Hermes"]}],
            "summary": "A week of foundational work.",
        })
        mock_call_llm.return_value = mock_response

        node_id = compress_d1_to_d2(store, "2026-W17", ["d1-2026-04-21", "d1-2026-04-22"])
        assert node_id == "d2-2026-W17"

        node = store.get_dag_node(node_id)
        assert node is not None
        assert node["depth"] == 2


# ── context_injector.py ─────────────────────────────────────────────────


class TestContextInjector:
    def test_assemble_context_with_data(self, store):
        from memory.context_injector import assemble_context

        # Create test data
        store.ensure_session("s1", source="test")
        store.create_episode(
            session_id="s1",
            topic="Telegram bot setup",
            summary="Set up the Telegram bot for Hermes",
            key_decisions='["Use python-telegram-bot"]',
        )
        store.upsert_entity("proj-hermes", "project", "Hermes", {"language": "Python"})

        context = assemble_context(store, "telegram setup")
        assert "Episodic Memory Context" in context
        assert "Telegram" in context or "telegram" in context

    def test_assemble_context_empty(self, store):
        from memory.context_injector import assemble_context

        context = assemble_context(store, "nonexistent topic xyz123")
        # May be empty or have minimal content
        assert isinstance(context, str)

    def test_assemble_recent_context(self, store):
        from memory.context_injector import assemble_recent_context

        store.ensure_session("s1", source="test")
        store.create_episode("s1", "Recent topic", "Recent summary")
        store.upsert_entity("ent-1", "tool", "TestTool", {"version": "1.0"})

        context = assemble_recent_context(store)
        assert isinstance(context, str)

    def test_token_budget_respected(self, store):
        from memory.context_injector import assemble_context

        # Create many episodes
        for i in range(20):
            store.ensure_session(f"s{i}", source="test")
            store.create_episode(
                session_id=f"s{i}",
                topic=f"Topic {i}" * 10,
                summary=f"Summary {i}" * 50,
            )

        context = assemble_context(store, "topic", max_tokens=100)
        # Should be truncated
        assert len(context) < 2000  # 100 tokens ≈ 400 chars, plus headers


# ── Recall Tools ─────────────────────────────────────────────────────────


class TestRecallTools:
    def test_memory_grep(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        # Add turns with specific content
        store.append_turn("test-session-1", "user", "I want to deploy to AWS Lambda")
        store.append_turn("test-session-1", "assistant", "Sure, let me set up the Lambda function")
        store.append_turn("test-session-1", "user", "What about Docker instead?")

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_grep({"query": "Lambda"}))
        assert result["matches"] >= 1
        assert any("Lambda" in r["content"] for r in result["results"])

    def test_memory_grep_empty_query(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_grep({"query": ""}))
        assert "error" in result

    def test_memory_expand_episode(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        # Create episode with source turns
        store.ensure_session("s1", source="test")
        tid1 = store.append_turn("s1", "user", "Let's build a CLI")
        tid2 = store.append_turn("s1", "assistant", "Great idea!")
        ep_id = store.create_episode("s1", "CLI project", "Started CLI build", source_turns=[tid1, tid2])

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_expand({"episode_id": ep_id}))
        assert "episode" in result
        assert "source_turns" in result
        assert result["turn_count"] == 2

    def test_memory_expand_dag_node(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        store.create_dag_node("d0-s1", [], 0, json.dumps({"summary": "test"}))
        store.create_dag_node("d1-day1", ["d0-s1"], 1, json.dumps({"summary": "day summary"}))

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_expand({"dag_node_id": "d1-day1"}))
        assert "node" in result
        assert len(result["parents"]) == 1
        assert result["parents"][0]["id"] == "d0-s1"

    def test_memory_expand_not_found(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_expand({"episode_id": 99999}))
        assert "error" in result

    def test_memory_describe_entity(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        store.upsert_entity("person-test", "person", "TestUser", {"role": "developer"})
        store.ensure_session("s1", source="test")
        store.create_episode("s1", "TestUser collaboration", "Worked with TestUser on project")

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_describe({"entity_id": "person-test"}))
        assert "entity" in result
        assert result["entity"]["name"] == "TestUser"
        assert result["related_count"] >= 1

    def test_memory_describe_episode(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        store.upsert_entity("proj-x", "project", "ProjectX", {})
        store.ensure_session("s1", source="test")
        ep_id = store.create_episode("s1", "ProjectX planning", "Discussed ProjectX roadmap")

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_describe({"episode_id": ep_id}))
        assert "episode" in result
        assert "related_entities" in result


# ── Provider Integration ─────────────────────────────────────────────────


class TestProviderIntegration:
    def test_extraction_trigger(self, store):
        """Test that extraction is triggered after N turns."""
        from memory.episodic_provider import EpisodicMemoryProvider

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True
        provider._session_id = "test-session-1"

        # Insert real turns and collect their IDs
        from memory.config import EXTRACT_BATCH_SIZE
        for i in range(EXTRACT_BATCH_SIZE):
            tid = store.append_turn("test-session-1", "user", f"Message {i}")
            provider._turn_ids.append(tid)
            provider._turns_since_extract += 1

        with patch("memory.episodic_provider.extract_from_turns") as mock_extract:
            mock_extract.return_value = {
                "entities": [{"name": "Test", "type": "concept", "attributes": {}}],
                "facts": [],
                "events": [],
            }
            provider._run_extraction()

            mock_extract.assert_called_once()
            assert len(provider._pending_extractions) == 1
            assert provider._turns_since_extract == 0

    def test_context_injection_in_prefetch(self, store):
        """Test that prefetch uses context_injector."""
        from memory.episodic_provider import EpisodicMemoryProvider

        store.ensure_session("s1", source="test")
        store.create_episode("s1", "Docker setup", "Set up Docker containers")

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        context = provider.prefetch("docker setup")
        assert isinstance(context, str)

    def test_provider_health_fails_loudly(self):
        """Test that is_available logs errors loudly on failure."""
        from memory.episodic_provider import EpisodicMemoryProvider

        with patch("memory.episodic_provider.EpisodicStore") as MockStore:
            MockStore.side_effect = Exception("DB locked")
            provider = EpisodicMemoryProvider()
            result = provider.is_available()
            assert result is False


class TestMemoryRuntimeConfig:
    def test_memory_model_settings_follow_main_config_for_wiki(self):
        from memory.config import get_memory_model_settings

        with patch("memory.config.load_config", return_value={
            "model": {"default": "gpt-5.4", "provider": "openai-codex"}
        }):
            provider, model = get_memory_model_settings("wiki")

        assert provider == "openai-codex"
        assert model == "gpt-5.4"

    def test_memory_model_settings_derive_mini_model_for_extract(self):
        from memory.config import get_memory_model_settings

        with patch("memory.config.load_config", return_value={
            "model": {"default": "gpt-5.4", "provider": "openai-codex"}
        }):
            provider, model = get_memory_model_settings("extract")

        assert provider == "openai-codex"
        assert model == "gpt-5.4-mini"


class TestEpisodeSearchIndex:
    def test_search_episodes_matches_participants(self, store):
        store.create_episode(
            session_id="test-session-1",
            topic="Release planning",
            summary="Discussed the release checklist.",
            participants=json.dumps(["Jefe", "Kronos"]),
        )

        episodes = store.search_episodes("Jefe", limit=5)

        assert len(episodes) == 1
        assert episodes[0]["topic"] == "Release planning"


class TestWikiDistiller:
    def test_filter_session_episodes_prefers_rich_summaries(self):
        from memory.wiki_distiller import filter_session_episodes

        episodes = [
            {
                "topic": "User said they were going to sleep.",
                "summary": "User said they were going to sleep.",
                "key_decisions": None,
                "unresolved": None,
                "participants": "Jefe",
                "source_turns_json": None,
            },
            {
                "topic": "Testing episodic memory and wrapping up before sleep",
                "summary": "The user asked Kronos to recall the difference between faring and fairing, validated the result, and noted the first distillation was two hours away.",
                "key_decisions": json.dumps(["Use the current interaction as a real-world test of episodic memory."]),
                "unresolved": json.dumps(["Whether the memory process will continue to perform correctly."]),
                "participants": json.dumps(["Kronos", "Jefe"]),
                "source_turns_json": json.dumps([1, 2, 3]),
            },
        ]

        filtered = filter_session_episodes(episodes)

        assert len(filtered) == 1
        assert filtered[0]["topic"] == "Testing episodic memory and wrapping up before sleep"

    def test_run_daily_distill_counts_failed_pages(self, tmp_path):
        from memory.wiki_distiller import run_daily_distill

        db_path = tmp_path / "index.db"
        store = EpisodicStore(db_path=db_path)
        store.ensure_session("test-session-1", source="test")
        store.upsert_entity("person-jefe", "person", "Jefe", {"role": "user"})
        store.create_episode(
            session_id="test-session-1",
            topic="Release planning",
            summary="Discussed the release checklist.",
            participants=json.dumps(["Jefe"]),
        )
        store.close()

        with patch("memory.wiki_distiller.WIKI_OUTPUT_DIR", tmp_path / "wiki"), \
             patch("memory.wiki_distiller.distill_entity_wiki", return_value=None), \
             patch("memory.wiki_distiller.distill_session_wiki", return_value=None):
            stats = run_daily_distill(db_path=db_path)

        assert stats["entities_distilled"] == 0
        assert stats["sessions_distilled"] == 0
        assert stats["errors"] == 1

    def test_safe_entity_filename_canonicalizes_email_person_to_name(self):
        from memory.wiki_distiller import _safe_entity_filename

        filename = _safe_entity_filename(
            "person",
            "darren@apex-z.com",
            {"preferred_name": "Jefe", "full_name": "Darren Aion"},
        )

        assert filename == "person-darren-aion.md"

    def test_safe_entity_filename_treats_urls_as_resources(self):
        from memory.wiki_distiller import _safe_entity_filename

        filename = _safe_entity_filename(
            "location",
            "https://docs.google.com/document/d/abc123/edit",
            {"title": "Episode Memory Release Notes", "platform": "Google Docs"},
        )

        assert filename == "resource-episode-memory-release-notes.md"

    def test_distill_entity_wiki_skips_sparse_profile_only_entity(self, store):
        from memory.wiki_distiller import distill_entity_wiki

        entity = {
            "id": "person-jefe",
            "type": "person",
            "name": "Jefe",
            "profile_json": {"role": "user", "_source": "extraction", "_extracted_at": 1776874948.240679},
        }

        assert distill_entity_wiki(store, entity) is None

    def test_distill_entity_wiki_uses_stub_for_useful_but_episode_sparse_entity(self, store):
        from memory.wiki_distiller import distill_entity_wiki

        entity = {
            "id": "person-darren-email",
            "type": "person",
            "name": "darren@apex-z.com",
            "profile_json": {
                "preferred_name": "Jefe",
                "full_name": "Darren Aion",
                "access": "writer",
                "aliases": ["Jefe", "darren@apex-z.com"],
                "_source": "extraction",
                "_extracted_at": 1776874948.240679,
            },
        }

        content = distill_entity_wiki(store, entity)

        assert content is not None
        assert "# Darren Aion" in content
        assert "No related episodes were provided" not in content
        assert "Extracted At" not in content
        assert "1776874948.240679" not in content

    @patch("memory.wiki_distiller._call_llm_with_fallback")
    def test_distill_entity_wiki_sends_normalized_prompt_to_llm(self, mock_call_llm, store):
        from memory.wiki_distiller import distill_entity_wiki

        mock_call_llm.return_value = "# Google Docs\n\n## Overview\nClean output."
        entity = {
            "id": "location-doc-link",
            "type": "location",
            "name": "https://docs.google.com/document/d/1B6Ag5RMw06e2npPt0YGX71mVJ-s_XL2F8CiFlnCy3xI/edit",
            "profile_json": {
                "role": "shared document URL",
                "platform": "Google Docs",
                "document_id": "1B6Ag5RMw06e2npPt0YGX71mVJ-s_XL2F8CiFlnCy3xI",
                "title": "Episode Memory Release Notes",
                "_source": "extraction",
                "_extracted_at": 1776874948.2405398,
            },
        }
        store.create_episode(
            session_id="test-session-1",
            topic="Episode Memory Release Notes",
            summary="Reviewed the Episode Memory Release Notes document in Google Docs.",
            participants=json.dumps(["Jefe"]),
        )

        content = distill_entity_wiki(store, entity)

        assert content == "# Google Docs\n\n## Overview\nClean output."
        user_msg = mock_call_llm.call_args.kwargs["messages"][1]["content"]
        assert '"type": "resource"' in user_msg
        assert '"canonical_name": "Episode Memory Release Notes"' in user_msg
        assert '"_extracted_at"' not in user_msg
        assert '"_source"' not in user_msg
        assert 'https://docs.google.com/document' not in user_msg

    def test_run_daily_distill_skips_sparse_entities_and_writes_canonicalized_files(self, tmp_path):
        from memory.wiki_distiller import run_daily_distill

        db_path = tmp_path / "index.db"
        store = EpisodicStore(db_path=db_path)
        store.ensure_session("test-session-1", source="test")
        store.upsert_entity("person-jefe", "person", "Jefe", {"role": "user"})
        store.upsert_entity(
            "person-darren-email",
            "person",
            "darren@apex-z.com",
            {"preferred_name": "Jefe", "full_name": "Darren Aion", "access": "writer"},
        )
        store.upsert_entity(
            "location-doc-link",
            "location",
            "https://docs.google.com/document/d/abc123/edit",
            {"title": "Episode Memory Release Notes", "platform": "Google Docs", "role": "shared document URL"},
        )
        store.create_episode(
            session_id="test-session-1",
            topic="Release planning",
            summary="Discussed the release checklist and shared the release notes doc.",
            participants=json.dumps(["Jefe"]),
        )
        store.close()

        with patch("memory.wiki_distiller.WIKI_OUTPUT_DIR", tmp_path / "wiki"), \
             patch("memory.wiki_distiller._call_llm_with_fallback", return_value="# Clean Page\n\nBody"):
            stats = run_daily_distill(db_path=db_path)

        assert stats["entities_distilled"] == 2
        assert not (tmp_path / "wiki" / "entities" / "person-jefe.md").exists()
        assert (tmp_path / "wiki" / "entities" / "person-darren-aion.md").exists()
        assert (tmp_path / "wiki" / "entities" / "resource-episode-memory-release-notes.md").exists()

    def test_run_daily_distill_overwrites_session_by_id_and_cleans_dated_copies(self, tmp_path):
        from memory.wiki_distiller import run_daily_distill

        db_path = tmp_path / "index.db"
        store = EpisodicStore(db_path=db_path)
        store.ensure_session("sess-abc-123", source="test")
        store.create_episode(
            session_id="sess-abc-123",
            topic="Testing session dedup",
            summary="This session tests that old dated copies are cleaned up.",
            key_decisions=json.dumps(["Use session-id filenames"]),
        )
        store.close()

        wiki_path = tmp_path / "wiki"
        sessions_dir = wiki_path / "sessions"
        sessions_dir.mkdir(parents=True)
        # Stale dated copy from a previous run
        stale_file = sessions_dir / "2026-04-22_sess-abc-123.md"
        stale_file.write_text("old content", encoding="utf-8")

        with patch("memory.wiki_distiller.WIKI_OUTPUT_DIR", wiki_path), \
             patch("memory.wiki_distiller._call_llm_with_fallback", return_value="# Session Dedup\n\nBody"):
            stats = run_daily_distill(db_path=db_path)

        assert stats["sessions_distilled"] == 1
        assert (sessions_dir / "sess-abc-123.md").exists()
        assert not stale_file.exists()


class TestEpisodeType:
    def test_create_episode_accepts_episode_type(self, store):
        eid = store.create_episode(
            session_id="test-session-1",
            topic="Release planning",
            summary="Discussed release checklist.",
            episode_type="substantive",
        )
        ep = store.get_episode(eid)
        assert ep["episode_type"] == "substantive"

    def test_create_episode_defaults_to_raw(self, store):
        eid = store.create_episode(
            session_id="test-session-1",
            topic="Facts extracted",
            summary="User prefers Python; project status active.",
        )
        ep = store.get_episode(eid)
        assert ep["episode_type"] == "raw"

    def test_classify_episode_banter(self):
        from memory.merge import classify_episode_type

        assert classify_episode_type(
            "User said they were going to sleep.",
            "User said they were going to sleep.",
            key_decisions=None,
            source_turns_json=None,
        ) == "banter"

    def test_classify_episode_chitchat(self):
        from memory.merge import classify_episode_type

        assert classify_episode_type(
            "Jokes and banter",
            "Kronos told a memory-themed dad joke and exchanged brief banter about memory.",
            key_decisions=None,
            source_turns_json=None,
        ) == "chitchat"

    def test_classify_episode_substantive(self):
        from memory.merge import classify_episode_type

        assert classify_episode_type(
            "Episodic memory provider activation",
            "The session confirmed that EpisodicMemoryProvider is live and healthy with key decisions about v0.2 and v0.3.",
            key_decisions=json.dumps(["Define v0.2 as Auto-Skill Detection", "Define v0.3 as Channel Provenance"]),
            source_turns_json=json.dumps([9, 10, 11]),
        ) == "substantive"

    def test_classify_episode_raw_with_significance(self):
        from memory.merge import classify_episode_type

        assert classify_episode_type(
            "User uses Ollama for project",
            "User prefers Ollama for local inference; project is active.",
            key_decisions=None,
            source_turns_json=None,
        ) == "raw"

    def test_merge_creates_episodes_with_banter_type(self, store):
        from memory.merge import merge_extracted_facts

        extracted = {
            "entities": [],
            "facts": [
                {"subject": "Kronos", "predicate": "told", "object": "a dad joke about memory", "confidence": "medium"},
            ],
            "events": [
                {"description": "User said goodnight and going to sleep.", "participants": ["Jefe"]},
            ],
        }
        stats = merge_extracted_facts(store, extracted, "test-session-1")
        episodes = store.get_recent_episodes(limit=5)
        # The event about going to sleep should be classified as banter
        banter_episodes = [ep for ep in episodes if ep.get("episode_type") == "banter"]
        assert len(banter_episodes) >= 1

    def test_compress_creates_substantive_episodes(self, store):
        from memory.compress import compress_session_to_d0
        from unittest.mock import MagicMock, patch

        for i in range(5):
            store.append_turn("test-session-1", "user", f"User message {i}")
            store.append_turn("test-session-1", "assistant", f"Assistant reply {i}")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "topic": "Test conversation",
            "summary": "A test conversation about testing.",
            "key_decisions": ["Use pytest"],
            "unresolved": [],
            "participants": ["user"],
        })

        with patch("memory.compress.call_llm", return_value=mock_response):
            turns = store.get_turns_for_session("test-session-1")
            compress_session_to_d0(store, "test-session-1", turns)

        episodes = store.get_recent_episodes(limit=5)
        assert any(ep.get("episode_type") == "substantive" for ep in episodes)

    def test_distiller_tags_banter_episodes(self, store):
        from memory.wiki_distiller import mark_banter_episodes
        from unittest.mock import patch

        eid_banter = store.create_episode(
            session_id="test-session-1",
            topic="User said they were going to sleep.",
            summary="User said they were going to sleep.",
            episode_type="raw",
        )
        eid_good = store.create_episode(
            session_id="test-session-1",
            topic="Release planning and key decisions",
            summary="Discussed release checklist with several decisions and open questions.",
            key_decisions=json.dumps(["Ship tomorrow"]),
            episode_type="raw",
        )

        count = mark_banter_episodes(store)
        assert count >= 1

        ep_banter = store.get_episode(eid_banter)
        assert ep_banter["episode_type"] == "banter"
        ep_good = store.get_episode(eid_good)
        assert ep_good["episode_type"] == "substantive"


class TestDistillerFallback:
    def test_is_retryable_error_429(self):
        from memory.wiki_distiller import _is_retryable_error
        assert _is_retryable_error(Exception("Error code: 429 - rate limit exceeded"))

    def test_is_retryable_error_529(self):
        from memory.wiki_distiller import _is_retryable_error
        assert _is_retryable_error(Exception("529 overloaded"))

    def test_is_retryable_error_503(self):
        from memory.wiki_distiller import _is_retryable_error
        assert _is_retryable_error(Exception("503 service unavailable"))

    def test_is_not_retryable_auth_error(self):
        from memory.wiki_distiller import _is_retryable_error
        assert not _is_retryable_error(Exception("401 unauthorized"))

    def test_get_fallback_providers_from_config(self):
        from memory.wiki_distiller import _get_fallback_providers

        with patch("memory.config.load_config", return_value={
            "fallback_providers": [
                {"provider": "openrouter", "model": "x-ai/grok-4.1-fast"},
            ]
        }):
            chain = _get_fallback_providers()

        assert len(chain) == 1
        assert chain[0]["provider"] == "openrouter"
        assert chain[0]["model"] == "x-ai/grok-4.1-fast"

    def test_call_llm_with_fallback_retries_on_429(self):
        from memory.wiki_distiller import _call_llm_with_fallback

        call_count = 0
        def fake_call_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Error code: 429 - rate limit")
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "ok"
            return resp

        with patch("memory.wiki_distiller.call_llm", side_effect=fake_call_llm), \
             patch("memory.wiki_distiller._RETRY_BASE_DELAY", 0.01), \
             patch("memory.wiki_distiller._INTER_REQUEST_DELAY", 0.01):
            result = _call_llm_with_fallback(
                provider="test", model="test", messages=[{"role": "user", "content": "hi"}]
            )

        assert result == "ok"
        assert call_count == 3

    def test_call_llm_with_fallback_falls_to_next_provider(self):
        from memory.wiki_distiller import _call_llm_with_fallback

        used_providers = []
        def fake_call_llm(**kwargs):
            used_providers.append(kwargs["provider"])
            if kwargs["provider"] == "openai-codex":
                raise Exception("400 - model not supported")
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "fallback"
            return resp

        with patch("memory.wiki_distiller.call_llm", side_effect=fake_call_llm), \
             patch("memory.wiki_distiller._get_fallback_providers", return_value=[
                 {"provider": "openrouter", "model": "x-ai/grok-4.1-fast"}
             ]), \
             patch("memory.wiki_distiller._INTER_REQUEST_DELAY", 0.01):
            result = _call_llm_with_fallback(
                provider="openai-codex", model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}]
            )

        assert result == "fallback"
        assert "openrouter" in used_providers


class TestSessionJournal:
    def test_extract_tags_from_github_tool(self):
        from memory.session_journal import _extract_tags
        turns = [
            {"role": "assistant", "content": "Pushing files", "tool_name": "mcp_github_push_files"},
        ]
        tags = _extract_tags(turns)
        assert "#github" in tags

    def test_extract_tags_from_terminal_test_command(self):
        from memory.session_journal import _extract_tags
        turns = [
            {"role": "assistant", "content": "Running pytest -xvs", "tool_name": "terminal_tool"},
        ]
        tags = _extract_tags(turns)
        assert "#test" in tags
        assert "#terminal" in tags

    def test_extract_tags_heavy_session(self):
        from memory.session_journal import _extract_tags
        turns = [{"role": "assistant", "content": "x", "tool_name": "terminal_tool"} for _ in range(12)]
        tags = _extract_tags(turns)
        assert "#heavy-session" in tags

    def test_summarize_turns(self):
        from memory.session_journal import _summarize_turns
        turns = [
            {"role": "user", "content": "Hello", "timestamp": 0},
            {"role": "assistant", "content": "Hi there", "timestamp": 60},
        ]
        summary, tool_calls = _summarize_turns(turns)
        assert "1 user turns" in summary
        assert "1 assistant turns" in summary
        assert "~1 min duration" in summary

    def test_write_session_journal_skips_empty(self, tmp_path):
        from memory.session_journal import write_session_journal
        result = write_session_journal("empty-session", turns=[], output_dir=tmp_path)
        assert result is None

    def test_write_session_journal_skips_skill_review_only(self, tmp_path):
        from memory.session_journal import write_session_journal
        turns = [
            {"role": "user", "content": "Review the conversation above and consider saving or updating a skill if appropriate."},
        ]
        result = write_session_journal("skill-review", turns=turns, output_dir=tmp_path)
        assert result is None

    def test_write_session_journal_creates_markdown(self, tmp_path):
        from memory.session_journal import write_session_journal
        turns = [
            {"role": "user", "content": "Build the thing\nMore details here."},
            {"role": "assistant", "content": "Running tests", "tool_name": "terminal_tool"},
        ]
        result = write_session_journal("test-sess-123", turns=turns, platform="telegram", output_dir=tmp_path)
        assert result is not None
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "# Session: test-sess-123" in content
        assert "**Platform:** telegram" in content
        assert "#terminal" in content
        assert "Build the thing" in content
        assert "`terminal_tool`" in content

    def test_write_session_journal_uses_weekly_folder(self, tmp_path):
        from memory.session_journal import write_session_journal, _iso_week_folder
        from datetime import datetime, timezone

        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = write_session_journal("week-test", turns=turns, output_dir=tmp_path)
        assert result is not None
        # Should be inside a year-Wxx folder
        assert result.parent.name.startswith("20") and "-W" in result.parent.name

    def test_journal_for_jsonl(self, tmp_path):
        from memory.session_journal import write_journal_for_jsonl
        jsonl_path = tmp_path / "my-session.jsonl"
        jsonl_path.write_text(
            '{"role": "user", "content": "Do the thing"}\n'
            '{"role": "assistant", "content": "Done", "tool_name": "execute_code"}\n',
            encoding="utf-8",
        )
        result = write_journal_for_jsonl(jsonl_path, platform="cli", output_dir=tmp_path)
        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "# Session: my-session" in content
        assert "#python" in content
