"""Tests for kensei_entity_extraction — LLM-based entity extraction & consolidation.

Three layers:
1. Unit tests — ExtractionCache, ExtractedFact, ConsolidationDecision data structures
2. Unit tests — extract_facts, consolidate_facts, extract_and_consolidate with mocked LLM
3. Integration tests — full pipeline through CLI entry point
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add the scripts directory to path so we can import the module
# Resolve relative to this test file so it works from any checkout / CI runner.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Mock Mnemosyne at module level so imports don't fail
# ---------------------------------------------------------------------------
_mnemosyne_stub = MagicMock()
sys.modules["mnemosyne"] = _mnemosyne_stub
sys.modules["mnemosyne.core"] = MagicMock()
sys.modules["mnemosyne.core.triples"] = MagicMock()
sys.modules["mnemosyne.core.embeddings"] = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Prevent accidental writes to real Mnemosyne data."""
    monkeypatch.setenv("KENSEI_MEMORY_BANK", "test_bank")
    monkeypatch.setenv("MNEMOSYNE_HOME", "/tmp/mnemosyne_test")
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes_test")


@pytest.fixture
def mock_triple_store():
    """A MagicMock that behaves like TripleStore."""
    store = MagicMock()
    store.add = MagicMock(return_value=42)
    store.conn = MagicMock()
    store.conn.execute = MagicMock(return_value=None)
    return store


# ---------------------------------------------------------------------------
# Unit tests: ExtractedFact and ConsolidationDecision data structures
# ---------------------------------------------------------------------------


class TestExtractedFact:
    def test_to_dict(self):
        from kensei_entity_extraction import ExtractedFact

        fact = ExtractedFact(
            subject="Alice",
            predicate="works_at",
            object="Acme Corp",
            confidence=0.9,
            source="test",
            valid_from="2026-01-01",
        )
        d = fact.to_dict()
        assert d["subject"] == "Alice"
        assert d["predicate"] == "works_at"
        assert d["object"] == "Acme Corp"
        assert d["confidence"] == 0.9
        assert d["source"] == "test"
        assert d["valid_from"] == "2026-01-01"

    def test_key(self):
        from kensei_entity_extraction import ExtractedFact

        fact = ExtractedFact(subject="Alice", predicate="works_at", object="Acme Corp")
        assert fact.key() == "Alice|works_at|Acme Corp"

    def test_defaults(self):
        from kensei_entity_extraction import ExtractedFact

        fact = ExtractedFact(subject="Bob", predicate="likes", object="Python")
        assert fact.confidence == 0.8
        assert fact.source == "extraction"
        assert fact.valid_from is None


class TestConsolidationDecision:
    def test_add_decision(self):
        from kensei_entity_extraction import ConsolidationDecision, ExtractedFact

        fact = ExtractedFact(subject="Alice", predicate="works_at", object="Acme Corp")
        decision = ConsolidationDecision(
            action="ADD", reason="Novel fact", new_fact=fact
        )
        assert decision.action == "ADD"
        assert decision.reason == "Novel fact"
        assert decision.new_fact is fact
        assert decision.existing_triple is None

    def test_update_decision(self):
        from kensei_entity_extraction import ConsolidationDecision, ExtractedFact

        fact = ExtractedFact(subject="Alice", predicate="works_at", object="New Corp")
        existing = {"subject": "Alice", "predicate": "works_at", "object": "Old Corp"}
        decision = ConsolidationDecision(
            action="UPDATE", reason="Changed employer", existing_triple=existing, new_fact=fact
        )
        assert decision.action == "UPDATE"
        assert decision.existing_triple["object"] == "Old Corp"


# ---------------------------------------------------------------------------
# Unit tests: ExtractionCache
# ---------------------------------------------------------------------------


class TestExtractionCache:
    def test_cache_hit_and_miss(self):
        from kensei_entity_extraction import ExtractionCache

        cache = ExtractionCache(maxsize=10, ttl=60)
        text = "Alice works at Acme Corp."

        # Miss on first access
        assert cache.get(text) is None
        assert cache.stats["misses"] == 1

        # Put and hit
        result = {"entities": ["Alice"], "triples": []}
        cache.put(text, result)
        cached = cache.get(text)
        assert cached == result
        assert cache.stats["hits"] == 1

    def test_cache_ttl_expiry(self):
        from kensei_entity_extraction import ExtractionCache

        cache = ExtractionCache(maxsize=10, ttl=0.1)  # 100ms TTL
        text = "Bob likes Python."
        result = {"entities": ["Bob"], "triples": []}

        cache.put(text, result)
        assert cache.get(text) is not None  # Hit

        time.sleep(0.15)
        assert cache.get(text) is None  # Expired

    def test_cache_eviction(self):
        from kensei_entity_extraction import ExtractionCache

        cache = ExtractionCache(maxsize=2, ttl=60)
        for i in range(3):
            cache.put(f"text_{i}", {"id": i})

        # The oldest entry should be evicted
        assert cache.get("text_0") is None
        assert cache.get("text_1") is not None
        assert cache.get("text_2") is not None

    def test_cache_invalidate(self):
        from kensei_entity_extraction import ExtractionCache

        cache = ExtractionCache(maxsize=10, ttl=60)
        cache.put("hello", {"data": 1})
        assert cache.get("hello") is not None
        cache.invalidate("hello")
        assert cache.get("hello") is None

    def test_cache_clear(self):
        from kensei_entity_extraction import ExtractionCache

        cache = ExtractionCache(maxsize=10, ttl=60)
        cache.put("a", {"data": 1})
        cache.put("b", {"data": 2})
        cache.clear()
        assert cache.stats["size"] == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_cache_stats(self):
        from kensei_entity_extraction import ExtractionCache

        cache = ExtractionCache(maxsize=10, ttl=60)
        stats = cache.stats
        assert stats["maxsize"] == 10
        assert stats["ttl_seconds"] == 60
        assert stats["hit_rate"] == 0.0

        cache.put("x", {"v": 1})
        cache.get("x")  # hit
        cache.get("y")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# Unit tests: extract_facts
# ---------------------------------------------------------------------------


class TestExtractFacts:
    def test_empty_text_returns_error(self):
        from kensei_entity_extraction import extract_facts

        result = extract_facts(text="", use_cache=False)
        assert result["status"] == "error"
        assert "text is required" in result["message"]

    def test_whitespace_text_returns_error(self):
        from kensei_entity_extraction import extract_facts

        result = extract_facts(text="   \n  ", use_cache=False)
        assert result["status"] == "error"

    @patch("kensei_entity_extraction._call_llm", return_value=None)
    def test_llm_failure_returns_error(self, mock_llm):
        from kensei_entity_extraction import extract_facts

        result = extract_facts(text="Alice works at Acme Corp.", use_cache=False)
        assert result["status"] == "error"
        assert "LLM extraction failed" in result["message"]

    @patch("kensei_entity_extraction._call_llm", return_value="not valid json")
    def test_unparseable_llm_response(self, mock_llm):
        from kensei_entity_extraction import extract_facts

        result = extract_facts(text="Alice works at Acme Corp.", use_cache=False)
        assert result["status"] == "error"
        assert "unparseable JSON" in result["message"]

    @patch("kensei_entity_extraction._call_llm")
    def test_successful_extraction(self, mock_llm):
        """Valid LLM response should parse and return entities and triples."""
        mock_llm.return_value = json.dumps({
            "entities": ["Alice", "Acme Corp"],
            "triples": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp", "confidence": 0.95},
            ],
        })

        from kensei_entity_extraction import extract_facts

        result = extract_facts(
            text="Alice works at Acme Corp.",
            use_cache=False,
        )

        assert result["status"] == "ok"
        assert result["entities"] == ["Alice", "Acme Corp"]
        assert len(result["triples"]) == 1
        assert result["triples"][0]["subject"] == "Alice"
        assert result["triples"][0]["predicate"] == "works_at"
        assert result["triples"][0]["object"] == "Acme Corp"
        assert result["triples"][0]["confidence"] == 0.95
        assert result["cached"] is False

    @patch("kensei_entity_extraction._call_llm")
    def test_handles_markdown_wrapped_json(self, mock_llm):
        """LLM response wrapped in ```json should be parsed correctly."""
        mock_llm.return_value = "```json\n{\"entities\": [\"Bob\"], \"triples\": []}\n```"

        from kensei_entity_extraction import extract_facts

        result = extract_facts(text="Bob is here.", use_cache=False)
        assert result["status"] == "ok"
        assert result["entities"] == ["Bob"]

    @patch("kensei_entity_extraction._call_llm")
    def test_invalid_triple_skipped(self, mock_llm):
        """Triples with missing fields should be skipped."""
        mock_llm.return_value = json.dumps({
            "entities": [],
            "triples": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"},
                {"subject": "", "predicate": "is", "object": ""},  # invalid
            ],
        })

        from kensei_entity_extraction import extract_facts

        result = extract_facts(text="Alice works at Acme Corp.", use_cache=False)
        assert result["status"] == "ok"
        assert len(result["triples"]) == 1

    @patch("kensei_entity_extraction._call_llm")
    def test_confidence_clamped(self, mock_llm):
        """Confidence should be clamped to [0.0, 1.0]."""
        mock_llm.return_value = json.dumps({
            "entities": [],
            "triples": [
                {"subject": "X", "predicate": "is", "object": "Y", "confidence": 2.5},
                {"subject": "A", "predicate": "is", "object": "B", "confidence": -0.5},
            ],
        })

        from kensei_entity_extraction import extract_facts

        result = extract_facts(text="Test.", use_cache=False)
        assert result["triples"][0]["confidence"] == 1.0
        assert result["triples"][1]["confidence"] == 0.0

    @patch("kensei_entity_extraction._call_llm")
    def test_cache_used_when_enabled(self, mock_llm):
        """When use_cache=True, the cache should be checked and results cached."""
        mock_llm.return_value = json.dumps({
            "entities": ["Alice"],
            "triples": [{"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"}],
        })

        from kensei_entity_extraction import extract_facts

        # First call — should call LLM
        result1 = extract_facts(text="Alice works at Acme Corp.", use_cache=True)
        assert result1["status"] == "ok"
        assert result1["cached"] is False
        assert mock_llm.call_count == 1

        # Second call with same text — should hit cache
        result2 = extract_facts(text="Alice works at Acme Corp.", use_cache=True)
        assert result2["status"] == "ok"
        assert result2["cached"] is True
        assert mock_llm.call_count == 1  # No additional LLM call


# ---------------------------------------------------------------------------
# Unit tests: consolidate_facts
# ---------------------------------------------------------------------------


class TestConsolidateFacts:
    def test_empty_new_facts_returns_empty(self):
        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        decisions = consolidate_facts([], bank="test_bank")
        assert decisions == []

    @patch("kensei_entity_extraction._get_existing_triples", return_value=[])
    def test_no_existing_facts_all_add(self, mock_get):
        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        facts = [ExtractedFact(subject="Alice", predicate="works_at", object="Acme Corp")]
        decisions = consolidate_facts(facts, bank="test_bank")
        assert len(decisions) == 1
        assert decisions[0].action == "ADD"
        assert "No existing facts" in decisions[0].reason

    @patch("kensei_entity_extraction._get_existing_triples")
    @patch("kensei_entity_extraction._get_embedding", return_value=None)
    def test_exact_match_noop(self, mock_emb, mock_get):
        """Exact same triple should result in NOOP."""
        mock_get.return_value = [
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp", "confidence": 1.0},
        ]

        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        facts = [ExtractedFact(subject="Alice", predicate="works_at", object="Acme Corp")]
        decisions = consolidate_facts(facts, bank="test_bank", use_llm_judge=False)
        assert len(decisions) == 1
        assert decisions[0].action == "NOOP"
        assert "Exact triple" in decisions[0].reason

    @patch("kensei_entity_extraction._get_existing_triples")
    @patch("kensei_entity_extraction._get_embedding", return_value=None)
    def test_same_subject_predicate_different_object_update(self, mock_emb, mock_get):
        """Same (subject, predicate) with different object should result in UPDATE."""
        mock_get.return_value = [
            {"subject": "Alice", "predicate": "works_at", "object": "Old Corp", "confidence": 1.0},
        ]

        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        facts = [ExtractedFact(subject="Alice", predicate="works_at", object="New Corp")]
        decisions = consolidate_facts(facts, bank="test_bank", use_llm_judge=False)
        assert len(decisions) == 1
        assert decisions[0].action == "UPDATE"
        assert "Old Corp" in decisions[0].reason
        assert "New Corp" in decisions[0].reason

    @patch("kensei_entity_extraction._get_existing_triples")
    @patch("kensei_entity_extraction._get_embedding", return_value=None)
    def test_novel_fact_add(self, mock_emb, mock_get):
        """Completely novel fact should result in ADD."""
        mock_get.return_value = [
            {"subject": "Bob", "predicate": "likes", "object": "Python", "confidence": 1.0},
        ]

        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        facts = [ExtractedFact(subject="Alice", predicate="works_at", object="Acme Corp")]
        decisions = consolidate_facts(facts, bank="test_bank", use_llm_judge=False)
        assert len(decisions) == 1
        assert decisions[0].action == "ADD"
        assert "Novel fact" in decisions[0].reason

    @patch("kensei_entity_extraction._get_existing_triples")
    @patch("kensei_entity_extraction._get_embedding")
    def test_high_similarity_noop(self, mock_emb, mock_get):
        """Semantically similar triples should result in NOOP."""
        mock_get.return_value = [
            {"subject": "Alice", "predicate": "employed_by", "object": "Acme Corporation", "confidence": 1.0},
        ]
        # Return a high-similarity vector
        mock_emb.return_value = [0.1, 0.2, 0.3]

        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        facts = [ExtractedFact(subject="Alice", predicate="works_at", object="Acme Corp")]
        decisions = consolidate_facts(facts, bank="test_bank", use_llm_judge=False, similarity_threshold=0.5)
        assert len(decisions) == 1
        assert decisions[0].action == "NOOP"
        assert "similar" in decisions[0].reason.lower()

    @patch("kensei_entity_extraction._get_existing_triples")
    @patch("kensei_entity_extraction._get_embedding")
    @patch("kensei_entity_extraction._call_llm")
    def test_llm_judge_called_when_enabled(self, mock_llm, mock_emb, mock_get):
        """When use_llm_judge=True and similar triple found, LLM judge should be called."""
        mock_get.return_value = [
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp", "confidence": 1.0},
        ]
        mock_emb.return_value = [0.1, 0.2, 0.3]
        mock_llm.return_value = json.dumps({
            "action": "NOOP",
            "reason": "Already represented",
            "confidence": 0.95,
        })

        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        facts = [ExtractedFact(subject="Alice", predicate="works_at", object="Acme Corp")]
        decisions = consolidate_facts(facts, bank="test_bank", use_llm_judge=True, similarity_threshold=0.5)
        assert len(decisions) == 1
        assert decisions[0].action == "NOOP"
        assert mock_llm.called

    @patch("kensei_entity_extraction._get_existing_triples")
    @patch("kensei_entity_extraction._get_embedding")
    @patch("kensei_entity_extraction._call_llm", return_value=None)
    def test_llm_judge_fallback_to_add(self, mock_llm, mock_emb, mock_get):
        """When LLM judge fails, should fall back to ADD."""
        mock_get.return_value = [
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp", "confidence": 1.0},
        ]
        mock_emb.return_value = [0.1, 0.2, 0.3]

        from kensei_entity_extraction import consolidate_facts, ExtractedFact

        facts = [ExtractedFact(subject="Alice", predicate="works_at", object="New Corp")]
        decisions = consolidate_facts(facts, bank="test_bank", use_llm_judge=True, similarity_threshold=0.5)
        assert len(decisions) == 1
        assert decisions[0].action == "ADD"
        assert "defaulting to ADD" in decisions[0].reason


# ---------------------------------------------------------------------------
# Unit tests: extract_and_consolidate (full pipeline)
# ---------------------------------------------------------------------------


class TestExtractAndConsolidate:
    @patch("kensei_entity_extraction._call_llm")
    @patch("kensei_entity_extraction._get_existing_triples", return_value=[])
    @patch("kensei_entity_extraction._get_triple_store")
    def test_full_pipeline_add(self, mock_get_store, mock_get_existing, mock_llm):
        """Full pipeline should extract, consolidate, and store new facts."""
        mock_llm.return_value = json.dumps({
            "entities": ["Alice", "Acme Corp"],
            "triples": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp", "confidence": 0.95},
            ],
        })
        mock_store = MagicMock()
        mock_store.add = MagicMock(return_value=42)
        mock_get_store.return_value = mock_store

        from kensei_entity_extraction import extract_and_consolidate

        result = extract_and_consolidate(
            text="Alice works at Acme Corp.",
            bank="test_bank",
            use_cache=False,
            use_llm_judge=False,
            store_results=True,
        )

        assert result["status"] == "ok"
        assert result["extraction"]["entities"] == ["Alice", "Acme Corp"]
        assert result["extraction"]["triples_extracted"] == 1
        assert result["stored"] == 1
        assert len(result["consolidation"]) == 1
        assert result["consolidation"][0]["action"] == "ADD"
        mock_store.add.assert_called_once()

    @patch("kensei_entity_extraction._call_llm")
    @patch("kensei_entity_extraction._get_existing_triples", return_value=[])
    def test_full_pipeline_no_store(self, mock_get_existing, mock_llm):
        """When store_results=False, should not call triple store."""
        mock_llm.return_value = json.dumps({
            "entities": ["Alice"],
            "triples": [{"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"}],
        })

        from kensei_entity_extraction import extract_and_consolidate

        result = extract_and_consolidate(
            text="Alice works at Acme Corp.",
            bank="test_bank",
            use_cache=False,
            use_llm_judge=False,
            store_results=False,
        )

        assert result["status"] == "ok"
        assert result["stored"] == 0

    @patch("kensei_entity_extraction._call_llm", return_value=None)
    def test_llm_failure_in_pipeline(self, mock_llm):
        """LLM failure in extraction should propagate through pipeline."""
        from kensei_entity_extraction import extract_and_consolidate

        result = extract_and_consolidate(
            text="Alice works at Acme Corp.",
            use_cache=False,
        )

        assert result["status"] == "error"
        assert "LLM extraction failed" in result["message"]

    @patch("kensei_entity_extraction._call_llm")
    @patch("kensei_entity_extraction._get_existing_triples", return_value=[])
    def test_no_facts_extracted(self, mock_get_existing, mock_llm):
        """When no facts are extracted, pipeline should return ok with zero stored."""
        mock_llm.return_value = json.dumps({
            "entities": [],
            "triples": [],
        })

        from kensei_entity_extraction import extract_and_consolidate

        result = extract_and_consolidate(
            text="Hello world.",
            use_cache=False,
        )

        assert result["status"] == "ok"
        assert result["stored"] == 0
        assert "No facts extracted" in result["message"]


# ---------------------------------------------------------------------------
# Unit tests: process_conversation_turn
# ---------------------------------------------------------------------------


class TestProcessConversationTurn:
    @patch("kensei_entity_extraction._call_llm")
    @patch("kensei_entity_extraction._get_existing_triples", return_value=[])
    @patch("kensei_entity_extraction._get_triple_store")
    def test_processes_user_and_assistant(self, mock_get_store, mock_get_existing, mock_llm):
        """process_conversation_turn should combine user + assistant text."""
        mock_llm.return_value = json.dumps({
            "entities": ["Alice", "Python"],
            "triples": [
                {"subject": "Alice", "predicate": "prefers", "object": "Python", "confidence": 0.9},
            ],
        })
        mock_store = MagicMock()
        mock_store.add = MagicMock(return_value=42)
        mock_get_store.return_value = mock_store

        from kensei_entity_extraction import process_conversation_turn

        result = process_conversation_turn(
            user_message="I like Python.",
            assistant_response="Alice prefers Python for data science.",
            bank="test_bank",
            use_cache=False,
            use_llm_judge=False,
        )

        assert result["status"] == "ok"
        assert result["stored"] == 1


# ---------------------------------------------------------------------------
# Unit tests: cache management
# ---------------------------------------------------------------------------


class TestCacheManagement:
    @patch("kensei_entity_extraction._extraction_cache")
    def test_get_cache_stats(self, mock_cache):
        mock_cache.stats = {"size": 5, "hits": 10, "misses": 2, "hit_rate": 0.833}
        from kensei_entity_extraction import get_cache_stats

        stats = get_cache_stats()
        assert stats["size"] == 5
        assert stats["hits"] == 10

    @patch("kensei_entity_extraction._extraction_cache")
    def test_clear_cache(self, mock_cache):
        from kensei_entity_extraction import clear_cache

        clear_cache()
        mock_cache.clear.assert_called_once()


# ---------------------------------------------------------------------------
# Unit tests: _parse_llm_json
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_plain_json(self):
        from kensei_entity_extraction import _parse_llm_json

        result = _parse_llm_json('{"entities": ["Alice"]}')
        assert result == {"entities": ["Alice"]}

    def test_markdown_json_block(self):
        from kensei_entity_extraction import _parse_llm_json

        result = _parse_llm_json('```json\n{"entities": ["Bob"]}\n```')
        assert result == {"entities": ["Bob"]}

    def test_markdown_code_block(self):
        from kensei_entity_extraction import _parse_llm_json

        result = _parse_llm_json('```\n{"entities": ["Charlie"]}\n```')
        assert result == {"entities": ["Charlie"]}

    def test_none_input(self):
        from kensei_entity_extraction import _parse_llm_json

        assert _parse_llm_json(None) is None

    def test_invalid_json(self):
        from kensei_entity_extraction import _parse_llm_json

        assert _parse_llm_json("not json at all") is None


# ---------------------------------------------------------------------------
# Unit tests: _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from kensei_entity_extraction import _cosine_similarity

        sim = _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 1.0

    def test_orthogonal_vectors(self):
        from kensei_entity_extraction import _cosine_similarity

        sim = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == 0.0

    def test_zero_vector(self):
        from kensei_entity_extraction import _cosine_similarity

        sim = _cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0

    def test_partial_similarity(self):
        from kensei_entity_extraction import _cosine_similarity

        sim = _cosine_similarity([1.0, 1.0], [1.0, 0.0])
        import math
        expected = 1.0 / math.sqrt(2)
        assert abs(sim - expected) < 0.0001


# ---------------------------------------------------------------------------
# Unit tests: _format_existing_for_prompt
# ---------------------------------------------------------------------------


class TestFormatExistingForPrompt:
    def test_formats_triples(self):
        from kensei_entity_extraction import _format_existing_for_prompt

        triples = [
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp", "confidence": 0.9},
            {"subject": "Bob", "predicate": "likes", "object": "Python", "confidence": 1.0},
        ]
        formatted = _format_existing_for_prompt(triples)
        assert "Alice" in formatted
        assert "works_at" in formatted
        assert "Acme Corp" in formatted
        assert "Bob" in formatted
        assert "Python" in formatted
        assert "1." in formatted
        assert "2." in formatted

    def test_empty_list(self):
        from kensei_entity_extraction import _format_existing_for_prompt

        assert _format_existing_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# CLI entry point tests
# ---------------------------------------------------------------------------


class TestCliEntryPoint:
    def test_cache_stats_flag(self):
        """--cache-stats should print cache stats and exit."""
        from kensei_entity_extraction import main

        with patch("kensei_entity_extraction.get_cache_stats") as mock_stats, \
             patch("builtins.print") as mock_print:
            mock_stats.return_value = {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}

            with patch("sys.argv", ["kensei_entity_extraction.py", "--cache-stats"]):
                main()

            mock_print.assert_called_once()
            args, _ = mock_print.call_args
            assert "size" in args[0]

    def test_clear_cache_flag(self):
        """--clear-cache should clear cache and print confirmation."""
        from kensei_entity_extraction import main

        with patch("kensei_entity_extraction.clear_cache") as mock_clear, \
             patch("builtins.print") as mock_print:

            with patch("sys.argv", ["kensei_entity_extraction.py", "--clear-cache"]):
                main()

            mock_clear.assert_called_once()
            mock_print.assert_called_once_with("Cache cleared")

    def test_dry_run_flag(self):
        """--dry-run should not store results."""
        from kensei_entity_extraction import main

        with patch("kensei_entity_extraction.extract_and_consolidate") as mock_pipeline, \
             patch("builtins.print"):

            with patch("sys.argv", ["kensei_entity_extraction.py", "test text", "--dry-run"]):
                main()

            mock_pipeline.assert_called_once()
            _, kwargs = mock_pipeline.call_args
            assert kwargs["store_results"] is False

    def test_no_text_shows_help(self):
        """No text argument should print help."""
        from kensei_entity_extraction import main

        with patch("argparse.ArgumentParser.print_help") as mock_help:
            with patch("sys.argv", ["kensei_entity_extraction.py"]):
                main()

            mock_help.assert_called_once()
