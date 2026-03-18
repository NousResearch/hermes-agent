"""Tests for cognitive_memory.extraction module."""

import time

import pytest

from cognitive_memory.extraction import (
    ExtractedFact,
    ForgettingManager,
    ForgettingResult,
    extract_facts,
    is_memorable,
    _split_sentences,
)
from cognitive_memory.store import CognitiveStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_extraction.db")
    s = CognitiveStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def forgetting(store):
    return ForgettingManager(store=store)


# ---------------------------------------------------------------------------
# is_memorable
# ---------------------------------------------------------------------------


class TestIsMemorable:
    def test_short_text_not_memorable(self):
        assert is_memorable("ok") is False
        assert is_memorable("yes") is False
        assert is_memorable("hi") is False

    def test_greeting_not_memorable(self):
        assert is_memorable("Hello") is False
        assert is_memorable("Good morning") is False

    def test_thanks_not_memorable(self):
        assert is_memorable("Thank you") is False
        assert is_memorable("Thanks") is False

    def test_preference_is_memorable(self):
        assert is_memorable("I prefer using Python for backend development") is True

    def test_name_is_memorable(self):
        assert is_memorable("My name is John and I work at Google") is True

    def test_remember_request_is_memorable(self):
        assert is_memorable("Remember that the API key expires monthly") is True

    def test_convention_is_memorable(self):
        assert is_memorable("We always use camelCase for JavaScript variables") is True

    def test_correction_is_memorable(self):
        assert is_memorable("Actually, the database is PostgreSQL not MySQL") is True

    def test_long_text_is_memorable(self):
        assert is_memorable("The project architecture follows a microservices pattern with each service deployed independently") is True

    def test_environment_fact_is_memorable(self):
        assert is_memorable("The project uses Python 3.11 and runs on Linux") is True

    def test_empty_not_memorable(self):
        assert is_memorable("") is False
        assert is_memorable("   ") is False


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_basic_split(self):
        result = _split_sentences("First sentence. Second sentence.")
        assert len(result) == 2

    def test_newline_split(self):
        result = _split_sentences("Line one\nLine two")
        assert len(result) == 2

    def test_single_sentence(self):
        result = _split_sentences("Just one sentence")
        assert len(result) == 1

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == []

    def test_multiple_punctuation(self):
        result = _split_sentences("Wow! Really? Yes.")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# extract_facts
# ---------------------------------------------------------------------------


class TestExtractFacts:
    def test_extracts_from_user_message(self):
        facts = extract_facts("I prefer dark mode. My name is Alice.")
        assert len(facts) >= 1

    def test_ignores_assistant_messages(self):
        facts = extract_facts("Here is the code you asked for.", role="assistant")
        assert facts == []

    def test_empty_text(self):
        assert extract_facts("") == []
        assert extract_facts("   ") == []
        assert extract_facts(None) == []

    def test_fact_has_encoding(self):
        facts = extract_facts("I always prefer using TypeScript over JavaScript")
        assert len(facts) >= 1
        fact = facts[0]
        assert isinstance(fact, ExtractedFact)
        assert len(fact.encoding.categories) > 0
        assert fact.encoding.importance > 0

    def test_scope_propagated(self):
        facts = extract_facts("I prefer Python for scripting", scope="/user")
        if facts:
            assert facts[0].scope == "/user"

    def test_ephemeral_not_extracted(self):
        facts = extract_facts("ok")
        assert facts == []

    def test_full_text_fallback(self):
        # A single memorable sentence that isn't split
        text = "Remember that the deployment key is stored in vault"
        facts = extract_facts(text)
        assert len(facts) >= 1


# ---------------------------------------------------------------------------
# ForgettingManager
# ---------------------------------------------------------------------------


class TestForgettingManager:
    def test_run_forgetting_cycle(self, store, forgetting):
        # Add some memories
        store.add_memory("important", importance=0.8)
        store.add_memory("weak", importance=0.03)
        store.add_memory("moderate", importance=0.5)

        result = forgetting.run_forgetting_cycle()
        assert isinstance(result, ForgettingResult)
        assert result.pruned_count >= 1  # "weak" should be pruned

    def test_decay_reduces_old_memories(self, store, forgetting):
        mid = store.add_memory("old memory", importance=1.0)
        # Backdate last_accessed
        conn = store._get_conn()
        old_time = time.time() - (30 * 86400)
        conn.execute(
            "UPDATE cognitive_memories SET last_accessed = ? WHERE id = ?",
            (old_time, mid),
        )
        conn.commit()

        result = forgetting.run_forgetting_cycle()
        assert result.decayed_count >= 1

        mem = store.get_memory(mid)
        assert mem.importance < 1.0

    def test_consolidation_merges_duplicates(self, store):
        emb = [1.0, 0.0, 0.0]
        store.add_memory("Python is great", embedding=emb, importance=0.8)
        store.add_memory("Python is great", embedding=emb, importance=0.5)

        manager = ForgettingManager(
            store=store,
            consolidation_similarity=0.99,
        )
        result = manager.run_forgetting_cycle()
        assert result.consolidated_count >= 1
        assert store.count() == 1  # One should be soft-deleted

    def test_consolidation_keeps_higher_importance(self, store):
        emb = [1.0, 0.0, 0.0]
        id1 = store.add_memory("dup A", embedding=emb, importance=0.9)
        id2 = store.add_memory("dup B", embedding=emb, importance=0.3)

        manager = ForgettingManager(
            store=store,
            consolidation_similarity=0.99,
        )
        manager.run_forgetting_cycle()

        # Higher importance one should survive
        mem1 = store.get_memory(id1)
        mem2 = store.get_memory(id2)
        assert mem1.forgotten is False
        assert mem2.forgotten is True

    def test_exempt_scopes(self, store, forgetting):
        mid = store.add_memory("user pref", importance=1.0, scope="/user")
        conn = store._get_conn()
        old_time = time.time() - (60 * 86400)
        conn.execute(
            "UPDATE cognitive_memories SET last_accessed = ? WHERE id = ?",
            (old_time, mid),
        )
        conn.commit()

        result = forgetting.run_forgetting_cycle(exempt_scopes=["/user"])
        mem = store.get_memory(mid)
        assert mem.importance == pytest.approx(1.0)

    def test_should_run_cycle_too_soon(self, store, forgetting):
        # Add enough memories
        for i in range(5):
            store.add_memory(f"mem-{i}")

        # Just ran - shouldn't run again
        now = time.time()
        assert forgetting.should_run_cycle(last_run_time=now) is False

    def test_should_run_cycle_enough_time(self, store, forgetting):
        for i in range(5):
            store.add_memory(f"mem-{i}")

        old_time = time.time() - (7 * 3600)  # 7 hours ago
        assert forgetting.should_run_cycle(last_run_time=old_time) is True

    def test_should_run_cycle_not_enough_memories(self, store, forgetting):
        store.add_memory("only one")
        assert forgetting.should_run_cycle() is False

    def test_should_run_cycle_no_last_run(self, store, forgetting):
        for i in range(5):
            store.add_memory(f"mem-{i}")
        assert forgetting.should_run_cycle(last_run_time=None) is True

    def test_prune_threshold_respected(self, store):
        store.add_memory("above", importance=0.2)
        store.add_memory("below", importance=0.01)

        manager = ForgettingManager(store=store, prune_threshold=0.1)
        result = manager.run_forgetting_cycle()
        assert result.pruned_count == 1
        assert result.total_active == 1

    def test_maybe_run_cycle_runs_when_due(self, store):
        for i in range(6):
            store.add_memory(f"mem-{i}", importance=0.8)

        manager = ForgettingManager(store=store)
        assert manager._last_cycle_run is None

        result = manager.maybe_run_cycle()
        assert result is not None
        assert manager._last_cycle_run is not None

    def test_maybe_run_cycle_skips_when_too_soon(self, store):
        for i in range(6):
            store.add_memory(f"mem-{i}", importance=0.8)

        manager = ForgettingManager(store=store)
        # First run should work
        result1 = manager.maybe_run_cycle()
        assert result1 is not None

        # Second run immediately should be skipped
        result2 = manager.maybe_run_cycle()
        assert result2 is None

    def test_maybe_run_cycle_skips_few_memories(self, store):
        store.add_memory("only one")
        manager = ForgettingManager(store=store)
        result = manager.maybe_run_cycle()
        assert result is None
        assert manager._last_cycle_run is None
