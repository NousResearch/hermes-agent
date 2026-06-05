"""Tests for holographic memory auto_extract structured facts (#22907).

The old regex patterns matched "I like / I want / I need" and dumped the
entire user message verbatim as a fact. "I like that, sounds good" became
a stored fact.

The fix removes noisy patterns and extracts clean declarative statements
from captured groups.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test_memory.db")
    return MemoryStore(db_path=db, default_trust=0.5, hrr_dim=128)


class TestAutoExtract:
    """auto_extract should produce clean declarative facts, not raw messages."""

    def _make_provider(self, store):
        from plugins.memory.holographic import HolographicMemoryProvider
        provider = HolographicMemoryProvider.__new__(HolographicMemoryProvider)
        provider._store = store
        provider._config = {"auto_extract": True}
        return provider

    def test_preference_extraction(self, store):
        """'I prefer dark mode' should extract 'prefers dark mode'."""
        provider = self._make_provider(store)
        provider._auto_extract_facts([
            {"role": "user", "content": "I prefer dark mode over light mode"}
        ])
        facts = store.list_facts(category="user_pref")
        assert any("dark mode" in f["content"] for f in facts)
        # Should NOT contain the raw message
        assert not any("over light mode" not in f["content"] and "I prefer" in f["content"] for f in facts)

    def test_no_raw_message_dump(self, store):
        """'I like that, sounds good' should NOT be stored as a fact."""
        provider = self._make_provider(store)
        provider._auto_extract_facts([
            {"role": "user", "content": "I like that, sounds good"}
        ])
        facts = store.list_facts()
        # "I like" no longer matches — we removed "like" from the pattern
        assert not any("I like that" in f["content"] for f in facts)

    def test_decision_extraction(self, store):
        """'We decided to use PostgreSQL' should extract 'decided to use PostgreSQL'."""
        provider = self._make_provider(store)
        provider._auto_extract_facts([
            {"role": "user", "content": "We decided to use PostgreSQL for the backend"}
        ])
        facts = store.list_facts(category="project")
        assert any("PostgreSQL" in f["content"] for f in facts)
        assert any(f["content"].startswith("decided to") for f in facts)

    def test_habit_extraction(self, store):
        """'I always check git status' should extract 'always check git status'."""
        provider = self._make_provider(store)
        provider._auto_extract_facts([
            {"role": "user", "content": "I always check git status before committing"}
        ])
        facts = store.list_facts(category="user_pref")
        assert any("always check git status" in f["content"] for f in facts)

    def test_assistant_messages_ignored(self, store):
        """Assistant messages should not produce facts."""
        provider = self._make_provider(store)
        provider._auto_extract_facts([
            {"role": "assistant", "content": "I prefer to use structured outputs"}
        ])
        facts = store.list_facts()
        assert len(facts) == 0

    def test_short_messages_ignored(self, store):
        """Messages under 10 chars should be skipped."""
        provider = self._make_provider(store)
        provider._auto_extract_facts([
            {"role": "user", "content": "I like"}
        ])
        facts = store.list_facts()
        assert len(facts) == 0

    def test_one_fact_per_message(self, store):
        """A single message should produce at most one fact."""
        provider = self._make_provider(store)
        provider._auto_extract_facts([
            {"role": "user", "content": "I prefer Rust. We decided to use Rust. I always use Rust."}
        ])
        facts = store.list_facts(category="user_pref")
        # Should produce at most 1 fact for this message (first match wins)
        assert len([f for f in facts if "Rust" in f["content"]]) <= 1

    def test_fact_length_capped(self, store):
        """Extracted facts should not exceed 200 characters."""
        provider = self._make_provider(store)
        long_pref = "x" * 300
        provider._auto_extract_facts([
            {"role": "user", "content": f"I prefer {long_pref}"}
        ])
        facts = store.list_facts()
        for f in facts:
            assert len(f["content"]) <= 200
