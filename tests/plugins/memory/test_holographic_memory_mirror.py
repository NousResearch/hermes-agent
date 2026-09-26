"""Tests for the built-in-memory mirror and for retrieval_count.

`on_memory_write` is how a write to the built-in memory file is mirrored into the
fact store. It mirrored only `add`, so a `replace` or `remove` left the superseded
text sitting beside the correction: the store kept feeding later sessions a claim
the user had retracted, and the two stores silently diverged. The previous text is
what identifies the row to amend, and it reaches the provider in `metadata` --
declaring `metadata` in the signature is load-bearing, because the manager inspects
the signature and uses a legacy 3-argument call when it is absent.

`retrieval_count` was declared in the schema and surfaced in every fact dict, but
never incremented anywhere, so it always read 0.

Covers: add/replace/remove mirroring, the previous_content and old_text sources,
missing metadata, and retrieval_count semantics.
"""

import sqlite3

import pytest

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    """Each test starts and ends with an empty shared-connection registry."""
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db")})
    p.initialize("test-session")
    yield p
    p.shutdown()


def stored_contents(provider) -> list[str]:
    rows = provider._store._conn.execute("SELECT content FROM facts ORDER BY fact_id").fetchall()
    return [row["content"] for row in rows]


class TestMirrorAdd:
    def test_add_creates_a_fact(self, provider):
        provider.on_memory_write("add", "user", "User prefers dark mode")

        assert stored_contents(provider) == ["User prefers dark mode"]

    def test_blank_add_is_ignored(self, provider):
        provider.on_memory_write("add", "user", "")

        assert stored_contents(provider) == []

    def test_unknown_action_is_ignored(self, provider):
        provider.on_memory_write("sideways", "user", "whatever")

        assert stored_contents(provider) == []


class TestMirrorReplace:
    def test_replace_amends_instead_of_appending(self, provider):
        """The superseded text must not survive beside the correction."""
        provider.on_memory_write("add", "user", "User prefers dark mode")

        provider.on_memory_write("replace", "user", "User prefers light mode",
                                 metadata={"previous_content": "User prefers dark mode"})

        assert stored_contents(provider) == ["User prefers light mode"]

    def test_replace_falls_back_to_old_text(self, provider):
        provider.on_memory_write("add", "user", "alpha note")

        provider.on_memory_write("replace", "user", "beta note",
                                 metadata={"old_text": "alpha note"})

        assert stored_contents(provider) == ["beta note"]

    def test_replace_without_a_previous_row_records_the_new_text(self, provider):
        """Nothing mirrored yet: record the current claim rather than silently keeping none."""
        provider.on_memory_write("replace", "user", "first ever content", metadata=None)

        assert stored_contents(provider) == ["first ever content"]


class TestMirrorRemove:
    def test_remove_deletes_the_mirrored_fact(self, provider):
        provider.on_memory_write("add", "user", "temp note")

        provider.on_memory_write("remove", "user", "temp note",
                                 metadata={"previous_content": "temp note"})

        assert stored_contents(provider) == []

    def test_remove_without_metadata_is_a_no_op(self, provider):
        provider.on_memory_write("add", "user", "kept note")

        provider.on_memory_write("remove", "user", "", metadata=None)

        assert stored_contents(provider) == ["kept note"]

    def test_remove_ignores_other_facts(self, provider):
        provider.on_memory_write("add", "user", "first")
        provider.on_memory_write("add", "user", "second")

        provider.on_memory_write("remove", "user", "first",
                                 metadata={"previous_content": "first"})

        assert stored_contents(provider) == ["second"]


class TestRetrievalCount:
    def test_search_increments_only_returned_facts(self, provider):
        wanted = provider._store.add_fact("Compaction threshold tuned to 0.85.", category="tool")
        other = provider._store.add_fact("Something entirely unrelated.", category="tool")

        provider._retriever.search("compaction threshold", limit=5)

        counts = {row["fact_id"]: row["retrieval_count"]
                  for row in provider._store._conn.execute("SELECT fact_id, retrieval_count FROM facts")}
        assert counts[wanted] == 1
        assert counts[other] == 0

    def test_repeated_retrievals_accumulate(self, provider):
        fact_id = provider._store.add_fact("Compaction threshold tuned to 0.85.", category="tool")

        for _ in range(3):
            provider._retriever.search("compaction threshold", limit=5)

        count = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()["retrieval_count"]
        assert count == 3

    def test_entity_queries_also_count(self, provider):
        provider._store.add_fact("Robosmart renewed the pilot.", category="project")
        fact_id = provider._store.add_fact("Robosmart asked for terms.", category="project")

        provider._retriever.probe("Robosmart", limit=5)

        count = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        assert count is not None and count["retrieval_count"] >= 1

    def test_record_retrieval_is_batched_and_empty_safe(self, provider):
        first = provider._store.add_fact("one")
        second = provider._store.add_fact("two")

        assert provider._store.record_retrieval([first, second]) == 2
        assert provider._store.record_retrieval([]) == 0


class TestCrossTargetCollapse:
    """`facts.content` is UNIQUE across the whole table, so a sentence mirrored for two targets
    is ONE row -- the first target's category wins, and `add_fact` returns that row untouched for
    the second target. The consequence is sharper than "one row": removing the sentence for the
    owning target removes it for every target, because there is only one row to remove.

    `fact_id_for_content()` is scoped by category, so the asymmetry is real in both directions:
    a remove for the NON-owning target cannot identify the row and is a no-op, even though that
    target's fact is the one that vanished from the owner's remove.
    """

    sentence = "User prefers dark mode"

    def test_two_targets_sharing_a_sentence_collapse_to_one_row(self, provider):
        provider.on_memory_write("add", "user", self.sentence)
        provider.on_memory_write("add", "memory", self.sentence)

        rows = provider._store._conn.execute(
            "SELECT content, category FROM facts WHERE content = ?", (self.sentence,)).fetchall()
        assert len(rows) == 1
        assert rows[0]["category"] == "user_pref"  # first writer's category wins
        # The second target has no row of its own to address later.
        assert provider._store.fact_id_for_content(self.sentence, "general") is None

    def test_removing_the_owning_target_deletes_the_other_targets_fact(self, provider):
        provider.on_memory_write("add", "user", self.sentence)
        provider.on_memory_write("add", "memory", self.sentence)

        provider.on_memory_write("remove", "user", self.sentence,
                                 metadata={"previous_content": self.sentence})

        assert stored_contents(provider) == []

    def test_removing_the_non_owning_target_is_a_no_op(self, provider):
        provider.on_memory_write("add", "user", self.sentence)
        provider.on_memory_write("add", "memory", self.sentence)

        provider.on_memory_write("remove", "memory", self.sentence,
                                 metadata={"previous_content": self.sentence})

        # The row is user_pref, the lookup is general: nothing to remove, so the sentence survives.
        assert stored_contents(provider) == [self.sentence]
