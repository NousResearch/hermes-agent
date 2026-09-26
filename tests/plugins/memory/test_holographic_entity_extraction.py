"""Tests for holographic entity extraction and the public entity-link API.

The extractor had three related defects, and they are not cosmetic: the entity
index is what probe()/related()/reason() answer from, so a junk entity invents a
party that does not exist and a missing one hides a real link.

  * "The Robosmart pilot renewed the contract." stored an entity literally named
    "The Robosmart". It co-occurs with the real "Robosmart", so related() reports a
    connection between two things that are the same thing.
  * `[A-Z][a-z]+` cannot match a name containing a digit, so "B2B Scaler" was never
    found at all.
  * The "X aka Y" pattern used a greedy ``\\w+`` for the right-hand side, so
    "Alice Cooper aka The Falcon joined the review" produced an entity called
    "The Falcon joined the review" -- the whole trailing clause.

Single-word names ("Robosmart", "Miranda") are still not discoverable by the
patterns -- that is deliberate, because matching every capitalised word is worse --
so they are covered by names the store already knows, and by link_entities() for a
name being seen for the first time. Both are tested here.

Covers: article stripping, digit-bearing names, non-greedy "aka", the known-name
gazetteer, link_entities(), reindex_entities(), and the absence of false positives.
"""

import sqlite3

import pytest

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
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


def entity_names(store: MemoryStore, fact_id: int) -> set[str]:
    """Entities linked to a fact, read straight from the index."""
    rows = store._conn.execute(
        "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id "
        "WHERE fe.fact_id = ?", (fact_id,)).fetchall()
    return {row["name"] for row in rows}


def all_entity_names(store: MemoryStore) -> set[str]:
    return {row["name"] for row in store._conn.execute("SELECT name FROM entities").fetchall()}


class TestExtractionJunk:
    """Entities that must never be created."""

    def test_leading_article_is_not_part_of_the_name(self, store):
        fact_id = store.add_fact("The Robosmart pilot renewed the contract.", category="project")

        assert entity_names(store, fact_id) == {"Robosmart"}
        assert "The Robosmart" not in all_entity_names(store)

    def test_article_stripping_also_applies_to_aka(self, store):
        fact_id = store.add_fact("The Robosmart aka The Robosmart", category="project")

        assert entity_names(store, fact_id) == {"Robosmart"}

    def test_aka_does_not_run_on_greedily(self, store):
        """The right-hand side must stop at the name, not swallow the next verb."""
        fact_id = store.add_fact("Alice Cooper aka The Falcon joined the review.", category="general")

        names = entity_names(store, fact_id)
        assert "Alice Cooper" in names
        assert "Falcon" in names
        assert not any("joined" in name for name in all_entity_names(store))

    def test_sentence_initial_ordinary_words_are_not_entities(self, store):
        fact_id = store.add_fact("The report was filed after lunch.", category="general")

        assert entity_names(store, fact_id) == set()

    def test_trailing_punctuation_is_not_captured(self, store):
        fact_id = store.add_fact("Acme Corp. closed the deal.", category="general")

        assert entity_names(store, fact_id) == {"Acme Corp"}


class TestDigitNames:
    """Names containing digits used to be invisible to the patterns."""

    def test_digit_bearing_name_is_extracted(self, store):
        fact_id = store.add_fact("B2B Scaler LLC signed with Revnet this quarter.", category="project")

        assert entity_names(store, fact_id) == {"B2B Scaler"}

    def test_no_spurious_sub_token_entity(self, store):
        """A multi-word match must not also yield its first token as a second entity."""
        store.add_fact("B2B Scaler LLC signed with Revnet this quarter.", category="project")

        assert "B2B" not in all_entity_names(store)


class TestSingleWordNames:
    """Single-word names: known ones resolve, new ones need link_entities()."""

    def test_known_single_word_name_is_linked_in_a_later_fact(self, store):
        store.link_entities(store.add_fact("A note with no proper nouns.", category="general"),
                            ["Robosmart"])
        fact_id = store.add_fact("Robosmart asked for the renewal terms.", category="project")

        assert "Robosmart" in entity_names(store, fact_id)

    def test_alias_resolves_to_the_canonical_name(self, store):
        fact_id = store.add_fact("A note with no proper nouns.", category="general")
        ids = store.link_entities(fact_id, ["Hermes"])
        store._write("UPDATE entities SET aliases = ? WHERE entity_id = ?", ("hermes-agent", ids[0]))

        later = store.add_fact("hermes-agent reported a clean run.", category="tool")

        assert entity_names(store, later) == {"Hermes"}


class TestLinkEntities:
    """The public write-side API."""

    def test_returns_ids_and_links_them(self, store):
        fact_id = store.add_fact("A note with no proper nouns.", category="general")

        ids = store.link_entities(fact_id, ["Zorblat"])

        assert ids and entity_names(store, fact_id) == {"Zorblat"}

    def test_is_idempotent(self, store):
        fact_id = store.add_fact("A note with no proper nouns.", category="general")
        first = store.link_entities(fact_id, ["Zorblat"])
        second = store.link_entities(fact_id, ["Zorblat"])

        assert first == second
        assert entity_names(store, fact_id) == {"Zorblat"}

    def test_strips_articles_and_blank_names(self, store):
        fact_id = store.add_fact("A note with no proper nouns.", category="general")

        store.link_entities(fact_id, ["The Zorblat", "   ", ""])

        assert entity_names(store, fact_id) == {"Zorblat"}


class TestReindexEntities:
    """Backfilling, which previously needed private calls plus raw SQL."""

    def test_reindexes_a_single_fact(self, store):
        fact_id = store.add_fact("The Robosmart pilot renewed the contract.", category="project")
        store._conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
        store._conn.commit()
        assert entity_names(store, fact_id) == set()

        result = store.reindex_entities(fact_id)

        assert result == {"facts": 1}
        assert entity_names(store, fact_id) == {"Robosmart"}

    def test_reindexes_every_fact_by_default(self, store):
        ids = [store.add_fact(f"B2B Scaler note number {n}.", category="project") for n in range(3)]
        store._conn.execute("DELETE FROM fact_entities")
        store._conn.commit()

        result = store.reindex_entities()

        assert result == {"facts": 3}
        assert all(entity_names(store, fact_id) == {"B2B Scaler"} for fact_id in ids)


class TestNoNewJunk:
    """A relaxed token class must not admit initials, initialisms, or split an identity.

    An earlier revision of this change accepted any capitalised token, which made
    "The CEO John Smith approved the deal." yield "CEO John Smith" -- one person
    split across two entities, so probe("John Smith") missed the first fact -- and
    turned ordinary prose into entities ("I Think", "OK Google", "A Rollback").
    The token rule requires a lowercase letter or a digit for exactly this reason.
    """

    def test_initialism_does_not_join_a_person_name(self, store):
        fact_id = store.add_fact("The CEO John Smith approved the deal.", category="general")

        assert entity_names(store, fact_id) == {"John Smith"}

    def test_bare_initial_does_not_start_an_entity(self, store):
        fact_id = store.add_fact("I Think the migration is safe.", category="general")

        assert entity_names(store, fact_id) == set()

    def test_all_caps_word_does_not_start_an_entity(self, store):
        fact_id = store.add_fact("OK Google set a timer.", category="general")

        assert entity_names(store, fact_id) == set()

    def test_single_letter_article_plus_common_noun(self, store):
        fact_id = store.add_fact("A Rollback was performed at midnight.", category="general")

        assert entity_names(store, fact_id) == set()


class TestGazetteerIsShared:
    """The gazetteer cache must be shared per database, not per instance.

    Several stores share one connection per process (the main agent plus every
    delegate_task subagent). An instance-local cache goes stale when a SIBLING
    creates an entity, and then silently stops matching a known name -- or, worse,
    matches a shorter pre-existing entity and invents a co-occurrence.
    """

    def test_sibling_instance_sees_a_newly_created_entity(self, tmp_path):
        first = MemoryStore(tmp_path / "memory_store.db")
        second = MemoryStore(tmp_path / "memory_store.db")
        assert first._conn is second._conn  # same DB shares one connection

        second._known_names_in("warm second's gazetteer while the table is empty")
        first.link_entities(first.add_fact("a note with no proper nouns"), ["Robosmart"])

        fact_id = second.add_fact("Robosmart asked for the terms.", category="project")

        assert "Robosmart" in entity_names(second, fact_id)


class TestReindexEntitiesPrune:
    """Pruning is what actually repairs a store polluted before the fix."""

    def test_prune_removes_links_extraction_would_not_produce(self, store):
        fact_id = store.add_fact("placeholder", category="project")
        store._conn.execute("UPDATE facts SET content = ? WHERE fact_id = ?",
                            ("The Robosmart pilot renewed the contract.", fact_id))
        store._conn.commit()
        junk = store._resolve_entity("The Robosmart")
        store._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                     (fact_id, junk))
        assert "The Robosmart" in entity_names(store, fact_id)

        store.reindex_entities(fact_id)                 # additive: junk survives
        assert "The Robosmart" in entity_names(store, fact_id)

        store.reindex_entities(fact_id, prune=True)     # repair
        assert entity_names(store, fact_id) == {"Robosmart"}


class TestLinkEntitiesInput:
    """Input hygiene: a bare string must not become one entity per letter."""

    def test_bare_string_is_treated_as_one_name(self, store):
        fact_id = store.add_fact("a note with no proper nouns")

        store.link_entities(fact_id, "Qux")

        assert entity_names(store, fact_id) == {"Qux"}

    def test_unknown_fact_id_raises(self, store):
        with pytest.raises(KeyError):
            store.link_entities(999999, ["Ghost"])
