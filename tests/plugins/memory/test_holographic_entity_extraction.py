"""Entity extraction tests for the holographic MemoryStore.

Covers the de-overlap rules that keep one phrase from yielding several entity
bindings (the reviewer's "The Thursday deployment" fixture), plus the
single-word / hyphenated / quoted / AKA patterns and the case-insensitive
dedup. Exact expected output is asserted, so a change to any pattern or to the
overlap rules fails here rather than silently degrading probe() recall.
"""

from __future__ import annotations

import pytest

from plugins.memory.holographic.store import MemoryStore

# A quoted run past the 48-character cap: a pasted block, not a name (the live store had a 426-char one).
LONG_QUOTE = 'He pasted "' + ("quoted text " * 5).strip() + '" into the note.'


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "memory_store.db")
    yield s
    s.close()


@pytest.mark.parametrize(
    "text,expected",
    [
        # Multi-word phrases stay one entity: no component words alongside them.
        ("Acme Corporation is the employer.", ["Acme Corporation"]),
        # ...wherever the phrase sits in the sentence, not just at the start of a word run.
        ("Alice works at Acme Corporation as a developer.", ["Acme Corporation", "Alice"]),
        # A leading stoplist word is stripped and the remainder is not ALSO bound separately.
        ("The Thursday deployment rollback failed.", ["Thursday"]),
        ("The Docker container restarted overnight.", ["Docker"]),
        # Hyphenated names are kept whole and do not leak their first syllable.
        ("Pi-hole blocks ads on the LAN.", ["Pi-hole"]),
        # Single capitalized words are extracted.
        ("Alice and Bob met Carol.", ["Alice", "Bob", "Carol"]),
        # Sentence-initial function words are not entities.
        ("The This That These Those are words.", []),
        # The stoplist filters function words as standalone candidates...
        ("Alice left, However.", ["Alice"]),
        # Non-ASCII names extract like ASCII ones: an accented capital is still a capital.
        ("Kőbánya is where József works on Kifőzde.", ["Kőbánya", "József", "Kifőzde"]),
        ("École Normale Supérieure is in Paris.", ["École Normale Supérieure", "Paris"]),
        ("Иван Петров работает в Москве.", ["Иван Петров", "Москве"]),
        # Language-independent literals are not entities either.
        ("The token was True.", []),
        # ...but a name that merely STARTS with one must survive intact (only determiners are trimmed,
        # because stripping every function word renamed real entities like these).
        ("One Horse Town is a phrase.", ["One Horse Town"]),
        ("Old Town Hall is the building.", ["Old Town Hall"]),
        # Arbitrary whitespace inside a phrase must not shift the claimed span (or eat the next word).
        ("The   Docker\ncontainer restarted.", ["Docker"]),
        # Words shorter than three characters are ignored (non-consecutive, so the multi-word rule is out).
        ("Pi and Go and Ok", []),
        # A hyphenated name with a one-letter prefix keeps its tail ("X-Gitlab-Token" -> "Gitlab-Token").
        ("The header is X-Gitlab-Token.", ["Gitlab-Token"]),
        # Quoted terms and AKA patterns.
        ('He called it "kubernetes cluster" once.', ["kubernetes cluster"]),
        # A quoted run past the 48-character cap is a pasted block, not an entity.
        (LONG_QUOTE, []),
        # A quoted term must carry a letter.
        ('The token was "12345".', []),
        ("Guido aka BDFL.", ["Guido", "BDFL"]),
        # Dedup is case-insensitive and keeps the first spelling seen.
        ("Alice told alice nothing.", ["Alice"]),
    ],
)
def test_extraction_exact_output(store, text, expected):
    assert store._extract_entities(text) == expected


def test_extraction_returns_nothing_for_plain_prose(store):
    assert store._extract_entities("the quick brown fox jumps over it") == []


def test_multi_word_phrase_does_not_bind_its_component_words(store):
    """Regression for the reviewer's ask: one phrase, one entity binding."""
    assert store._extract_entities("The Thursday deployment rollback failed.") == ["Thursday"]
    assert "The" not in store._extract_entities("The Thursday deployment rollback failed.")
    assert "The Thursday" not in store._extract_entities("The Thursday deployment rollback failed.")


def test_hyphenated_entity_does_not_bind_its_prefix(store):
    """Regression: "Pi-hole" must not also create "Pi"."""
    extracted = store._extract_entities("Pi-hole runs on the Pi")
    assert extracted == ["Pi-hole"]


def test_extracted_entities_are_linked_to_the_fact(store):
    """Extraction feeds fact_entities, which is the structural ground truth probe() relies on."""
    fact_id = store.add_fact("Sierra renewal intake is a greenfield project.", category="project")
    linked = [row["name"] for row in store._conn.execute(
        "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
        (fact_id,)).fetchall()]
    assert "Sierra" in linked
