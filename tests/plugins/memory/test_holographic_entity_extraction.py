"""Regression tests for precision-oriented holographic entity extraction."""

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    instance = MemoryStore(tmp_path / "entity-extraction.db")
    yield instance
    instance.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("John Doe works at Acme Labs.", ["John Doe", "Acme Labs"]),
        (
            'The migration targets "PostgreSQL" and \'pgvector\'.',
            ["PostgreSQL", "pgvector"],
        ),
        (
            "Guido van Rossum aka BDFL maintains Python.",
            ["Guido van Rossum", "BDFL"],
        ),
        ("Active Directory authenticates users.", ["Active Directory"]),
        ("New York hosts the service.", ["New York"]),
        ("General Motors builds vehicles.", ["General Motors"]),
        ("Project Gutenberg archives books.", ["Project Gutenberg"]),
        ('The label is "Active Project".', ["Active Project"]),
        ('The title is "No Country for Old Men".', ["No Country for Old Men"]),
        (
            "The user prefers John Doe aka JD for demos.",
            ["John Doe", "JD"],
        ),
        ("John Doe aka JD.", ["John Doe", "JD"]),
        ("pytest aka py.test is configured.", ["pytest", "py.test"]),
        ("pytest aka py.test.", ["pytest", "py.test"]),
        (
            "The user's editor is 'vim' and shell is 'zsh'.",
            ["vim", "zsh"],
        ),
        ('The runtime is "R".', ["R"]),
        (
            "Use \u201cGraph Memory\u201d with \u2018sqlite_vec\u2019 today.",
            ["Graph Memory", "sqlite_vec"],
        ),
    ],
)
def test_extract_entities_preserves_high_signal_candidates(store, text, expected):
    assert store._extract_entities(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Active Project uses Python.",
        "Concept Mapping improves recall.",
        "Confirmed Decision changes tomorrow.",
        "No Foo should be retained.",
        "Current Status remains unchanged.",
        "Alice's editor and Bob's shell are configured.",
        "The code is 'x'.",
    ],
)
def test_extract_entities_rejects_structural_noise(store, text):
    assert store._extract_entities(text) == []


def test_extract_entities_deduplicates_separator_variants(store):
    assert store._extract_entities(
        'Use "Some-Project" and "Some Project" together.'
    ) == ["Some-Project"]


def test_add_fact_does_not_persist_possessive_fragments(store):
    fact_id = store.add_fact("Alice's editor and Bob's shell are configured.")

    links = store._conn.execute(
        "SELECT e.name FROM entities e "
        "JOIN fact_entities fe ON fe.entity_id = e.entity_id "
        "WHERE fe.fact_id = ?",
        (fact_id,),
    ).fetchall()

    assert links == []
