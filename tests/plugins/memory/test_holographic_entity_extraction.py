"""Regression tests for apostrophes being treated as single-quote delimiters.

``_RE_SINGLE_ENTITY`` matched ``'([^']+)'``, so an apostrophe inside a word
opened a quoted-entity match that ran to the next apostrophe anywhere in the
text. Ordinary English triggers it:

    "...requested restarts aren't incidents; anything with no rollback needs a
     dry run plus sign-off on blast radius first - 'yes' to a summary isn't that."

The apostrophe in ``aren't`` opened a match that closed on the one in
``isn't``, storing the whole intervening span as an entity name. Possessives
did the same: ``EDI's role ...`` opened a match that ran to the next
apostrophe in the fact.

Every fact containing a contraction or possessive polluted the entity graph,
and because ``_compute_hrr_vector`` encodes linked entities as roles, the junk
was baked into each fact's similarity vector as well.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "m.db"))
    yield s
    s.close()


def test_contractions_do_not_open_a_quoted_entity(store):
    text = (
        "Requested restarts aren't incidents; anything with no rollback needs a dry run "
        "plus sign-off on blast radius first - 'yes' to a summary isn't that."
    )
    names = store._extract_entities(text)
    assert "yes" in [n.lower() for n in names]
    assert all(len(n) <= 60 for n in names)
    assert not any("incidents" in n or "summary" in n for n in names)


def test_possessive_does_not_open_a_quoted_entity(store):
    text = "EDI's role is to assist with reviews. Ben's fleet runs on migrant."
    assert store._extract_entities(text) == []


def test_genuine_single_quoted_term_is_still_extracted(store):
    assert "post it" in store._extract_entities("Ben says to 'post it' when ready.")


def test_capitalized_and_double_quoted_entities_are_unaffected(store):
    names = store._extract_entities('John Smith approved "Cloud One" today.')
    assert "John Smith" in names and "Cloud One" in names


def test_overlong_quoted_span_is_rejected(store):
    text = "'" + "x" * 200 + "'"
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("typographic", ["’", "'"])
def test_apostrophe_variants_do_not_create_fragments(store, typographic):
    text = f"It{typographic}s fine. Nothing here is a name, it{typographic}s just prose."
    assert all(len(n) <= 60 for n in store._extract_entities(text))
