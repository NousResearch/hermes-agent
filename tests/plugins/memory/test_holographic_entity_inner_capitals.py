"""Regression test: entity extraction must recognize names with inner capitals.

"McKellar" (capital K mid-word) previously failed the capitalized-phrase
regex ``[A-Z][a-z]+``, so names like "Sharon McKellar" produced no entity.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "memory_store.db"


def _entity_names(store):
    return [
        r["name"]
        for r in store._conn.execute("SELECT name FROM entities ORDER BY name")
    ]


def test_inner_capital_surname_creates_entity(db_path):
    with MemoryStore(db_path) as store:
        store.add_fact("Sharon McKellar is Ian's wife.", category="family")
        assert "Sharon McKellar" in _entity_names(store)


def test_multiword_inner_capital_names_all_extracted(db_path):
    with MemoryStore(db_path) as store:
        store.add_fact(
            "Lillian McKellar and Matilda McKellar are twins.", category="family"
        )
        names = _entity_names(store)
        assert "Lillian McKellar" in names
        assert "Matilda McKellar" in names
