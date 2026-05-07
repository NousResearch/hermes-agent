"""Tests for MemoryStore.rename_entity — the canonicalization invariant.

Renaming an entity must:
  - update entities.name in the row
  - merge the old name into entities.aliases (so legacy probes still resolve)
  - re-encode every linked fact's hrr_vector against the new canonical name
  - leave encoding_version at the current version
  - be atomic: a mid-rename failure leaves no partial state
  - reject empty new_name
  - reject unknown entity_id
"""

from __future__ import annotations

import os
import tempfile

import pytest

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import (
    MemoryStore,
    _CURRENT_ENCODING_VERSION,
)


DIM = 2048


@pytest.fixture
def store_and_retriever():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = MemoryStore(db_path=path, hrr_dim=DIM, default_trust=0.85)
    retriever = FactRetriever(
        store=store,
        temporal_decay_half_life=0,
        reinforce_on_retrieval=False,
        hrr_dim=DIM,
        hrr_weight=0.3,
    )
    yield store, retriever
    store.close()
    try:
        os.unlink(path)
    except OSError:
        pass


def _raw_sim(score: float, fact: dict) -> float:
    base = float(fact.get("trust_score", 0) or 0)
    hc = int(fact.get("helpful_count", 0) or 0)
    mult = base * (1.0 + 0.1 * min(hc, 20))
    return (score / mult) if mult > 0 else 0.0


def _entity_id_for(store: MemoryStore, name: str) -> int:
    row = store._conn.execute(
        "SELECT entity_id FROM entities WHERE LOWER(name) = LOWER(?)", (name,)
    ).fetchone()
    assert row is not None, f"entity {name!r} not found"
    return int(row["entity_id"])


def test_rename_updates_name_and_merges_alias(store_and_retriever):
    store, _ = store_and_retriever
    store.add_fact("Apollo Energy Resources is a multi-division business.",
                   category="identity")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    result = store.rename_entity(eid, "Apollo Energy Group", add_aliases=["AEG"])

    row = store._conn.execute(
        "SELECT name, aliases FROM entities WHERE entity_id = ?", (eid,)
    ).fetchone()
    assert row["name"] == "Apollo Energy Group"
    aliases = {a.strip().lower() for a in (row["aliases"] or "").split(",")}
    assert "apollo energy resources" in aliases
    assert "aeg" in aliases

    assert result["old_name"] == "Apollo Energy Resources"
    assert result["new_name"] == "Apollo Energy Group"
    assert result["facts_re_encoded"] >= 1


def test_rename_re_encodes_facts(store_and_retriever):
    """After rename, hrr_vector for linked facts must reflect the new name."""
    store, _ = store_and_retriever
    fid = store.add_fact("Apollo Energy Resources is a multi-division business.",
                         category="identity")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    old_vec = store._conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()["hrr_vector"]

    store.rename_entity(eid, "Apollo Energy Group")

    new_vec = store._conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()["hrr_vector"]

    # Vector bytes must change (entity atom in the bundle changed).
    assert old_vec != new_vec
    # And the new vector must encode the new entity.
    new_vec_phases = hrr.bytes_to_phases(new_vec)
    role_entity = hrr.encode_atom("__hrr_role_entity__", DIM)
    new_atom = hrr.encode_atom("apollo energy group", DIM)
    residual = hrr.unbind(new_vec_phases, role_entity)
    assert hrr.similarity(residual, new_atom) > 0.30


def test_rename_keeps_encoding_version_current(store_and_retriever):
    store, _ = store_and_retriever
    fid = store.add_fact("Apollo Energy Resources owns trucks.", category="general")
    eid = _entity_id_for(store, "Apollo Energy Resources")
    # Force encoding_version to 0 to simulate an old encoding.
    store._conn.execute(
        "UPDATE facts SET encoding_version = 0 WHERE fact_id = ?", (fid,)
    )
    store._conn.commit()

    store.rename_entity(eid, "Apollo Energy Group")

    row = store._conn.execute(
        "SELECT encoding_version FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert row["encoding_version"] == _CURRENT_ENCODING_VERSION


def test_rename_then_probe_by_new_name_recovers(store_and_retriever):
    """End-to-end Strategy A recovery via the new canonical name."""
    store, retriever = store_and_retriever
    fid = store.add_fact("Apollo Energy Resources is a multi-division business.",
                         category="identity")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    store.rename_entity(eid, "Apollo Energy Group")

    results = retriever.probe(entity="Apollo Energy Group", limit=5)
    by_id = {r["fact_id"]: r for r in results}
    assert fid in by_id
    sim = _raw_sim(by_id[fid]["score"], by_id[fid])
    assert sim > 0.30, f"new-name probe raw_sim={sim:.4f}; expected > 0.30"


def test_rename_rejects_unknown_entity(store_and_retriever):
    store, _ = store_and_retriever
    with pytest.raises(KeyError):
        store.rename_entity(99999, "Anything")


def test_rename_rejects_empty_name(store_and_retriever):
    store, _ = store_and_retriever
    store.add_fact("Apollo Energy Resources owns trucks.", category="general")
    eid = _entity_id_for(store, "Apollo Energy Resources")
    with pytest.raises(ValueError):
        store.rename_entity(eid, "   ")


def test_rename_atomic_on_compute_failure(store_and_retriever, monkeypatch):
    """If _compute_hrr_vector raises mid-loop, rollback leaves entity unchanged.

    Patches encode_fact to fail; the rename must abort and revert the
    entities.name write.
    """
    store, _ = store_and_retriever
    store.add_fact("Apollo Energy Resources is a multi-division business.",
                   category="identity")
    store.add_fact("Apollo Energy Resources runs LNG logistics.",
                   category="general")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    call_count = {"n": 0}
    real_encode = hrr.encode_fact

    def flaky_encode(content, entities, dim):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated encode failure")
        return real_encode(content, entities, dim)

    monkeypatch.setattr(
        "plugins.memory.holographic.store.hrr.encode_fact",
        flaky_encode,
    )

    with pytest.raises(RuntimeError, match="simulated encode failure"):
        store.rename_entity(eid, "Apollo Energy Group")

    # Name must be unchanged.
    row = store._conn.execute(
        "SELECT name FROM entities WHERE entity_id = ?", (eid,)
    ).fetchone()
    assert row["name"] == "Apollo Energy Resources"
