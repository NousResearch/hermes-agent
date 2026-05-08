"""Tests for MemoryStore.merge_entities — entity dedup primitive.

Merging source into target must:
  - re-point every fact_entities row from source_id to target_id
  - collapse pre-existing dual links (no duplicate fact_entities rows)
  - union source.aliases + source.name into target.aliases (case-insensitive
    dedup; target.name itself stays out of its own aliases)
  - delete the source entity row
  - re-encode every formerly-source-linked fact's hrr_vector
  - leave encoding_version current
  - be atomic: any mid-merge failure rolls back to the prior state
  - reject source==target (ValueError) and missing ids (KeyError)
  - keep probes by the old source name resolving (via the merged alias)
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


def _entity_id_for(store: MemoryStore, name: str) -> int:
    row = store._conn.execute(
        "SELECT entity_id FROM entities WHERE LOWER(name) = LOWER(?)", (name,)
    ).fetchone()
    assert row is not None, f"entity {name!r} not found"
    return int(row["entity_id"])


def _raw_sim(score: float, fact: dict) -> float:
    base = float(fact.get("trust_score", 0) or 0)
    hc = int(fact.get("helpful_count", 0) or 0)
    mult = base * (1.0 + 0.1 * min(hc, 20))
    return (score / mult) if mult > 0 else 0.0


def test_merge_re_points_facts_and_deletes_source(store_and_retriever):
    store, _ = store_and_retriever
    fa = store.add_fact("Apollo Energy Resources is a multi-division business.",
                        category="identity")
    fb = store.add_fact("Apollo Energy Resources runs LNG logistics.",
                        category="general")
    src = _entity_id_for(store, "Apollo Energy Resources")
    # Create the target by adding a fact that mentions it directly.
    fc = store.add_fact("Apollo Energy Group has divisions across LNG.",
                        category="identity")
    tgt = _entity_id_for(store, "Apollo Energy Group")
    assert src != tgt

    result = store.merge_entities(src, tgt)

    # Source row gone.
    assert store._conn.execute(
        "SELECT 1 FROM entities WHERE entity_id = ?", (src,)
    ).fetchone() is None

    # All formerly-source-linked facts now point at target.
    rows = store._conn.execute(
        "SELECT fact_id FROM fact_entities WHERE entity_id = ?", (tgt,)
    ).fetchall()
    fids_at_tgt = {r["fact_id"] for r in rows}
    assert fa in fids_at_tgt and fb in fids_at_tgt and fc in fids_at_tgt

    # No fact_entities row still references the source.
    assert store._conn.execute(
        "SELECT COUNT(*) FROM fact_entities WHERE entity_id = ?", (src,)
    ).fetchone()[0] == 0

    assert result["source_id"] == src
    assert result["target_id"] == tgt
    assert result["facts_re_encoded"] == 2  # fa + fb were source-linked


def test_merge_unions_aliases_case_insensitive(store_and_retriever):
    store, _ = store_and_retriever
    # Build two entities by hand so the test doesn't depend on the
    # capitalized-phrase extractor (which can't see "Apollo Energy Resources LLC"
    # as a single entity due to the all-caps "LLC" suffix).
    store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES (?, ?)",
        ("Apollo Energy Resources LLC", "AER,Apollo Energy"),
    )
    store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES (?, ?)",
        ("Apollo Energy Group", "apollo energy,AEG"),
    )
    store._conn.commit()
    src = _entity_id_for(store, "Apollo Energy Resources LLC")
    tgt = _entity_id_for(store, "Apollo Energy Group")

    store.merge_entities(src, tgt)

    row = store._conn.execute(
        "SELECT aliases FROM entities WHERE entity_id = ?", (tgt,)
    ).fetchone()
    aliases_lower = {a.strip().lower() for a in row["aliases"].split(",")}
    # Union: aer, apollo energy, aeg, apollo energy resources llc (source.name).
    assert "aer" in aliases_lower
    assert "apollo energy" in aliases_lower
    assert "aeg" in aliases_lower
    assert "apollo energy resources llc" in aliases_lower
    # Target's own name must NOT appear in its aliases.
    assert "apollo energy group" not in aliases_lower


def test_merge_collapses_dual_links_no_duplicates(store_and_retriever):
    """Fact linked to BOTH source and target ends up linked to target only."""
    store, _ = store_and_retriever
    fid = store.add_fact("Walker Anderson manages logistics.", category="general")
    # Manually add a second entity and link it to the same fact.
    store._conn.execute("INSERT INTO entities (name) VALUES (?)", ("Walker A.",))
    src = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = 'Walker A.'"
    ).fetchone()["entity_id"]
    store._conn.execute(
        "INSERT INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
        (fid, src),
    )
    store._conn.commit()
    tgt = _entity_id_for(store, "Walker Anderson")

    store.merge_entities(src, tgt)

    # Exactly one row for (fid, tgt). The dual link collapsed.
    rows = store._conn.execute(
        "SELECT entity_id FROM fact_entities WHERE fact_id = ?", (fid,)
    ).fetchall()
    eids = [r["entity_id"] for r in rows]
    assert eids.count(tgt) == 1
    assert src not in eids


def test_merge_empty_source_only_unions_aliases(store_and_retriever):
    """Source with zero linked facts: aliases merge, source row deleted, no re-encode."""
    store, _ = store_and_retriever
    store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES (?, ?)",
        ("Apollo Sales", "ApolloSales"),
    )
    src = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = 'Apollo Sales'"
    ).fetchone()["entity_id"]
    store.add_fact("Apollo Energy Group runs sales out of Houston.",
                   category="general")
    tgt = _entity_id_for(store, "Apollo Energy Group")
    store._conn.commit()

    result = store.merge_entities(src, tgt)
    assert result["facts_re_encoded"] == 0

    # Source gone; target aliases include Apollo Sales + ApolloSales.
    assert store._conn.execute(
        "SELECT 1 FROM entities WHERE entity_id = ?", (src,)
    ).fetchone() is None
    aliases = store._conn.execute(
        "SELECT aliases FROM entities WHERE entity_id = ?", (tgt,)
    ).fetchone()["aliases"]
    aliases_lower = {a.strip().lower() for a in aliases.split(",") if a.strip()}
    assert "apollo sales" in aliases_lower
    assert "apollosales" in aliases_lower


def test_merge_rejects_self_merge(store_and_retriever):
    store, _ = store_and_retriever
    store.add_fact("Walker Anderson is a person.", category="identity")
    eid = _entity_id_for(store, "Walker Anderson")
    with pytest.raises(ValueError):
        store.merge_entities(eid, eid)


def test_merge_rejects_missing_ids(store_and_retriever):
    store, _ = store_and_retriever
    store.add_fact("Walker Anderson is a person.", category="identity")
    eid = _entity_id_for(store, "Walker Anderson")
    with pytest.raises(KeyError):
        store.merge_entities(99999, eid)
    with pytest.raises(KeyError):
        store.merge_entities(eid, 99999)


def test_merge_re_encodes_facts_to_current_version(store_and_retriever):
    store, _ = store_and_retriever
    fa = store.add_fact("Apollo Energy Resources runs trucks.", category="general")
    src = _entity_id_for(store, "Apollo Energy Resources")
    store.add_fact("Apollo Energy Group runs trucks too.", category="general")
    tgt = _entity_id_for(store, "Apollo Energy Group")
    # Force the source-linked fact's encoding_version to 0.
    store._conn.execute(
        "UPDATE facts SET encoding_version = 0 WHERE fact_id = ?", (fa,)
    )
    store._conn.commit()

    store.merge_entities(src, tgt)

    row = store._conn.execute(
        "SELECT encoding_version FROM facts WHERE fact_id = ?", (fa,)
    ).fetchone()
    assert row["encoding_version"] == _CURRENT_ENCODING_VERSION


def test_merge_then_probe_old_name_still_resolves(store_and_retriever):
    """After merge, probing by source.name resolves via aliases → finds target's facts."""
    store, retriever = store_and_retriever
    fa = store.add_fact("Apollo Energy Resources is a logistics company.",
                        category="identity")
    src = _entity_id_for(store, "Apollo Energy Resources")
    store.add_fact("Apollo Energy Group is the canonical entity.",
                   category="identity")
    tgt = _entity_id_for(store, "Apollo Energy Group")

    store.merge_entities(src, tgt)

    # Probe by the old source name — alias resolution should map to target.
    results = retriever.probe(entity="Apollo Energy Resources", limit=5)
    by_id = {r["fact_id"]: r for r in results}
    assert fa in by_id
    sim = _raw_sim(by_id[fa]["score"], by_id[fa])
    assert sim > 0.30, f"merged-source-name probe raw_sim={sim:.4f}; expected > 0.30"


def test_merge_atomic_on_compute_failure(store_and_retriever, monkeypatch):
    """Simulated mid-merge encode failure: source row + links must remain."""
    store, _ = store_and_retriever
    fa = store.add_fact("Apollo Energy Resources runs trucks.", category="general")
    fb = store.add_fact("Apollo Energy Resources owns a fleet.", category="general")
    src = _entity_id_for(store, "Apollo Energy Resources")
    store.add_fact("Apollo Energy Group is canonical.", category="identity")
    tgt = _entity_id_for(store, "Apollo Energy Group")

    call_count = {"n": 0}
    real_encode = hrr.encode_fact

    def flaky(content, entities, dim):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated encode failure")
        return real_encode(content, entities, dim)

    monkeypatch.setattr(
        "plugins.memory.holographic.store.hrr.encode_fact", flaky
    )

    with pytest.raises(RuntimeError, match="simulated encode failure"):
        store.merge_entities(src, tgt)

    # Source row still present.
    row = store._conn.execute(
        "SELECT 1 FROM entities WHERE entity_id = ?", (src,)
    ).fetchone()
    assert row is not None
    # Source still linked to its facts.
    eids = {
        r["entity_id"] for r in store._conn.execute(
            "SELECT entity_id FROM fact_entities WHERE fact_id IN (?, ?)",
            (fa, fb),
        ).fetchall()
    }
    assert src in eids
