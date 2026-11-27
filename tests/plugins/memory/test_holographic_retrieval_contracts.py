"""Real-store retrieval contracts and coverage regression for #36488."""

import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic import retrieval
from plugins.memory.holographic.store import MemoryStore


@pytest.mark.parametrize("min_trust", [0.0, 0.3, 0.8])
@pytest.mark.parametrize("half_life", [0, 30])
def test_hybrid_recall_filters_ranks_and_decays_persisted_facts(
    tmp_path, monkeypatch, min_trust, half_life
):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(retrieval, "datetime", Clock)
    dates = ["2026-09-05", "2026-10-05", "2026-10-06", "not-a-date", None]
    trusts = [0.9, 0.6, 0.7, 0.4, 0.2]
    with MemoryStore(tmp_path / "recall.db", hrr_dim=64) as store:
        ids = []
        for i, (date, trust) in enumerate(zip(dates, trusts)):
            fid = store.add_fact(f"deployment plan {i}", category="work", tags="release")
            store._conn.execute(
                "UPDATE facts SET trust_score=?, updated_at=?, created_at=? WHERE fact_id=?",
                (trust, date, date, fid),
            )
            ids.append(fid)
        outside = store.add_fact("deployment outside this project", category="personal")
        retriever = retrieval.FactRetriever(store, hrr_dim=64, temporal_decay_half_life=half_life)
        results = retriever.search("deployment", category="work", min_trust=min_trust)
        base = retrieval.FactRetriever(store, hrr_dim=64).search(
            "deployment", category="work", min_trust=min_trust
        )
        base_scores = {row["fact_id"]: row["score"] for row in base}
        assert {row["fact_id"] for row in results} == {
            fid for fid, trust in zip(ids, trusts) if trust >= min_trust
        }
        assert outside not in {row["fact_id"] for row in results}
        assert [row["score"] for row in results] == sorted(
            (row["score"] for row in results), reverse=True
        )
        for row in results:
            factor = 0.5 if half_life and row["fact_id"] == ids[0] else 1.0
            assert row["score"] == pytest.approx(base_scores[row["fact_id"]] * factor)
            assert 0 <= row["score"] <= row["trust_score"]
            assert "hrr_vector" not in row
        assert retriever.search("deployment", category="work", min_trust=min_trust, limit=1) == results[:1]
        assert retriever.search('"', category="work") == []
        assert retriever.search("", category="work") == []
        assert retrieval.FactRetriever(store, hrr_weight=0).search("release", category="work")
        json.dumps(results)
        if hrr._HAS_NUMPY:
            assert store._conn.execute("SELECT hrr_vector FROM facts WHERE fact_id=?", (ids[0],)).fetchone()[0]


@pytest.mark.parametrize("strategy", ["probe", "related", "reason", "contradict"])
@pytest.mark.parametrize("mode", ["vectors", "legacy", "keyword"])
def test_compositional_recall_preserves_scope_and_keyword_fallback(
    tmp_path, monkeypatch, strategy, mode
):
    if mode == "keyword":
        monkeypatch.setattr(hrr, "_HAS_NUMPY", False)
    with MemoryStore(tmp_path / "compositional.db", hrr_dim=64) as store:
        retriever = retrieval.FactRetriever(store, hrr_dim=64)
        args = {
            "probe": ("Orion",), "related": ("Orion",),
            "reason": (["Orion", "Vega"],), "contradict": (),
        }[strategy]
        kwargs = {"threshold": 0.0} if strategy == "contradict" else {}
        query = getattr(retriever, strategy)
        assert query(*args, category="work", **kwargs) == []
        selected = {
            store.add_fact('"Orion" and "Vega" deploy a service', category="work"),
            store.add_fact('"Orion" and "Vega" cancel the launch', category="work"),
            store.add_fact('"Other" and "Altair" monitor operations', category="work"),
            store.add_fact("plain deployment checklist", category="work"),
        }
        outside = store.add_fact('"Orion" and "Vega" personal appointment', category="personal")
        if mode == "legacy":
            store._conn.execute("UPDATE facts SET hrr_vector=NULL")
            store._conn.execute("DELETE FROM memory_banks")
        results = query(*args, category="work", limit=10, **kwargs)
        if strategy == "contradict":
            assert bool(results) == (mode == "vectors")
            for pair in results:
                assert {pair["fact_a"]["fact_id"], pair["fact_b"]["fact_id"]} <= selected
                assert pair["shared_entities"] == ["orion", "vega"]
                assert "hrr_vector" not in pair["fact_a"]
                assert "hrr_vector" not in pair["fact_b"]
            scores = [pair["contradiction_score"] for pair in results]
        else:
            assert results
            assert {row["fact_id"] for row in results} <= selected
            assert all(0 <= row["score"] <= row["trust_score"] for row in results)
            assert all("hrr_vector" not in row for row in results)
            scores = [row["score"] for row in results]
            assert query(*args, category="work", limit=1, **kwargs) == results[:1]
            assert outside in {row["fact_id"] for row in query(*args, limit=10, **kwargs)}
        assert scores == sorted(scores, reverse=True)
        assert retriever.reason([], category="work") == []
        json.dumps(results)
