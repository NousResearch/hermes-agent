"""Entity-index recall through the same search path used by prefetch."""
import pytest

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    value = MemoryStore(str(tmp_path / "facts.db"))
    yield value
    value.close()


def link(store, fact_id, name, aliases=""):
    entity_id = store._resolve_entity(name)
    store._write("UPDATE entities SET aliases = ? WHERE entity_id = ?", (aliases, entity_id))
    store._write("INSERT OR IGNORE INTO fact_entities VALUES (?, ?)", (fact_id, entity_id))


@pytest.mark.parametrize("query", ["morning briefing", "DAILY DIGEST", "nightly summary"])
def test_prefetch_recovers_entity_names_and_aliases(store, query):
    content = "Rebuild indexes before opening the app."
    fact_id = store.add_fact(content)
    link(store, fact_id, "Morning Briefing", "daily digest,nightly summary")
    provider = HolographicMemoryProvider(config={"min_trust_threshold": 0.3})
    provider._store = store
    provider._retriever = FactRetriever(store)
    assert content in provider.prefetch(query)


def test_search_filters_and_deduplicates_before_limiting(store):
    lexical = store.add_fact("briefing keyword match", category="project")
    hidden = store.add_fact("private unrelated content", category="general")
    low = store.add_fact("untrusted content", category="project")
    target = store.add_fact("Rebuild indexes at dawn.", category="project")
    for fid in [lexical, hidden, low, target]:
        link(store, fid, "Morning Briefing")
        link(store, fid, "Briefing Routine")
    store.update_fact(low, trust_delta=-0.5)
    results = FactRetriever(store).search("briefing", category="project", limit=2)
    assert {r["fact_id"] for r in results} == {lexical, target}
    assert len(results) == 2


@pytest.mark.parametrize("query", ["", "the and of", "unknown topic"])
def test_empty_or_unmatched_query_does_not_recall_arbitrary_entities(store, query):
    fid = store.add_fact("Rebuild indexes at dawn.")
    link(store, fid, "The Morning Briefing")
    assert FactRetriever(store).search(query) == []


def test_full_lexical_result_is_preserved(store):
    fid = store.add_fact("briefing keyword match")
    other = store.add_fact("Rebuild indexes at dawn.")
    link(store, other, "Morning Briefing")
    assert [r["fact_id"] for r in FactRetriever(store).search("briefing", limit=1)] == [fid]


@pytest.mark.parametrize("query", ["morning briefing", "daily digest"])
@pytest.mark.parametrize("noise_count", [0, 4, 5, 12, 20])
def test_entity_recall_survives_lexical_noise(store, query, noise_count):
    content = "Rebuild indexes before opening the app."
    target = store.add_fact(content)
    link(store, target, "Morning Briefing", "daily digest,nightly summary")
    for index in range(noise_count):
        store.add_fact(f"{query.split()[0]} unrelated weather report number {index}")
    retriever = FactRetriever(store)
    assert target in {row["fact_id"] for row in retriever.search(query, limit=5)}
    provider = HolographicMemoryProvider(config={"min_trust_threshold": 0.3})
    provider._store, provider._retriever = store, retriever
    assert content in provider.prefetch(query)


@pytest.mark.parametrize("no_numpy", [False, True])
def test_entity_ranking_preserves_filters_and_lexical_weights(store, monkeypatch, no_numpy):
    from plugins.memory.holographic import holographic as hrr
    for index in range(8):
        fid = store.add_fact(f"unrelated note {index}", category="project")
        link(store, fid, f"Morning routine {index}")
    target = store.add_fact("Rebuild indexes at dawn.", category="project")
    link(store, target, "Morning Briefing", "daily digest,nightly summary")
    link(store, target, "Briefing Morning")
    hidden = store.add_fact("private instructions", category="general")
    link(store, hidden, "Morning Briefing")
    low = store.add_fact("untrusted instructions", category="project")
    link(store, low, "Morning Briefing")
    store.update_fact(low, trust_delta=-0.5)
    if no_numpy:
        monkeypatch.setattr(hrr, "_HAS_NUMPY", False)
    retriever = FactRetriever(store, hrr_weight=0)
    results = retriever.search("morning briefing", category="project", limit=1)
    assert [row["fact_id"] for row in results] == [target]
    assert "entity_rank" not in results[0]
    assert retriever.search("morning briefing", min_trust=0.9) == []
    # The bridge is not a new unconditional score: disabling lexical weights
    # disables its contribution too, including in the no-numpy configuration.
    disabled = FactRetriever(store, fts_weight=0, jaccard_weight=0, hrr_weight=0)
    assert all(row["score"] == 0 for row in disabled.search("morning briefing"))


def test_entity_index_failure_preserves_lexical_results(store):
    fid = store.add_fact("briefing keyword match")
    store._write("DROP TABLE entities")
    assert [r["fact_id"] for r in FactRetriever(store).search("briefing")] == [fid]


def test_caller_trust_and_no_numpy_are_respected(store, monkeypatch):
    from plugins.memory.holographic import holographic as hrr
    fid = store.add_fact("Rebuild indexes at dawn.")
    link(store, fid, "Morning Briefing")
    monkeypatch.setattr(hrr, "_HAS_NUMPY", False)
    retriever = FactRetriever(store)
    assert [r["fact_id"] for r in retriever.search("briefing", min_trust=0.4)] == [fid]
    assert retriever.search("briefing", min_trust=0.9) == []
