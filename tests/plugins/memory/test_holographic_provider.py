from plugins.memory.holographic import HolographicMemoryProvider


def test_prefetch_falls_back_to_entity_probe_for_sparse_keyword_results(tmp_path):
    provider = HolographicMemoryProvider({
        "db_path": str(tmp_path / "memory_store.db"),
        "hrr_dim": 64,
    })
    provider.initialize("test-session")
    provider._store.add_fact('"X99" route: transfer at Central, walk to Terminal')

    recall = provider.prefetch("What about my X99 route?")

    assert "## Holographic Memory" in recall
    assert "X99" in recall
    assert "transfer at Central" in recall


def test_extract_query_entities_keeps_codes_without_common_words():
    entities = HolographicMemoryProvider._extract_query_entities("What about my X99 route?")

    assert entities == ["X99"]
