from __future__ import annotations

import sqlite3

from vector_core.embeddings import EmbeddingCache as VectorCoreEmbeddingCache
from vector_core.embeddings import EmbeddingClient as VectorCoreEmbeddingClient

from hermes_wiki.config import WikiConfig
from hermes_wiki.search import EmbeddingCache, WikiSearch, _EmbeddingClient


class _FakeVectorCoreClient:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.closed = False

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), float(idx)] for idx, text in enumerate(texts)]

    async def close(self):
        self.closed = True


class _FakeBridge:
    def run(self, coro):
        return _run_for_test(coro)

    def close(self):
        return None


def _run_for_test(coro):
    import asyncio

    return asyncio.run(coro)


def _embedder(tmp_path, *, max_entries=100):
    config = WikiConfig(
        embedding_model="test-model",
        embedding_dim=2,
        embedding_cache_path=tmp_path / "embeddings.db",
        embedding_cache_max_entries=max_entries,
    )
    embedder = _EmbeddingClient.__new__(_EmbeddingClient)
    embedder._client = _FakeVectorCoreClient()
    embedder._bridge = _FakeBridge()
    embedder._model = config.embedding_model
    embedder._dim = config.embedding_dim
    embedder._cache = EmbeddingCache(config.embedding_cache_path, max_entries=config.embedding_cache_max_entries)
    return embedder


def test_embedding_cache_wraps_vector_core_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db", max_entries=10)

    assert isinstance(cache._vector_cache, VectorCoreEmbeddingCache)
    assert cache.cache_path == tmp_path / "cache.db"


def test_embedding_client_instantiates_vector_core_embedding_client(tmp_path):
    config = WikiConfig(
        embedding_url="http://embedding.test/v1",
        embedding_model="test-model",
        embedding_dim=2,
        embedding_cache_path=tmp_path / "embeddings.db",
    )

    embedder = _EmbeddingClient(config)

    assert isinstance(embedder._client, VectorCoreEmbeddingClient)
    assert embedder._client.base_url == "http://embedding.test"
    assert embedder._client.model == "test-model"
    assert embedder._client.dim == 2


def test_embedding_cache_round_trips_json_vectors_and_tracks_stats(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db", max_entries=10)
    key = cache.key_for("hello", model="model-a", dim=2)

    assert cache.get(key) is None
    cache.set(key, [1.0, 2.5], model="model-a", dim=2)

    assert cache.get(key) == [1.0, 2.5]
    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_embedding_cache_keys_include_model_and_dimension(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")

    text = "same text"
    assert cache.key_for(text, model="model-a", dim=2) != cache.key_for(text, model="model-b", dim=2)
    assert cache.key_for(text, model="model-a", dim=2) != cache.key_for(text, model="model-a", dim=4)


def test_embedding_cache_deletes_corrupt_rows_as_misses(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    key = cache.key_for("bad", model="model-a", dim=2)
    with sqlite3.connect(cache.cache_path) as conn:
        conn.execute(
            "INSERT INTO embeddings (content_hash, embedding, model, dim, created_at, accessed_at) VALUES (?, ?, ?, ?, '', '')",
            (key, b"not-json", "model-a", 2),
        )

    assert cache.get(key) is None
    with sqlite3.connect(cache.cache_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE content_hash = ?", (key,)).fetchone()[0]
    assert count == 0


def test_embedding_cache_lru_eviction_respects_max_entries(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db", max_entries=2)
    for text in ["one", "two", "three"]:
        key = cache.key_for(text, model="model-a", dim=2)
        cache.set(key, [1.0, 2.0], model="model-a", dim=2)

    assert cache.stats()["entries"] == 2
    assert cache.get(cache.key_for("three", model="model-a", dim=2)) == [1.0, 2.0]


def test_embedding_client_uses_persistent_cache_for_duplicate_texts(tmp_path):
    embedder = _embedder(tmp_path)

    assert embedder.embed_batch(["alpha", "alpha", "beta"]) == [[5.0, 0.0], [5.0, 0.0], [4.0, 1.0]]
    assert embedder._client.calls == [["alpha", "beta"]]

    assert embedder.embed_batch(["alpha", "beta"]) == [[5.0, 0.0], [4.0, 1.0]]
    assert embedder._client.calls == [["alpha", "beta"]]


def test_embedding_client_cache_survives_new_embedder_instance(tmp_path):
    first = _embedder(tmp_path)
    assert first.embed_single("alpha") == [5.0, 0.0]
    assert first._client.calls == [["alpha"]]

    second = _embedder(tmp_path)
    assert second.embed_single("alpha") == [5.0, 0.0]
    assert second._client.calls == []


def test_embedding_client_can_disable_cache_for_read_only_contexts(tmp_path):
    config = WikiConfig(
        embedding_model="test-model",
        embedding_dim=2,
        embedding_cache_path=tmp_path / "embeddings.db",
    )
    embedder = _EmbeddingClient.__new__(_EmbeddingClient)
    embedder._client = _FakeVectorCoreClient()
    embedder._bridge = _FakeBridge()
    embedder._model = config.embedding_model
    embedder._dim = config.embedding_dim
    embedder._cache = None

    assert embedder.embed_batch(["alpha", "alpha"]) == [[5.0, 0.0], [5.0, 0.0]]
    assert embedder._client.calls == [["alpha"]]
    assert not config.embedding_cache_path.exists()


def test_wiki_search_read_only_disables_embedding_cache(monkeypatch, tmp_path):
    captured = {}

    class FakeQdrantClient:
        def __init__(self, url):
            self.url = url

    class FakeEmbeddingClient:
        def __init__(self, config, *, cache_enabled=True):
            captured["cache_enabled"] = cache_enabled

    monkeypatch.setattr("hermes_wiki.search._EmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("qdrant_client.QdrantClient", FakeQdrantClient)

    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    WikiSearch(config, ensure_collection=False, read_only=True)

    assert captured["cache_enabled"] is False
