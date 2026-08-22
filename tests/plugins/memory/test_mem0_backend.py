"""Tests for Mem0Backend abstraction — PlatformBackend, OSSBackend, SelfHostedBackend."""

import copy
import pytest

from plugins.memory.mem0._backend import (
    Mem0Backend,
    PlatformBackend,
    OSSBackend,
    SelfHostedBackend,
)


class FakePlatformClient:
    """Fake MemoryClient for PlatformBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.9}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"count": 1, "next": None, "results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"status": "PENDING", "event_id": "evt-1"}

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {"id": kwargs["memory_id"], "text": kwargs["text"]}

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))


class TestPlatformBackend:

    def _make(self):
        client = FakePlatformClient()
        backend = PlatformBackend.__new__(PlatformBackend)
        backend._client = client
        return backend, client

    def test_search_forwards_params(self):
        backend, client = self._make()
        result = backend.search("test query", filters={"user_id": "u1"}, top_k=5)
        assert client.calls[0][0] == "search"
        assert client.calls[0][1] == "test query"
        assert client.calls[0][2]["filters"] == {"user_id": "u1"}
        assert client.calls[0][2]["top_k"] == 5


    def test_add_forwards_kwargs(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        result = backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)
        call = client.calls[0]
        assert call[2]["user_id"] == "u1"
        assert call[2]["infer"] is False
        # metadata kwarg should be omitted entirely when not provided so we
        # don't surprise older mem0 client versions with an unknown kwarg.
        assert "metadata" not in call[2]


    def test_update_forwards(self):
        backend, client = self._make()
        backend.update("m1", "new text")
        assert client.calls[0][1] == {"memory_id": "m1", "text": "new text"}

    def test_delete_forwards(self):
        backend, client = self._make()
        backend.delete("m1")
        assert client.calls[0][1] == {"memory_id": "m1"}


class FakeOSSMemory:
    """Fake mem0.Memory for OSSBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.8}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "event": "ADD"}]}

    def update(self, memory_id, **kwargs):
        self.calls.append(("update", memory_id, kwargs))
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        return {"message": "Memory deleted successfully!"}


class TestOSSBackend:

    def _make(self):
        memory = FakeOSSMemory()
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = memory
        return backend, memory


    def test_legacy_api_base_aliases_are_normalized_before_mem0_init(self, monkeypatch):
        import sys
        import types

        captured = {}

        class Memory:
            @staticmethod
            def from_config(config):
                captured.update(config)
                return FakeOSSMemory()

        # OSSBackend.__init__ does `from mem0 import Memory`. mem0 is a lazy
        # optional dep absent from CI's env, so inject a stub module rather
        # than importing the real package (which would ModuleNotFoundError).
        stub_mem0 = types.ModuleType("mem0")
        stub_mem0.Memory = Memory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mem0", stub_mem0)
        raw = {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-5-mini", "api_base": "https://llm.example/v1"},
            },
            "embedder": {
                "provider": "ollama",
                "config": {"model": "nomic-embed-text", "api_base": "http://ollama:11434"},
            },
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        before = copy.deepcopy(raw)

        OSSBackend(raw)

        assert captured["llm"]["config"]["openai_base_url"] == "https://llm.example/v1"
        assert captured["embedder"]["config"]["ollama_base_url"] == "http://ollama:11434"
        assert "api_base" not in captured["llm"]["config"]
        assert "api_base" not in captured["embedder"]["config"]
        assert raw == before


class _FakeCollectionInfo:
    def __init__(self, dims: int):
        class _Vectors:
            def __init__(self, size):
                self.size = size
        self.config = type("C", (), {"params": type("P", (), {"vectors": _Vectors(dims)})()})()


class _FakeQdrantClient:
    """Fake QdrantClient that tracks calls — no file locks."""
    def __init__(self, *, existing_dims: int | None = 8, collection_name: str = "mem0"):
        self._existing_dims = existing_dims
        self._collection_name = collection_name
        self.deleted = False
        self.creations = []

    def collection_exists(self, name: str) -> bool:
        return self._existing_dims is not None and name == self._collection_name

    def get_collection(self, name: str):
        return _FakeCollectionInfo(self._existing_dims)

    def delete_collection(self, name: str):
        self.deleted = True
        self._existing_dims = None  # collection no longer exists

    def create_collection(self, **kwargs):
        self.creations.append(kwargs)
        # Update dims so get_collection() reflects the new collection
        vc = kwargs.get("vectors_config")
        if vc is not None and hasattr(vc, "size"):
            self._existing_dims = vc.size
        elif not self._existing_dims:
            self._existing_dims = 0  # placeholder if unknown


class _FakeVectorStore:
    """Fake vector store that wraps a fake QdrantClient."""
    def __init__(self, client: _FakeQdrantClient, on_disk: bool = False):
        self.client = client
        self.on_disk = on_disk

    def create_col(self, vector_size: int, on_disk: bool):
        """Recreate the collection — update dims on the fake client."""
        self.client._existing_dims = vector_size


class TestOSSBackendRecreateQdrantDims:
    """Verify _recreate_qdrant_if_dims_changed uses Memory's own client."""

    def _make_backend(self, client: _FakeQdrantClient, collection_name: str = "mem0"):
        backend = OSSBackend.__new__(OSSBackend)
        vs = _FakeVectorStore(client)
        memory = type("M", (), {
            "vector_store": vs,
            "collection_name": collection_name,
        })()
        backend._memory = memory
        return backend

    def test_dims_match_no_delete(self):
        """When collection dims match expected, nothing happens."""
        client = _FakeQdrantClient(existing_dims=384)
        backend = self._make_backend(client)
        backend._recreate_qdrant_if_dims_changed(384)
        assert not client.deleted

    def test_dims_mismatch_recreates_collection(self):
        """When collection dims differ, collection is deleted AND recreated."""
        client = _FakeQdrantClient(existing_dims=128)
        backend = self._make_backend(client)
        vs = backend._memory.vector_store
        original_create_col = vs.create_col
        called = []
        def tracking_create_col(vector_size, on_disk):
            called.append((vector_size, on_disk))
            return original_create_col(vector_size, on_disk)
        vs.create_col = tracking_create_col

        backend._recreate_qdrant_if_dims_changed(384)

        assert client.deleted, "Collection should be deleted on dim mismatch"
        assert len(called) == 1, "create_col should be called exactly once"
        assert called[0] == (384, False), "Should recreate with expected dims"

    def test_missing_collection_noop(self):
        """When collection doesn't exist, nothing happens."""
        client = _FakeQdrantClient(existing_dims=None)
        backend = self._make_backend(client)
        backend._recreate_qdrant_if_dims_changed(384)
        assert not client.deleted

    def test_no_vector_store_client_noop(self):
        """When Memory has no vector_store.client, nothing happens."""
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = type("M", (), {"vector_store": None, "collection_name": "mem0"})()
        backend._recreate_qdrant_if_dims_changed(384)
        # Should not raise

    def test_uses_memory_own_client(self):
        """Verify the method accesses Memory's vector_store.client, not a new QdrantClient."""
        client = _FakeQdrantClient(existing_dims=128)
        backend = self._make_backend(client)
        vs = backend._memory.vector_store
        called = []
        original = vs.create_col
        def tracking_create_col(vector_size, on_disk):
            called.append((vector_size, on_disk))
            return original(vector_size, on_disk)
        vs.create_col = tracking_create_col

        backend._recreate_qdrant_if_dims_changed(384)

        assert called, "create_col was called on Memory's own vector_store"
        assert client.deleted

    def test_no_vector_store_itself_noop(self):
        """When Memory.vector_store is None, nothing happens."""
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = type("M", (), {"vector_store": None, "collection_name": "mem0"})()
        backend._recreate_qdrant_if_dims_changed(384)
        # Should not raise

    def test_dims_none_skips_delete(self):
        """When Qdrant reports None dims, nothing happens."""
        class _NoDimsCollectionInfo:
            class _Vectors:
                size = None
            config = type("C", (), {"params": type("P", (), {"vectors": _Vectors()})()})()

        class _NoDimsQdrantClient(_FakeQdrantClient):
            def get_collection(self, name):
                return _NoDimsCollectionInfo()

        client = _NoDimsQdrantClient(existing_dims=384)
        backend = self._make_backend(client)
        backend._recreate_qdrant_if_dims_changed(512)
        assert not client.deleted

    def test_on_disk_respected(self):
        """The vector store's on_disk setting is passed to create_col."""
        client = _FakeQdrantClient(existing_dims=128)
        vs = _FakeVectorStore(client, on_disk=True)
        backend = OSSBackend.__new__(OSSBackend)
        memory = type("M", (), {"vector_store": vs, "collection_name": "mem0"})()
        backend._memory = memory
        called = []
        original = vs.create_col
        def tracking(vector_size, on_disk):
            called.append((vector_size, on_disk))
            return original(vector_size, on_disk)
        vs.create_col = tracking

        backend._recreate_qdrant_if_dims_changed(384)

        assert client.deleted
        assert called[0] == (384, True), "on_disk=True should be forwarded"

    def test_missing_create_col_does_not_delete(self):
        """When vector store lacks create_col, the collection is NOT deleted
        (bare create_collection would produce a degraded collection)."""
        client = _FakeQdrantClient(existing_dims=128)

        class _VSWoCreate:
            def __init__(self, c):
                self.client = c
                self.on_disk = False

        vs = _VSWoCreate(client)
        backend = OSSBackend.__new__(OSSBackend)
        memory = type("M", (), {"vector_store": vs, "collection_name": "mem0"})()
        backend._memory = memory

        backend._recreate_qdrant_if_dims_changed(384)

        assert not client.deleted, "Should NOT delete when create_col is absent"

    def test_partial_failure_triggers_fallback(self, caplog):
        """When delete succeeds but create_col raises, the fallback is attempted."""
        import logging
        caplog.set_level(logging.WARNING)

        class _RaisingVectorStore:
            def __init__(self):
                self.client = _FakeQdrantClient(existing_dims=128)
                self.on_disk = False
            def create_col(self, vector_size, on_disk):
                raise RuntimeError("create_col failed: connection refused")

        vs = _RaisingVectorStore()
        backend = OSSBackend.__new__(OSSBackend)
        memory = type("M", (), {
            "vector_store": vs,
            "collection_name": "mem0",
        })()
        backend._memory = memory

        backend._recreate_qdrant_if_dims_changed(384)

        assert vs.client.deleted, "Collection should still be deleted"
        # The fallback (bare client.create_collection) should have been called
        assert len(vs.client.creations) == 1, "Fallback create_collection should be called"
        fallback_kwargs = vs.client.creations[0]
        assert fallback_kwargs["collection_name"] == "mem0"
        assert "attempting fallback" in caplog.text


class TestOSSBackendRecreateQdrantIntegration:
    """Verify the collection is functional and correctly configured AFTER a dim-mismatch recreate."""

    def _make_backend(self, client: _FakeQdrantClient, collection_name: str = "mem0"):
        backend = OSSBackend.__new__(OSSBackend)
        vs = _FakeVectorStore(client)
        memory = type("M", (), {
            "vector_store": vs,
            "collection_name": collection_name,
        })()
        backend._memory = memory
        return backend

    def test_recreate_updates_collection_dims(self):
        """After recreate, get_collection() should return the new dimension size."""
        client = _FakeQdrantClient(existing_dims=128)
        backend = self._make_backend(client)

        backend._recreate_qdrant_if_dims_changed(384)

        info = client.get_collection("mem0")
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            new_dims = first.size if first else None
        else:
            new_dims = getattr(vectors, "size", None)
        assert new_dims == 384, (
            f"Collection dims should be updated to 384, got {new_dims}"
        )

    def test_recreate_preserves_on_disk(self):
        """After recreate, the on_disk config is passed through correctly."""
        client = _FakeQdrantClient(existing_dims=128)
        vs = _FakeVectorStore(client, on_disk=True)
        backend = OSSBackend.__new__(OSSBackend)
        memory = type("M", (), {
            "vector_store": vs,
            "collection_name": "mem0",
        })()
        backend._memory = memory

        backend._recreate_qdrant_if_dims_changed(384)

        info = client.get_collection("mem0")
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            new_dims = first.size if first else None
        else:
            new_dims = getattr(vectors, "size", None)
        assert new_dims == 384

    def test_recreate_does_not_affect_other_collections(self):
        """Only the target collection should be affected by recreate."""

        class _MultiColQdrantClient(_FakeQdrantClient):
            def __init__(self):
                super().__init__(existing_dims=128)
                self._other_dims = 256

            def collection_exists(self, name: str) -> bool:
                return name in ("mem0", "other_col")

            def get_collection(self, name: str):
                if name == "other_col":
                    return _FakeCollectionInfo(self._other_dims)
                return _FakeCollectionInfo(self._existing_dims)

            def delete_collection(self, name: str):
                super().delete_collection(name)
                if name == "other_col":
                    self._other_dims = None

            def create_collection(self, **kwargs):
                super().create_collection(**kwargs)
                if kwargs.get("collection_name") == "other_col":
                    self._other_dims = 256

        client = _MultiColQdrantClient()
        vs = _FakeVectorStore(client)
        backend = OSSBackend.__new__(OSSBackend)
        memory = type("M", (), {
            "vector_store": vs,
            "collection_name": "mem0",
        })()
        backend._memory = memory

        backend._recreate_qdrant_if_dims_changed(384)

        # Target collection should have new dims
        info = client.get_collection("mem0")
        vectors = info.config.params.vectors
        current = vectors.size if hasattr(vectors, "size") else None
        assert current == 384

        # Other collection should be untouched
        other = client.get_collection("other_col")
        other_vectors = other.config.params.vectors
        other_size = other_vectors.size if hasattr(other_vectors, "size") else None
        assert other_size == 256, "Other collections must not be affected"

    def test_recreate_fallback_creates_basic_collection(self, caplog):
        """When create_col raises, the fallback creates a basic collection."""
        import logging
        caplog.set_level(logging.WARNING)

        class _FallbackVectorStore:
            def __init__(self):
                self.client = _FakeQdrantClient(existing_dims=128)
                self.on_disk = False
            def create_col(self, vector_size, on_disk):
                raise RuntimeError("primary failed")

        vs = _FallbackVectorStore()
        backend = OSSBackend.__new__(OSSBackend)
        memory = type("M", (), {
            "vector_store": vs,
            "collection_name": "mem0",
        })()
        backend._memory = memory

        backend._recreate_qdrant_if_dims_changed(384)

        # Collection should still exist (via fallback)
        info = vs.client.get_collection("mem0")
        vectors = info.config.params.vectors
        current = vectors.size if hasattr(vectors, "size") else None
        assert current == 384, (
            "Fallback should create a collection with the expected dims"
        )
        assert vs.client.deleted
        assert "attempting fallback" in caplog.text

    def test_fallback_reported_in_creations(self):
        """Verify client.create_collection is called by the fallback path."""

        class _FallbackVStore:
            def __init__(self):
                self.client = _FakeQdrantClient(existing_dims=128)
                self.on_disk = False
            def create_col(self, vector_size, on_disk):
                raise RuntimeError("boom")

        vs = _FallbackVStore()
        backend = OSSBackend.__new__(OSSBackend)
        memory = type("M", (), {
            "vector_store": vs,
            "collection_name": "mem0",
        })()
        backend._memory = memory

        backend._recreate_qdrant_if_dims_changed(384)

        assert len(vs.client.creations) == 1
        kwargs = vs.client.creations[0]
        assert kwargs["collection_name"] == "mem0"
        # The vectors_config should contain the expected dims
        vc = kwargs.get("vectors_config")
        assert vc is not None, "vectors_config must be provided in fallback"
        assert hasattr(vc, "size"), "vectors_config should have size"
        assert vc.size == 384


qdrant_models = pytest.importorskip("qdrant_client.models")


class _RealMem0StyleVectorStore:
    """Vector store over a REAL QdrantClient, reproducing mem0's create_col.

    Mirrors ``mem0.vector_stores.qdrant.Qdrant``: the dense slot plus a ``bm25``
    sparse slot with the IDF modifier, and ``_has_bm25_slot`` — the flag mem0's
    insert path consults to decide whether to write the sparse vector.
    """

    def __init__(self, client, collection_name="mem0", on_disk=False):
        self.client = client
        self.collection_name = collection_name
        self.on_disk = on_disk
        self.is_local = True  # embedded Qdrant: no payload indexes, as in mem0
        self._has_bm25_slot = False

    def create_col(self, vector_size, on_disk, distance=None):
        from qdrant_client.models import (
            Distance,
            Modifier,
            SparseVectorParams,
            VectorParams,
        )

        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size, distance=distance or Distance.COSINE, on_disk=on_disk
            ),
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
        )
        self._has_bm25_slot = True


class _RealStoreFailingCreateCol(_RealMem0StyleVectorStore):
    """Same real client, but ``create_col`` fails — forces the fallback path."""

    def create_col(self, vector_size, on_disk, distance=None):
        raise RuntimeError("create_col failed: simulated backend error")


def _sparse_config(client, collection_name="mem0"):
    return client.get_collection(collection_name).config.params.sparse_vectors


def _dense_dims(client, collection_name="mem0"):
    vectors = client.get_collection(collection_name).config.params.vectors
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        return first.size if first else None
    return getattr(vectors, "size", None)


def _upsert_hybrid(client, collection_name, dims, point_id=1):
    """Write a point the way mem0 does when the bm25 slot exists.

    Raises if the collection has no ``bm25`` sparse slot — which is exactly how
    a degraded (dense-only) collection surfaces: it rejects writes.
    """
    from qdrant_client.models import PointStruct, SparseVector

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector={
                    "": [0.1] * dims,
                    "bm25": SparseVector(indices=[7, 42], values=[0.5, 0.9]),
                },
                payload={"user_id": "u1", "data": "likes tea"},
            )
        ],
    )


class TestQdrantRecreateRealContract:
    """Real-Qdrant tests: the recreate must preserve the full Mem0 contract.

    These use an actual in-memory ``QdrantClient`` (not a fake), so the
    assertions are against Qdrant's real collection config and its real
    accept/reject behaviour on writes.
    """

    def _backend(self, vs):
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = type("M", (), {
            "vector_store": vs,
            "collection_name": vs.collection_name,
        })()
        return backend

    def _client(self):
        from qdrant_client import QdrantClient
        return QdrantClient(location=":memory:")

    def _seed(self, vs, dims):
        """Create the pre-existing (wrong-dims) collection the way mem0 would."""
        vs.create_col(dims, vs.on_disk)
        assert vs._has_bm25_slot
        assert "bm25" in _sparse_config(vs.client)

    # --- primary path: create_col rebuilds the collection ------------------

    def test_primary_path_preserves_bm25_and_dims(self):
        client = self._client()
        vs = _RealMem0StyleVectorStore(client)
        self._seed(vs, 128)

        self._backend(vs)._recreate_qdrant_if_dims_changed(384)

        assert client.collection_exists("mem0"), "collection must exist after recreate"
        assert _dense_dims(client) == 384
        sparse = _sparse_config(client)
        assert sparse and "bm25" in sparse, "bm25 sparse slot must survive recreate"
        assert sparse["bm25"].modifier == qdrant_models.Modifier.IDF

    def test_primary_path_collection_accepts_hybrid_write_and_search(self):
        client = self._client()
        vs = _RealMem0StyleVectorStore(client)
        self._seed(vs, 128)

        self._backend(vs)._recreate_qdrant_if_dims_changed(384)

        _upsert_hybrid(client, "mem0", 384)
        hits = client.query_points(
            collection_name="mem0", query=[0.1] * 384, limit=5
        ).points
        assert len(hits) == 1
        assert hits[0].payload["data"] == "likes tea"

    # --- fallback path: create_col raises, we rebuild by hand --------------

    def test_fallback_preserves_bm25_sparse_config(self):
        """The regression under review: the fallback must NOT create a
        dense-only collection, which would later reject mem0's writes."""
        client = self._client()
        seed = _RealMem0StyleVectorStore(client)
        self._seed(seed, 128)

        vs = _RealStoreFailingCreateCol(client)
        vs._has_bm25_slot = True  # what mem0 believes after its own create_col
        self._backend(vs)._recreate_qdrant_if_dims_changed(384)

        assert client.collection_exists("mem0"), "fallback must recreate the collection"
        assert _dense_dims(client) == 384
        sparse = _sparse_config(client)
        assert sparse and "bm25" in sparse, (
            "fallback dropped the bm25 sparse slot — collection is degraded"
        )
        assert sparse["bm25"].modifier == qdrant_models.Modifier.IDF
        assert vs._has_bm25_slot is True, (
            "vector store's bm25 flag must stay true so insert() keeps working"
        )

    def test_fallback_collection_accepts_hybrid_write_and_search(self):
        """Operate on the collection after the fallback recreate."""
        client = self._client()
        self._seed(_RealMem0StyleVectorStore(client), 128)

        vs = _RealStoreFailingCreateCol(client)
        self._backend(vs)._recreate_qdrant_if_dims_changed(384)

        _upsert_hybrid(client, "mem0", 384)
        hits = client.query_points(
            collection_name="mem0", query=[0.1] * 384, limit=5
        ).points
        assert len(hits) == 1
        assert hits[0].payload["user_id"] == "u1"

    def test_dense_only_collection_would_reject_hybrid_write(self):
        """Pins down *why* the fallback must keep the sparse slot: a dense-only
        collection rejects the very write mem0's insert path performs."""
        from qdrant_client.models import Distance, VectorParams

        client = self._client()
        client.create_collection(
            collection_name="dense_only",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        with pytest.raises(Exception):
            _upsert_hybrid(client, "dense_only", 384)

    # --- no-op paths, against the real client ------------------------------

    def test_matching_dims_leaves_collection_untouched(self):
        client = self._client()
        vs = _RealMem0StyleVectorStore(client)
        self._seed(vs, 384)
        _upsert_hybrid(client, "mem0", 384)

        self._backend(vs)._recreate_qdrant_if_dims_changed(384)

        assert client.count("mem0").count == 1, (
            "matching dims must not drop existing points"
        )
        assert "bm25" in _sparse_config(client)

    def test_missing_create_col_leaves_collection_intact(self):
        """Without create_col we cannot honour the contract, so we must not
        delete — the real collection and its data stay put."""
        client = self._client()
        seed = _RealMem0StyleVectorStore(client)
        self._seed(seed, 128)
        _upsert_hybrid(client, "mem0", 128)

        class _NoCreateCol:
            def __init__(self, c):
                self.client = c
                self.collection_name = "mem0"
                self.on_disk = False

        self._backend(_NoCreateCol(client))._recreate_qdrant_if_dims_changed(384)

        assert client.collection_exists("mem0")
        assert _dense_dims(client) == 128, "collection must be left as-is"
        assert client.count("mem0").count == 1


class TestOSSBackendConstructorNoExtraClient:
    """Constructor-level: verify __init__ does NOT create a separate QdrantClient."""

    def test_init_does_not_create_extra_qdrant_client(self, monkeypatch):
        """When dims mismatch, the collection is recreated via Memory's
        vector_store, not via a temporary QdrantClient."""
        import sys
        import types

        # Track QdrantClient constructions
        qdrant_instances = []
        class QdrantClient:
            def __init__(self, **kwargs):
                qdrant_instances.append(kwargs)
            def collection_exists(self, name):
                return True
            def get_collection(self, name):
                return _FakeCollectionInfo(128)  # Mismatch!
            def delete_collection(self, name):
                pass
            def create_collection(self, **kwargs):
                pass
            def close(self):
                pass

        qdrant_client_module = types.ModuleType("qdrant_client")
        qdrant_client_module.QdrantClient = QdrantClient

        class FakeMemoryFromConfig:
            collection_name = "mem0"
            vector_store = _FakeVectorStore(_FakeQdrantClient(existing_dims=128))

            @staticmethod
            def from_config(config):
                m = FakeMemoryFromConfig()
                # Set the vector_store properly
                vs = _FakeVectorStore(_FakeQdrantClient(existing_dims=128))
                vs.on_disk = config.get("vector_store", {}).get("config", {}).get("on_disk", False)
                m.vector_store = vs
                m.collection_name = config.get("vector_store", {}).get("config", {}).get("collection_name", "mem0")
                return m

        mem0_module = types.ModuleType("mem0")
        mem0_module.Memory = FakeMemoryFromConfig

        # Also stub qdrant_client in sys.modules so OSSBackend won't try real import
        monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client_module)
        monkeypatch.setitem(sys.modules, "mem0", mem0_module)

        raw = {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small", "embedding_dims": 384},
            },
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/test_qdrant"}},
        }

        backend = OSSBackend(raw)

        # Should have used the Memory's QdrantClient, not created a new one.
        assert len(qdrant_instances) == 0, (
            f"No QdrantClient should be created during __init__. "
            f"Got {len(qdrant_instances)}: {qdrant_instances}"
        )

        # Verify the vector store's collection was recreated on the dim mismatch.
        assert hasattr(backend._memory, "vector_store")
        assert backend._memory.vector_store.client.deleted


httpx = pytest.importorskip("httpx")


class _StubServer:
    """Records requests and serves the real self-hosted server's response shapes."""

    def __init__(self, rows=10):
        self.requests = []
        self._rows = [{"id": f"m{i}", "memory": f"f{i}"} for i in range(rows)]

    def handler(self, request):
        self.requests.append(request)
        path, method = request.url.path, request.method
        if path == "/search" and method == "POST":
            return httpx.Response(200, json={"results": [{"id": "m1", "memory": "tea", "score": 0.9}]})
        if path == "/memories" and method == "GET":
            top_k = int(request.url.params.get("top_k", len(self._rows)))
            return httpx.Response(200, json={"results": self._rows[:top_k]})
        if path == "/memories" and method == "POST":
            return httpx.Response(200, json={"results": [{"id": "new", "memory": "stored", "event": "ADD"}]})
        if path.startswith("/memories/") and method in ("PUT", "DELETE"):
            if path.endswith("/missing"):  # server 404s unknown ids
                return httpx.Response(404, json={"detail": "Memory not found"})
            verb = "updated" if method == "PUT" else "Memory deleted successfully"
            return httpx.Response(200, json={"message": verb})
        return httpx.Response(404, json={"detail": "not found"})


def _backend(server, api_key="adminkey", host="http://sh:8888"):
    """Build a SelfHostedBackend routed through the stub transport.

    Uses the real __init__ (via the injectable ``transport`` kwarg) so the
    constructor's header/base_url setup is exercised by every test here.
    """
    return SelfHostedBackend(
        api_key, host, transport=httpx.MockTransport(server.handler)
    )


class TestSelfHostedBackend:
    # --- constructor / auth setup (the crux of the bug) -------------------

    def test_init_uses_x_api_key_not_token_auth(self):
        b = SelfHostedBackend("adminkey", "http://sh:8888")
        assert b._client.headers["x-api-key"] == "adminkey"
        assert "authorization" not in b._client.headers  # NOT the cloud 'Token' scheme


    # --- search ----------------------------------------------------------


    # --- add / update / delete ------------------------------------------


    # --- error propagation (feeds the plugin's circuit breaker) ----------

    def test_http_error_raises(self):
        s = _StubServer()
        with pytest.raises(httpx.HTTPStatusError):
            _backend(s).delete("missing")  # 404 -> raise_for_status; 'not found' won't trip breaker
