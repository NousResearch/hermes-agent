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

    def test_search_forwards_rerank(self):
        backend, client = self._make()
        backend.search("q", filters={}, rerank=False)
        assert client.calls[0][2]["rerank"] is False

    def test_search_rerank_default_false(self):
        backend, client = self._make()
        backend.search("q", filters={})
        assert client.calls[0][2]["rerank"] is False

    def test_search_returns_list(self):
        backend, _ = self._make()
        result = backend.search("q", filters={})
        assert isinstance(result, list)
        assert result[0]["id"] == "m1"

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

    def test_add_forwards_metadata_when_present(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(
            msgs,
            user_id="u1",
            agent_id="hermes",
            infer=False,
            metadata={"channel": "telegram"},
        )
        assert client.calls[0][2]["metadata"] == {"channel": "telegram"}

    def test_add_omits_empty_metadata(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(msgs, user_id="u1", agent_id="hermes", infer=False, metadata={})
        assert "metadata" not in client.calls[0][2]

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

    def test_search_returns_list(self):
        backend, _ = self._make()
        result = backend.search("test", filters={"user_id": "u1"})
        assert isinstance(result, list)
        assert result[0]["id"] == "m1"

    def test_search_passes_filters(self):
        backend, memory = self._make()
        backend.search("q", filters={"user_id": "u1"}, top_k=3)
        assert memory.calls[0][2]["filters"] == {"user_id": "u1"}
        assert memory.calls[0][2]["top_k"] == 3

    def test_search_ignores_rerank(self):
        """OSS backend accepts rerank param but does not forward it to Memory."""
        backend, memory = self._make()
        backend.search("q", filters={}, rerank=True)
        assert "rerank" not in memory.calls[0][2]

    def test_add_forwards_kwargs(self):
        backend, memory = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)
        assert memory.calls[0][2]["user_id"] == "u1"
        assert memory.calls[0][2]["infer"] is False

    def test_update_maps_text_to_data(self):
        """OSS Memory.update uses `data=` param, not `text=`."""
        backend, memory = self._make()
        backend.update("m1", "new text")
        assert memory.calls[0][0] == "update"
        assert memory.calls[0][1] == "m1"
        assert memory.calls[0][2] == {"data": "new text"}

    def test_delete_positional_arg(self):
        backend, memory = self._make()
        backend.delete("m1")
        assert memory.calls[0] == ("delete", "m1")

    def test_update_normalizes_response(self):
        backend, _ = self._make()
        result = backend.update("m1", "text")
        assert result == {"result": "Memory updated.", "memory_id": "m1"}

    def test_delete_normalizes_response(self):
        backend, _ = self._make()
        result = backend.delete("m1")
        assert result == {"result": "Memory deleted.", "memory_id": "m1"}

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

    def create_collection(self, **kwargs):
        self.creations.append(kwargs)


class _FakeVectorStore:
    """Fake vector store that wraps a fake QdrantClient."""
    def __init__(self, client: _FakeQdrantClient, on_disk: bool = False):
        self.client = client
        self.on_disk = on_disk

    def create_col(self, vector_size: int, on_disk: bool):
        pass  # Just validate it's called


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

    def test_partial_failure_does_not_hide_error(self, caplog):
        """When delete succeeds but create_col raises, the exception is logged."""
        import logging
        caplog.set_level(logging.ERROR)

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
        assert "Failed to recreate Qdrant collection" in caplog.text
        assert "create_col failed" in caplog.text


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

    def test_init_strips_trailing_slash(self):
        b = SelfHostedBackend("k", "http://sh:8888/")
        assert str(b._client.base_url) == "http://sh:8888"

    def test_init_omits_api_key_header_when_blank(self):
        b = SelfHostedBackend("", "http://sh:8888")  # AUTH_DISABLED server
        assert "x-api-key" not in b._client.headers

    # --- search ----------------------------------------------------------

    def test_search_posts_to_search_with_filters_in_body(self):
        s = _StubServer()
        results = _backend(s).search("drink", filters={"user_id": "u1"}, top_k=5)
        req = s.requests[-1]
        assert (req.method, req.url.path) == ("POST", "/search")
        import json
        body = json.loads(req.content)
        assert body == {"query": "drink", "top_k": 5, "filters": {"user_id": "u1"}}
        assert results == [{"id": "m1", "memory": "tea", "score": 0.9}]

    def test_search_sends_x_api_key_header(self):
        s = _StubServer()
        _backend(s).search("q", filters={"user_id": "u1"})
        req = s.requests[-1]
        assert req.headers["x-api-key"] == "adminkey"
        assert "authorization" not in req.headers

    # --- add / update / delete ------------------------------------------

    def test_add_posts_messages_and_identity(self):
        s = _StubServer()
        msgs = [{"role": "user", "content": "likes tea"}]
        result = _backend(s).add(msgs, user_id="u1", agent_id="hermes", infer=False, metadata={"channel": "cli"})
        req = s.requests[-1]
        assert (req.method, req.url.path) == ("POST", "/memories")
        import json
        body = json.loads(req.content)
        assert body == {"messages": msgs, "user_id": "u1", "agent_id": "hermes",
                        "infer": False, "metadata": {"channel": "cli"}}
        assert result["results"][0]["id"] == "new"

    def test_update_puts_text_to_memory_id(self):
        s = _StubServer()
        result = _backend(s).update("abc", "new text")
        req = s.requests[-1]
        assert (req.method, req.url.path) == ("PUT", "/memories/abc")
        import json
        assert json.loads(req.content) == {"text": "new text"}
        assert result == {"result": "Memory updated.", "memory_id": "abc"}

    def test_delete_calls_delete_endpoint(self):
        s = _StubServer()
        result = _backend(s).delete("abc")
        req = s.requests[-1]
        assert (req.method, req.url.path) == ("DELETE", "/memories/abc")
        assert result == {"result": "Memory deleted.", "memory_id": "abc"}

    # --- error propagation (feeds the plugin's circuit breaker) ----------

    def test_http_error_raises(self):
        s = _StubServer()
        with pytest.raises(httpx.HTTPStatusError):
            _backend(s).delete("missing")  # 404 -> raise_for_status; 'not found' won't trip breaker
