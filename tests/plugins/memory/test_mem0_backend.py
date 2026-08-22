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


    def test_oss_pgvector_password_reads_from_env_when_absent(self, monkeypatch):
        import sys
        import types

        captured = {}

        class Memory:
            @staticmethod
            def from_config(config):
                captured.update(config)
                return FakeOSSMemory()

        stub_mem0 = types.ModuleType("mem0")
        stub_mem0.Memory = Memory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mem0", stub_mem0)
        # Unknown embedder model -> dims unknown -> dims-migration check skipped
        # (avoids a real psycopg2 connection attempt in the test).
        raw = {
            "llm": {"provider": "openai", "config": {"model": "gpt-5-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "custom-unknown-model"}},
            "vector_store": {"provider": "pgvector", "config": {"host": "localhost"}},
        }
        monkeypatch.setenv("MEM0_PGVECTOR_PASSWORD", "s3cret")

        OSSBackend(raw)

        assert captured["vector_store"]["config"]["password"] == "s3cret"

    def test_oss_pgvector_password_keeps_mem0_json_value(self, monkeypatch):
        import sys
        import types

        captured = {}

        class Memory:
            @staticmethod
            def from_config(config):
                captured.update(config)
                return FakeOSSMemory()

        stub_mem0 = types.ModuleType("mem0")
        stub_mem0.Memory = Memory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mem0", stub_mem0)
        raw = {
            "llm": {"provider": "openai", "config": {"model": "gpt-5-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "custom-unknown-model"}},
            "vector_store": {"provider": "pgvector", "config": {"host": "localhost", "password": "legacy"}},
        }

        OSSBackend(raw)

        # Backward compat: a password already in mem0.json (written by older
        # wizard versions) still wins over the env var.
        assert captured["vector_store"]["config"]["password"] == "legacy"

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

def _make_pg_stub(row, monkeypatch):
    """Stub psycopg2 (incl. its ``sql`` submodule) whose cursor returns ``row``."""
    import sys
    import types

    class _Sql:
        @staticmethod
        def SQL(s):
            return s

        @staticmethod
        def Identifier(s):
            return s

    class _Cursor:
        def __init__(self):
            self.statements = []

        def execute(self, sql, params=None):
            self.statements.append(str(sql))

        def fetchone(self):
            return row

        def close(self):
            pass

    class _Conn:
        def __init__(self):
            self.autocommit = False
            self._c = _Cursor()

        def cursor(self):
            return self._c

        def close(self):
            pass

    class _Module:
        sql = _Sql

        def __init__(self):
            self._conn = _Conn()

        def connect(self, **kwargs):
            return self._conn

    mod = _Module()
    monkeypatch.setitem(sys.modules, "psycopg2", mod)
    return mod


class TestRecreateCollectionDims:
    """Regression: pgvector ``atttypmod`` stores dims + 4 (VARHDRSZ header),
    so a same-dims collection must NOT be dropped on every startup."""

    def _call_pgvector(self, row, monkeypatch):
        mod = _make_pg_stub(row, monkeypatch)
        OSSBackend._recreate_collection_if_dims_changed(
            "pgvector",
            {"collection_name": "mem0", "host": "localhost", "port": 5432},
            1536,
        )
        return mod._conn._c.statements

    def test_pgvector_unchanged_dims_does_not_drop_table(self, monkeypatch):
        # atttypmod for vector(1536) is 1540, not 1536.
        stmts = self._call_pgvector((1536 + 4,), monkeypatch)
        assert not any("DROP TABLE" in s for s in stmts)

    def test_pgvector_changed_dims_drops_table(self, monkeypatch):
        # Collection was built for 768-dims embeddings; drop is the intended
        # migration so the new dims can be recreated cleanly.
        stmts = self._call_pgvector((768 + 4,), monkeypatch)
        assert any("DROP TABLE" in s for s in stmts)

    def test_pgvector_missing_column_does_not_drop(self, monkeypatch):
        stmts = self._call_pgvector(None, monkeypatch)
        assert not any("DROP TABLE" in s for s in stmts)

    def test_qdrant_unchanged_dims_does_not_delete_collection(self, monkeypatch):
        import sys
        import types
        from types import SimpleNamespace

        calls = []

        class _FakeQdrant:
            def __init__(self, **kwargs):
                pass

            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return SimpleNamespace(
                    config=SimpleNamespace(
                        params=SimpleNamespace(vectors=SimpleNamespace(size=1536))
                    )
                )

            def delete_collection(self, name):
                calls.append(name)

            def close(self):
                pass

        mod = types.ModuleType("qdrant_client")
        mod.QdrantClient = _FakeQdrant
        monkeypatch.setitem(sys.modules, "qdrant_client", mod)

        OSSBackend._recreate_collection_if_dims_changed(
            "qdrant", {"collection_name": "mem0", "path": "~/.hermes/mem0_qdrant"}, 1536
        )
        assert calls == []

