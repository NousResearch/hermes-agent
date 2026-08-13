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
