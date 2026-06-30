"""Tests for Mem0Backend abstraction — PlatformBackend, OSSBackend, SelfHostedBackend."""

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

    def test_search_rerank_default_true(self):
        backend, client = self._make()
        backend.search("q", filters={})
        assert client.calls[0][2]["rerank"] is True

    def test_search_returns_list(self):
        backend, _ = self._make()
        result = backend.search("q", filters={})
        assert isinstance(result, list)
        assert result[0]["id"] == "m1"

    def test_get_all_forwards_pagination(self):
        backend, client = self._make()
        result = backend.get_all(filters={"user_id": "u1"}, page=2, page_size=50)
        assert client.calls[0][1]["page"] == 2
        assert client.calls[0][1]["page_size"] == 50
        assert "count" in result

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

    def test_get_all_ignores_pagination(self):
        """OSSBackend accepts page/page_size but does NOT forward to Memory.get_all()."""
        backend, memory = self._make()
        result = backend.get_all(filters={"user_id": "u1"}, page=2, page_size=50)
        call_kwargs = memory.calls[0][1]
        assert "page" not in call_kwargs
        assert "page_size" not in call_kwargs
        assert result["count"] == 1

    def test_get_all_returns_envelope(self):
        backend, _ = self._make()
        result = backend.get_all(filters={"user_id": "u1"})
        assert "results" in result
        assert "count" in result

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
    """Build a SelfHostedBackend whose client routes through the stub transport.

    Mirrors __init__'s header setup but injects MockTransport so no real socket
    is opened.
    """
    backend = SelfHostedBackend.__new__(SelfHostedBackend)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    backend._client = httpx.Client(
        base_url=host.rstrip("/"), headers=headers,
        transport=httpx.MockTransport(server.handler),
    )
    return backend


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

    # --- get_all / pagination -------------------------------------------

    def test_get_all_passes_user_id_as_query_param(self):
        s = _StubServer()
        _backend(s).get_all(filters={"user_id": "u1"}, page=1, page_size=3)
        req = s.requests[-1]
        assert req.method == "GET" and req.url.path == "/memories"
        assert req.url.params["user_id"] == "u1"

    def test_get_all_slices_requested_page(self):
        s = _StubServer(rows=10)
        out = _backend(s).get_all(filters={"user_id": "u1"}, page=2, page_size=3)
        # top_k = page*page_size = 6 -> server returns m0..m5 -> slice[3:6]
        assert [r["id"] for r in out["results"]] == ["m3", "m4", "m5"]
        assert out["count"] == 6

    def test_get_all_caps_top_k_at_server_limit(self):
        s = _StubServer(rows=10)
        _backend(s).get_all(filters={"user_id": "u1"}, page=100, page_size=200)
        req = s.requests[-1]
        assert int(req.url.params["top_k"]) == SelfHostedBackend._MAX_TOP_K

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
