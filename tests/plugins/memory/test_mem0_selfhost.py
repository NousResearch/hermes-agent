"""Tests for the self-hosted Mem0 direct-REST backend."""

import io
import json
import sys
import types
import urllib.error
import urllib.request
from email.message import Message
from urllib.parse import parse_qs, urlparse

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider


class _HTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if self._payload is None:
            return b""
        return json.dumps(self._payload).encode("utf-8")


def _header(request, name):
    name = name.lower()
    for key, value in request.header_items():
        if key.lower() == name:
            return value
    return None


def _json_body(request):
    if not request.data:
        return None
    return json.loads(request.data.decode("utf-8"))


def _install_exploding_memory_client(monkeypatch):
    # Patch MemoryClient ON the real mem0 module rather than replacing sys.modules["mem0"]
    # with a fake ModuleType. Replacing the whole module deletes the real one (and its
    # submodules) on monkeypatch revert when absent, polluting other tests that import mem0
    # (caused order-dependent hindsight failures under pytest-randomly). setattr on the real
    # module is reverted cleanly and leaves the module graph intact.
    import mem0 as _real_mem0

    class ExplodingMemoryClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MemoryClient must not be constructed for MEM0_HOST")

    monkeypatch.setattr(_real_mem0, "MemoryClient", ExplodingMemoryClient)


def _selfhost_provider(monkeypatch, tmp_path, *, destructive=False):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_HOST", "http://mem0.test")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("MEM0_USER_ID", "ace")
    monkeypatch.setenv("MEM0_AGENT_ID", "daedalus")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.setenv("MEM0_DESTRUCTIVE_TOOLS", "true" if destructive else "false")
    provider = Mem0MemoryProvider()
    provider.initialize("test-session")
    return provider


def test_selfhost_tools_use_direct_rest_with_api_key_and_response_mapping(monkeypatch, tmp_path):
    _install_exploding_memory_client(monkeypatch)
    calls = []

    def fake_urlopen(request, timeout=0):
        parsed = urlparse(request.full_url)
        call = {
            "method": request.get_method(),
            "path": parsed.path,
            "query": parse_qs(parsed.query),
            "headers": {k.lower(): v for k, v in request.header_items()},
            "api_key": _header(request, "X-API-Key"),
            "body": _json_body(request),
            "timeout": timeout,
        }
        calls.append(call)
        if call["method"] == "POST" and call["path"] == "/memories":
            return _HTTPResponse({"results": [{"id": "m-add", "memory": "stored fact"}]})
        if call["method"] == "POST" and call["path"] == "/search":
            return _HTTPResponse({"results": [{"id": "m-search", "memory": "matched fact", "score": 0.87}]})
        if call["method"] == "GET" and call["path"] == "/memories":
            return _HTTPResponse({"results": [{"id": "m-profile", "memory": "profile fact"}]})
        raise AssertionError(f"unexpected HTTP call: {call}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = _selfhost_provider(monkeypatch, tmp_path)

    conclude = json.loads(provider.handle_tool_call("mem0_conclude", {"conclusion": "store this"}))
    search = json.loads(provider.handle_tool_call("mem0_search", {"query": "needle", "top_k": 3}))
    profile = json.loads(provider.handle_tool_call("mem0_profile", {}))

    assert conclude == {"result": "Fact stored."}
    assert search == {"results": [{"memory": "matched fact", "score": 0.87}], "count": 1}
    assert profile == {"result": "profile fact", "count": 1}

    assert [(c["method"], c["path"]) for c in calls] == [
        ("POST", "/memories"),
        ("POST", "/search"),
        ("GET", "/memories"),
    ]
    assert all(c["api_key"] == "admin-key" for c in calls)

    add_call, search_call, profile_call = calls
    assert add_call["body"]["messages"] == [{"role": "user", "content": "store this"}]
    assert add_call["body"]["user_id"] == "ace"
    assert add_call["body"]["agent_id"] == "daedalus"
    assert search_call["body"]["query"] == "needle"
    assert search_call["body"]["user_id"] == "ace"
    assert search_call["body"]["agent_id"] == "daedalus"
    assert profile_call["query"] == {"user_id": ["ace"], "agent_id": ["daedalus"]}


def test_selfhost_delete_uses_rest_delete_after_read_before_destroy(monkeypatch, tmp_path):
    _install_exploding_memory_client(monkeypatch)
    calls = []

    def fake_urlopen(request, timeout=0):
        parsed = urlparse(request.full_url)
        call = {
            "method": request.get_method(),
            "path": parsed.path,
            "api_key": _header(request, "X-API-Key"),
            "body": _json_body(request),
        }
        calls.append(call)
        if call["method"] == "GET" and call["path"] == "/memories/m-delete":
            return _HTTPResponse({"id": "m-delete", "memory": "doomed", "metadata": {}})
        if call["method"] == "DELETE" and call["path"] == "/memories/m-delete":
            return _HTTPResponse({"id": "m-delete"})
        raise AssertionError(f"unexpected HTTP call: {call}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = _selfhost_provider(monkeypatch, tmp_path, destructive=True)

    result = json.loads(provider.handle_tool_call("mem0_delete", {"memory_id": "m-delete"}))

    assert result["deleted"] == 1
    assert result["results"] == [{"id": "m-delete", "outcome": "deleted", "was": "doomed"}]
    assert [(c["method"], c["path"]) for c in calls] == [
        ("GET", "/memories/m-delete"),
        ("DELETE", "/memories/m-delete"),
    ]
    assert all(c["api_key"] == "admin-key" for c in calls)


@pytest.mark.parametrize("host_value", [None, "", "   "])
def test_unset_or_blank_host_uses_existing_memoryclient_path(monkeypatch, tmp_path, host_value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "cloud-key")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "admin-key")
    if host_value is None:
        monkeypatch.delenv("MEM0_HOST", raising=False)
    else:
        monkeypatch.setenv("MEM0_HOST", host_value)

    constructed = []
    import mem0 as _real_mem0

    class FakeMemoryClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    # setattr on the real module (reverted cleanly) instead of replacing sys.modules["mem0"]
    # — the whole-module swap pollutes other tests under random ordering (see helper above).
    monkeypatch.setattr(_real_mem0, "MemoryClient", FakeMemoryClient)

    provider = Mem0MemoryProvider()
    provider.initialize("test-session")
    provider._get_client()

    assert constructed == [{"api_key": "cloud-key"}]


def test_mem0_json_overrides_selfhost_env_config(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_HOST", "http://env-host")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "env-key")
    monkeypatch.setenv("MEM0_USER_ID", "env-user")
    monkeypatch.setenv("MEM0_AGENT_ID", "env-agent")
    (tmp_path / "mem0.json").write_text(json.dumps({
        "host": "http://file-host/",
        "admin_api_key": "file-key",
        "user_id": "file-user",
        "agent_id": "file-agent",
    }))

    def fake_urlopen(request, timeout=0):
        parsed = urlparse(request.full_url)
        calls.append({
            "url": request.full_url,
            "netloc": parsed.netloc,
            "api_key": _header(request, "X-API-Key"),
            "body": _json_body(request),
        })
        return _HTTPResponse({"results": [{"id": "m1", "memory": "file scoped", "score": 0.9}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = Mem0MemoryProvider()
    provider.initialize("test-session")

    result = json.loads(provider.handle_tool_call("mem0_search", {"query": "scope"}))

    assert result["count"] == 1
    assert calls[0]["netloc"] == "file-host"
    assert calls[0]["api_key"] == "file-key"
    assert calls[0]["body"]["user_id"] == "file-user"
    assert calls[0]["body"]["agent_id"] == "file-agent"


def test_selfhost_401_surfaces_error_and_records_failure_without_fabricated_memory(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(b'{"detail":"bad key"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = _selfhost_provider(monkeypatch, tmp_path)

    result = json.loads(provider.handle_tool_call("mem0_search", {"query": "needle"}))

    assert "error" in result
    assert "401" in result["error"]
    assert "results" not in result
    assert "No relevant memories found" not in result["error"]
    assert provider._consecutive_failures == 1


def test_direct_rest_client_scopes_user_and_agent_even_when_caller_omits_them():
    """B3/B4: on a shared multi-agent store, add/search/get_all with NO explicit scope
    must still be constrained to the client's configured user_id AND agent_id —
    never querying or writing globally."""
    from importlib import import_module
    mod = import_module("plugins.memory.mem0")
    client = mod._DirectRestMem0Client(
        host="http://mem0.test", admin_api_key="k", agent_id="daedalus", user_id="ace"
    )
    sent = []

    def _fake_request(method, path, *, body=None, params=None):
        sent.append({"method": method, "path": path, "body": body, "params": params})
        return {"results": []}

    client._request = _fake_request

    # add() with no user_id/agent_id kwargs must inject both
    client.add([{"role": "user", "content": "x"}])
    assert sent[-1]["body"]["user_id"] == "ace"
    assert sent[-1]["body"]["agent_id"] == "daedalus"

    # search() with no filters must inject both
    client.search(query="q")
    assert sent[-1]["body"]["user_id"] == "ace"
    assert sent[-1]["body"]["agent_id"] == "daedalus"

    # get_all() with no filters must inject both
    client.get_all()
    assert sent[-1]["params"]["user_id"] == "ace"
    assert sent[-1]["params"]["agent_id"] == "daedalus"

    # an explicit caller filter is respected (not overridden)
    client.search(query="q", filters={"user_id": "other"})
    assert sent[-1]["body"]["user_id"] == "other"
    assert sent[-1]["body"]["agent_id"] == "daedalus"
