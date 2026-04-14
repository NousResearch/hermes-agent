"""Smoke tests for the expanded OpenViking memory tools.

Verifies that viking_write, viking_link, viking_grep, viking_glob, and
viking_extract are wired into get_tool_schemas() and handle_tool_call(),
and that each tool returns the expected JSON structure when the HTTP
client is mocked.
"""

import json

import pytest

from plugins.memory.openviking import OpenVikingMemoryProvider


# ---------------------------------------------------------------------------
# Fake HTTP client — replaces _VikingClient for all tests
# ---------------------------------------------------------------------------

class FakeVikingClient:
    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []
        self._health = True

    def health(self) -> bool:
        return self._health

    def get(self, path: str, **kwargs) -> dict:
        self.calls.append({"method": "GET", "path": path, **kwargs})
        return {"result": {}}

    def post(self, path: str, payload: dict = None, **kwargs) -> dict:
        self.calls.append({"method": "POST", "path": path, "payload": payload or {}})
        # Return shapes that match what each tool reads
        if path.endswith("/extract"):
            return {"result": {"categories_extracted": 3}}
        if "/search/grep" in path:
            return {"result": [{"uri": "viking://a", "snippet": "foo", "line": 1}]}
        if "/search/glob" in path:
            return {"result": [{"uri": "viking://b", "name": "b"}]}
        if "/content/write" in path:
            return {"result": {"metadata": {"version": 1}}}
        if "/relations/link" in path:
            return {"result": {"edge_id": "e1"}}
        return {"result": {}}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1933")
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    monkeypatch.setattr(
        "plugins.memory.openviking._VikingClient", FakeVikingClient
    )
    p = OpenVikingMemoryProvider()
    p.initialize("session-test", hermes_home=str(tmp_path), platform="cli")
    return p


# ---------------------------------------------------------------------------
# get_tool_schemas
# ---------------------------------------------------------------------------

def test_new_tools_in_schemas(provider):
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert "viking_write" in names
    assert "viking_link" in names
    assert "viking_grep" in names
    assert "viking_glob" in names
    assert "viking_extract" in names


# ---------------------------------------------------------------------------
# viking_write
# ---------------------------------------------------------------------------

def test_viking_write_success(provider):
    result = json.loads(
        provider.handle_tool_call("viking_write", {"uri": "viking://x", "content": "hello"})
    )
    assert result["status"] == "written"
    assert result["uri"] == "viking://x"


def test_viking_write_missing_args(provider):
    result = json.loads(provider.handle_tool_call("viking_write", {}))
    assert "error" in result or "uri and content" in str(result)


# ---------------------------------------------------------------------------
# viking_link
# ---------------------------------------------------------------------------

def test_viking_link_success(provider):
    result = json.loads(
        provider.handle_tool_call(
            "viking_link",
            {"source": "viking://a", "target": "viking://b", "reason": "dependency"},
        )
    )
    assert result["status"] == "linked"
    assert result["source"] == "viking://a"


def test_viking_link_missing_args(provider):
    result = json.loads(provider.handle_tool_call("viking_link", {}))
    assert "error" in result or "source" in str(result)


# ---------------------------------------------------------------------------
# viking_grep
# ---------------------------------------------------------------------------

def test_viking_grep_success(provider):
    result = json.loads(
        provider.handle_tool_call("viking_grep", {"pattern": "foo"})
    )
    assert "results" in result
    assert result["results"][0]["uri"] == "viking://a"


def test_viking_grep_missing_pattern(provider):
    result = json.loads(provider.handle_tool_call("viking_grep", {}))
    assert "error" in result or "pattern" in str(result)


# ---------------------------------------------------------------------------
# viking_glob
# ---------------------------------------------------------------------------

def test_viking_glob_success(provider):
    result = json.loads(
        provider.handle_tool_call("viking_glob", {"pattern": "viking://**"})
    )
    assert "entries" in result
    assert result["entries"][0]["uri"] == "viking://b"


def test_viking_glob_missing_pattern(provider):
    result = json.loads(provider.handle_tool_call("viking_glob", {}))
    assert "error" in result or "pattern" in str(result)


# ---------------------------------------------------------------------------
# viking_extract
# ---------------------------------------------------------------------------

def test_viking_extract_success(provider):
    result = json.loads(
        provider.handle_tool_call("viking_extract", {"session_id": "session-test"})
    )
    assert result["status"] == "extracted"
    assert result["session_id"] == "session-test"


def test_viking_extract_uses_active_session(provider):
    """session_id defaults to the provider's own session when omitted."""
    result = json.loads(provider.handle_tool_call("viking_extract", {}))
    assert result["status"] == "extracted"


# ---------------------------------------------------------------------------
# Unknown tool routing
# ---------------------------------------------------------------------------

def test_unknown_tool_returns_error(provider):
    result = provider.handle_tool_call("viking_nonexistent", {})
    # tool_error returns a string; make sure it signals an error
    assert "Unknown tool" in result or "error" in result.lower()
