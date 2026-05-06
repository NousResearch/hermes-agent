import json
from pathlib import Path
from unittest.mock import MagicMock

from plugins.memory.openviking import OpenVikingMemoryProvider, _VikingClient


class _FakeFileReadClient:
    def __init__(self):
        self.calls = []

    def get(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if path == "/api/v1/fs/stat":
            return {"result": {"isDir": False}}
        if path == "/api/v1/content/read":
            return {"result": "file body"}
        raise AssertionError(f"Unexpected GET call: {path}")


class _FakeUploadResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttpx:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeUploadResponse({"result": {"temp_file_id": "temp-123"}})


class _FakeResourceClient:
    def __init__(self):
        self._httpx = _FakeHttpx()
        self.post_calls = []

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "X-OpenViking-Account": "default",
            "X-OpenViking-User": "hermes",
            "X-OpenViking-Agent": "hermes",
            "X-API-Key": "dev",
        }

    def _url(self, path):
        return f"http://127.0.0.1:1933{path}"

    def post(self, path, payload=None, **kwargs):
        self.post_calls.append((path, payload, kwargs))
        if path == "/api/v1/resources":
            return {"result": {"root_uri": "viking://resources/test-upload"}}
        raise AssertionError(f"Unexpected POST call: {path}")


class _FakeRememberClient:
    def __init__(self):
        self.post_calls = []

    def post(self, path, payload=None, **kwargs):
        self.post_calls.append((path, payload, kwargs))
        return {"result": {}}


class _FakePrefetchClient:
    def post(self, path, payload=None, **kwargs):
        assert path == "/api/v1/search/find"
        return {"result": {"memories": [], "resources": []}}


def test_viking_client_defaults_use_default_user(monkeypatch):
    monkeypatch.delenv("OPENVIKING_ACCOUNT", raising=False)
    monkeypatch.delenv("OPENVIKING_USER", raising=False)
    monkeypatch.delenv("OPENVIKING_AGENT", raising=False)

    client = _VikingClient("http://127.0.0.1:1933", api_key="dev")

    assert client._headers()["X-OpenViking-Account"] == "default"
    assert client._headers()["X-OpenViking-User"] == "default"
    assert client._headers()["X-OpenViking-Agent"] == "hermes"



def test_provider_initial_defaults_use_default_user():
    provider = OpenVikingMemoryProvider()

    assert provider._account == "default"
    assert provider._user == "default"
    assert provider._agent == "hermes"



def test_tool_search_sorts_by_raw_score_across_buckets():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/1", "score": 0.9003, "abstract": "memory result"},
            ],
            "resources": [
                {"uri": "viking://resources/1", "score": 0.9004, "abstract": "resource result"},
            ],
            "skills": [
                {"uri": "viking://skills/1", "score": 0.8999, "abstract": "skill result"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://resources/1",
        "viking://memories/1",
        "viking://skills/1",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.9, 0.9, 0.9]
    assert result["total"] == 3



def test_tool_search_sorts_missing_raw_score_after_negative_scores():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/missing", "abstract": "missing score"},
            ],
            "resources": [
                {"uri": "viking://resources/negative", "score": -0.25, "abstract": "negative score"},
            ],
            "skills": [
                {"uri": "viking://skills/positive", "score": 0.1, "abstract": "positive score"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://skills/positive",
        "viking://memories/missing",
        "viking://resources/negative",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.1, 0.0, -0.25]
    assert result["total"] == 3



def test_tool_read_uses_full_for_file_uri():
    provider = OpenVikingMemoryProvider()
    provider._client = _FakeFileReadClient()

    payload = json.loads(
        provider._tool_read(
            {"uri": "viking://user/hermes/memories/profile.md", "level": "overview"}
        )
    )

    assert payload["level"] == "full"
    assert payload["content"] == "file body"



def test_tool_add_resource_uploads_local_file_via_temp_upload(tmp_path):
    provider = OpenVikingMemoryProvider()
    provider._client = _FakeResourceClient()

    test_file = tmp_path / "resource.md"
    test_file.write_text("hello")

    payload = json.loads(provider._tool_add_resource({"url": str(test_file), "reason": "unit-test"}))

    upload_url, upload_kwargs = provider._client._httpx.calls[0]
    assert upload_url.endswith("/api/v1/resources/temp_upload")
    assert "file" in upload_kwargs["files"]
    assert payload["status"] == "added"
    assert payload["root_uri"] == "viking://resources/test-upload"
    assert provider._client.post_calls[0][1]["temp_file_id"] == "temp-123"



def test_tool_remember_commits_and_returns_fallback_uri(monkeypatch):
    provider = OpenVikingMemoryProvider()
    provider._client = _FakeRememberClient()
    provider._session_id = "sess-123"
    provider._ensure_session = MagicMock(return_value=True)
    provider._store_explicit_memory_resource = MagicMock(return_value="viking://resources/hermes_explicit_memories/case_1")

    payload = json.loads(provider._tool_remember({"content": "remember me", "category": "case"}))

    assert provider._client.post_calls[0][0] == "/api/v1/sessions/sess-123/messages"
    assert provider._client.post_calls[1][0] == "/api/v1/sessions/sess-123/commit"
    assert payload["fallback_uri"] == "viking://resources/hermes_explicit_memories/case_1"



def test_queue_prefetch_includes_explicit_fallback_hits():
    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._build_client = MagicMock(return_value=_FakePrefetchClient())
    provider._search_explicit_fallback = MagicMock(return_value=[
        {
            "uri": "viking://resources/hermes_explicit_memories/test-note.md",
            "score": 0.97,
            "abstract": "important fallback note",
        }
    ])

    provider.queue_prefetch("openviking restart")
    if provider._prefetch_thread and provider._prefetch_thread.is_alive():
        provider._prefetch_thread.join(timeout=3)

    result = provider.prefetch("openviking restart")

    assert "important fallback note" in result
    assert "viking://resources/hermes_explicit_memories/test-note.md" in result
