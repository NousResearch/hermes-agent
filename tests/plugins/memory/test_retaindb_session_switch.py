"""RetainDB session ownership through real discovery, manager hooks and HTTP payloads."""

import json
import threading
from urllib.parse import urlsplit

import pytest
import requests

from agent.memory_manager import MemoryManager
from plugins.memory import load_memory_provider


def _response(payload):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode()
    return response


@pytest.fixture
def memory(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("RETAINDB_API_KEY", "offline-test-key")
    monkeypatch.setenv("RETAINDB_BASE_URL", "https://memory.invalid")
    monkeypatch.setenv("RETAINDB_PROJECT", "test-project")
    provider = load_memory_provider("retaindb", register_skills=False)
    assert provider is not None
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all("parent", hermes_home=str(home), platform="cli", user_id="user")
    try:
        yield manager, provider
    finally:
        manager.shutdown_all()


def test_session_switch_rebinds_tools_without_relabeling_accepted_turns(memory, monkeypatch):
    manager, _ = memory
    requests_seen = []
    parent_started = threading.Event()
    release_parent = threading.Event()
    child_written = threading.Event()

    def request(method, url, **kwargs):
        path, body = urlsplit(url).path, kwargs.get("json")
        requests_seen.append((path, body))
        if path == "/v1/memory/ingest/session":
            content = body["messages"][0]["content"]
            if content == "queued parent turn":
                parent_started.set()
                assert release_parent.wait(5)
            if content == "child turn":
                child_written.set()
        return _response({"id": "memory-id", "memories": [], "results": []})

    monkeypatch.setattr(requests, "request", request)
    try:
        manager.sync_all("queued parent turn", "answer", session_id="parent")
        assert parent_started.wait(5)
        manager.on_session_switch("child", parent_session_id="parent", reason="compression")
        for tool, args in (
            ("retaindb_remember", {"content": "child fact"}),
            ("retaindb_search", {"query": "child fact"}),
            ("retaindb_context", {"query": "child fact"}),
        ):
            assert "error" not in json.loads(manager.handle_tool_call(tool, args))
        manager.on_memory_write("add", "memory", "child mirror")
        # Work accepted by the manager before a boundary can reach the provider
        # afterwards. Its explicit session parameter must still own the write.
        manager.sync_all("delayed parent turn", "answer", session_id="parent")
        manager.sync_all("child turn", "answer", session_id="child")
        release_parent.set()
        assert child_written.wait(5)
        manager.on_session_switch("reset-session", reset=True)
        assert "error" not in json.loads(manager.handle_tool_call("retaindb_remember", {"content": "reset fact"}))
    finally:
        release_parent.set()
        manager.shutdown_all()

    expected_owners = {
        ("/v1/memory", "child fact"): "child",
        ("/v1/memory/search", "child fact"): "child",
        ("/v1/context/query", "child fact"): "child",
        ("/v1/memory", "child mirror"): "child",
        ("/v1/memory", "reset fact"): "reset-session",
        ("/v1/memory/ingest/session", "queued parent turn"): "parent",
        ("/v1/memory/ingest/session", "delayed parent turn"): "parent",
        ("/v1/memory/ingest/session", "child turn"): "child",
    }
    scoped_paths = {path for path, _ in expected_owners}
    observed = set()
    for path, body in requests_seen:
        if path not in scoped_paths:
            continue
        assert isinstance(body, dict), (path, body)
        assert {"session_id", "user_id", "project"} <= body.keys(), (path, body)
        if path == "/v1/memory/ingest/session":
            content = body["messages"][0]["content"]
        else:
            content = body.get("content", body.get("query"))
        operation = (path, content)
        assert operation in expected_owners
        assert body["session_id"] == expected_owners[operation]
        assert body["user_id"] == "user"
        assert body["project"] == "test-project"
        observed.add(operation)
    assert observed == expected_owners.keys()


def test_session_switch_and_rewind_discard_prior_prefetch_results(memory, monkeypatch):
    manager, provider = memory
    requests_seen = []
    recall_started = threading.Event()
    release_recall = threading.Event()

    def request(method, url, **kwargs):
        path, body = urlsplit(url).path, kwargs.get("json") or {}
        requests_seen.append((path, body))
        if path == "/v1/context/query":
            query = body["query"]
            if query in {"before switch", "before undo"}:
                recall_started.set()
                assert release_recall.wait(5)
            return _response({"results": [{"content": query}]})
        if path.endswith("/ask"):
            return _response({"answer": body["query"]})
        if path.endswith("/model"):
            return _response({"memory_count": 1, "persona": "remembered persona"})
        return _response({"memories": []})

    def finish_prefetch():
        for thread in provider._prefetch_threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    monkeypatch.setattr(requests, "request", request)
    try:
        provider.queue_prefetch("before switch", session_id="parent")
        assert recall_started.wait(5)
        manager.on_session_switch("child", reset=True)
        release_recall.set()
        finish_prefetch()
        assert provider.prefetch("after switch", session_id="child") == ""

        # An old manager task arriving after the boundary must not recall its
        # old query against the new owner or fill that owner's cache.
        provider.queue_prefetch("delayed parent query", session_id="parent")
        finish_prefetch()
        assert provider.prefetch("after switch", session_id="child") == ""

        provider.queue_prefetch("child fact", session_id="child")
        finish_prefetch()
        assert provider.prefetch("delayed parent read", session_id="parent") == ""
        assert "child fact" in provider.prefetch("child fact", session_id="child")
        recall_started.clear()
        release_recall.clear()
        provider.queue_prefetch("before undo", session_id="child")
        assert recall_started.wait(5)
        manager.on_session_switch("child", rewound=True)
        release_recall.set()
        finish_prefetch()
        assert provider.prefetch("after undo", session_id="child") == ""
    finally:
        release_recall.set()
        finish_prefetch()

    for path, body in requests_seen:
        if path == "/v1/context/query":
            expected = "parent" if body["query"] in {"before switch", "delayed parent query"} else "child"
            assert body["session_id"] == expected
