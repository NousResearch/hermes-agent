import json
import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from plugins.memory.jiuwenmemory import (  # noqa: E402
    JiuwenMemoryProvider,
    _JiuwenMemoryClient,
    _extract_results,
    _is_embedding_failure,
    _load_config,
)


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("JIUWENMEMORY_API_KEY", raising=False)
    monkeypatch.delenv("JIUWENMEMORY_BASE_URL", raising=False)
    monkeypatch.delenv("JIUWENMEMORY_SCOPE_ID", raising=False)
    monkeypatch.delenv("JIUWENMEMORY_USER_ID", raising=False)
    monkeypatch.delenv("MEMORY_API_KEY", raising=False)
    monkeypatch.delenv("DREAMING_ENABLED", raising=False)
    monkeypatch.delenv("DREAMING_SCOPE_ID", raising=False)
    monkeypatch.delenv("DREAMING_USER_ID", raising=False)
    monkeypatch.delenv("DREAMING_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("DREAMING_MIN_SESSION_ROUNDS", raising=False)
    monkeypatch.delenv("DREAMING_MAX_SESSIONS_PER_SWEEP", raising=False)
    monkeypatch.delenv("DREAMING_MAX_COMPRESS_TOKENS", raising=False)
    monkeypatch.delenv("DREAMING_MAX_ITEMS_PER_SESSION", raising=False)


class TestJiuwenMemoryConfig:
    def test_load_config_defaults(self, tmp_path):
        config = _load_config(str(tmp_path / ".hermes"))
        assert config["base_url"] == "http://127.0.0.1:8000"
        assert config["scope_id"] == "hermes"
        assert config["user_id"] == ""
        assert config["threshold"] == 0.3
        assert config["max_recall_results"] == 8
        assert config["max_summary_results"] == 4
        assert config["local_fallback"] is True
        assert config["fallback_page_size"] == 50
        assert config["fallback_max_pages"] == 4
        assert config["enable_long_term_mem"] is True
        assert "paths" not in config

    def test_load_config_merges_file_and_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "jiuwenmemory.json").write_text(
            json.dumps(
                {
                    "base_url": "http://file.example///",
                    "scope_id": "file-scope",
                    "user_id": "file-user",
                    "threshold": 1.5,
                    "max_recall_results": 99,
                    "max_summary_results": -1,
                    "auto_recall": "false",
                    "local_fallback": "false",
                    "fallback_page_size": 500,
                    "fallback_max_pages": 0,
                    "enable_summary_memory": False,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("JIUWENMEMORY_BASE_URL", "http://env.example/")
        monkeypatch.setenv("JIUWENMEMORY_SCOPE_ID", "env-scope")
        monkeypatch.setenv("JIUWENMEMORY_USER_ID", "env-user")
        config = _load_config(str(hermes_home))
        assert config["base_url"] == "http://env.example"
        assert config["scope_id"] == "env-scope"
        assert config["user_id"] == "env-user"
        assert config["threshold"] == 1.0
        assert config["max_recall_results"] == 20
        assert config["max_summary_results"] == 0
        assert config["auto_recall"] is False
        assert config["local_fallback"] is False
        assert config["fallback_page_size"] == 200
        assert config["fallback_max_pages"] == 1
        assert config["enable_summary_memory"] is False

    def test_load_config_maps_legacy_collection_to_scope(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "jiuwenmemory.json").write_text(
            json.dumps({"collection": "legacy-scope"}),
            encoding="utf-8",
        )
        config = _load_config(str(hermes_home))
        assert config["scope_id"] == "legacy-scope"

    def test_extract_results_accepts_common_shapes(self):
        payload = {"data": [{"memory": "one", "score": 0.8}, {"text": "two"}]}
        assert [item["content"] for item in _extract_results(payload)] == ["one", "two"]
        assert _extract_results(["three"])[0]["content"] == "three"
        nested = {"results": [{"mem_info": {"content": "nested"}, "score": 0.7}]}
        assert _extract_results(nested)[0]["content"] == "nested"

    def test_timeout_counts_as_backend_failure_for_fallback(self):
        assert _is_embedding_failure(RuntimeError("timed out")) is True
        assert _is_embedding_failure(TimeoutError("request timeout")) is True


class FakeClient:
    def __init__(self):
        self.scope_id = "hermes"
        self.user_id = "user-1"
        self.base_url = "http://fake"
        self.search_memory_calls = []
        self.search_summary_calls = []
        self.store_calls = []
        self.store_session_ids = []
        self.sync_turn_calls = []
        self.turn_event = threading.Event()

    def search_memory(self, query, limit):
        self.search_memory_calls.append((query, limit))
        return [{"content": "User prefers practical integrations", "score": 0.91}]

    def search_summary(self, query, limit):
        self.search_summary_calls.append((query, limit))
        return [{"content": "Recent work focused on plugin APIs", "score": 0.82}]

    def store(self, content, *, session_id=""):
        self.store_calls.append(content)
        self.store_session_ids.append(session_id)
        return {"status": "success", "content": content}

    def status(self):
        return {"status": "healthy"}

    def sync_turn(self, user, assistant, *, session_id=""):
        self.sync_turn_calls.append((user, assistant, session_id))
        self.turn_event.wait(0.2)
        return {"status": "success", "messages": [user, assistant]}

    def diagnostic_status(self):
        return {"status": "healthy", "memory_search": "ok"}

    def lexical_search_from_pages(self, query, limit, *, page_size, max_pages):
        del query, limit, page_size, max_pages
        return []


class BrokenEmbeddingClient(FakeClient):
    def search_memory(self, query, limit):
        del query, limit
        raise RuntimeError("retrieval embedding_request call failed, Failed to get embedding after 3 attempts")

    def store(self, content, *, session_id=""):
        del session_id
        del content
        raise RuntimeError("retrieval embedding_request call failed, Failed to get embedding after 3 attempts")

    def diagnostic_status(self):
        return {"status": "healthy", "memory_search": "failed", "fallback_search": "available"}


class TestJiuwenMemoryProvider:

    def _provider(self, tmp_path):
        provider = JiuwenMemoryProvider()
        provider.initialize("sess-1", hermes_home=str(tmp_path / ".hermes"), user_id="user-1")
        fake = FakeClient()
        provider._client = cast(Any, fake)
        return provider, fake

    def test_name_and_availability(self):
        provider = JiuwenMemoryProvider()
        assert provider.name == "jiuwenmemory"
        assert provider.is_available() is True

    def test_system_prompt_block(self, tmp_path):
        provider, _fake = self._provider(tmp_path)
        block = provider.system_prompt_block()
        assert "JiuwenMemory" in block
        assert "Scope: hermes" in block
        assert "jiuwenmemory_search" in block

    def test_prefetch_formats_memory_and_summary_context(self, tmp_path):
        provider, fake = self._provider(tmp_path)
        result = provider.prefetch("what does user prefer")
        assert "jiuwenmemory-context" in result
        assert "Relevant memories" in result
        assert "practical integrations" in result
        assert "Relevant conversation summaries" in result
        assert "plugin APIs" in result
        assert fake.search_memory_calls == [("what does user prefer", 8)]
        assert fake.search_summary_calls == [("what does user prefer", 4)]

    def test_search_tool_dispatch(self, tmp_path):
        provider, fake = self._provider(tmp_path)
        result = json.loads(provider.handle_tool_call("jiuwenmemory_search", {"query": "prefs", "limit": 50}))
        assert result["results"][0]["content"] == "User prefers practical integrations"
        assert fake.search_memory_calls == [("prefs", 20)]
        assert fake.search_summary_calls == []

    def test_store_tool_dispatch(self, tmp_path):
        provider, fake = self._provider(tmp_path)
        result = json.loads(provider.handle_tool_call("jiuwenmemory_store", {"content": "remember this"}))
        assert result["status"] == "success"
        assert fake.store_calls == ["remember this"]
        assert fake.store_session_ids == ["sess-1"]

    def test_status_tool_adds_connection_defaults(self, tmp_path):
        provider, _fake = self._provider(tmp_path)
        result = json.loads(provider.handle_tool_call("jiuwenmemory_status", {}))
        assert result["status"] == "healthy"
        assert result["memory_search"] == "ok"
        assert result["scope_id"] == "hermes"
        assert result["user_id"] == "user-1"
        assert result["base_url"] == "http://fake"

    def test_missing_required_tool_args_return_error(self, tmp_path):
        provider, _fake = self._provider(tmp_path)
        assert "query is required" in json.loads(provider.handle_tool_call("jiuwenmemory_search", {}))["error"]
        assert "content is required" in json.loads(provider.handle_tool_call("jiuwenmemory_store", {}))["error"]

    def test_search_tool_uses_local_fallback_after_embedding_failure(self, tmp_path):
        provider, fake = self._provider(tmp_path)
        provider._client = cast(Any, BrokenEmbeddingClient())
        assert provider.handle_tool_call("jiuwenmemory_store", {"content": "User likes focused tests"})

        result = json.loads(provider.handle_tool_call("jiuwenmemory_search", {"query": "focused tests"}))

        assert result["fallback"] is True
        assert "server_search_error" in result
        assert result["results"][0]["content"] == "User likes focused tests"
        assert (tmp_path / ".hermes" / "jiuwenmemory_fallback.jsonl").exists()
        assert fake.store_calls == []

    def test_sync_turn_is_non_blocking(self, tmp_path):
        provider, fake = self._provider(tmp_path)
        started = time.time()
        provider.sync_turn("hello", "world")
        elapsed = time.time() - started
        fake.turn_event.set()
        provider.shutdown()
        assert elapsed < 0.1
        assert fake.sync_turn_calls == [("hello", "world", "sess-1")]

    def test_sync_turn_forwards_runtime_session_id(self, tmp_path):
        provider, fake = self._provider(tmp_path)
        provider.sync_turn("hello", "world", session_id="runtime-session")
        fake.turn_event.set()
        provider.shutdown()
        assert fake.sync_turn_calls == [("hello", "world", "runtime-session")]

    def test_on_memory_write_mirrors_add_and_replace(self, tmp_path):
        provider, fake = self._provider(tmp_path)
        provider.on_memory_write("add", "user", "User likes tests", {"origin": "unit"})
        provider.shutdown()
        provider.on_memory_write("replace", "user", "User likes focused tests", {"origin": "unit"})
        provider.shutdown()
        assert fake.store_calls == ["User likes tests", "User likes focused tests"]
        assert fake.store_session_ids == ["sess-1", "sess-1"]


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_client_uses_official_paths_and_payloads(monkeypatch):
    calls = []
    responses = [
        {"results": [{"content": "memory"}]},
        {"results": [{"content": "summary"}]},
        {"status": "success"},
        {"status": "success"},
        {"results": [], "total": 0},
        {"status": "healthy"},
    ]

    def fake_urlopen(req, timeout):
        calls.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": {key.lower(): value for key, value in req.header_items()},
                "body": json.loads(req.data.decode("utf-8")) if req.data else None,
                "timeout": timeout,
            }
        )
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = _JiuwenMemoryClient(
        "Bearer test-key",
        "http://example",
        "scope-a",
        "user-a",
        0.42,
        1.0,
        {
            "enable_long_term_mem": True,
            "enable_user_profile": False,
            "enable_semantic_memory": True,
            "enable_episodic_memory": False,
            "enable_summary_memory": True,
        },
    )

    assert client.search_memory("prefs", 3)[0]["content"] == "memory"
    assert client.search_summary("history", 2)[0]["content"] == "summary"
    assert client.store("remember this", session_id="sess-store")["status"] == "success"
    assert client.sync_turn("hello", "world", session_id="sess-turn")["status"] == "success"
    assert client.get_user_mem_by_page(page_size=5, page_idx=2, memory_type="semantic_memory")["total"] == 0
    assert client.status()["status"] == "healthy"

    assert [call["url"] for call in calls] == [
        "http://example/search_memory/",
        "http://example/search_user_history_summary/",
        "http://example/add_messages/",
        "http://example/add_messages/",
        "http://example/get_user_mem_by_page/",
        "http://example/health",
    ]
    assert [call["method"] for call in calls] == ["POST", "POST", "POST", "POST", "POST", "GET"]
    assert calls[0]["body"] == {
        "query": "prefs",
        "num": 3,
        "scope_id": "scope-a",
        "user_id": "user-a",
        "threshold": 0.42,
    }
    assert calls[1]["body"] == {
        "query": "history",
        "num": 2,
        "scope_id": "scope-a",
        "user_id": "user-a",
        "threshold": 0.42,
    }
    assert calls[2]["body"] == {
        "messages": [{"role": "user", "content": "remember this"}],
        "scope_id": "scope-a",
        "user_id": "user-a",
        "enable_long_term_mem": True,
        "enable_user_profile": False,
        "enable_semantic_memory": True,
        "enable_episodic_memory": False,
        "enable_summary_memory": True,
        "session_id": "sess-store",
    }
    assert calls[3]["body"]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    assert calls[3]["body"]["session_id"] == "sess-turn"
    assert calls[4]["body"] == {
        "scope_id": "scope-a",
        "user_id": "user-a",
        "page_size": 5,
        "page_idx": 2,
        "memory_type": "semantic_memory",
    }
    assert calls[5]["body"] is None
    assert calls[0]["headers"]["authorization"] == "Bearer test-key"
    assert "x-api-key" not in calls[0]["headers"]
    assert all(call["timeout"] == 1.0 for call in calls)


def test_client_diagnostic_status_reports_search_failures(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        del timeout
        calls.append(req.full_url)
        if req.full_url.endswith("/health"):
            return FakeResponse({"status": "healthy", "message": "ok"})
        if req.full_url.endswith("/get_user_mem_by_page/"):
            return FakeResponse({"results": [], "total": 0})
        raise RuntimeError("Failed to get embedding after 3 attempts")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = _JiuwenMemoryClient("", "http://example", "scope-a", "", 0.3, 1.0, {})
    result = client.diagnostic_status()

    assert result["status"] == "healthy"
    assert result["memory_search"] == "failed"
    assert result["fallback_search"] == "available"
    assert "Failed to get embedding" in result["memory_search_error"]
    assert "EMBED_MODEL_NAME" in result["diagnostic"]
    assert calls == [
        "http://example/health",
        "http://example/search_memory/",
        "http://example/get_user_mem_by_page/",
    ]


def test_client_diagnostic_status_reports_search_ok(monkeypatch):
    responses = [{"status": "healthy"}, {"results": []}]

    def fake_urlopen(req, timeout):
        del req, timeout
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = _JiuwenMemoryClient("", "http://example", "scope-a", "", 0.3, 1.0, {})

    assert client.diagnostic_status()["memory_search"] == "ok"


def test_client_omits_optional_user_id_when_unset(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        del timeout
        calls.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse({"results": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = _JiuwenMemoryClient("", "http://example", "scope-a", "", 0.3, 1.0, {})
    client.search_memory("prefs", 3)
    assert calls == [{"query": "prefs", "num": 3, "scope_id": "scope-a", "threshold": 0.3}]


def test_client_omits_optional_session_id_when_unset(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        del timeout
        calls.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse({"status": "success"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = _JiuwenMemoryClient("", "http://example", "scope-a", "", 0.3, 1.0, {})
    client.add_messages([{"role": "user", "content": "remember"}])
    assert calls == [
        {
            "messages": [{"role": "user", "content": "remember"}],
            "scope_id": "scope-a",
            "enable_long_term_mem": True,
            "enable_user_profile": True,
            "enable_semantic_memory": True,
            "enable_episodic_memory": True,
            "enable_summary_memory": True,
        }
    ]


def test_client_headers_include_optional_bearer_api_key():
    client = _JiuwenMemoryClient("Bearer test-key", "http://example", "hermes", "", 0.3, 1.0, {})
    headers = client._headers()
    assert headers["Authorization"] == "Bearer test-key"
    assert "X-API-Key" not in headers


def test_memory_server_add_messages_forwards_session_id(monkeypatch):
    from jiuwen_memory.server import memory_server

    class FakeMemoryEngine:
        def __init__(self):
            self.kwargs = None

        async def add_messages(self, messages, agent_config, *, user_id, scope_id, session_id):
            self.kwargs = {
                "messages": messages,
                "agent_config": agent_config,
                "user_id": user_id,
                "scope_id": scope_id,
                "session_id": session_id,
            }

    fake = FakeMemoryEngine()
    monkeypatch.setattr(memory_server, "memory_engine", fake)
    request = memory_server.AddMessagesRequest(
        messages=[{"role": "user", "content": "hello"}],
        user_id="user-a",
        scope_id="scope-a",
        session_id="sess-a",
    )

    result = asyncio.run(memory_server.add_messages_endpoint(request))

    assert result["status"] == "success"
    assert fake.kwargs is not None
    assert fake.kwargs["session_id"] == "sess-a"
    assert fake.kwargs["user_id"] == "user-a"
    assert fake.kwargs["scope_id"] == "scope-a"
    assert fake.kwargs["messages"][0].content == "hello"


class FakeTurboMemoryEngine:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    async def add_messages(self, messages, agent_config, *, user_id, scope_id, session_id):
        del agent_config
        self.calls.append(
            {
                "messages": [(message.role, message.content) for message in messages],
                "user_id": user_id,
                "scope_id": scope_id,
                "session_id": session_id,
            }
        )
        if self.fail:
            raise RuntimeError("turbo failure token=super-secret")


def test_memory_server_turbo_routes_are_present_and_root_lists_them():
    from jiuwen_memory.server import memory_server

    paths = {getattr(route, "path", "") for route in memory_server.app.routes}
    assert {"/turbo/add_messages_async", "/turbo/status", "/turbo/process_once"} <= paths

    root_payload = asyncio.run(memory_server.root())
    assert {
        "POST /turbo/add_messages_async",
        "GET /turbo/status",
        "POST /turbo/process_once",
    } <= set(root_payload["endpoints"])


def test_memory_server_turbo_add_messages_async_accepts_and_persists_raw_messages(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    monkeypatch.setattr(memory_server, "_TURBO_DB_PATH", tmp_path / "memory_turbo.sqlite3")
    started = []
    monkeypatch.setattr(memory_server, "_turbo_start_worker", lambda: started.append(True))
    request = memory_server.AddMessagesRequest(
        messages=[{"role": "user", "content": "remember turbo"}],
        user_id="user-a",
        scope_id="scope-a",
        session_id="turbo-session-a",
    )

    started_at = time.time()
    result = asyncio.run(memory_server.turbo_add_messages_async_endpoint(request))
    elapsed = time.time() - started_at

    assert elapsed < 0.5
    assert result["status"] == "accepted"
    assert result["queued"] is True
    assert result["session_id"] == "turbo-session-a"
    assert started == [True]
    with memory_server._turbo_connect() as conn:
        row = conn.execute("SELECT * FROM turbo_jobs WHERE job_id = ?", (result["job_id"],)).fetchone()
    assert row is not None
    assert row["status"] == "queued"
    assert json.loads(row["messages_json"]) == [{"role": "user", "content": "remember turbo"}]
    payload = json.loads(row["payload_json"])
    assert payload["session_id"] == "turbo-session-a"
    assert payload["scope_id"] == "scope-a"
    assert payload["user_id"] == "user-a"


def test_memory_server_turbo_process_once_completes_and_updates_status(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    monkeypatch.setattr(memory_server, "_TURBO_DB_PATH", tmp_path / "memory_turbo.sqlite3")
    fake = FakeTurboMemoryEngine()
    monkeypatch.setattr(memory_server, "memory_engine", fake)
    queued = memory_server._turbo_enqueue(
        memory_server.AddMessagesRequest(
            messages=[{"role": "user", "content": "finish turbo"}],
            user_id="user-b",
            scope_id="scope-b",
            session_id="turbo-session-b",
        )
    )

    result = asyncio.run(memory_server.turbo_process_once_endpoint())
    status = asyncio.run(memory_server.turbo_status_endpoint())

    assert result["processed"] is True
    assert result["status"] == "completed"
    assert result["job"]["job_id"] == queued["job_id"]
    assert result["job"]["error"] is None
    assert fake.calls == [
        {
            "messages": [("user", "finish turbo")],
            "user_id": "user-b",
            "scope_id": "scope-b",
            "session_id": "turbo-session-b",
        }
    ]
    assert status["queued"] == 0
    assert status["processing"] == 0
    assert status["completed"] == 1
    assert status["failed"] == 0
    assert status["recent_jobs"][0]["status"] == "completed"
    assert "messages_json" not in status["recent_jobs"][0]
    assert "payload_json" not in status["recent_jobs"][0]


def test_memory_server_turbo_process_once_records_failure_without_secret(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    monkeypatch.setattr(memory_server, "_TURBO_DB_PATH", tmp_path / "memory_turbo.sqlite3")
    fake = FakeTurboMemoryEngine(fail=True)
    monkeypatch.setattr(memory_server, "memory_engine", fake)
    memory_server._turbo_enqueue(
        memory_server.AddMessagesRequest(
            messages=[{"role": "user", "content": "fail turbo"}],
            user_id="user-c",
            scope_id="scope-c",
            session_id="turbo-session-c",
        )
    )

    result = asyncio.run(memory_server.turbo_process_once_endpoint())
    status = asyncio.run(memory_server.turbo_status_endpoint())

    assert result["processed"] is True
    assert result["status"] == "failed"
    assert "turbo failure" in result["job"]["error"]
    assert "super-secret" not in result["job"]["error"]
    assert status["queued"] == 0
    assert status["processing"] == 0
    assert status["completed"] == 0
    assert status["failed"] == 1
    assert status["recent_jobs"][0]["status"] == "failed"
    assert "super-secret" not in status["recent_jobs"][0]["error"]


def _use_temp_swarm_db(memory_server, tmp_path, monkeypatch):
    monkeypatch.setattr(memory_server, "_SWARM_DB_PATH", tmp_path / "swarm_memory.sqlite3")


def test_memory_server_swarm_routes_are_present_and_root_lists_them():
    from jiuwen_memory.server import memory_server

    paths = {getattr(route, "path", "") for route in memory_server.app.routes}
    assert {"/swarm/promote", "/swarm/status", "/swarm/search"} <= paths

    root_payload = asyncio.run(memory_server.root())
    assert {
        "POST /swarm/promote",
        "GET /swarm/status",
        "POST /swarm/search",
    } <= set(root_payload["endpoints"])


def test_memory_server_swarm_promote_persists_safe_memory(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    _use_temp_swarm_db(memory_server, tmp_path, monkeypatch)
    result = asyncio.run(
        memory_server.swarm_promote_endpoint(
            memory_server.SwarmPromoteRequest(
                source_scope="personal:user-a",
                target_scope="team:alpha",
                content="Release checklist requires staged rollout.",
                reason="Team-relevant project procedure",
                source_user_id="user-a",
                target_user_id="user-b",
            )
        )
    )

    assert result["status"] == "success"
    assert result["promoted"] is True
    assert result["scope_convention"]["personal"].startswith("personal:<user_id>")
    assert result["record"]["status"] == "promoted"
    assert result["record"]["source_scope_kind"] == "personal"
    assert result["record"]["target_scope_kind"] == "team"
    assert "content" not in result["record"]

    with memory_server._swarm_connect() as conn:
        row = conn.execute("SELECT * FROM swarm_promotions WHERE promotion_id = ?", (result["record"]["promotion_id"],)).fetchone()
    assert row is not None
    assert row["status"] == "promoted"
    assert row["content"] == "Release checklist requires staged rollout."


def test_memory_server_swarm_promote_skips_sensitive_memory_without_storing_content(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    _use_temp_swarm_db(memory_server, tmp_path, monkeypatch)
    result = asyncio.run(
        memory_server.swarm_promote_endpoint(
            memory_server.SwarmPromoteRequest(
                source_scope="personal:user-a",
                target_scope="team:alpha",
                content="The password field is present in this note.",
                reason="Should be rejected by the PoC guard",
                source_user_id="user-a",
            )
        )
    )

    assert result["status"] == "skipped"
    assert result["promoted"] is False
    assert result["skip_reason"] == "sensitive_content"
    assert result["record"]["status"] == "skipped_sensitive"
    assert "password" in result["record"]["sensitive_findings"]
    assert "content" not in result["record"]

    with memory_server._swarm_connect() as conn:
        row = conn.execute("SELECT * FROM swarm_promotions WHERE promotion_id = ?", (result["record"]["promotion_id"],)).fetchone()
    assert row is not None
    assert row["content"] == ""


def test_memory_server_swarm_search_returns_promoted_records_matching_query_tokens_and_scope(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    _use_temp_swarm_db(memory_server, tmp_path, monkeypatch)

    async def fail_native_search(request):
        del request
        raise RuntimeError("native unavailable")

    monkeypatch.setattr(memory_server, "search_memory_endpoint", fail_native_search)

    team_content = "Release checklist requires staged rollout. Swarm marker swarm-e2e-20260708-team."
    org_content = "Release policy requires incident notes. Swarm marker swarm-e2e-20260708-org."
    sensitive_content = "Release note contains a password and Swarm marker that must stay private."

    for source_scope, target_scope, content, reason in [
        ("personal:user-a", "team:alpha", team_content, "Team release evidence"),
        ("personal:user-a", "org:acme", org_content, "Org release evidence"),
        ("personal:user-a", "personal:user-a", "Release private reminder stays personal.", "Personal-only evidence"),
        ("personal:user-a", "team:alpha", sensitive_content, "Sensitive content should not leak"),
    ]:
        asyncio.run(
            memory_server.swarm_promote_endpoint(
                memory_server.SwarmPromoteRequest(
                    source_scope=source_scope,
                    target_scope=target_scope,
                    content=content,
                    source_user_id="user-a",
                    target_user_id="user-b",
                    reason=reason,
                )
            )
        )

    query = "Release Swarm marker staged incident"
    result = asyncio.run(
        memory_server.swarm_search_endpoint(
            memory_server.SwarmSearchRequest(
                scopes=["team:alpha", "org:acme"],
                query=query,
                user_id="user-b",
            )
        )
    )
    personal_result = asyncio.run(
        memory_server.swarm_search_endpoint(
            memory_server.SwarmSearchRequest(
                scopes=["personal:user-b"],
                query=query,
                user_id="user-b",
            )
        )
    )

    by_scope = {
        group["scope"]: group["memories"]
        for group in result["scopes"]
    }
    all_content = "\n".join(memory.get("content", "") for memories in by_scope.values() for memory in memories)
    assert result["status"] == "success"
    assert result["scope_convention"]["team"].startswith("team:<team_id>")
    assert by_scope["team:alpha"][0]["content"] == team_content
    assert by_scope["team:alpha"][0]["evidence"] == "Team release evidence"
    assert by_scope["team:alpha"][0]["status"] == "promoted"
    assert by_scope["org:acme"][0]["content"] == org_content
    assert by_scope["org:acme"][0]["evidence"] == "Org release evidence"
    assert by_scope["org:acme"][0]["status"] == "promoted"
    assert "private reminder" not in all_content
    assert sensitive_content not in json.dumps(result, sort_keys=True)
    assert personal_result["results_by_scope"]["personal:user-b"] == []
    assert {group["scope_kind"] for group in result["scopes"]} == {"team", "org"}
    assert len(result["native_search"]["errors"]) == 2


def test_memory_server_swarm_status_counts_and_recent_records_omit_content(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    _use_temp_swarm_db(memory_server, tmp_path, monkeypatch)
    requests = [
        memory_server.SwarmPromoteRequest(
            source_scope="personal:user-a",
            target_scope="team:alpha",
            content="Release checklist requires staged rollout.",
        ),
        memory_server.SwarmPromoteRequest(
            source_scope="personal:user-a",
            target_scope="team:alpha",
            content="Contains an API key label and must be skipped.",
        ),
        memory_server.SwarmPromoteRequest(
            source_scope="team:alpha",
            target_scope="org:acme",
            content="Release policy requires incident notes.",
        ),
    ]
    for request in requests:
        asyncio.run(memory_server.swarm_promote_endpoint(request))

    status = asyncio.run(memory_server.swarm_status_endpoint())
    counts = {item["target_scope"]: item["counts"] for item in status["counts_by_target_scope"]}

    assert status["status"] == "available"
    assert counts["team:alpha"]["promoted"] == 1
    assert counts["team:alpha"]["skipped_sensitive"] == 1
    assert counts["org:acme"]["promoted"] == 1
    assert status["totals"]["promoted"] == 2
    assert status["totals"]["skipped_sensitive"] == 1
    assert len(status["recent_promotions"]) == 3
    assert all("content" not in record for record in status["recent_promotions"])
    assert status["scope_convention"]["org"].startswith("org:<org_id>")


class FakeDreamingOrchestrator:
    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.running = True

    @property
    def health(self):
        return {"running": self.running, "interval_seconds": self.interval_seconds}


class FakeDreamingEngine:
    def __init__(self):
        self._dreaming_orchestrators = {}
        self.start_calls = []
        self.stop_calls = []

    async def start_dreaming(self, scope_id, user_id, *, config):
        self.start_calls.append((scope_id, user_id, config))
        orchestrator = FakeDreamingOrchestrator(config.interval_seconds)
        self._dreaming_orchestrators[(scope_id, user_id)] = orchestrator
        return orchestrator

    async def stop_dreaming(self, scope_id=None, user_id=None):
        self.stop_calls.append((scope_id, user_id))
        for key, orchestrator in list(self._dreaming_orchestrators.items()):
            if (scope_id is None or key[0] == scope_id) and (user_id is None or key[1] == user_id):
                orchestrator.running = False
                self._dreaming_orchestrators.pop(key, None)


def test_memory_server_dreaming_endpoints_control_engine(monkeypatch):
    from jiuwen_memory.server import memory_server

    paths = {getattr(route, "path", "") for route in memory_server.app.routes}
    assert {"/dreaming/status", "/dreaming/start", "/dreaming/stop"} <= paths

    fake = FakeDreamingEngine()
    monkeypatch.setattr(memory_server, "memory_engine", fake)
    monkeypatch.setenv("DREAMING_ENABLED", "true")
    monkeypatch.setenv("DREAMING_INTERVAL_SECONDS", "180")
    start_request = memory_server.DreamingStartRequest(
        scope_id="scope-a",
        user_id="user-a",
        interval_seconds=90.0,
        min_session_rounds=2,
        max_sessions_per_sweep=3,
        max_compress_tokens=4096,
        max_items_per_session=4,
    )

    started = asyncio.run(memory_server.dreaming_start_endpoint(start_request))
    status = asyncio.run(memory_server.dreaming_status_endpoint())
    stopped = asyncio.run(memory_server.dreaming_stop_endpoint(memory_server.DreamingStopRequest()))

    assert started["status"] == "started"
    assert started["scope_id"] == "scope-a"
    assert started["user_id"] == "user-a"
    assert started["running"] is True
    assert status["configured_enabled"] is True
    assert status["orchestrators"][0]["scope_id"] == "scope-a"
    assert status["orchestrators"][0]["health"]["interval_seconds"] == 90.0
    assert stopped["status"] == "stopped"
    assert stopped["running"] is False
    assert fake.start_calls[0][2].min_session_rounds == 2
    assert fake.start_calls[0][2].max_sessions_per_sweep == 3
    assert fake.start_calls[0][2].max_compress_tokens == 4096
    assert fake.start_calls[0][2].max_items_per_session == 4
    assert fake.stop_calls == [(None, None)]


def test_memory_server_startup_starts_dreaming_from_env(monkeypatch):
    from jiuwen_memory.server import memory_server

    class FakeStartupEngine(FakeDreamingEngine):
        def __init__(self):
            super().__init__()
            self.registered = False
            self.configured = False

        async def register_store(self, *, kv_store, db_store, vector_store, embedding_model):
            del kv_store, db_store, vector_store, embedding_model
            self.registered = True

        def set_config(self, config):
            del config
            self.configured = True

    fake = FakeStartupEngine()
    monkeypatch.setattr(memory_server, "memory_engine", fake)
    monkeypatch.setattr(memory_server, "create_async_engine_from_env", lambda: object())
    monkeypatch.setattr(memory_server, "create_kv_store", lambda engine: object())
    monkeypatch.setattr(memory_server, "create_db_store", lambda engine: object())
    monkeypatch.setattr(memory_server, "create_vector_store", lambda: object())
    monkeypatch.setattr(memory_server, "APIEmbedding", lambda config: object())
    monkeypatch.setenv("DREAMING_ENABLED", "true")
    monkeypatch.setenv("DREAMING_SCOPE_ID", "scope-env")
    monkeypatch.setenv("DREAMING_USER_ID", "user-env")
    monkeypatch.setenv("DREAMING_INTERVAL_SECONDS", "240")
    monkeypatch.setenv("DREAMING_MIN_SESSION_ROUNDS", "5")
    monkeypatch.setenv("DREAMING_MAX_SESSIONS_PER_SWEEP", "6")
    monkeypatch.setenv("DREAMING_MAX_COMPRESS_TOKENS", "7000")
    monkeypatch.setenv("DREAMING_MAX_ITEMS_PER_SESSION", "8")

    asyncio.run(memory_server.startup_event())

    assert fake.registered is True
    assert fake.configured is True
    assert len(fake.start_calls) == 1
    scope_id, user_id, config = fake.start_calls[0]
    assert (scope_id, user_id) == ("scope-env", "user-env")
    assert config.enabled is True
    assert config.interval_seconds == 240
    assert config.min_session_rounds == 5
    assert config.max_sessions_per_sweep == 6
    assert config.max_compress_tokens == 7000
    assert config.max_items_per_session == 8


def test_memory_server_graph_routes_are_present_and_root_lists_them():
    from jiuwen_memory.server import memory_server

    paths = {getattr(route, "path", "") for route in memory_server.app.routes}
    assert {"/graph/status", "/graph/extract", "/graph/query"} <= paths

    root_payload = asyncio.run(memory_server.root())
    assert {
        "GET /graph/status",
        "POST /graph/extract",
        "POST /graph/query",
    } <= set(root_payload["endpoints"])


def test_memory_server_graph_natural_language_chain_query(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    monkeypatch.setattr(memory_server, "_GRAPH_DB_PATH", tmp_path / "graph_memory.sqlite3")
    text = (
        "Aurora 项目依赖 PaymentGateway。"
        "PaymentGateway 负责人是 Alice。"
        "PaymentGateway 回归测试失败导致 Aurora 发布延期。"
    )
    extract_request = memory_server.GraphExtractRequest(
        scope_id="scope-a",
        user_id="user-a",
        session_id="graph-session-a",
        text=text,
    )

    extracted = asyncio.run(memory_server.graph_extract_endpoint(extract_request))
    status = asyncio.run(memory_server.graph_status_endpoint())
    result = asyncio.run(
        memory_server.graph_query_endpoint(
            memory_server.GraphQueryRequest(
                scope_id="scope-a",
                user_id="user-a",
                query="Aurora 相关负责人是谁？为什么延期？",
                limit=20,
            )
        )
    )

    assert extracted["status"] == "success"
    assert extracted["session_id"] == "graph-session-a"
    assert extracted["entities_written"] >= 3
    assert extracted["relations_written"] >= 3
    assert status["entities"] >= 3
    assert status["relations"] >= 3
    relation_pairs = {(item["source"], item["relation_type"], item["target"]) for item in result["relations"]}
    assert ("Aurora", "depends_on", "PaymentGateway") in relation_pairs
    assert ("PaymentGateway", "owner", "Alice") in relation_pairs
    assert ("PaymentGateway", "caused_delay_of", "Aurora") in relation_pairs
    assert any(item["session_id"] == "graph-session-a" for item in result["relations"])
    assert any("PaymentGateway" in item for item in result["evidence"])
    assert {entity["name"] for entity in result["entities"]} >= {"Aurora", "PaymentGateway", "Alice"}
    assert {entity["name"] for entity in result["matching_entities"]} == {"Aurora"}
    assert result["session_id"] == "graph-session-a"
    assert result["session_ids"] == ["graph-session-a"]
    chain_paths = [chain["entities"] for chain in result["chains"]]
    assert ["Aurora", "PaymentGateway", "Alice"] in chain_paths
    assert any(
        [relation["relation_type"] for relation in chain["relations"]] == ["caused_delay_of", "owner"]
        for chain in result["chains"]
    )


def test_memory_server_graph_explicit_extract_deduplicates(tmp_path, monkeypatch):
    from jiuwen_memory.server import memory_server

    monkeypatch.setattr(memory_server, "_GRAPH_DB_PATH", tmp_path / "graph_memory.sqlite3")
    request = memory_server.GraphExtractRequest(
        scope_id="scope-b",
        user_id="user-b",
        session_id="graph-session-b",
        text="Aurora uses FastAPI",
        entities=[
            memory_server.GraphEntityRequest(name="Aurora", type="project"),
            memory_server.GraphEntityRequest(name="FastAPI", type="framework"),
        ],
        relations=[
            memory_server.GraphRelationRequest(
                source="Aurora",
                target="FastAPI",
                relation_type="uses",
                evidence="Aurora uses FastAPI",
            )
        ],
    )

    asyncio.run(memory_server.graph_extract_endpoint(request))
    asyncio.run(memory_server.graph_extract_endpoint(request))
    status = asyncio.run(memory_server.graph_status_endpoint())
    result = asyncio.run(
        memory_server.graph_query_endpoint(
            memory_server.GraphQueryRequest(scope_id="scope-b", user_id="user-b", query="Aurora 使用什么框架？")
        )
    )

    assert status["entities"] == 2
    assert status["relations"] == 1
    assert status["episodes"] == 1
    assert result["relations"][0]["relation_type"] == "uses"
    assert result["relations"][0]["evidence"] == "Aurora uses FastAPI"
    assert result["relations"][0]["session_id"] == "graph-session-b"
    assert result["session_id"] == "graph-session-b"
    assert result["chains"][0]["session_id"] == "graph-session-b"
    assert {entity["session_id"] for entity in result["entities"]} == {"graph-session-b"}
    assert {entity["evidence"] for entity in result["entities"]} == {"Aurora uses FastAPI"}
