import importlib
import json
import os


client_module = importlib.import_module("plugins.memory.nowledge-mem.client")
provider_module = importlib.import_module("plugins.memory.nowledge-mem.provider")
loader_module = importlib.import_module("plugins.memory")

NowledgeMemProvider = provider_module.NowledgeMemProvider


class FakeClient:
    available = True

    def __init__(self, timeout=30, space=None):
        self.timeout = timeout
        self.space = space
        self.saved = []
        self.import_calls = []
        self.append_calls = []
        self.fail_import = False
        self.fail_append = False

    @staticmethod
    def is_available():
        return FakeClient.available

    def health(self):
        return True

    def working_memory(self):
        return {"content": "Focus: ship plugin quality"}

    def search(self, query="", *, limit=10, filter_labels=None, mode=None):
        return {
            "memories": [
                {
                    "title": "Shipping focus",
                    "content": f"Query was {query}",
                    "labels": ["release"],
                    "score": 0.92,
                }
            ]
        }

    def save(self, content, **kwargs):
        self.saved.append({"content": content, **kwargs})
        return {"id": "mem_123", "content": content}

    def update(self, memory_id, **kwargs):
        return {"memory_id": memory_id, **kwargs}

    def delete(self, memory_id):
        return {"deleted": [memory_id]}

    def delete_many(self, memory_ids):
        return {"deleted": memory_ids}

    def thread_search(self, query="", *, limit=10, source=None):
        return {"threads": [{"id": "thr_1", "query": query, "source": source}]}

    def thread_messages(self, thread_id, *, limit=50, offset=0):
        return {"thread_id": thread_id, "limit": limit, "offset": offset}

    def import_thread(self, thread_id, messages, *, title=None, source="hermes"):
        if self.fail_import:
            raise RuntimeError("import failed")
        self.import_calls.append(
            {
                "thread_id": thread_id,
                "messages": messages,
                "title": title,
                "source": source,
            }
        )
        return {"success": True, "thread_id": thread_id}

    def append_thread(self, thread_id, messages):
        if self.fail_append:
            raise RuntimeError("append failed")
        self.append_calls.append({"thread_id": thread_id, "messages": messages})
        return {"success": True, "thread_id": thread_id}


def test_load_memory_provider_nowledge_mem(monkeypatch):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = loader_module.load_memory_provider("nowledge-mem")
    assert provider is not None
    assert provider.name == "nowledge-mem"
    assert provider.is_available() is True


def test_initialize_and_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    prompt = provider.system_prompt_block()
    assert "Nowledge Mem" in prompt
    assert "ship plugin quality" in prompt


def test_initialize_falls_back_on_invalid_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    (tmp_path / "nowledge-mem.json").write_text(json.dumps({"timeout": "abc"}))

    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    assert provider._client.timeout == 30


def test_initialize_passes_resolved_space_to_client(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    (tmp_path / "nowledge-mem.json").write_text(
        json.dumps(
            {
                "space_by_identity": {
                    "research": "Research Agent",
                }
            }
        )
    )

    provider = NowledgeMemProvider()
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="research",
    )

    assert provider._client.space == "Research Agent"
    assert "## Active Space" in provider.system_prompt_block()


def test_tool_schemas_are_available_before_initialize(monkeypatch):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()

    schemas = provider.get_tool_schemas()
    assert {schema["name"] for schema in schemas} == {
        "nmem_search",
        "nmem_save",
        "nmem_update",
        "nmem_delete",
        "nmem_thread_search",
        "nmem_thread_messages",
    }


def test_prefetch_formats_results(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    result = provider.prefetch("what are we shipping?")
    assert "Recalled from Nowledge Mem" in result
    assert "Shipping focus" in result


def test_handle_tool_call_search(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    payload = json.loads(provider.handle_tool_call("nmem_search", {"query": "release"}))
    assert payload["success"] is True
    assert payload["memories"][0]["title"] == "Shipping focus"


def test_on_memory_write_mirrors_user_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    provider.on_memory_write("add", "user", "User prefers concise reviews")
    provider.shutdown()

    assert provider._client.saved[0]["content"] == "User prefers concise reviews"


def test_handle_tool_call_accepts_list_args(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    save_payload = json.loads(
        provider.handle_tool_call(
            "nmem_save",
            {
                "content": "Hermes plugin integration test passed",
                "labels": ["hermes", "integration"],
            },
        )
    )
    assert save_payload["id"] == "mem_123"
    assert provider._client.saved[-1]["labels"] == ["hermes", "integration"]

    search_payload = json.loads(
        provider.handle_tool_call(
            "nmem_search",
            {
                "query": "OpenClaw",
                "filter_labels": ["release", "plugin"],
            },
        )
    )
    assert search_payload["success"] is True
    assert search_payload["memories"][0]["title"] == "Shipping focus"


def test_handle_tool_call_returns_structured_error(monkeypatch, tmp_path):
    class FailingClient(FakeClient):
        def save(self, content, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(provider_module, "NowledgeMemClient", FailingClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    payload = json.loads(provider.handle_tool_call("nmem_save", {"content": "x"}))
    assert payload["success"] is False
    assert "nmem_save failed" in payload["error"]


def test_handle_tool_call_falls_back_without_registry_helpers(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    monkeypatch.setattr(provider_module, "_registry_tool_error", None)
    monkeypatch.setattr(provider_module, "_registry_tool_result", None)

    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    payload = json.loads(provider.handle_tool_call("nmem_search", {"query": "release"}))
    assert payload["success"] is True
    assert payload["memories"][0]["title"] == "Shipping focus"


def test_resolve_space_explicit_empty_beats_environment():
    previous = os.environ.get("NMEM_SPACE")
    os.environ["NMEM_SPACE"] = "Env Space"
    try:
        resolved = NowledgeMemProvider._resolve_space({"space": ""}, {})
        assert resolved == ""
    finally:
        if previous is None:
            os.environ.pop("NMEM_SPACE", None)
        else:
            os.environ["NMEM_SPACE"] = previous


def test_resolve_space_configured_beats_environment():
    previous = os.environ.get("NMEM_SPACE")
    os.environ["NMEM_SPACE"] = "Env Space"
    try:
        resolved = NowledgeMemProvider._resolve_space(
            {"space": "Research Agent"},
            {"agent_identity": "research"},
        )
        assert resolved == "Research Agent"
    finally:
        if previous is None:
            os.environ.pop("NMEM_SPACE", None)
        else:
            os.environ["NMEM_SPACE"] = previous


def test_resolve_space_template_falls_back_when_no_mapping():
    resolved = NowledgeMemProvider._resolve_space(
        {"space_template": "agent-{identity}"},
        {"agent_identity": "ops"},
    )
    assert resolved == "agent-ops"


def test_resolve_space_non_string_falls_through_to_identity():
    resolved = NowledgeMemProvider._resolve_space(
        {
            "space": None,
            "space_by_identity": {"research": "Research Agent"},
        },
        {"agent_identity": "research"},
    )
    assert resolved == "Research Agent"


def test_resolve_space_missing_identity_does_not_synthesize_space(monkeypatch):
    monkeypatch.delenv("NMEM_SPACE", raising=False)
    monkeypatch.delenv("NMEM_SPACE_ID", raising=False)

    resolved = NowledgeMemProvider._resolve_space(
        {
            "space_by_identity": {"default": "Default Agent"},
            "space_template": "agent-{identity}",
        },
        {},
    )

    assert resolved is None


def test_cli_explicit_empty_space_clears_inherited_environment(monkeypatch):
    captured = {}

    def _fake_run(_cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})

        class _Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        return _Result()

    monkeypatch.setattr(client_module.subprocess, "run", _fake_run)
    monkeypatch.setenv("NMEM_SPACE", "Env Space")
    monkeypatch.setenv("NMEM_SPACE_ID", "Env Space")

    client = client_module.NowledgeMemClient(space="")
    client.working_memory()

    assert "NMEM_SPACE" not in captured["env"]
    assert "NMEM_SPACE_ID" not in captured["env"]


def test_cli_explicit_space_sets_subprocess_environment(monkeypatch):
    captured = {}

    def _fake_run(_cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})

        class _Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        return _Result()

    monkeypatch.setattr(client_module.subprocess, "run", _fake_run)
    monkeypatch.setenv("NMEM_SPACE", "Env Space")
    monkeypatch.setenv("NMEM_SPACE_ID", "Env Space")

    client = client_module.NowledgeMemClient(space="Research Agent")
    client.working_memory()

    assert captured["env"]["NMEM_SPACE"] == "Research Agent"
    assert captured["env"]["NMEM_SPACE_ID"] == "Research Agent"


def test_on_session_end_imports_clean_messages_then_appends_delta(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    first_messages = [
        {"role": "system", "content": "skip"},
        {"role": "user", "content": [{"type": "text", "text": " hello "}]},
        {"role": "assistant", "content": "hi there"},
        {"role": "tool", "content": "skip"},
        {"role": "assistant", "content": {"text": " more detail "}},
    ]

    provider.on_session_end(first_messages)

    assert provider._client.import_calls == [
        {
            "thread_id": "session-1",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "assistant", "content": "more detail"},
            ],
            "title": "hello",
            "source": "hermes",
        }
    ]
    assert provider._client.append_calls == []

    next_messages = first_messages + [{"role": "user", "content": "next step"}]
    provider.on_session_end(next_messages)

    assert provider._client.append_calls == [
        {
            "thread_id": "session-1",
            "messages": [{"role": "user", "content": "next step"}],
        }
    ]


def test_on_session_end_failed_import_does_not_advance_count(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-2", hermes_home=str(tmp_path), platform="cli")
    provider._client.fail_import = True

    provider.on_session_end(
        [
            "bad",
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": {"content": "world"}},
        ]
    )

    assert provider._saved_message_count == 0
    assert provider._client.import_calls == []


def test_on_session_end_failed_append_does_not_advance_count(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_module, "NowledgeMemClient", FakeClient)
    provider = NowledgeMemProvider()
    provider.initialize("session-3", hermes_home=str(tmp_path), platform="cli")
    provider._saved_message_count = 2
    provider._client.fail_append = True

    provider.on_session_end(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "next step"},
        ]
    )

    assert provider._saved_message_count == 2
    assert provider._client.append_calls == []


def test_client_thread_import_posts_payload_without_subprocess_argv(monkeypatch):
    captured = {}

    def _bad_run(*_args, **_kwargs):
        raise AssertionError("thread import must not send transcript through argv")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"success": true}'

    def _fake_urlopen(request, **_kwargs):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        return _Response()

    monkeypatch.setattr(client_module.subprocess, "run", _bad_run)
    monkeypatch.setattr(client_module.urlrequest, "urlopen", _fake_urlopen)
    monkeypatch.setenv("NMEM_API_URL", "http://mem.test")
    monkeypatch.setenv("NMEM_API_KEY", "")

    client = client_module.NowledgeMemClient(space="Research Agent")
    result = client.import_thread(
        "hermes-session-1",
        [{"role": "user", "content": "x" * 100_000}],
        title="Long session",
    )

    payload = json.loads(captured["body"])
    assert result["success"] is True
    assert captured["url"] == "http://mem.test/threads/import"
    assert payload["thread_id"] == "hermes-session-1"
    assert payload["metadata"]["space_id"] == "Research Agent"
    assert payload["messages"][0]["content"] == "x" * 100_000


def test_client_thread_append_posts_payload_without_subprocess_argv(monkeypatch):
    captured = {}

    def _bad_run(*_args, **_kwargs):
        raise AssertionError("thread append must not send transcript through argv")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"success": true, "messages_added": 1}'

    def _fake_urlopen(request, **_kwargs):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        return _Response()

    monkeypatch.setattr(client_module.subprocess, "run", _bad_run)
    monkeypatch.setattr(client_module.urlrequest, "urlopen", _fake_urlopen)
    monkeypatch.setenv("NMEM_API_URL", "http://mem.test")
    monkeypatch.setenv("NMEM_API_KEY", "")

    client = client_module.NowledgeMemClient()
    result = client.append_thread(
        "hermes/session 1",
        [{"role": "assistant", "content": "y" * 100_000}],
    )

    payload = json.loads(captured["body"])
    assert result["messages_added"] == 1
    assert captured["url"] == "http://mem.test/threads/hermes%2Fsession%201/append"
    assert payload["messages"][0]["content"] == "y" * 100_000
    assert "space_id" not in payload


def test_client_thread_append_includes_space_when_configured(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"success": true, "messages_added": 1}'

    def _fake_urlopen(request, **_kwargs):
        captured["body"] = request.data.decode("utf-8")
        return _Response()

    monkeypatch.setattr(
        client_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("thread append must not send transcript through argv")
        ),
    )
    monkeypatch.setattr(client_module.urlrequest, "urlopen", _fake_urlopen)
    monkeypatch.setenv("NMEM_API_URL", "http://mem.test")
    monkeypatch.setenv("NMEM_API_KEY", "")

    client = client_module.NowledgeMemClient(space="Research Agent")
    result = client.append_thread(
        "hermes/session 1",
        [{"role": "assistant", "content": "hello"}],
    )

    payload = json.loads(captured["body"])
    assert result["messages_added"] == 1
    assert payload["space_id"] == "Research Agent"


def test_retry_urls_strip_remote_api_without_credential_query_params():
    urls = client_module.NowledgeMemClient._retry_urls(
        "https://mem.example.com/remote-api/threads/import?existing=1"
    )

    assert urls == ["https://mem.example.com/threads/import?existing=1"]
    assert all("nmem_api_key" not in url.lower() for url in urls)


def test_api_post_retries_remote_api_path_with_auth_headers_only(monkeypatch):
    client = client_module.NowledgeMemClient()
    seen = []

    monkeypatch.setattr(client, "_api_url", lambda: "https://mem.example.com/remote-api")
    monkeypatch.setattr(client, "_api_key", lambda: "secret-token")

    def _fake_request_json(url, body, headers):
        seen.append((url, body, headers.copy()))
        if len(seen) == 1:
            raise RuntimeError("proxy path failed")
        return {"success": True}

    monkeypatch.setattr(client, "_request_json", _fake_request_json)

    assert client._api_post("/threads/import", {"messages": []}) == {"success": True}
    assert [url for url, _body, _headers in seen] == [
        "https://mem.example.com/remote-api/threads/import",
        "https://mem.example.com/threads/import",
    ]
    assert all("nmem_api_key" not in url.lower() for url, _body, _headers in seen)
    assert all(headers["Authorization"] == "Bearer secret-token" for _url, _body, headers in seen)
    assert all(headers["X-NMEM-API-Key"] == "secret-token" for _url, _body, headers in seen)
