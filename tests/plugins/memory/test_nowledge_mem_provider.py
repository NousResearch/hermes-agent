import importlib
import json
import os


provider_module = importlib.import_module("plugins.memory.nowledge-mem.provider")
loader_module = importlib.import_module("plugins.memory")

NowledgeMemProvider = provider_module.NowledgeMemProvider


class FakeClient:
    available = True

    def __init__(self, timeout=30, space=None):
        self.timeout = timeout
        self.space = space
        self.saved = []

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


def test_resolve_space_non_string_falls_through_to_identity():
    resolved = NowledgeMemProvider._resolve_space(
        {
            "space": None,
            "space_by_identity": {"research": "Research Agent"},
        },
        {"agent_identity": "research"},
    )
    assert resolved == "Research Agent"
