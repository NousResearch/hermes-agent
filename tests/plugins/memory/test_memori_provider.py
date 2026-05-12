import json
import sys
import types
from unittest.mock import patch

import pytest

from plugins.memory.memori import (
    MemoriMemoryProvider,
    _load_config,
    register,
)
from plugins.memory.memori.client import MemoriAgentClient
from plugins.memory.memori.tools import MEMORI_RECALL_SCHEMA


class FakeMemoriSdk:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.config = types.SimpleNamespace(request_secs_timeout=None)
        self.agent = types.SimpleNamespace(default_api=FakeDefaultApi())
        self.capture_calls = []
        self.recall_calls = []
        self.summary_calls = []
        self.feedback_calls = []

    def attribution(self, entity_id, process_id):
        self.entity_id = entity_id
        self.process_id = process_id
        return self

    def capture_agent_turn(self, **kwargs):
        self.capture_calls.append(kwargs)

    def agent_recall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return {"memories": [{"content": "User prefers concise replies"}]}

    def agent_recall_summary(self, **kwargs):
        self.summary_calls.append(kwargs)
        return {"summary": "Project status"}

    def agent_feedback(self, content):
        self.feedback_calls.append(content)


class FakeDefaultApi:
    def __init__(self):
        self.posts = []

    def get(self, path):
        return {"path": path, "quota": 42}

    def post(self, path, payload):
        self.posts.append((path, payload))
        return {"path": path, "payload": payload}


@pytest.fixture
def fake_memori_module(monkeypatch):
    module = types.ModuleType("memori")
    module.Memori = FakeMemoriSdk
    monkeypatch.setitem(sys.modules, "memori", module)
    return module


def test_provider_imports_without_memori_sdk_installed():
    provider = MemoriMemoryProvider()

    assert provider.name == "memori"


def test_recall_schema_uses_python_snake_case_parameters():
    properties = MEMORI_RECALL_SCHEMA["parameters"]["properties"]

    assert "project_id" in properties
    assert "session_id" in properties
    assert "date_start" in properties
    assert "date_end" in properties
    assert "projectId" not in properties
    assert "sessionId" not in properties
    assert "dateStart" not in properties
    assert "dateEnd" not in properties


def test_client_reports_missing_memori_dependency():
    def fake_import(name, *args, **kwargs):
        if name == "memori":
            raise ModuleNotFoundError("No module named 'memori'", name="memori")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="pip install memori"):
            MemoriAgentClient(
                api_key="test-key",
                entity_id="user-123",
                project_id="project-123",
            )


def test_load_config_merges_env_and_file(monkeypatch, tmp_path):
    (tmp_path / "memori.json").write_text(
        json.dumps({"entity_id": "file-user", "project_id": "file-project"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")
    monkeypatch.setenv("MEMORI_PROJECT_ID", "env-project")

    config = _load_config(tmp_path)

    assert config is not None
    assert config.api_key == "test-key"
    assert config.entity_id == "file-user"
    assert config.project_id == "env-project"


def test_load_config_accepts_legacy_camel_case_file(monkeypatch, tmp_path):
    (tmp_path / "memori.json").write_text(
        json.dumps({"entityId": "file-user", "projectId": "file-project"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")

    config = _load_config(tmp_path)

    assert config is not None
    assert config.entity_id == "file-user"
    assert config.project_id == "file-project"


def test_load_config_leaves_project_unset_without_user_value(monkeypatch, tmp_path):
    (tmp_path / "memori.json").write_text(
        json.dumps({"entity_id": "file-user"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")

    config = _load_config(tmp_path)

    assert config is not None
    assert config.project_id is None


def test_is_available_requires_memori_sdk(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "memori", raising=False)
    (tmp_path / "memori.json").write_text(
        json.dumps({"entity_id": "file-user"}),
        encoding="utf-8",
    )
    provider = MemoriMemoryProvider()

    with patch("importlib.util.find_spec", return_value=None):
        assert provider.is_available() is False


def test_save_config_writes_profile_scoped_json(tmp_path):
    (tmp_path / "memori.json").write_text(
        json.dumps({"base_url": "https://api.example.test", "process_id": "agent-1"}),
        encoding="utf-8",
    )
    provider = MemoriMemoryProvider()

    provider.save_config(
        {"entity_id": "user-123", "project_id": "hermes-project"},
        str(tmp_path),
    )

    data = json.loads((tmp_path / "memori.json").read_text(encoding="utf-8"))
    assert data == {
        "base_url": "https://api.example.test",
        "entity_id": "user-123",
        "process_id": "agent-1",
        "project_id": "hermes-project",
    }


def test_handle_tool_call_applies_project_defaults(fake_memori_module, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")
    (tmp_path / "memori.json").write_text(
        json.dumps({"entity_id": "user-123", "project_id": "hermes-project"}),
        encoding="utf-8",
    )
    provider = MemoriMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    payload = json.loads(provider.handle_tool_call("memori_recall", {"query": "prefs"}))

    assert payload == {"memories": [{"content": "User prefers concise replies"}]}
    assert provider._client.memori.recall_calls == [  # noqa: SLF001
        {
            "query": "prefs",
            "date_start": None,
            "date_end": None,
            "project_id": "hermes-project",
            "session_id": None,
            "signal": None,
            "source": None,
        }
    ]


def test_initialize_uses_agent_workspace_when_project_is_unset(
    fake_memori_module,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")
    (tmp_path / "memori.json").write_text(
        json.dumps({"entity_id": "user-123"}),
        encoding="utf-8",
    )
    provider = MemoriMemoryProvider()
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_workspace="workspace-project",
    )

    provider.sync_turn("hello", "hi", session_id="session-1")
    provider.shutdown()

    assert provider._client.project_id == "workspace-project"  # noqa: SLF001
    assert provider._client.memori.capture_calls == [  # noqa: SLF001
        {
            "user_content": "hello",
            "assistant_content": "hi",
            "project_id": "workspace-project",
            "session_id": "session-1",
            "platform": "hermes",
        }
    ]


def test_sync_turn_passes_trace_to_client(fake_memori_module, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")
    (tmp_path / "memori.json").write_text(
        json.dumps({"entity_id": "user-123", "project_id": "hermes-project"}),
        encoding="utf-8",
    )
    provider = MemoriMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    trace = {
        "version": 1,
        "capture_policy": "full_raw_after_hermes_processing",
        "tool_calls": [{"name": "terminal", "result_content": "raw output"}],
    }

    provider.sync_turn("hello", "hi", session_id="session-1", trace=trace)
    provider.shutdown()

    assert provider._client.memori.capture_calls == [  # noqa: SLF001
        {
            "user_content": "hello",
            "assistant_content": "hi",
            "project_id": "hermes-project",
            "session_id": "session-1",
            "platform": "hermes",
            "trace": trace,
        }
    ]


def test_client_delegates_capture_and_feedback(fake_memori_module):
    client = MemoriAgentClient(
        api_key="test-key",
        entity_id="user-123",
        project_id="project-123",
        process_id="agent-1",
    )

    client.capture_turn(
        user_content="hello",
        assistant_content="hi",
        session_id="session-1",
        platform="cli",
    )
    assert client.feedback("useful recall") == {}

    memori = client.memori
    assert memori.capture_calls == [
        {
            "user_content": "hello",
            "assistant_content": "hi",
            "project_id": "project-123",
            "session_id": "session-1",
            "platform": "hermes",
        }
    ]
    assert memori.feedback_calls == ["useful recall"]


def test_memori_plugin_loads_end_to_end_through_memory_manager(
    fake_memori_module,
    tmp_path,
    monkeypatch,
):
    from agent.memory_manager import MemoryManager
    from plugins.memory import discover_memory_providers, load_memory_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEMORI_API_KEY", "test-key")
    (tmp_path / "memori.json").write_text(
        json.dumps({"entity_id": "user-123", "project_id": "hermes-project"}),
        encoding="utf-8",
    )

    discovered = {name: available for name, _, available in discover_memory_providers()}
    assert discovered["memori"] is True

    provider = load_memory_provider("memori")
    assert provider is not None
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    manager = MemoryManager()
    manager.add_provider(provider)

    assert manager.has_tool("memori_recall")
    payload = json.loads(manager.handle_tool_call("memori_recall", {"query": "prefs"}))

    assert payload == {"memories": [{"content": "User prefers concise replies"}]}
    assert provider._client.memori.recall_calls == [  # noqa: SLF001
        {
            "query": "prefs",
            "date_start": None,
            "date_end": None,
            "project_id": "hermes-project",
            "session_id": None,
            "signal": None,
            "source": None,
        }
    ]


def test_register_exposes_memori_skill():
    class FakeContext:
        def __init__(self):
            self.providers = []
            self.skills = []

        def register_memory_provider(self, provider):
            self.providers.append(provider)

        def register_skill(self, name, path, description=""):
            self.skills.append((name, path, description))

    ctx = FakeContext()

    register(ctx)

    skill_name, skill_path, skill_description = ctx.skills[0]
    assert len(ctx.providers) == 1
    assert skill_name == "memory"
    assert skill_path.name == "SKILL.md"
    assert skill_path.exists()
    assert skill_description == (
        "Use Memori recall, summaries, quota, signup, and feedback tools."
    )
