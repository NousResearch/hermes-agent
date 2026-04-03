import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from plugins.memory import load_memory_provider
from plugins.memory.cyber_memory import (
    CyberMemoryProvider,
    _load_config,
    _default_db_path,
)


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, **_ignored):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._alive = False

    def start(self):
        self._alive = True
        if self._target:
            self._target(*self._args, **self._kwargs)
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        return None


def test_load_provider_by_name():
    provider = load_memory_provider("cyber_memory")
    assert provider is not None
    assert provider.name == "cyber_memory"


def test_is_available_true_when_binary_and_mcp_present():
    provider = CyberMemoryProvider()
    with patch("plugins.memory.cyber_memory._load_config", return_value={"command": "cyber-memory"}), \
         patch("plugins.memory.cyber_memory._resolve_command", return_value="/usr/local/bin/cyber-memory"), \
         patch("plugins.memory.cyber_memory.cyber_memory_mcp_available", return_value=True):
        assert provider.is_available() is True


def test_is_available_false_without_mcp():
    provider = CyberMemoryProvider()
    with patch("plugins.memory.cyber_memory._load_config", return_value={"command": "cyber-memory"}), \
         patch("plugins.memory.cyber_memory._resolve_command", return_value="/usr/local/bin/cyber-memory"), \
         patch("plugins.memory.cyber_memory.cyber_memory_mcp_available", return_value=False):
        assert provider.is_available() is False


def test_save_and_load_config_roundtrip(tmp_path):
    provider = CyberMemoryProvider()
    provider.save_config(
        {
            "command": "/usr/local/bin/cyber-memory",
            "db_path": str(tmp_path / "profile" / "db.sqlite3"),
        },
        str(tmp_path),
    )

    loaded = _load_config(str(tmp_path))
    assert loaded["command"] == "/usr/local/bin/cyber-memory"
    assert loaded["db_path"] == str(tmp_path / "profile" / "db.sqlite3")


def test_initialize_uses_profile_scoped_db_path(tmp_path):
    provider = CyberMemoryProvider()
    fake_client = MagicMock()
    fake_client.list_tools.return_value = [
        "memory_store", "memory_recall", "memory_search", "memory_relate",
        "memory_graph", "memory_update", "memory_forget", "memory_stats",
    ]

    with patch("plugins.memory.cyber_memory._resolve_command", return_value="/usr/local/bin/cyber-memory"), \
         patch("plugins.memory.cyber_memory.CyberMemoryClient", return_value=fake_client):
        provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    expected_db = str(Path(_default_db_path(tmp_path)).expanduser())
    start_args = fake_client.start.call_args[0]
    assert start_args[0] == "/usr/local/bin/cyber-memory"
    assert start_args[2]["CYBER_MEMORY_DB"] == expected_db
    assert provider._db_path == expected_db


def test_initialize_fails_if_backend_missing_required_tools(tmp_path):
    provider = CyberMemoryProvider()
    fake_client = MagicMock()
    fake_client.list_tools.return_value = ["memory_store", "memory_recall"]

    with patch("plugins.memory.cyber_memory._resolve_command", return_value="/usr/local/bin/cyber-memory"), \
         patch("plugins.memory.cyber_memory.CyberMemoryClient", return_value=fake_client):
        try:
            provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
            assert False, "expected initialize() to fail"
        except RuntimeError as exc:
            assert "missing required MCP tools" in str(exc)

    fake_client.close.assert_called_once()


def test_handle_tool_call_maps_to_backend_tool():
    provider = CyberMemoryProvider()
    provider._client = MagicMock()
    provider._client.call_tool.return_value = {"result": "ok"}

    result = json.loads(provider.handle_tool_call("cyber_memory_search", {"query": "oauth"}))
    provider._client.call_tool.assert_called_once_with("memory_search", {"query": "oauth"})
    assert result["result"] == "ok"


def test_queue_prefetch_formats_results():
    provider = CyberMemoryProvider()
    provider._client = MagicMock()
    provider._client.call_tool.return_value = {
        "results": [
            {"content": "User prefers terse answers", "score": 0.92},
            {"content": "Project uses uv", "score": 0.75},
        ]
    }

    with patch("plugins.memory.cyber_memory.threading.Thread", _ImmediateThread):
        provider.queue_prefetch("what does the user prefer?")

    prefetched = provider.prefetch("what does the user prefer?")
    assert "Cyber Memory" in prefetched
    assert "terse answers" in prefetched


def test_sync_turn_stores_conversation():
    provider = CyberMemoryProvider()
    provider._client = MagicMock()
    provider._client.call_tool.return_value = {"result": "stored"}

    with patch("plugins.memory.cyber_memory.threading.Thread", _ImmediateThread):
        provider.sync_turn("remember this preference", "Got it, I will.")

    provider._client.call_tool.assert_called_once()
    tool_name, payload = provider._client.call_tool.call_args[0]
    assert tool_name == "memory_store"
    assert "User:" in payload["content"]
    assert payload["kind"] == "episodic"


def test_on_memory_write_mirrors_to_backend():
    provider = CyberMemoryProvider()
    provider._client = MagicMock()
    provider._client.call_tool.return_value = {"result": "stored"}

    with patch("plugins.memory.cyber_memory.threading.Thread", _ImmediateThread):
        provider.on_memory_write("add", "user", "Timezone: America/New_York")

    tool_name, payload = provider._client.call_tool.call_args[0]
    assert tool_name == "memory_store"
    assert payload["source"] == "hermes_user"
    assert "user" in payload["tags"]


def test_shutdown_closes_client():
    provider = CyberMemoryProvider()
    client = MagicMock()
    provider._client = client
    provider.shutdown()
    client.close.assert_called_once()
