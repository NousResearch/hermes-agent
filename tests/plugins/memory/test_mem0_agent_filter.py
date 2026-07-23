"""Tests for filter_by_agent_id option in _read_filters()."""

import json

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider, _load_config


class TestReadFiltersDefault:
    """Default behavior when filter_by_agent_id is not set or False."""

    def test_read_filters_user_id_only_by_default(self, monkeypatch):
        """Without filter_by_agent_id, _read_filters() returns only user_id."""
        provider = Mem0MemoryProvider()
        monkeypatch.setenv("MEM0_USER_ID", "soji-client")
        monkeypatch.setenv("MEM0_AGENT_ID", "soji-hisyo")
        provider.initialize("test-session")

        filters = provider._read_filters()

        assert filters == {"user_id": "soji-client"}
        assert "agent_id" not in filters

    def test_read_filters_user_id_only_when_false(self, monkeypatch, tmp_path):
        """filter_by_agent_id: false → user_id only."""
        mem0_json = tmp_path / "mem0.json"
        mem0_json.write_text(json.dumps({
            "user_id": "soji-client",
            "agent_id": "soji-hisyo",
            "filter_by_agent_id": False,
        }))

        monkeypatch.setattr(
            "plugins.memory.mem0._load_config",
            lambda: json.loads(mem0_json.read_text()),
        )

        provider = Mem0MemoryProvider()
        provider.initialize("test-session")

        filters = provider._read_filters()
        assert filters == {"user_id": "soji-client"}
        assert "agent_id" not in filters


class TestReadFiltersWithAgentId:
    """Behavior when filter_by_agent_id is True."""

    def test_read_filters_includes_agent_id_when_true(self, monkeypatch):
        """Setting the normalized attribute adds agent_id to filters."""
        provider = Mem0MemoryProvider()
        monkeypatch.setenv("MEM0_USER_ID", "soji-client")
        monkeypatch.setenv("MEM0_AGENT_ID", "soji-hisyo")
        provider.initialize("test-session")
        provider._filter_by_agent_id = True

        filters = provider._read_filters()

        assert filters == {"user_id": "soji-client", "agent_id": "soji-hisyo"}

    def test_agent_id_from_mem0_json(self, monkeypatch, tmp_path):
        """mem0.json with filter_by_agent_id: true includes agent_id in filters."""
        mem0_json = tmp_path / "mem0.json"
        mem0_json.write_text(json.dumps({
            "user_id": "soji-client",
            "agent_id": "soji-eigyo",
            "filter_by_agent_id": True,
        }))

        monkeypatch.setattr(
            "plugins.memory.mem0._load_config",
            lambda: json.loads(mem0_json.read_text()),
        )

        provider = Mem0MemoryProvider()
        provider.initialize("test-session")

        filters = provider._read_filters()
        assert filters == {"user_id": "soji-client", "agent_id": "soji-eigyo"}


class TestConfigLoading:
    """_load_config() correctly reads filter_by_agent_id."""

    def test_load_config_defaults_filter_by_agent_id_false(self, monkeypatch):
        """No mem0.json → filter_by_agent_id defaults to False."""
        monkeypatch.setenv("MEM0_USER_ID", "u1")
        config = _load_config()
        assert config.get("filter_by_agent_id") is False

    def test_load_config_reads_true_from_json(self, tmp_path, monkeypatch):
        """mem0.json with filter_by_agent_id: true → True is read."""
        mem0_json = tmp_path / "mem0.json"
        mem0_json.write_text(json.dumps({
            "user_id": "u1",
            "filter_by_agent_id": True,
        }))
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )
        config = _load_config()
        assert config.get("filter_by_agent_id") is True
