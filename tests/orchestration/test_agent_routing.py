"""Tests for agent_routing.py — AgentRoutingTable (CICS PCT model)."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.orchestration.agent_routing import AgentRoutingTable


# ── helpers ──────────────────────────────────────────────────────────


def _write_routing_config(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


SAMPLE_CONFIG = {
    "routing": [
        {"topic": "telegram:-100X:101", "agent": "coding-agent"},
        {"topic": "telegram:-100X:201", "agent": "cs-agent"},
        {"dm": "telegram:user123", "agent": "personal-agent"},
    ],
    "default_agent": "fallback-agent",
}


@pytest.fixture
def routing_table(tmp_path):
    """AgentRoutingTable backed by a temp YAML file."""
    config = tmp_path / "agent_routing.yaml"
    _write_routing_config(config, SAMPLE_CONFIG)
    return AgentRoutingTable(config_path=config, default_agent="fallback-agent")


# ── tests ────────────────────────────────────────────────────────────


class TestAgentRoutingTable:

    def test_exact_topic_match(self, routing_table):
        """Topic key with all three components matches first."""
        agent = routing_table.resolve("telegram", "-100X", topic_id="101")
        assert agent == "coding-agent"

    def test_other_topic_match(self, routing_table):
        agent = routing_table.resolve("telegram", "-100X", topic_id="201")
        assert agent == "cs-agent"

    def test_dm_fallback_when_topic_unmatched(self, routing_table):
        """When topic_id doesn't match any topic entry, fall through to dm."""
        agent = routing_table.resolve("telegram", "user123", topic_id="999")
        assert agent == "personal-agent"

    def test_default_fallback_when_nothing_matches(self, routing_table):
        agent = routing_table.resolve("discord", "guild789")
        assert agent == "fallback-agent"

    def test_empty_file_uses_default(self, tmp_path):
        config = tmp_path / "empty.yaml"
        config.write_text("")
        rt = AgentRoutingTable(config_path=config, default_agent="dflt")
        assert rt.resolve("telegram", "any") == "dflt"

    def test_hot_reload(self, tmp_path):
        """Changing the YAML file is reflected on next resolve()."""
        config = tmp_path / "live.yaml"
        _write_routing_config(config, {
            "routing": [{"dm": "telegram:u1", "agent": "old-agent"}],
            "default_agent": "dflt",
        })
        rt = AgentRoutingTable(config_path=config)
        assert rt.resolve("telegram", "u1") == "old-agent"

        # update the file
        _write_routing_config(config, {
            "routing": [{"dm": "telegram:u1", "agent": "new-agent"}],
            "default_agent": "dflt",
        })
        assert rt.resolve("telegram", "u1") == "new-agent"
