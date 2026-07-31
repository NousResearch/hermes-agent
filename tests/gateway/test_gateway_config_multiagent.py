"""Regression tests for multi-agent config parsing in GatewayConfig.

Covers the defensive branches added by the single-gateway-multi-agent PR:
malformed ``agents`` / ``routes`` / ``default_agent`` values must degrade to
safe defaults rather than mis-routing every inbound message.

``GatewayConfig.from_dict`` is the config-validation chokepoint for the whole
feature, so each malformed-input branch is exercised directly.

``TestNestedGatewayAgentsLoad`` covers the ``load_gateway_config()`` YAML
plumbing above it: the documented multi-agent shape is nested
``gateway.agents`` (website/docs/user-guide/messaging/buzz.md), which must
populate the registry just like top-level ``agents`` does.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

from gateway.config import GatewayConfig


class TestMultiAgentAgentsField:
    def test_agents_non_dict_becomes_empty_dict(self):
        # A list where a mapping is expected must not leak through.
        cfg = GatewayConfig.from_dict({"agents": ["research", "main"]})
        assert cfg.agents == {}

    def test_agents_string_becomes_empty_dict(self):
        cfg = GatewayConfig.from_dict({"agents": "research"})
        assert cfg.agents == {}

    def test_agents_missing_defaults_to_empty_dict(self):
        cfg = GatewayConfig.from_dict({})
        assert cfg.agents == {}


class TestMultiAgentRoutesField:
    def test_routes_non_list_becomes_empty_list(self):
        # A dict (non-list) where a list is expected degrades to [].
        cfg = GatewayConfig.from_dict({"routes": {"match": "x"}})
        assert cfg.routes == []

    def test_routes_string_becomes_empty_list(self):
        cfg = GatewayConfig.from_dict({"routes": "research"})
        assert cfg.routes == []

    def test_non_dict_route_entries_filtered_out(self):
        # Only dict entries survive; scalars/None/lists inside the list drop.
        cfg = GatewayConfig.from_dict(
            {
                "routes": [
                    {"match": "keyword: research", "agent": "research"},
                    "not-a-dict",
                    None,
                    ["also", "not", "a", "dict"],
                    {"match": "keyword: ops", "agent": "ops"},
                ]
            }
        )
        assert cfg.routes == [
            {"match": "keyword: research", "agent": "research"},
            {"match": "keyword: ops", "agent": "ops"},
        ]

    def test_routes_missing_defaults_to_empty_list(self):
        cfg = GatewayConfig.from_dict({})
        assert cfg.routes == []


class TestMultiAgentDefaultAgentField:
    def test_blank_default_agent_falls_back_to_main(self):
        cfg = GatewayConfig.from_dict({"default_agent": "   "})
        assert cfg.default_agent == "main"

    def test_empty_default_agent_falls_back_to_main(self):
        cfg = GatewayConfig.from_dict({"default_agent": ""})
        assert cfg.default_agent == "main"

    def test_non_str_default_agent_falls_back_to_main(self):
        cfg = GatewayConfig.from_dict({"default_agent": 123})
        assert cfg.default_agent == "main"

    def test_missing_default_agent_is_main(self):
        cfg = GatewayConfig.from_dict({})
        assert cfg.default_agent == "main"

    def test_valid_default_agent_is_stripped(self):
        cfg = GatewayConfig.from_dict({"default_agent": "  research  "})
        assert cfg.default_agent == "research"


class TestMultiAgentHappyPath:
    def test_wellformed_agents_routes_default_parse(self):
        data = {
            "agents": {
                "main": {"model": "anthropic/claude-opus-4.8"},
                "research": {"model": "anthropic/claude-opus-4.8", "toolset": "web"},
            },
            "routes": [
                {"match": "keyword: research", "agent": "research"},
                {"match": "channel: 12345", "agent": "main"},
            ],
            "default_agent": "research",
        }
        cfg = GatewayConfig.from_dict(data)

        assert cfg.agents == {
            "main": {"model": "anthropic/claude-opus-4.8"},
            "research": {"model": "anthropic/claude-opus-4.8", "toolset": "web"},
        }
        assert cfg.routes == [
            {"match": "keyword: research", "agent": "research"},
            {"match": "channel: 12345", "agent": "main"},
        ]
        assert cfg.default_agent == "research"

    def test_happy_path_survives_to_dict_round_trip(self):
        data = {
            "agents": {"research": {"model": "m"}},
            "routes": [{"match": "keyword: x", "agent": "research"}],
            "default_agent": "research",
        }
        restored = GatewayConfig.from_dict(GatewayConfig.from_dict(data).to_dict())
        assert restored.agents == {"research": {"model": "m"}}
        assert restored.routes == [{"match": "keyword: x", "agent": "research"}]
        assert restored.default_agent == "research"


def _load_with_yaml_dict(yaml_dict: dict):
    """Patch the filesystem so load_gateway_config() sees *yaml_dict* as
    config.yaml (same harness as tests/test_gateway_streaming_nested_config.py)."""
    from gateway.config import load_gateway_config

    fake_home = Path("/tmp/fake_hermes_home_agents_nested")

    def fake_exists(self):
        return str(self).endswith("config.yaml")

    with patch("gateway.config.get_hermes_home", return_value=fake_home), \
         patch.object(Path, "exists", fake_exists), \
         patch("builtins.open", create=True) as mock_file:
        mock_file.return_value.__enter__ = lambda s: s
        mock_file.return_value.__exit__ = MagicMock(return_value=False)
        with patch("yaml.safe_load", return_value=yaml_dict):
            return load_gateway_config()


class TestNestedGatewayAgentsLoad:
    """Nested ``gateway.agents`` must reach the registry (sweeper finding on
    #71686): the docs show the ``gateway.agents`` shape, so a config written
    that way must not silently produce an empty agent registry."""

    _AGENTS = {
        "chip": {"home_dir": "~/.hermes/agents/chip",
                 "buzz": {"nsec_env": "CHIP_BUZZ_NSEC"}},
        "scout": {"home_dir": "~/.hermes/agents/scout",
                  "buzz": {"nsec_env": "SCOUT_BUZZ_NSEC"}},
    }

    def test_nested_gateway_agents_populates_registry(self):
        # The exact documented shape: agents ONLY under gateway:.
        cfg = _load_with_yaml_dict({"gateway": {"agents": self._AGENTS}})
        assert cfg.agents == self._AGENTS

    def test_top_level_agents_still_load(self):
        cfg = _load_with_yaml_dict({"agents": self._AGENTS})
        assert cfg.agents == self._AGENTS

    def test_top_level_wins_when_both_present(self):
        # Matches the repo-wide precedence (streaming, reset_triggers, ...):
        # top-level key wins, nested gateway.* is the fallback.
        cfg = _load_with_yaml_dict({
            "agents": {"chip": {"model": "top-level"}},
            "gateway": {"agents": {"chip": {"model": "nested"}}},
        })
        assert cfg.agents == {"chip": {"model": "top-level"}}

    def test_neither_shape_means_empty_registry(self):
        cfg = _load_with_yaml_dict({"gateway": {}})
        assert cfg.agents == {}
