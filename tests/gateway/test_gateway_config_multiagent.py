"""Regression tests for multi-agent config parsing in GatewayConfig.

Covers the defensive branches added by the single-gateway-multi-agent PR:
malformed ``agents`` / ``routes`` / ``default_agent`` values must degrade to
safe defaults rather than mis-routing every inbound message.

``GatewayConfig.from_dict`` is the config-validation chokepoint for the whole
feature, so each malformed-input branch is exercised directly.
"""

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
