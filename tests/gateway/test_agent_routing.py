"""Tests for gateway/agent_routing.py — inbound message -> agent_id resolver."""

import pytest
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.agent_routing import resolve_agent_id, _route_matches


class TestRouteMatches:
    def test_platform_match(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert _route_matches({"platform": "telegram"}, source) is True

    def test_platform_mismatch(self):
        source = SessionSource(platform=Platform.DISCORD, chat_id="123")
        assert _route_matches({"platform": "telegram"}, source) is False

    def test_chat_id_match(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert _route_matches({"chat_id": "123"}, source) is True

    def test_chat_id_mismatch(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert _route_matches({"chat_id": "456"}, source) is False

    def test_multiple_keys_all_match(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234",
            thread_id="42",
        )
        assert _route_matches(
            {"platform": "telegram", "chat_id": "-1001234", "thread_id": "42"},
            source,
        ) is True

    def test_multiple_keys_one_mismatch(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234",
            thread_id="42",
        )
        assert _route_matches(
            {"platform": "telegram", "chat_id": "-1001234", "thread_id": "99"},
            source,
        ) is False

    def test_empty_match_returns_false(self):
        """Empty match dict should not match everything — it's a guard rail."""
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert _route_matches({}, source) is False

    def test_unknown_match_key_returns_false(self):
        """Unknown keys are rejected to catch typos."""
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert _route_matches({"platfrom": "telegram"}, source) is False

    def test_missing_source_attribute_returns_false(self):
        """If source lacks the attribute, route doesn't match."""
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert _route_matches({"user_id": "456"}, source) is False

    def test_guild_id_match(self):
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            guild_id="T0ABC",
        )
        assert _route_matches({"guild_id": "T0ABC"}, source) is True

    def test_user_id_match(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            user_id="456",
        )
        assert _route_matches({"user_id": "456"}, source) is True

    def test_numeric_chat_id_coerced(self):
        """Match values are stringified for comparison."""
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345")
        assert _route_matches({"chat_id": 12345}, source) is True


class TestResolveAgentId:
    def test_no_routes_returns_default(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert resolve_agent_id(source, [], default="main") == "main"

    def test_no_routes_no_default_returns_none(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert resolve_agent_id(source, [], default=None) is None

    def test_first_match_wins(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234",
            thread_id="42",
        )
        routes = [
            {"match": {"platform": "telegram", "chat_id": "-1001234", "thread_id": "42"}, "agent": "coder"},
            {"match": {"platform": "telegram", "chat_id": "-1001234"}, "agent": "research"},
        ]
        assert resolve_agent_id(source, routes) == "coder"

    def test_falls_through_to_less_specific(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234",
            thread_id="99",
        )
        routes = [
            {"match": {"platform": "telegram", "chat_id": "-1001234", "thread_id": "42"}, "agent": "coder"},
            {"match": {"platform": "telegram", "chat_id": "-1001234"}, "agent": "research"},
        ]
        assert resolve_agent_id(source, routes) == "research"

    def test_no_match_returns_default(self):
        source = SessionSource(platform=Platform.DISCORD, chat_id="123")
        routes = [
            {"match": {"platform": "telegram"}, "agent": "coder"},
        ]
        assert resolve_agent_id(source, routes, default="main") == "main"

    def test_routes_none_treated_as_empty(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert resolve_agent_id(source, None, default="main") == "main"

    def test_invalid_route_entries_skipped(self):
        """Non-dict routes, missing agent, or missing match are skipped."""
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        routes = [
            "not-a-dict",
            {"match": {"platform": "telegram"}},  # missing agent
            {"agent": "coder"},  # missing match
            {"match": {"platform": "telegram"}, "agent": "coder"},
        ]
        assert resolve_agent_id(source, routes) == "coder"

    def test_empty_agent_string_skipped(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        routes = [
            {"match": {"platform": "telegram"}, "agent": "  "},
            {"match": {"platform": "telegram"}, "agent": "coder"},
        ]
        assert resolve_agent_id(source, routes) == "coder"

    def test_agent_id_stripped(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        routes = [
            {"match": {"platform": "telegram"}, "agent": "  coder  "},
        ]
        assert resolve_agent_id(source, routes) == "coder"

    def test_slack_workspace_route(self):
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            guild_id="T0ABC123",
        )
        routes = [
            {"match": {"platform": "slack", "guild_id": "T0ABC123"}, "agent": "coder"},
        ]
        assert resolve_agent_id(source, routes) == "coder"

    def test_user_id_route(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            user_id="999",
        )
        routes = [
            {"match": {"platform": "telegram", "user_id": "999"}, "agent": "vip"},
            {"match": {"platform": "telegram"}, "agent": "standard"},
        ]
        assert resolve_agent_id(source, routes) == "vip"

    def test_declaration_order_matters(self):
        """More specific routes must be declared before general ones."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234",
            thread_id="42",
        )
        # Wrong order: general first
        bad_routes = [
            {"match": {"platform": "telegram", "chat_id": "-1001234"}, "agent": "research"},
            {"match": {"platform": "telegram", "chat_id": "-1001234", "thread_id": "42"}, "agent": "coder"},
        ]
        assert resolve_agent_id(source, bad_routes) == "research"
