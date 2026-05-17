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


class TestMatrixRouting:
    """Matrix -> code agent routing (E2E scenario TC-01 / TC-02)."""

    def test_matrix_dm_routes_to_code(self):
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!u00jd7u1b1WqHly1:localhost",
            chat_type="dm",
            user_id="@testuser:localhost",
        )
        routes = [
            {"match": {"platform": "wecom"}, "agent": "wecom-agent"},
            {"match": {"platform": "matrix"}, "agent": "code"},
        ]
        assert resolve_agent_id(source, routes, default="main") == "code"

    def test_matrix_room_routes_to_code(self):
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="#test-room:localhost",
            chat_type="channel",
            user_id="@testuser:localhost",
        )
        routes = [
            {"match": {"platform": "matrix"}, "agent": "code"},
        ]
        assert resolve_agent_id(source, routes, default="main") == "code"

    def test_weixin_still_routes_to_main(self):
        """Regression: weixin must continue using default main agent."""
        source = SessionSource(
            platform=Platform.WEIXIN,
            chat_id="wx123456",
            chat_type="dm",
            user_id="wxuser_abc",
        )
        routes = [
            {"match": {"platform": "wecom"}, "agent": "wecom-agent"},
            {"match": {"platform": "matrix"}, "agent": "code"},
        ]
        assert resolve_agent_id(source, routes, default="main") == "main"

    def test_wecom_still_routes_to_wecom_agent(self):
        """Regression: explicit wecom route must remain intact."""
        source = SessionSource(
            platform=Platform.WECOM,
            chat_id="wc123456",
            chat_type="dm",
            user_id="wcuser_abc",
        )
        routes = [
            {"match": {"platform": "wecom"}, "agent": "wecom-agent"},
            {"match": {"platform": "matrix"}, "agent": "code"},
        ]
        assert resolve_agent_id(source, routes, default="main") == "wecom-agent"

    def test_matrix_user_id_specific_route(self):
        """VIP user on Matrix gets premium agent, others get code."""
        routes = [
            {"match": {"platform": "matrix", "user_id": "@boss:localhost"}, "agent": "premium"},
            {"match": {"platform": "matrix"}, "agent": "code"},
        ]
        vip = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:localhost",
            user_id="@boss:localhost",
        )
        regular = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:localhost",
            user_id="@peon:localhost",
        )
        assert resolve_agent_id(vip, routes) == "premium"
        assert resolve_agent_id(regular, routes) == "code"


class TestProfileIsolation:
    """AgentProfile home_dir / ContextVar / session path isolation."""

    def test_agent_profile_resolved_home(self, tmp_path):
        from agent.profile import AgentProfile
        profile = AgentProfile(id="code", home_dir=tmp_path / "code")
        assert profile.resolved_home == tmp_path / "code"
        assert profile.soul_md_path == tmp_path / "code" / "SOUL.md"
        assert profile.memory_dir == tmp_path / "code" / "memories"
        assert profile.skills_dir == tmp_path / "code" / "skills"
        assert profile.sessions_path == tmp_path / "code" / "sessions.json"

    def test_main_profile_falls_back_to_process_home(self, monkeypatch):
        from agent.profile import AgentProfile, get_hermes_home
        monkeypatch.setenv("HERMES_HOME", "/fake/hermes")
        profile = AgentProfile(id="main")
        assert profile.resolved_home == get_hermes_home()

    def test_contextvar_scopes_profile(self):
        from agent.profile import AgentProfile, use_profile, get_active_profile
        code_profile = AgentProfile(id="code", home_dir="/tmp/code")
        main_profile = AgentProfile(id="main", home_dir="/tmp/main")

        # Default is None
        assert get_active_profile() is None

        with use_profile(code_profile):
            assert get_active_profile() == code_profile
            # Nested override
            with use_profile(main_profile):
                assert get_active_profile() == main_profile
            assert get_active_profile() == code_profile

        assert get_active_profile() is None

    def test_load_agent_registry_from_config(self):
        from agent.profile import load_agent_registry, AgentProfile, DEFAULT_AGENT_ID

        class FakeConfig:
            agents = {
                "main": {},
                "code": {
                    "model": "kimi-for-coding",
                    "provider": "moonshot",
                    "home_dir": "/root/.hermes/profiles/code",
                },
                "wecom-agent": {
                    "home_dir": "/root/.hermes/profiles/wecom-agent",
                },
            }

        registry = load_agent_registry(FakeConfig())
        assert set(registry.keys()) == {"main", "code", "wecom-agent"}
        assert registry["code"].model == "kimi-for-coding"
        assert registry["code"].provider == "moonshot"
        assert str(registry["code"].home_dir) == "/root/.hermes/profiles/code"

    def test_load_agent_registry_ensures_default(self):
        from agent.profile import load_agent_registry, DEFAULT_AGENT_ID

        class FakeConfig:
            agents = {"code": {}}

        registry = load_agent_registry(FakeConfig())
        assert DEFAULT_AGENT_ID in registry

    def test_load_agent_registry_skips_non_dict_entries(self):
        from agent.profile import load_agent_registry

        class FakeConfig:
            agents = {"bad": "not-a-dict", "good": {}}

        registry = load_agent_registry(FakeConfig())
        assert "bad" not in registry
        assert "good" in registry
        assert "main" in registry


class TestBackwardCompatibility:
    """Legacy single-agent installs must see zero behavioral change."""

    def test_no_routes_no_agents_returns_main(self):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
        assert resolve_agent_id(source, [], default="main") == "main"

    def test_session_source_agent_id_field_exists(self):
        """SessionSource must carry agent_id for downstream session key building."""
        source = SessionSource(platform=Platform.MATRIX, chat_id="!room:localhost")
        assert hasattr(source, "agent_id")
        assert source.agent_id is None  # Default before routing

    def test_empty_config_registry_has_main(self):
        from agent.profile import load_agent_registry, DEFAULT_AGENT_ID
        registry = load_agent_registry(None)
        assert DEFAULT_AGENT_ID in registry
        assert registry[DEFAULT_AGENT_ID].home_dir is None
