"""Tests for CLI/Desktop fallback and concurrency-safe scope resolution."""

import pytest
from unittest.mock import patch

from agent.memory_scope import (
    resolve_scope_key,
    resolve_scope_suffix,
    resolve_active_scope_suffix,
    scope_hash,
)


class TestConversationScopeCLIFallback:
    """conversation scope falls back to session_id for CLI/Desktop."""

    def test_conversation_falls_back_to_session_id(self):
        """Without gateway context, CLI sessions get session-scoped memory."""
        result = resolve_scope_key(
            "conversation",
            session_id="20260711_120000_abc",
        )
        assert result is not None
        assert result == scope_hash("hermes-memory:v1:conversation:local:20260711_120000_abc")

    def test_conversation_different_cli_sessions_isolated(self):
        r1 = resolve_scope_key("conversation", session_id="sess-1")
        r2 = resolve_scope_key("conversation", session_id="sess-2")
        assert r1 != r2

    def test_conversation_gateway_takes_priority_over_session(self):
        """When both gateway_session_key and session_id exist, gateway wins."""
        gw_key = "agent:main:whatsapp:dm:123@lid"
        r_gw = resolve_scope_key(
            "conversation",
            gateway_session_key=gw_key,
            session_id="sess-1",
        )
        r_sess = resolve_scope_key(
            "conversation",
            session_id="sess-1",
        )
        assert r_gw != r_sess
        assert r_gw == scope_hash(f"hermes-memory:v1:conversation:gateway:{gw_key}")

    def test_conversation_cli_persists_across_same_session(self):
        """Same session_id = same scope key (resumed sessions)."""
        r1 = resolve_scope_key("conversation", session_id="stable-sess")
        r2 = resolve_scope_key("conversation", session_id="stable-sess")
        assert r1 == r2


class TestResolveActiveScopeSuffix:
    """Test the concurrency-safe resolver used by secondary paths."""

    def test_identity_scope_returns_none(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            result = resolve_active_scope_suffix()
            assert result is None

    def test_conversation_scope_with_session_context(self):
        cfg = {"memory": {"scope": "conversation"}}
        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("gateway.session_context.get_session_env") as mock_get,
        ):
            mock_get.side_effect = lambda key, default="": {
                "HERMES_SESSION_KEY": "agent:main:whatsapp:dm:123@lid",
                "HERMES_SESSION_ID": "",
                "HERMES_SESSION_PLATFORM": "whatsapp",
                "HERMES_SESSION_USER_ID": "",
            }.get(key, default)

            result = resolve_active_scope_suffix()
            assert result is not None
            assert "default_" in result

    def test_conversation_scope_cli_fallback(self):
        """Without gateway session key, uses session_id."""
        cfg = {"memory": {"scope": "conversation"}}
        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("gateway.session_context.get_session_env") as mock_get,
        ):
            mock_get.side_effect = lambda key, default="": {
                "HERMES_SESSION_KEY": "",
                "HERMES_SESSION_ID": "20260711_120000_abc",
                "HERMES_SESSION_PLATFORM": "cli",
                "HERMES_SESSION_USER_ID": "",
            }.get(key, default)

            result = resolve_active_scope_suffix()
            assert result is not None
            assert result.startswith("default_")

    def test_no_session_context_returns_none(self):
        cfg = {"memory": {"scope": "conversation"}}
        with (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("gateway.session_context.get_session_env") as mock_get,
        ):
            mock_get.return_value = ""
            result = resolve_active_scope_suffix()
            assert result is None


class TestConcurrencySafeResolution:
    """Verify that resolve_active_scope_suffix reads ContextVars, not os.environ."""

    def test_env_var_not_used(self):
        """HERMES_MEMORY_SCOPE_SUFFIX should not exist as a mechanism."""
        import os
        assert "HERMES_MEMORY_SCOPE_SUFFIX" not in os.environ

    def test_two_concurrent_sessions_resolve_differently(self):
        """Simulate two concurrent gateway chats with different session keys."""
        cfg = {"memory": {"scope": "conversation"}}

        def make_resolver(session_key, session_id):
            def mock_get(key, default=""):
                return {
                    "HERMES_SESSION_KEY": session_key,
                    "HERMES_SESSION_ID": session_id,
                    "HERMES_SESSION_PLATFORM": "whatsapp",
                    "HERMES_SESSION_USER_ID": "",
                }.get(key, default)
            return mock_get

        with patch("hermes_cli.config.load_config", return_value=cfg):
            with patch("gateway.session_context.get_session_env",
                       side_effect=make_resolver("agent:main:whatsapp:dm:userA", "sess-a")):
                result_a = resolve_active_scope_suffix()

            with patch("gateway.session_context.get_session_env",
                       side_effect=make_resolver("agent:main:whatsapp:dm:userB", "sess-b")):
                result_b = resolve_active_scope_suffix()

        assert result_a != result_b
