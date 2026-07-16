"""Tests for the memory scope resolver."""

import pytest

from agent.memory_scope import (
    DEFAULT_SCOPE,
    VALID_SCOPES,
    resolve_scope_key,
    resolve_scope_suffix,
    scope_hash,
)


class TestScopeHash:
    def test_deterministic(self):
        assert scope_hash("test-value") == scope_hash("test-value")

    def test_different_inputs_produce_different_hashes(self):
        assert scope_hash("a") != scope_hash("b")

    def test_length(self):
        assert len(scope_hash("x")) == 16

    def test_no_pii_leak(self):
        h = scope_hash("secret-phone-123456")
        assert "secret" not in h
        assert "123456" not in h


class TestResolveScopeKeyIdentity:
    def test_identity_returns_none(self):
        result = resolve_scope_key("identity", agent_identity="default")
        assert result is None

    def test_identity_ignores_all_context(self):
        result = resolve_scope_key(
            "identity",
            agent_identity="default",
            platform="whatsapp",
            gateway_session_key="agent:main:whatsapp:dm:123",
            session_id="sess-1",
        )
        assert result is None


class TestResolveScopeKeyUser:
    def test_user_with_user_id(self):
        result = resolve_scope_key("user", platform="whatsapp", user_id="123456@lid")
        assert result is not None
        assert len(result) == 16

    def test_user_prefers_user_id_alt(self):
        r1 = resolve_scope_key("user", platform="telegram", user_id="123")
        r2 = resolve_scope_key("user", platform="telegram", user_id="123", user_id_alt="789")
        assert r1 != r2
        assert r2 == scope_hash("hermes-memory:v1:user:telegram:789")

    def test_user_stable_across_chats(self):
        r1 = resolve_scope_key("user", platform="whatsapp", user_id="abc@lid")
        r2 = resolve_scope_key("user", platform="whatsapp", user_id="abc@lid")
        assert r1 == r2

    def test_user_different_platforms_isolated(self):
        r1 = resolve_scope_key("user", platform="whatsapp", user_id="abc")
        r2 = resolve_scope_key("user", platform="telegram", user_id="abc")
        assert r1 != r2

    def test_user_no_identifiers_falls_back(self):
        result = resolve_scope_key("user", platform="whatsapp")
        assert result is None


class TestResolveScopeKeyConversation:
    def test_conversation_with_gateway_session_key(self):
        result = resolve_scope_key(
            "conversation", gateway_session_key="agent:main:whatsapp:dm:123@lid"
        )
        assert result is not None
        assert len(result) == 16

    def test_conversation_stable_across_session_resets(self):
        key = "agent:main:whatsapp:dm:123@lid"
        r1 = resolve_scope_key("conversation", gateway_session_key=key)
        r2 = resolve_scope_key("conversation", gateway_session_key=key)
        assert r1 == r2

    def test_conversation_different_chats_isolated(self):
        r1 = resolve_scope_key(
            "conversation", gateway_session_key="agent:main:whatsapp:dm:user-a"
        )
        r2 = resolve_scope_key(
            "conversation", gateway_session_key="agent:main:whatsapp:dm:user-b"
        )
        assert r1 != r2

    def test_conversation_fallback_to_components(self):
        r1 = resolve_scope_key(
            "conversation", platform="whatsapp", chat_type="dm", chat_id="123@lid"
        )
        assert r1 is not None
        assert r1 == scope_hash("hermes-memory:v1:conversation:components:whatsapp:dm:123@lid")

    def test_conversation_fallback_with_thread(self):
        r1 = resolve_scope_key(
            "conversation", platform="telegram", chat_type="dm",
            chat_id="-100123", thread_id="42",
        )
        assert r1 is not None
        assert r1 == scope_hash("hermes-memory:v1:conversation:components:telegram:dm:-100123:42")

    def test_conversation_group_vs_dm_isolated(self):
        r1 = resolve_scope_key(
            "conversation", gateway_session_key="agent:main:whatsapp:dm:user-a"
        )
        r2 = resolve_scope_key(
            "conversation", gateway_session_key="agent:main:whatsapp:group:group-x"
        )
        assert r1 != r2

    def test_conversation_no_context_falls_back(self):
        result = resolve_scope_key("conversation")
        assert result is None

    def test_conversation_cli_fallback_to_session_id(self):
        """Without gateway context, CLI sessions get per-session scope."""
        result = resolve_scope_key("conversation", session_id="20260711_120000_abc")
        assert result is not None
        assert result == scope_hash("hermes-memory:v1:conversation:local:20260711_120000_abc")

    def test_same_raw_value_is_domain_separated_between_scopes(self):
        assert resolve_scope_key("session", session_id="same") != resolve_scope_key(
            "conversation", session_id="same"
        )

    def test_conversation_gateway_beats_session_id(self):
        """gateway_session_key takes priority over session_id fallback."""
        r_gw = resolve_scope_key(
            "conversation",
            gateway_session_key="agent:main:whatsapp:dm:123",
            session_id="sess-1",
        )
        r_sess = resolve_scope_key("conversation", session_id="sess-1")
        assert r_gw != r_sess


class TestResolveScopeKeySession:
    def test_session_with_session_id(self):
        result = resolve_scope_key("session", session_id="20260711_120000_abc")
        assert result is not None
        assert len(result) == 16

    def test_session_different_sessions_isolated(self):
        r1 = resolve_scope_key("session", session_id="sess-1")
        r2 = resolve_scope_key("session", session_id="sess-2")
        assert r1 != r2

    def test_session_no_session_id_falls_back(self):
        result = resolve_scope_key("session")
        assert result is None


class TestResolveScopeSuffix:
    def test_identity_scope(self):
        result = resolve_scope_suffix("identity", None, "default")
        assert result == "default"

    def test_scoped_suffix(self):
        key = "a4c981e7f2b3d5c9"
        result = resolve_scope_suffix("conversation", key, "default")
        assert result == "default_a4c981e7f2b3d5c9"

    def test_none_key_returns_identity(self):
        result = resolve_scope_suffix("user", None, "coder")
        assert result == "coder"


class TestUnknownScope:
    def test_default_config_preserves_identity_scope(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["memory"]["scope"] == "identity"

    def test_unknown_scope_falls_back_to_identity(self):
        result = resolve_scope_key("bogus", agent_identity="default")
        assert result is None

    def test_unknown_scope_suffix(self):
        result = resolve_scope_suffix("bogus", None, "default")
        assert result == "default"
