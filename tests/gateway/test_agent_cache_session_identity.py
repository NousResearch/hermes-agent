"""Regression tests for gateway cached-agent session identity.

A platform session_key can stay stable while compression rotates the underlying
Hermes session_id. Reusing the old cached AIAgent across that boundary keeps the
context engine bound to stale session state.
"""

from gateway.run import GatewayRunner


def _signature(*, session_id: str, context_engine: str = "lcm") -> str:
    return GatewayRunner._agent_config_signature(
        "test-model",
        {"provider": "test", "api_mode": "chat", "base_url": "https://example.invalid"},
        ["terminal"],
        "system prompt",
        cache_keys={"context.engine": context_engine},
        user_id="telegram-user",
        user_id_alt=None,
        session_id=session_id,
        context_engine=context_engine,
    )


def test_agent_cache_signature_changes_when_session_id_rotates_under_same_session_key():
    """A compressed child session must not reuse the parent session's cached AIAgent."""

    parent_sig = _signature(session_id="session-parent")
    child_sig = _signature(session_id="session-child")

    assert parent_sig != child_sig


def test_agent_cache_signature_changes_when_context_engine_identity_changes():
    """Switching context engines must invalidate cached AIAgent/context bindings."""

    lcm_sig = _signature(session_id="same-session", context_engine="lcm")
    compressor_sig = _signature(session_id="same-session", context_engine="compressor")

    assert lcm_sig != compressor_sig
