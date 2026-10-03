"""Regression tests for gateway session-context binding."""

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import get_session_env, reset_session_vars


def test_gateway_session_env_includes_session_id():
    """Gateway per-turn context must expose the canonical session id to tools.

    The gateway builds ``SessionContext`` with both ``session_key`` and
    ``session_id``. Tools and subprocesses read these values through
    ``gateway.session_context.get_session_env`` / the local env bridge. If
    ``_set_session_env`` forgets to bind ``session_id``, cards and other
    side-effects created from a gateway turn cannot reliably mirror or resume
    the originating session.
    """
    reset_session_vars()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-123",
        chat_name="test-chat",
        chat_type="group",
        thread_id="thread-17",
        user_id="user-42",
        user_name="Test User",
        message_id="msg-983",
    )
    context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key="agent:main:telegram:group:-1001234567890:17",
        session_id="20260705_120000_deadbe",
    )

    tokens = runner._set_session_env(context)
    try:
        assert get_session_env("HERMES_SESSION_KEY") == context.session_key
        assert get_session_env("HERMES_SESSION_ID") == context.session_id
        assert get_session_env("HERMES_SESSION_CHAT_ID") == source.chat_id
        assert get_session_env("HERMES_SESSION_THREAD_ID") == source.thread_id
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == source.message_id
    finally:
        runner._clear_session_env(tokens)

    assert get_session_env("HERMES_SESSION_ID") == ""
