"""Regression tests for gateway source/owner lock prompt injection."""

from gateway.config import Platform
from gateway.session import SessionContext, SessionSource, build_session_context_prompt


def test_threaded_gateway_context_includes_source_owner_lock_for_recovery():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-a",
        chat_name="Ops group",
        chat_type="group",
        user_id="user-a",
        user_name="Vladimir",
        thread_id="current-thread",
    )
    context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        shared_multi_user_session=True,
        session_key="telegram:group:chat-a:thread:current-thread",
        session_id="session-current",
    )

    prompt = build_session_context_prompt(context)
    lower = prompt.lower()

    assert "thread: current-thread" in prompt
    assert "source/owner lock" in lower
    assert "current session context" in lower
    assert "different chat/thread/session" in lower
    assert "explicit handoff" in lower
    assert "generic recovery" in lower
