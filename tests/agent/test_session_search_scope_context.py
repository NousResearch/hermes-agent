from types import SimpleNamespace

from agent.tool_executor import _session_search_scope_context


def test_session_search_scope_context_uses_public_agent_contract():
    expected = {
        "current_session_id": "session-1",
        "current_platform": "telegram",
        "current_session_key": "telegram:chat-1:topic-7",
    }
    agent = SimpleNamespace(
        session_search_scope_context=lambda: expected,
        _chat_id="private-chat-must-not-be-read",
        _thread_id="private-thread-must-not-be-read",
    )

    assert _session_search_scope_context(agent) == expected


def test_session_search_scope_context_has_safe_public_fallback():
    agent = SimpleNamespace(session_id="session-1", platform="cli")

    assert _session_search_scope_context(agent) == {
        "current_session_id": "session-1",
        "current_platform": "cli",
        "current_session_key": None,
    }
