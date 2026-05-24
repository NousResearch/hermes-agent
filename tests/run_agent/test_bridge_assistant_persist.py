"""Regression tests for #31269 — bridge-worker silently drops assistant replies.

The fix ensures `final_response` is injected into the messages list before
`_persist_session()` when exit paths like partial_stream_recovery or
max_iterations set final_response without appending a structured assistant
message dict.
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(messages=None, final_response="Hello!"):
    """Build a minimal mock agent with the attributes run_conversation uses."""
    agent = MagicMock()
    agent.messages = messages or [
        {"role": "user", "content": "hi"},
    ]
    agent.model = "test-model"
    agent.tools = []
    agent.provider = MagicMock()
    agent.provider.model = "test-model"
    agent._last_flushed_db_idx = 0
    agent._session_db = MagicMock()
    agent._drop_trailing_empty_response_scaffolding = MagicMock()
    agent._persist_session = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFinalResponseInjectionBeforePersist:
    """Verify assistant message is injected when missing from messages."""

    def test_final_response_injected_when_messages_has_no_assistant(self):
        """If messages only has user msg and final_response is set, assistant
        should be appended before _persist_session."""
        import agent.conversation_loop as cl

        messages = [{"role": "user", "content": "hello"}]
        agent = _make_agent(messages=messages)

        # Simulate the tail of run_conversation: drop scaffolding, then persist.
        with patch.object(cl, "final_response", "Hello from assistant", create=True):
            # Directly test the injection logic
            cl._ensure_final_response_in_messages(messages, "Hello from assistant")

        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Hello from assistant"

    def test_no_duplicate_when_assistant_already_present(self):
        """If messages tail already has the assistant response, don't re-inject."""
        import agent.conversation_loop as cl

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello from assistant"},
        ]
        original_len = len(messages)

        cl._ensure_final_response_in_messages(messages, "Hello from assistant")

        assert len(messages) == original_len

    def test_empty_final_response_not_injected(self):
        """Empty or whitespace-only final_response should NOT be injected."""
        import agent.conversation_loop as cl

        messages = [{"role": "user", "content": "hello"}]

        cl._ensure_final_response_in_messages(messages, "   ")

        assert len(messages) == 1

    def test_placeholder_empty_not_injected(self):
        """The '(empty)' placeholder should NOT be injected."""
        import agent.conversation_loop as cl

        messages = [{"role": "user", "content": "hello"}]

        cl._ensure_final_response_in_messages(messages, "(empty)")

        assert len(messages) == 1

    def test_non_string_final_response_not_injected(self):
        """Non-string final_response (e.g. None) should not be injected."""
        import agent.conversation_loop as cl

        messages = [{"role": "user", "content": "hello"}]

        cl._ensure_final_response_in_messages(messages, None)

        assert len(messages) == 1

    def test_different_assistant_tail_still_injects(self):
        """If messages tail is an assistant message but with DIFFERENT content,
        the new final_response should still be injected."""
        import agent.conversation_loop as cl

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "old response"},
        ]

        cl._ensure_final_response_in_messages(messages, "new response")

        assert messages[-1]["content"] == "new response"
        assert messages[-2]["content"] == "old response"
