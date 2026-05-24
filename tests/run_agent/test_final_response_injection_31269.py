"""Regression tests for issue #31269.

The bridge worker's ``state.db`` flush path
(``_flush_messages_to_session_db``) writes ``messages[idx:]`` — anything
not in the structured ``messages`` list silently never reaches disk.
A few break paths in ``run_conversation`` set ``final_response`` from
already-streamed bytes (partial-stream recovery, prior-turn content
fallback) without appending the matching structured assistant message
dict, so the user *saw* the reply in the WebUI but the database row
was never written.

The fix is a safety net in ``conversation_loop._ensure_final_response_in_messages``
called right before the final ``_persist_session``: any non-empty
``final_response`` that isn't already at the messages tail gets
appended as ``{"role": "assistant", "content": <text>,
"_injected_from_final_response": True}``.  These tests guard the
helper's behaviour and the end-to-end persistence path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _ensure_final_response_in_messages


class TestEnsureFinalResponseInMessages:
    """Direct unit tests for the injection helper."""

    def test_injects_when_messages_tail_lacks_assistant_reply(self):
        """The original #31269 case — bridge tail is the user message only."""
        messages = [
            {"role": "user", "content": "what's 2+2?"},
        ]

        injected = _ensure_final_response_in_messages(messages, "2+2 is 4.")

        assert injected is True
        assert messages[-1] == {
            "role": "assistant",
            "content": "2+2 is 4.",
            "_injected_from_final_response": True,
        }

    def test_injects_after_tool_result_partial_stream_recovery(self):
        """Partial-stream-recovery: tool turns ran, model went silent, callback
        already streamed prior content; final_response gets recovered from
        the streamed buffer but never appended to messages.
        """
        messages = [
            {"role": "user", "content": "search for X"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "results..."},
        ]

        injected = _ensure_final_response_in_messages(messages, "Found 3 results.")

        assert injected is True
        assert len(messages) == 4
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Found 3 results."
        assert messages[-1]["_injected_from_final_response"] is True

    def test_no_op_when_assistant_tail_already_has_matching_content(self):
        """Happy text-response path: ``_build_assistant_message`` already
        appended the structured dict — no double-append.
        """
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        original_len = len(messages)

        injected = _ensure_final_response_in_messages(messages, "hello there")

        assert injected is False
        assert len(messages) == original_len

    def test_no_op_when_assistant_tail_matches_after_strip(self):
        """Whitespace differences between final_response and messages tail
        shouldn't trigger spurious double-injection.
        """
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "  hello there  "},
        ]
        original_len = len(messages)

        injected = _ensure_final_response_in_messages(messages, "hello there\n")

        assert injected is False
        assert len(messages) == original_len

    @pytest.mark.parametrize("value", [None, "", "   ", "\n\t  "])
    def test_no_op_when_final_response_is_empty_or_whitespace(self, value):
        """Empty/whitespace final_response means no real reply was produced."""
        messages = [{"role": "user", "content": "hi"}]
        original = list(messages)

        injected = _ensure_final_response_in_messages(messages, value)

        assert injected is False
        assert messages == original

    def test_no_op_for_empty_response_sentinel(self):
        """The ``(empty)`` sentinel is a user-facing failure marker; the
        empty-response scaffolding path handles its own persistence and
        we must not append it as a real assistant reply.
        """
        messages = [{"role": "user", "content": "hi"}]
        original = list(messages)

        injected = _ensure_final_response_in_messages(messages, "(empty)")

        assert injected is False
        assert messages == original

    def test_no_op_when_final_response_not_a_string(self):
        """Defence against caller passing a non-string (e.g. None object)."""
        messages = [{"role": "user", "content": "hi"}]
        original = list(messages)

        for value in (123, [], {"role": "assistant"}, object()):
            injected = _ensure_final_response_in_messages(messages, value)
            assert injected is False
            assert messages == original

    def test_inject_after_assistant_tool_calls_only(self):
        """Tail is ``assistant(tool_calls=...)`` with no content — the helper
        must still append (this is a different turn's text, not the same one).
        """
        messages = [
            {"role": "user", "content": "do X"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "function": {"name": "X", "arguments": "{}"}}],
            },
        ]

        injected = _ensure_final_response_in_messages(messages, "I did X.")

        assert injected is True
        assert len(messages) == 3
        assert messages[-1]["content"] == "I did X."

    def test_inject_when_messages_empty(self):
        """Edge case: empty messages list with a recovered final_response."""
        messages: list = []

        injected = _ensure_final_response_in_messages(messages, "hi")

        assert injected is True
        assert len(messages) == 1
        assert messages[0]["content"] == "hi"

    def test_inject_when_assistant_tail_has_different_content(self):
        """The recovered final_response disagrees with what's at the tail
        (e.g. partial stream that produced different text than the
        already-appended ``(empty)`` recovery message has been popped) —
        the recovered text wins.
        """
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "earlier content"},
        ]

        injected = _ensure_final_response_in_messages(messages, "recovered later content")

        assert injected is True
        assert messages[-1]["content"] == "recovered later content"
        assert messages[-1]["_injected_from_final_response"] is True
        # Earlier assistant message preserved
        assert messages[-2]["content"] == "earlier content"


class TestEndToEndBridgePersistence:
    """End-to-end check: ``_persist_session`` writes the injected reply
    to ``state.db`` exactly the way the bridge worker would.
    """

    def test_injected_assistant_message_reaches_session_db(self):
        """Reproduces the diagnostic from the issue: messages list ends at
        the user turn, ``_last_flushed_db_idx`` already covers it; without
        the fix, ``state.db`` gets nothing on this persist call.
        """
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._session_db = MagicMock()
        agent._session_db_created = True
        agent.session_id = "session-31269"
        agent._last_flushed_db_idx = 1   # bridge already flushed the user turn
        agent._persist_user_message_idx = 0
        agent._persist_user_message_override = None
        agent._session_messages = []
        agent.save_session_log = False

        # Pre-state matches the diagnostic: 1 user message, no assistant
        # reply yet, but final_response holds the streamed bytes.
        messages = [{"role": "user", "content": "what's 2+2?"}]
        final_response = "2+2 is 4."

        injected = _ensure_final_response_in_messages(messages, final_response)
        assert injected is True

        agent._persist_session(messages, conversation_history=[])

        # The new assistant message must have been written.
        write_kwargs_list = [
            c.kwargs for c in agent._session_db.append_message.call_args_list
        ]
        roles = [k.get("role") for k in write_kwargs_list]
        contents = [k.get("content") for k in write_kwargs_list]
        assert "assistant" in roles, (
            "Bridge worker would have lost the assistant reply — #31269 "
            "regression. Rows actually written: %r" % write_kwargs_list
        )
        idx = roles.index("assistant")
        assert contents[idx] == "2+2 is 4."

    def test_happy_path_no_double_persist(self):
        """When the conversation loop already appended the assistant message
        (the normal text-response path), the safety-net injection must not
        cause a duplicate row.
        """
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._session_db = MagicMock()
        agent._session_db_created = True
        agent.session_id = "session-happy"
        agent._last_flushed_db_idx = 1
        agent._persist_user_message_idx = 0
        agent._persist_user_message_override = None
        agent._session_messages = []
        agent.save_session_log = False

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},   # already appended
        ]

        injected = _ensure_final_response_in_messages(messages, "hello")
        assert injected is False

        agent._persist_session(messages, conversation_history=[])

        write_kwargs_list = [
            c.kwargs for c in agent._session_db.append_message.call_args_list
        ]
        assistant_writes = [
            k for k in write_kwargs_list if k.get("role") == "assistant"
        ]
        assert len(assistant_writes) == 1, (
            "Happy path must produce exactly one assistant row — got %d"
            % len(assistant_writes)
        )
        assert assistant_writes[0]["content"] == "hello"
