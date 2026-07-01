"""Regression: _flush_messages_to_session_db must not mutate the caller's
live messages list when a persist-override is configured. (#56303)

The tool-execution loop re-uses the same ``messages`` list for subsequent
API calls — if the override rewrites ``content`` in-place, later iterations
see the transcript-clean text instead of the original API-facing prompt.
"""

from unittest.mock import MagicMock

from run_agent import AIAgent


class TestFlushOverrideMutation:
    """Persist-override in _flush_messages_to_session_db must not leak back."""

    def _make_agent(self):
        agent = AIAgent.__new__(AIAgent)
        agent._persist_user_message_idx = 0
        agent._persist_user_message_override = "Hello there"
        agent._persist_user_message_timestamp = None
        agent._session_db = MagicMock()
        agent._session_db_created = True
        agent._last_flushed_db_idx = 0
        agent.session_id = "session-123"
        agent._flushed_db_message_session_id = "session-123"
        agent._flushed_db_message_ids = set()
        agent._persist_disabled = False
        return agent

    def test_flush_does_not_mutate_caller_messages(self):
        """Caller's messages list must retain original content after flush."""
        agent = self._make_agent()
        original_content = (
            "[Voice input - respond concisely and conversationally, "
            "2-3 sentences max. No code blocks or markdown.] Hello there"
        )
        messages = [
            {"role": "user", "content": original_content},
            {"role": "assistant", "content": "Hi!"},
        ]

        agent._flush_messages_to_session_db(messages, [])

        # The live messages list used by the tool-execution loop must
        # retain the original content — override should only affect
        # the persisted copy.
        assert messages[0]["content"] == original_content

    def test_flush_still_persists_override_content(self):
        """DB writes must still receive the override content."""
        agent = self._make_agent()
        original_content = (
            "[Voice input - respond concisely and conversationally, "
            "2-3 sentences max.] Hello there"
        )
        messages = [
            {"role": "user", "content": original_content},
            {"role": "assistant", "content": "Hi!"},
        ]

        agent._flush_messages_to_session_db(messages, [])

        # DB should receive the overridden content
        calls = agent._session_db.append_message.call_args_list
        user_writes = [c for c in calls if c.kwargs.get("role") == "user"]
        assert len(user_writes) >= 1
        assert user_writes[0].kwargs["content"] == "Hello there"

    def test_flush_timestamp_override_does_not_mutate_caller(self):
        """Timestamp override must also not leak into caller's messages."""
        from datetime import datetime, timezone

        agent = self._make_agent()
        agent._persist_user_message_timestamp = datetime(
            2026, 4, 28, 13, 40, 53, tzinfo=timezone.utc
        )
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
        ]

        agent._flush_messages_to_session_db(messages, [])

        # Caller's message must NOT have the timestamp injected
        assert "timestamp" not in messages[0]

    def test_repeated_flush_does_not_duplicate_user_message(self):
        """Second flush of the same messages list must not re-persist the
        override target.  (#56318)

        Before the fix, the _DB_PERSISTED_MARKER was stamped on the copy
        (not the caller's live dict), so a second flush saw the original
        as unmarked and appended it to the DB again.
        """
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "synthetic clean"},
        ]

        agent._flush_messages_to_session_db(messages, [])
        agent._flush_messages_to_session_db(messages, [])

        user_calls = [
            c
            for c in agent._session_db.append_message.call_args_list
            if c.kwargs.get("role") == "user"
        ]
        assert len(user_calls) == 1, (
            f"Expected exactly 1 user DB write, got {len(user_calls)}"
        )
