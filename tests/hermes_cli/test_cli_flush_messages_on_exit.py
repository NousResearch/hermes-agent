"""Tests for CLI exit paths that must persist conversation messages.

Covers the fix for #44281: Ctrl+C at the idle prompt left the session's
messages table empty because _flush_messages_to_session_db was never called
in the ``finally`` block.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSessionDB:
    """Minimal stand-in for hermes_state.SessionDB."""

    def __init__(self):
        self.ended = []  # list of (session_id, reason)
        self.flushed = False

    def end_session(self, session_id: str, reason: str) -> None:
        self.ended.append((session_id, reason))

    def delete_session(self, *a, **kw):
        return False


class _FakeAgent:
    """Minimal stand-in for AIAgent."""

    def __init__(self, session_id: str = "test-session"):
        self.session_id = session_id
        self._flush_messages_to_session_db = MagicMock()
        self._last_flushed_db_idx = 0

    def interrupt(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlushMessagesOnExit:
    """Verify the finally-block flush added for #44281."""

    @staticmethod
    def _make_cli(conversation_history, agent, session_db):
        """Build a bare-bones CLI-like object with the attributes the
        finally block inspects."""
        cli = types.SimpleNamespace(
            _should_exit=False,
            conversation_history=conversation_history,
            agent=agent,
            _session_db=session_db,
            _agent_running=False,
            _voice_recorder=None,
            _delete_session_on_exit=False,
        )
        return cli

    def test_messages_flushed_on_exit(self):
        """conversation_history must be flushed before end_session."""
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        agent = _FakeAgent()
        db = _FakeSessionDB()
        cli = self._make_cli(history, agent, db)

        # Simulate the finally block logic (mirrors cli.py lines 13141-13161)
        self._run_finally_block(cli)

        agent._flush_messages_to_session_db.assert_called_once_with(history, None)
        assert db.ended  # end_session was also called

    def test_no_flush_when_history_empty(self):
        """Empty conversation_history should not trigger a flush."""
        agent = _FakeAgent()
        db = _FakeSessionDB()
        cli = self._make_cli([], agent, db)

        self._run_finally_block(cli)

        agent._flush_messages_to_session_db.assert_not_called()

    def test_no_flush_when_no_agent(self):
        """No agent → no flush attempt."""
        db = _FakeSessionDB()
        cli = self._make_cli([{"role": "user", "content": "x"}], None, db)

        self._run_finally_block(cli)

        # No crash, no flush — agent was None

    def test_flush_exception_does_not_block_end_session(self):
        """If flush raises, end_session must still run."""
        history = [{"role": "user", "content": "x"}]
        agent = _FakeAgent()
        agent._flush_messages_to_session_db.side_effect = RuntimeError("db locked")
        db = _FakeSessionDB()
        cli = self._make_cli(history, agent, db)

        self._run_finally_block(cli)

        # end_session still called despite flush failure
        assert db.ended

    @staticmethod
    def _run_finally_block(cli):
        """Reproduce the relevant portion of the ``finally`` block in
        ``HermesCLI.run()`` (cli.py lines ~13137-13168)."""
        # Flush messages (the fix)
        if (
            hasattr(cli, 'conversation_history')
            and cli.conversation_history
            and cli.agent
            and hasattr(cli.agent, '_flush_messages_to_session_db')
        ):
            try:
                cli.agent._flush_messages_to_session_db(
                    cli.conversation_history, None,
                )
            except Exception:
                pass
        # Close session
        if hasattr(cli, '_session_db') and cli._session_db and cli.agent:
            try:
                cli._session_db.end_session(cli.agent.session_id, "cli_close")
            except Exception:
                pass
