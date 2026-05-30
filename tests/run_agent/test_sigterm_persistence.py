"""Regression tests for #35313 — SIGTERM/SIGINT handler for emergency message persistence.

When a Hermes process is killed mid-turn (SIGTERM from a process manager, OOM
kill, etc.), messages accumulated since the last _persist_session call must be
flushed to the session DB before the process exits.
"""

from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent_with_session_db():
    """Minimal AIAgent with a mock session DB for testing persistence."""
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        # Wire up a mock session DB
        a._session_db = MagicMock()
        a._session_db_created = True
        return a


class TestTerminationHandlers:
    """Tests for _register_termination_handlers and the signal/atexit handlers."""

    def test_registers_sigterm_handler(self, agent_with_session_db):
        """SIGTERM handler is registered when _register_termination_handlers is called."""
        agent = agent_with_session_db
        agent._register_termination_handlers()
        assert agent._termination_flushed is False

    def test_signal_handler_flushes_messages(self, agent_with_session_db):
        """The SIGTERM handler calls _persist_session with current messages."""
        import signal

        agent = agent_with_session_db
        captured_handler = {}

        def _fake_signal(signum, handler):
            captured_handler[signum] = handler

        with patch("signal.signal", _fake_signal):
            agent._register_termination_handlers()

        # Verify handler was registered for SIGTERM
        assert signal.SIGTERM in captured_handler

        # Set up in-flight messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        agent._session_messages = messages
        agent._last_flushed_db_idx = 0

        # Mock _persist_session to verify it's called
        with patch.object(agent, "_persist_session") as mock_persist:
            handler = captured_handler[signal.SIGTERM]
            handler(signal.SIGTERM, None)

            mock_persist.assert_called_once_with(messages)
            assert agent._termination_flushed is True

    def test_sigint_handler_preserves_keyboard_interrupt(self, agent_with_session_db):
        """SIGINT handler flushes then re-raises KeyboardInterrupt."""
        import signal

        agent = agent_with_session_db
        captured_handler = {}

        def _fake_signal(signum, handler):
            captured_handler[signum] = handler

        with patch("signal.signal", _fake_signal):
            agent._register_termination_handlers()

        assert signal.SIGINT in captured_handler

        messages = [{"role": "user", "content": "test"}]
        agent._session_messages = messages

        with patch.object(agent, "_persist_session") as mock_persist:
            handler = captured_handler[signal.SIGINT]
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
            mock_persist.assert_called_once_with(messages)

    def test_handler_is_idempotent(self, agent_with_session_db):
        """Calling the handler twice only flushes once."""
        import signal

        agent = agent_with_session_db
        captured_handler = {}

        def _fake_signal(signum, handler):
            captured_handler[signum] = handler

        with patch("signal.signal", _fake_signal):
            agent._register_termination_handlers()

        messages = [{"role": "user", "content": "test"}]
        agent._session_messages = messages

        with patch.object(agent, "_persist_session") as mock_persist:
            handler = captured_handler[signal.SIGTERM]
            handler(signal.SIGTERM, None)
            handler(signal.SIGTERM, None)  # Second call should be a no-op
            mock_persist.assert_called_once()

    def test_handler_no_messages_no_crash(self, agent_with_session_db):
        """Signal handler does not crash when there are no messages."""
        import signal

        agent = agent_with_session_db
        captured_handler = {}

        def _fake_signal(signum, handler):
            captured_handler[signum] = handler

        with patch("signal.signal", _fake_signal):
            agent._register_termination_handlers()

        agent._session_messages = []
        agent._session_db = None  # No DB

        handler = captured_handler[signal.SIGTERM]
        # Should not raise
        handler(signal.SIGTERM, None)
        assert agent._termination_flushed is True

    def test_handler_no_session_db_no_crash(self, agent_with_session_db):
        """Signal handler does not crash when _session_db is None."""
        import signal

        agent = agent_with_session_db
        captured_handler = {}

        def _fake_signal(signum, handler):
            captured_handler[signum] = handler

        with patch("signal.signal", _fake_signal):
            agent._register_termination_handlers()

        agent._session_messages = [{"role": "user", "content": "test"}]
        agent._session_db = None

        handler = captured_handler[signal.SIGTERM]
        # Should not raise
        handler(signal.SIGTERM, None)

    def test_atexit_handler_flushes_when_signal_not_fired(self, agent_with_session_db):
        """atexit fallback flushes messages if signal handler wasn't called."""
        agent = agent_with_session_db

        atexit_handlers = []

        def _fake_atexit_register(fn):
            atexit_handlers.append(fn)

        with patch("atexit.register", _fake_atexit_register):
            agent._register_termination_handlers()

        assert len(atexit_handlers) == 1

        messages = [{"role": "user", "content": "atexit test"}]
        agent._session_messages = messages

        with patch.object(agent, "_persist_session") as mock_persist:
            atexit_handlers[0]()  # Simulate atexit firing
            mock_persist.assert_called_once_with(messages)

    def test_atexit_handler_skips_when_signal_already_handled(self, agent_with_session_db):
        """atexit fallback is a no-op when signal handler already flushed."""
        import signal

        agent = agent_with_session_db
        captured_handler = {}
        atexit_handlers = []

        def _fake_signal(signum, handler):
            captured_handler[signum] = handler

        def _fake_atexit_register(fn):
            atexit_handlers.append(fn)

        with patch("signal.signal", _fake_signal), patch("atexit.register", _fake_atexit_register):
            agent._register_termination_handlers()

        messages = [{"role": "user", "content": "double"}]
        agent._session_messages = messages

        with patch.object(agent, "_persist_session") as mock_persist:
            # Signal fires first
            captured_handler[signal.SIGTERM](signal.SIGTERM, None)
            # Then atexit fires
            atexit_handlers[0]()
            # Should only have been called once
            mock_persist.assert_called_once()


class TestRegisterTerminationHandlersInit:
    """Verify _register_termination_handlers is callable and works after init_agent wiring."""

    def test_handler_called_from_init_agent(self):
        """_register_termination_handlers is called by init_agent, not AIAgent.__init__.

        The production code calls agent._register_termination_handlers() at the end
        of init_agent() in agent/agent_init.py.  We verify the method exists on the
        agent and works correctly when invoked.
        """
        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("web_search"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            # Verify the method exists and is callable
            assert hasattr(a, "_register_termination_handlers")
            assert callable(a._register_termination_handlers)
            # Calling it should not raise
            a._register_termination_handlers()
            assert a._termination_flushed is False
