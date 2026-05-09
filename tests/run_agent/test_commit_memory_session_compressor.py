"""Regression test for commit_memory_session() missing context_compressor.on_session_end().

Issue #22394: commit_memory_session() called memory_manager.on_session_end() but did NOT
call context_compressor.on_session_end(), causing plugin context engines (e.g. hermes-lcm)
to miss the final session-end flush on /new, gateway session expiry, etc.

shutdown_memory_provider() correctly called both — commit_memory_session() was inconsistent.
"""

from unittest.mock import MagicMock, patch


def _make_agent_with_mocks(*, memory_manager=None, context_compressor=None, session_id="sess-1"):
    """Build a minimal AIAgent with controlled memory/compressor attributes.

    Patches away init side-effects (DB, network, files). Only sets the attributes
    touched by commit_memory_session.
    """
    with (
        patch("hermes_cli.config.load_config", return_value={"agent": {}}),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.example.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._memory_manager = memory_manager
    agent.context_compressor = context_compressor
    agent.session_id = session_id
    return agent


class TestCommitMemorySessionCompressor:
    def test_context_compressor_on_session_end_called(self):
        """commit_memory_session() must call context_compressor.on_session_end()."""
        mm = MagicMock()
        cc = MagicMock()
        agent = _make_agent_with_mocks(memory_manager=mm, context_compressor=cc, session_id="abc")

        agent.commit_memory_session(["msg1"])

        cc.on_session_end.assert_called_once_with("abc", ["msg1"])

    def test_context_compressor_receives_empty_messages_by_default(self):
        """commit_memory_session() with no args passes [] to context_compressor."""
        mm = MagicMock()
        cc = MagicMock()
        agent = _make_agent_with_mocks(memory_manager=mm, context_compressor=cc, session_id="xyz")

        agent.commit_memory_session()

        cc.on_session_end.assert_called_once_with("xyz", [])

    def test_memory_manager_still_called(self):
        """Adding compressor call must not drop the existing memory_manager call."""
        mm = MagicMock()
        cc = MagicMock()
        agent = _make_agent_with_mocks(memory_manager=mm, context_compressor=cc)

        msgs = ["a", "b"]
        agent.commit_memory_session(msgs)

        mm.on_session_end.assert_called_once_with(msgs)

    def test_no_context_compressor_does_not_raise(self):
        """When context_compressor is None, commit_memory_session() must not raise."""
        mm = MagicMock()
        agent = _make_agent_with_mocks(memory_manager=mm, context_compressor=None)

        agent.commit_memory_session(["msg"])  # must not raise

        mm.on_session_end.assert_called_once()

    def test_compressor_exception_is_swallowed(self):
        """Exceptions from context_compressor.on_session_end() must not propagate."""
        mm = MagicMock()
        cc = MagicMock()
        cc.on_session_end.side_effect = RuntimeError("lcm flush failed")
        agent = _make_agent_with_mocks(memory_manager=mm, context_compressor=cc)

        agent.commit_memory_session()  # must not raise

    def test_no_memory_manager_skips_both(self):
        """When _memory_manager is None, context_compressor.on_session_end() is also skipped."""
        cc = MagicMock()
        agent = _make_agent_with_mocks(memory_manager=None, context_compressor=cc)

        agent.commit_memory_session()

        cc.on_session_end.assert_not_called()
