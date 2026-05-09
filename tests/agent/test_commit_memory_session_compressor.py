"""Regression tests for AIAgent.commit_memory_session — must notify both the
memory manager and the context compressor (#22394).

Without the fix, plugin context engines like hermes-lcm miss the final
session-end flush on /new and gateway session expiry, losing the messages
that arrived after the last compress() call and leaving the LCM session
marked as active forever.
"""

from types import MethodType
from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_stub_agent(*, memory_manager=None, context_compressor=None, session_id="sess-1"):
    """Build a bare AIAgent stub with only the attributes the method needs."""
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = memory_manager
    agent.session_id = session_id
    if context_compressor is not None:
        agent.context_compressor = context_compressor
    agent.commit_memory_session = MethodType(AIAgent.commit_memory_session, agent)
    return agent


class TestCommitMemorySessionFanOut:
    def test_notifies_both_memory_manager_and_context_compressor(self):
        """Both backends must be flushed on session rotation."""
        mm = MagicMock()
        cc = MagicMock()
        msgs = [{"role": "user", "content": "hi"}]
        agent = _make_stub_agent(memory_manager=mm, context_compressor=cc)

        agent.commit_memory_session(msgs)

        mm.on_session_end.assert_called_once_with(msgs)
        cc.on_session_end.assert_called_once_with("sess-1", msgs)

    def test_compressor_flushed_even_when_memory_manager_absent(self):
        """A configured context engine must still get the flush even if
        no external memory provider is wired up."""
        cc = MagicMock()
        msgs = [{"role": "assistant", "content": "bye"}]
        agent = _make_stub_agent(memory_manager=None, context_compressor=cc)

        agent.commit_memory_session(msgs)

        cc.on_session_end.assert_called_once_with("sess-1", msgs)

    def test_memory_manager_failure_does_not_block_compressor(self):
        """A throwing provider must not prevent the compressor flush."""
        mm = MagicMock()
        mm.on_session_end.side_effect = RuntimeError("boom")
        cc = MagicMock()
        agent = _make_stub_agent(memory_manager=mm, context_compressor=cc)

        agent.commit_memory_session([])

        cc.on_session_end.assert_called_once_with("sess-1", [])

    def test_compressor_failure_is_swallowed(self):
        """A throwing compressor must not surface to the caller."""
        mm = MagicMock()
        cc = MagicMock()
        cc.on_session_end.side_effect = RuntimeError("kaput")
        agent = _make_stub_agent(memory_manager=mm, context_compressor=cc)

        # Must not raise.
        agent.commit_memory_session([])

        mm.on_session_end.assert_called_once()

    def test_no_op_when_neither_backend_present(self):
        """Method must be safe to call on a minimal agent."""
        agent = _make_stub_agent(memory_manager=None, context_compressor=None)
        # Must not raise even when the attribute is missing entirely.
        agent.commit_memory_session([])

    def test_uses_empty_string_session_id_when_none(self):
        """When session_id is None, compressor still receives an empty string."""
        cc = MagicMock()
        agent = _make_stub_agent(memory_manager=None, context_compressor=cc, session_id=None)

        agent.commit_memory_session([{"role": "user", "content": "x"}])

        cc.on_session_end.assert_called_once_with("", [{"role": "user", "content": "x"}])
