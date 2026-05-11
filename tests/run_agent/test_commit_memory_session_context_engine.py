"""Regression tests for AIAgent.commit_memory_session.

Issue #22394: commit_memory_session was calling MemoryManager.on_session_end
but never ContextEngine.on_session_end. Context engines that accumulate
per-session state (LCM-style DAGs, summary stores) leaked that state from a
rotated-out session into whatever continued under the same compressor
instance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_minimal_agent(memory_manager, context_compressor, session_id="abc"):
    """Build an object with just enough surface for commit_memory_session to run.

    AIAgent.__init__ is too heavy for a focused unit test — bind the method
    to a SimpleNamespace-style object that has the attributes the method
    actually touches.
    """
    from run_agent import AIAgent

    obj = SimpleNamespace(
        _memory_manager=memory_manager,
        context_compressor=context_compressor,
        session_id=session_id,
    )
    obj.commit_memory_session = AIAgent.commit_memory_session.__get__(obj)
    return obj


def test_commit_memory_session_notifies_context_engine():
    """Both the memory manager AND the context engine receive on_session_end."""
    mm = MagicMock()
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-42")

    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    agent.commit_memory_session(msgs)

    mm.on_session_end.assert_called_once_with(msgs)
    ctx.on_session_end.assert_called_once_with("sess-42", msgs)


def test_commit_memory_session_with_no_messages_passes_empty_list():
    """Empty/None messages must still fire both hooks with an empty list."""
    mm = MagicMock()
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-7")

    agent.commit_memory_session(None)

    mm.on_session_end.assert_called_once_with([])
    ctx.on_session_end.assert_called_once_with("sess-7", [])


def test_commit_memory_session_no_memory_manager_still_notifies_context_engine():
    """If only the context engine is configured, it still gets the hook."""
    ctx = MagicMock()
    agent = _make_minimal_agent(None, ctx, session_id="sess-9")

    agent.commit_memory_session([{"role": "user", "content": "x"}])

    ctx.on_session_end.assert_called_once_with("sess-9", [{"role": "user", "content": "x"}])


def test_commit_memory_session_no_context_engine_still_notifies_memory_manager():
    """If only the memory manager is configured, it still gets the hook."""
    mm = MagicMock()
    agent = _make_minimal_agent(mm, None, session_id="sess-3")

    agent.commit_memory_session([{"role": "user", "content": "x"}])

    mm.on_session_end.assert_called_once_with([{"role": "user", "content": "x"}])


def test_commit_memory_session_tolerates_memory_manager_failure():
    """A raising memory manager must not block the context engine notification."""
    mm = MagicMock()
    mm.on_session_end.side_effect = RuntimeError("boom")
    ctx = MagicMock()
    agent = _make_minimal_agent(mm, ctx, session_id="sess-X")

    # Must not raise
    agent.commit_memory_session([{"role": "user", "content": "x"}])

    ctx.on_session_end.assert_called_once_with("sess-X", [{"role": "user", "content": "x"}])


def test_commit_memory_session_tolerates_context_engine_failure():
    """A raising context engine must not surface the exception."""
    mm = MagicMock()
    ctx = MagicMock()
    ctx.on_session_end.side_effect = RuntimeError("boom")
    agent = _make_minimal_agent(mm, ctx, session_id="sess-Y")

    # Must not raise
    agent.commit_memory_session([{"role": "user", "content": "x"}])

    mm.on_session_end.assert_called_once()
