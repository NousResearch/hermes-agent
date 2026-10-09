"""Invariant: stale signal leak between consecutive compress_context calls."""
from unittest.mock import MagicMock, patch

import pytest


def test_signal_cleared_on_entry_between_calls(monkeypatch):
    """Call 1: lock held → signal set. Call 2: lock available → signal must
    be None after call 2 because the entry code cleared it. This prevents
    a prior auto-compress lock-skip from causing a subsequent successful
    manual /compress to falsely report 'Compression already in progress'."""
    from agent.conversation_compression import compress_context

    agent = MagicMock()
    agent._cached_system_prompt = ""
    agent.tools = None
    agent._memory_manager = None
    agent._build_system_prompt = MagicMock(return_value="sys prompt")
    agent._emit_warning = MagicMock()

    msgs = [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
    ]

    monkeypatch.setattr(
        "agent.conversation_compression._compression_lock_holder",
        lambda a: "pid=test:holder",
    )

    # --- Call 1: lock held ---
    db1 = MagicMock()
    db1.try_acquire_compression_lock.return_value = False
    db1.get_compression_lock_holder.return_value = "pid=holder1"
    agent._session_db = db1

    compress_context(agent, msgs, "", approx_tokens=100, force=True)

    # Call 1: signal set (lock was held).
    assert agent._compression_skipped_due_to_lock is not None

    # --- Call 2: lock available, compressor succeeds ---
    db2 = MagicMock()
    db2.try_acquire_compression_lock.return_value = True
    agent._session_db = db2
    agent.context_compressor = MagicMock()
    compressed = [
        {"role": "user", "content": "[summary]"},
        {"role": "assistant", "content": "ok"},
    ]
    agent.context_compressor.compress = MagicMock(return_value=compressed)
    agent.context_compressor.compression_count = 0
    agent.context_compressor.last_compression_rough_tokens = 0

    compress_context(agent, msgs, "", approx_tokens=100, force=True)

    # Call 2: signal must be None — entry code cleared stale Call 1 signal.
    assert agent._compression_skipped_due_to_lock is None, (
        "stale signal from lock-skip call 1 leaked into successful call 2"
    )
