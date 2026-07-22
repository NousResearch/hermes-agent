"""Tests for AIAgent.close() calling shutdown_memory_provider (#46082)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest


class _FakeMemoryManager:
    """Tracks shutdown_memory_provider calls."""

    def __init__(self):
        self.on_session_end_called = False
        self.shutdown_all_called = False
        self.received_messages = None

    def on_session_end(self, messages):
        self.on_session_end_called = True
        self.received_messages = messages

    def shutdown_all(self):
        self.shutdown_all_called = True

    def on_session_switch(self, *args, **kwargs):
        pass


class _FakeCompressor:
    def __init__(self):
        self.on_session_end_called = False
        self.received_messages = None

    def on_session_end(self, sid, messages):
        self.on_session_end_called = True
        self.received_messages = messages


def test_close_calls_shutdown_memory_provider():
    """AIAgent.close() must call shutdown_memory_provider() (#46082).

    Prior to the fix, close() cleaned up subprocesses, terminals, browser
    sessions, child agents, and the httpx client — but never called
    shutdown_memory_provider(). This left per-session memory provider threads
    (Honcho's honcho-async-writer, honcho-prewarm-dialectic) alive after
    session teardown, causing ~2 leaked threads per dashboard session.
    """
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "test-close-shutdown"
    agent._session_messages = []
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    agent._end_session_on_close = False
    agent.background_review_callback = None
    agent.memory_notifications = "on"

    mm = _FakeMemoryManager()
    agent._memory_manager = mm
    agent.context_compressor = _FakeCompressor()

    # Call close()
    agent.close()

    # shutdown_memory_provider should have been called,
    # which calls on_session_end + shutdown_all
    assert mm.on_session_end_called, (
        "shutdown_memory_provider() was not called by close() — "
        "memory manager on_session_end was never triggered"
    )
    assert mm.shutdown_all_called, (
        "shutdown_memory_provider() was not called by close() — "
        "memory manager shutdown_all was never triggered"
    )


def test_close_is_idempotent_with_memory_provider():
    """close() can be called multiple times; shutdown is idempotent."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "test-idempotent"
    agent._session_messages = []
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    agent._end_session_on_close = False
    agent.background_review_callback = None
    agent.memory_notifications = "on"

    mm = _FakeMemoryManager()
    agent._memory_manager = mm
    agent.context_compressor = _FakeCompressor()

    agent.close()
    assert mm.shutdown_all_called

    # Second call should not raise
    mm.shutdown_all_called = False
    agent.close()
    # shutdown_all is called again (idempotent in real MemoryManager;
    # our fake just tracks it)
    assert mm.shutdown_all_called


def test_close_passes_non_empty_messages_to_providers():
    """close() must pass _session_messages to shutdown_memory_provider (#46082 review).

    shutdown_memory_provider() runs BEFORE _session_messages is cleared,
    so providers receive the actual conversation transcript for
    on_session_end() extraction — not an empty list.
    """
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "test-msg-ordering"
    agent._session_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    agent._end_session_on_close = False
    agent.background_review_callback = None
    agent.memory_notifications = "on"

    mm = _FakeMemoryManager()
    agent._memory_manager = mm
    comp = _FakeCompressor()
    agent.context_compressor = comp

    agent.close()

    # Providers should have received the actual messages, not []
    assert mm.received_messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ], (
        "Memory manager received empty messages — shutdown_memory_provider() "
        "was called after _session_messages was cleared"
    )
    assert comp.received_messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ], (
        "Context compressor received empty messages — shutdown_memory_provider() "
        "was called after _session_messages was cleared"
    )

    # _session_messages should be cleared after close()
    assert agent._session_messages == []
