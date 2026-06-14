"""Unit tests for early turn-start session persistence hardening.

These tests verify the three failure modes addressed in PR #45110:
1. Retry on transient DB session creation failure (SQLite lock).
2. Reset of the flush cursor to prevent skipping the user message.
3. Structured logging when persistence still fails after retry.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, call, patch

import pytest

from agent.turn_context import build_turn_context


class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def __init__(self):
        self.reset_called = False

    def reset_for_turn(self):
        self.reset_called = True


class _FakeAgent:
    """Minimal stand-in covering only what the prologue touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        # Attributes the prologue assigns.
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_user_message_idx = None
        self._last_flushed_db_idx = -1
        self._session_db_created = True
        # Track calls to _persist_session and _ensure_db_session for verification.
        self._persist_session_calls = []
        self._ensure_db_session_calls = 0

    def _ensure_db_session(self):
        self._ensure_db_session_calls += 1

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, conversation_history):
        # Capture the messages list at the time of the call.
        self._persist_session_calls.append({
            "messages": list(messages),
            "conversation_history": conversation_history,
        })


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """Stub out auxiliary_client.set_runtime_main to keep tests hermetic."""
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def test_early_persist_includes_user_message():
    """Verify the user message appears in the first persist call, before any model response."""
    agent = _FakeAgent()
    ctx = _build(agent)

    # Early persist should have been called once.
    assert len(agent._persist_session_calls) == 1

    # The persisted messages list should contain the user message.
    persisted_messages = agent._persist_session_calls[0]["messages"]
    assert len(persisted_messages) >= 1
    assert persisted_messages[-1] == {"role": "user", "content": "hello"}


def test_flush_cursor_reset_prevents_skip():
    """Verify _last_flushed_db_idx is reset to the user message index.

    This prevents a prior turn's cursor from causing the user message to be
    skipped in the flush operation due to overshoot.
    """
    agent = _FakeAgent()
    # Simulate a prior turn's overshoot: _last_flushed_db_idx > len(messages).
    agent._last_flushed_db_idx = 999

    ctx = _build(agent)

    # After build_turn_context, _last_flushed_db_idx should be reset to point at
    # the user message (current_turn_user_idx).
    assert agent._last_flushed_db_idx == ctx.current_turn_user_idx
    assert agent._last_flushed_db_idx == len(ctx.messages) - 1


def test_early_persist_retries_on_db_failure():
    """Verify early persist retries _ensure_db_session on transient failure."""
    agent = _FakeAgent()

    # Mock _persist_session to fail on first call, succeed on second.
    call_count = [0]

    def _persist_session_with_failure(messages, conversation_history):
        call_count[0] += 1
        agent._persist_session_calls.append({
            "messages": list(messages),
            "conversation_history": conversation_history,
        })
        if call_count[0] == 1:
            raise RuntimeError("Transient DB failure (SQLite lock)")

    agent._persist_session = _persist_session_with_failure

    with patch.object(agent, "_ensure_db_session") as mock_ensure:
        ctx = _build(agent)

        # _ensure_db_session should have been called twice: once at prologue start,
        # once during early persist retry on first persist failure.
        assert mock_ensure.call_count == 2

    # Despite the first persist failing, the second attempt should succeed.
    assert call_count[0] == 2
    assert len(agent._persist_session_calls) == 2


def test_early_persist_logs_warning_after_retry_failure():
    """Verify early persist logs a structured warning if persistence still fails."""
    agent = _FakeAgent()

    # Mock _persist_session to always fail.
    def _persist_session_with_persistent_failure(messages, conversation_history):
        raise RuntimeError("Persistent DB failure")

    agent._persist_session = _persist_session_with_persistent_failure

    # Mock logger.warning to capture the call.
    with patch("agent.turn_context.logger.warning") as mock_warning:
        with patch.object(agent, "_ensure_db_session"):
            ctx = _build(agent)

        # logger.warning should have been called with the retry-failed message.
        assert mock_warning.called
        call_args = mock_warning.call_args
        assert "Early turn-start session persistence failed after retry" in call_args[0][0]
        assert agent.session_id in call_args[0]


def test_early_persist_user_message_idx_set():
    """Verify _persist_user_message_idx is set to the user message index."""
    agent = _FakeAgent()
    ctx = _build(agent)

    assert agent._persist_user_message_idx == ctx.current_turn_user_idx
    assert agent._persist_user_message_idx == len(ctx.messages) - 1
