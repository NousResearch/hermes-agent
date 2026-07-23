"""Tests for issue #40170: Honcho memory injection guard on customer-facing platforms.

The security fix suppresses the automatic external-memory ``prefetch_all()``
call for customer-facing gateways so operator-level memory context cannot leak
to customers via indirect prompt injection. Memory *tools* stay available; only
the per-turn auto-injection is skipped when ``agent._skip_memory_injection`` is
set.

These tests exercise the real production path — ``build_turn_context`` in
``agent/turn_context.py``, where the prefetch actually happens — rather than
re-implementing the conditional in the test body. If the guard is removed, the
prefetch site moves again, or the wiring regresses, these fail.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import TurnContext, build_turn_context


class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class _FakeAgent:
    """Minimal stand-in covering only what the prologue touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "telegram"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = False
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
        self._invalid_tool_retries = -1
        self._vision_supported = None

    def _ensure_db_session(self):
        pass

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

    def _persist_session(self, *_a, **_k):
        pass


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main`` as a
    production side effect; stub it so these unit tests stay hermetic."""
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _make_memory_manager():
    mm = MagicMock()
    mm.prefetch_all = MagicMock(return_value="OPERATOR MEMORY CONTEXT")
    mm.on_turn_start = MagicMock()
    return mm


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


def test_prefetch_runs_by_default():
    """Without the guard flag, memory is prefetched and cached into the turn."""
    agent = _FakeAgent()
    agent._memory_manager = _make_memory_manager()

    ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._memory_manager.prefetch_all.assert_called_once()
    assert ctx.ext_prefetch_cache == "OPERATOR MEMORY CONTEXT"


def test_skip_flag_suppresses_prefetch():
    """With ``_skip_memory_injection`` set, prefetch_all is never called and no
    operator memory is injected into the turn context."""
    agent = _FakeAgent()
    agent._memory_manager = _make_memory_manager()
    agent._skip_memory_injection = True

    ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._memory_manager.prefetch_all.assert_not_called()
    assert ctx.ext_prefetch_cache == ""


def test_skip_flag_leaves_turn_start_notification_intact():
    """Skipping injection only suppresses prefetch — providers are still told a
    new turn started, so memory tools remain functional."""
    agent = _FakeAgent()
    agent._memory_manager = _make_memory_manager()
    agent._skip_memory_injection = True

    _build(agent)

    agent._memory_manager.on_turn_start.assert_called_once()
    agent._memory_manager.prefetch_all.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
