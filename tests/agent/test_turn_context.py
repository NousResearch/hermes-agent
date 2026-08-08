"""Unit tests for the extracted turn prologue (``agent/turn_context.py``).

These exercise ``build_turn_context`` against a lightweight fake agent to
confirm the prologue produces the right ``TurnContext`` and applies the
``agent`` side effects the loop relies on — without spinning up a real
``AIAgent`` or hitting any provider.
"""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.turn_context import TurnContext, build_turn_context
from hermes_state import SessionDB


class _FakeTodoStore:
    def __init__(self):
        self.items = []

    def has_items(self):
        return bool(self.items)

    def read(self):
        return [item.copy() for item in self.items]

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
        self.requested_provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
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
        # Make the fake compressor honour the ContextEngine contract that the
        # real code now relies on (should_compress_info returns a (bool, reason)
        # tuple). Without it build_turn_context raises AttributeError.
        def _fake_should_compress(tokens=None):
            return False

        def _fake_should_compress_info(tokens=None):
            return (False, None)

        self.context_compressor.should_compress = _fake_should_compress
        self.context_compressor.should_compress_info = _fake_should_compress_info
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._emit_warning = MagicMock()
        self._last_ctx_overflow_warn = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        # Attributes the prologue assigns; recorded for assertions.
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0
        self._session_messages = []
        self._pending_cli_user_message = None
        self._session_persist_lock = threading.RLock()
        # Records _cached_system_prompt at the moment _ensure_db_session()
        # is called (regression guard for #45499 turn-setup ordering).
        self._ensure_db_prompt_at_call = "<unset>"

    def _warn_context_overflow_blocked(self, reason, preflight_tokens, threshold_tokens):
        # Mirror the real AIAgent helper so tests can assert the warning fired.
        _warn_kind = (reason or "unknown").split(":", 1)[0]
        _warn_key = ("ctx_overflow_blocked", _warn_kind)
        if self._last_ctx_overflow_warn != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            self._emit_warning(
                f"⚠ Context is over the compression threshold "
                f"(~{preflight_tokens:,} tokens >= {threshold_tokens:,}) "
                f"but compression is currently blocked ({reason})."
            )

    def _clear_context_overflow_warn(self):
        self._last_ctx_overflow_warn = None

    # --- methods the prologue calls ---
    def _ensure_db_session(self):
        self._ensure_db_prompt_at_call = self._cached_system_prompt

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
        self._persist_calls += 1


def _make_agent_with_cooldown(db_path, session_id, *, cooldown_until=None):
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._emit_status = MagicMock()
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM")
    )

    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="cli")
    if cooldown_until is not None:
        db.record_compression_failure_cooldown(session_id, cooldown_until, "timeout")

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    agent.context_compressor = compressor
    agent._session_db = db
    return agent


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main`` as a
    production side effect (telling aux tools the live main provider/model).
    That writes a module-level global these unit tests don't care about and
    which would otherwise leak into sibling tests (e.g. provider-parity
    resolution) when the per-test process isolation plugin is disabled. Stub
    it out so the prologue tests stay hermetic.
    """
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


def test_returns_turn_context_with_user_message_appended():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert isinstance(ctx, TurnContext)
    assert ctx.user_message == "hello"
    # The user turn was appended and indexed.
    assert ctx.messages[-1] == {"role": "user", "content": "hello"}
    assert ctx.current_turn_user_idx == len(ctx.messages) - 1
    assert ctx.active_system_prompt == "SYSTEM"


# ── Trivial-prompt prefetch gate (PR #25350 salvage) ─────────────────────────
#
# The prologue is the ONLY place the per-turn synchronous
# memory_manager.prefetch_all() fires; a bare greeting must not block the
# turn on provider network round-trips, while a substantive question must
# still prefetch. These assert the gate at the call site (the classifier
# itself is covered in tests/agent/test_memory_provider.py).


def _agent_with_memory_manager():
    agent = _FakeAgent()
    mm = MagicMock()
    mm.prefetch_all.return_value = "REMEMBERED CONTEXT"
    agent._memory_manager = mm
    return agent, mm


def test_prefetch_skipped_for_trivial_user_message():
    agent, mm = _agent_with_memory_manager()
    ctx = _build(agent, user_message="hi!")
    mm.prefetch_all.assert_not_called()
    assert ctx.ext_prefetch_cache == ""


def test_prefetch_runs_for_substantive_user_message():
    agent, mm = _agent_with_memory_manager()
    query = "what did we decide about the deploy pipeline?"
    ctx = _build(agent, user_message=query)
    mm.prefetch_all.assert_called_once_with(query)
    assert ctx.ext_prefetch_cache == "REMEMBERED CONTEXT"


def test_turn_start_replaces_stale_parent_history_with_compression_child():
    agent = _FakeAgent()
    stale_history = [{"role": "user", "content": "stale parent"}]
    compacted_history = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "child tail"},
    ]

    def _recover(_agent):
        _agent.session_id = "compression-child"
        return compacted_history

    log_context = MagicMock()
    with patch(
        "agent.turn_context.recover_rotated_compression_session",
        side_effect=_recover,
    ):
        ctx = _build(
            agent,
            conversation_history=stale_history,
            set_session_context=log_context,
        )

    assert agent.session_id == "compression-child"
    assert agent._current_turn_id.startswith("compression-child:")
    log_context.assert_called_once_with("compression-child")
    assert ctx.conversation_history == compacted_history
    assert ctx.messages == compacted_history + [{"role": "user", "content": "hello"}]
    assert all(message.get("content") != "stale parent" for message in ctx.messages)


def test_applies_agent_side_effects():
    agent = _FakeAgent()
    _build(agent)
    # Retry counters reset, guardrails reset, vision re-armed, turn counted.
    assert agent._invalid_tool_retries == 0
    assert agent._tool_guardrails.reset_called is True
    assert agent._vision_supported is True
    assert agent._user_turn_count == 1
    # Crash-resilience persistence fired once.
    assert agent._persist_calls == 1
    # task/turn ids assigned on the agent.
    assert agent._current_task_id
    assert agent._current_turn_id


def test_resume_hydration_reaches_pre_llm_hook_before_agent_work():
    agent = _FakeAgent()
    hydrated = [
        {"id": "resume", "content": "continue prior work", "status": "in_progress"}
    ]

    def _hydrate(_history):
        agent._todo_store.items = [item.copy() for item in hydrated]

    agent._hydrate_todo_store = _hydrate
    observed = {}

    def _hook(name, **kwargs):
        if name == "pre_llm_call":
            observed.update(kwargs)
        return []

    with patch("hermes_cli.lifecycle.invoke_hook", side_effect=_hook):
        _build(agent, conversation_history=[{"role": "user", "content": "prior"}])

    assert observed["session_id"] == "sess-1"
    assert observed["todos"] == hydrated
    assert observed["todos"] is not agent._todo_store.items










def test_pending_cli_message_uses_clean_override_for_api_local_note():
    """A noted API message reuses the clean staged dict and its DB marker."""
    agent = _FakeAgent()
    staged = {"role": "user", "content": "clean prompt", "_db_persisted": True}
    agent._pending_cli_user_message = staged

    ctx = _build(
        agent,
        user_message="[MODEL NOTE]\n\nclean prompt",
        persist_user_message="clean prompt",
    )

    assert ctx.messages[-1] is staged
    assert ctx.messages[-1]["content"] == "[MODEL NOTE]\n\nclean prompt"
    assert ctx.messages[-1]["_db_persisted"] is True
    assert agent._pending_cli_user_message is None








def test_ensure_db_session_runs_after_system_prompt_restore():
    """Regression for #45499.

    On a fresh API/gateway agent (``_cached_system_prompt is None``) the DB
    session row must be created AFTER the system prompt is restored/built, so
    the persisted snapshot is written non-NULL. If ``_ensure_db_session()``
    ran first it would insert ``system_prompt=NULL`` and trip the misleading
    "stored system prompt is null; rebuilding" warning plus a first-turn
    prefix cache miss.
    """
    agent = _FakeAgent()
    agent._cached_system_prompt = None  # fresh agent, no cached prompt yet

    def _restore(_agent, _system_message, _history):
        _agent._cached_system_prompt = "REBUILT-SYSTEM"

    _build(agent, restore_or_build_system_prompt=_restore)

    # The prompt was populated before the DB row was created.
    assert agent._ensure_db_prompt_at_call == "REBUILT-SYSTEM"
    assert agent._cached_system_prompt == "REBUILT-SYSTEM"


# ── Between-turns MCP refresh (cache-safe late-binding) ──────────────────────
#
# A slow MCP server that connects after the agent's build-time tool snapshot
# must become callable by the user's NEXT turn — without mutating an in-flight
# turn's cached request prefix. The prologue is exactly that boundary, so the
# refresh hook lives here. These assert the contract (R1/R2/R6 in the spec),
# not timing permutations.


def test_between_turns_refresh_adds_late_tool_when_servers_registered():
    """R1: a tool that registered since build lands in this turn's snapshot."""
    agent = _FakeAgent()

    new_def = {"type": "function", "function": {"name": "mcp_x_tool", "description": "", "parameters": {}}}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions", return_value=[new_def]):
        _build(agent)

    assert "mcp_x_tool" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_x_tool" for t in agent.tools)












