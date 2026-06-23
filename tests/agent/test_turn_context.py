"""Unit tests for the extracted turn prologue (``agent/turn_context.py``).

These exercise ``build_turn_context`` against a lightweight fake agent to
confirm the prologue produces the right ``TurnContext`` and applies the
``agent`` side effects the loop relies on — without spinning up a real
``AIAgent`` or hitting any provider.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from agent.turn_context import TurnContext, build_turn_context


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
        # Attributes the prologue assigns; recorded for assertions.
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0

    # --- methods the prologue calls ---
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
        self._persist_calls += 1


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


def test_task_id_passthrough():
    agent = _FakeAgent()
    ctx = _build(agent, task_id="fixed-task")
    assert ctx.effective_task_id == "fixed-task"
    assert agent._current_task_id == "fixed-task"


def test_persist_user_message_becomes_original():
    agent = _FakeAgent()
    ctx = _build(agent, user_message="api-prefixed", persist_user_message="clean")
    # original_user_message tracks the clean persist override.
    assert ctx.original_user_message == "clean"
    # but the appended user turn carries the full (sanitized) message.
    assert ctx.messages[-1]["content"] == "api-prefixed"


def test_memory_nudge_fires_at_interval():
    agent = _FakeAgent()
    agent._memory_nudge_interval = 1
    agent.valid_tool_names = {"memory"}
    agent._memory_store = object()
    ctx = _build(agent)
    assert ctx.should_review_memory is True
    assert agent._turns_since_memory == 0  # reset after firing


def test_no_review_when_memory_disabled():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert ctx.should_review_memory is False


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


def test_between_turns_refresh_skipped_when_no_servers():
    """R6: the common case (no MCP servers) never walks the registry."""
    agent = _FakeAgent()
    import model_tools

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=False), \
         patch.object(model_tools, "get_tool_definitions") as gtd:
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_skipped_when_skip_flag_set():
    """Internal forks (background_review) set _skip_mcp_refresh to keep tools[]
    byte-identical to the parent for cache parity — the hook must honor it even
    when MCP servers are registered."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = True
    import model_tools

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions") as gtd:
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_no_churn_when_unchanged():
    """R2: an unchanged tool set leaves the snapshot object identity intact
    (no needless swap → nothing for the next request prefix to diff against)."""
    agent = _FakeAgent()
    same = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}]
    agent.tools = same
    agent.valid_tool_names = {"a"}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(
             model_tools, "get_tool_definitions",
             return_value=[{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}],
         ):
        _build(agent)

    assert agent.tools is same  # not replaced → no churn


# ── pre_llm_call model-swap + short-circuit return keys (generic engine seam) ─
#
# The ``pre_llm_call`` hook may now return two new optional bundles. These are
# generic — any plugin can drive a model swap or a short-circuit reply. The
# tests stub the hook dispatch and assert the engine acts on the returned dict.


def _patch_hook(results):
    """Patch the lazily-imported plugins.invoke_hook to yield *results*."""
    return patch("hermes_cli.plugins.invoke_hook", lambda *a, **k: list(results))


def test_pre_llm_call_passes_agent_and_not_redundant_session_key():
    """The hook must receive the live ``agent`` kwarg.

    The redundant ``session_key`` kwarg (it only duplicated ``session_id``) was
    dropped — routing now keys off the durable ``agent._user_model_pin`` flag,
    not a session key. ``session_id`` is still passed.
    """
    agent = _FakeAgent()
    seen = {}

    def _spy(hook_name, **kw):
        seen.update(kw)
        return []

    with patch("hermes_cli.plugins.invoke_hook", _spy):
        _build(agent)

    assert seen.get("agent") is agent
    assert "session_id" in seen
    assert "session_key" not in seen  # redundant kwarg removed


def test_pre_llm_call_model_bundle_triggers_switch_model():
    """A ``{model, provider, api_key, base_url, api_mode}`` bundle swaps the model."""
    agent = _FakeAgent()
    bundle = {
        "model": "example/model-pro",
        "provider": "exampleprovider",
        "api_key": "sk-z",
        "base_url": "https://api.example.com/v1",
        "api_mode": None,
    }
    calls = []

    def _fake_switch(a, new_model, new_provider, api_key="", base_url="", api_mode=""):
        calls.append((new_model, new_provider, api_key, base_url, api_mode))
        a.model = new_model
        a.provider = new_provider

    with _patch_hook([bundle]), \
         patch("agent.agent_runtime_helpers.switch_model", _fake_switch):
        ctx = _build(agent)

    # api_mode None is normalized to "" for switch_model's contract (it then
    # derives the real api_mode from provider/base_url).
    assert calls == [("example/model-pro", "exampleprovider", "sk-z",
                      "https://api.example.com/v1", "")]
    assert agent.model == "example/model-pro"
    # A model swap does NOT short-circuit the turn.
    assert ctx.short_circuit_response is None


def test_pre_llm_call_switch_model_failure_is_fail_open():
    """If switch_model raises, the prologue logs and keeps the profile model."""
    agent = _FakeAgent()
    bundle = {"model": "example/model-pro", "provider": "exampleprovider", "api_key": "sk-z",
              "base_url": "https://x", "api_mode": None}

    def _boom(*a, **k):
        raise RuntimeError("bad key")

    with _patch_hook([bundle]), \
         patch("agent.agent_runtime_helpers.switch_model", _boom):
        ctx = _build(agent)

    assert agent.model == "test/model"  # unchanged — fail-open
    assert ctx.short_circuit_response is None


def test_pre_llm_call_final_response_sets_short_circuit():
    """A ``{final_response}`` bundle sets TurnContext.short_circuit_response."""
    agent = _FakeAgent()
    with _patch_hook([{"final_response": "It is 3:00 PM in Chicago."}]):
        ctx = _build(agent)

    assert ctx.short_circuit_response == "It is 3:00 PM in Chicago."


def test_pre_llm_call_final_response_wins_over_model_bundle():
    """When both keys are present, final_response wins and no swap happens."""
    agent = _FakeAgent()
    results = [
        {"model": "example/model-pro", "provider": "exampleprovider", "api_key": "sk-z",
         "base_url": "https://x", "api_mode": None},
        {"final_response": "answered"},
    ]
    calls = []
    with _patch_hook(results), \
         patch("agent.agent_runtime_helpers.switch_model",
               lambda *a, **k: calls.append(1)):
        ctx = _build(agent)

    assert ctx.short_circuit_response == "answered"
    assert calls == []  # final_response present → swap skipped
    assert agent.model == "test/model"


def test_pre_llm_call_context_key_still_works():
    """Existing {context} / str behavior is unchanged (back-compat)."""
    agent = _FakeAgent()
    with _patch_hook([{"context": "recalled fact"}, "plain string"]):
        ctx = _build(agent)

    assert "recalled fact" in ctx.plugin_user_context
    assert "plain string" in ctx.plugin_user_context
    assert ctx.short_circuit_response is None

