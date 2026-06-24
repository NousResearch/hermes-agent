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
        # Records _cached_system_prompt at the moment _ensure_db_session()
        # is called (regression guard for #45499 turn-setup ordering).
        self._ensure_db_prompt_at_call = "<unset>"

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



# ── Context windowing (_window_conversation_history) ──────────────────────

def test_window_noop_when_session_shorter_than_max():
    """Session with fewer messages than max_verbatim_messages → no windowing.
    All messages returned as-is, summary is None."""
    from agent.turn_context import _window_conversation_history

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you"},
    ]
    result, summary = _window_conversation_history(
        messages, session_id="sess-1", max_verbatim_messages=10
    )
    assert result is messages  # same list object — no copy needed
    assert summary is None


def test_window_noop_when_max_verbatim_zero():
    """max_verbatim_messages=0 is treated as disabled — no windowing."""
    from agent.turn_context import _window_conversation_history

    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    result, summary = _window_conversation_history(
        messages, session_id="sess-1", max_verbatim_messages=0
    )
    assert result is messages
    assert summary is None


def test_window_applies_when_session_longer_than_max(tmp_path):
    """Session longer than max_verbatim_messages → windowing applied.
    Last N messages kept verbatim, earlier messages replaced by summary."""
    from agent.turn_context import _window_conversation_history
    from unittest.mock import patch

    messages = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "msg2"},
        {"role": "user", "content": "msg3"},
        {"role": "assistant", "content": "msg4"},
        {"role": "user", "content": "msg5"},
        {"role": "assistant", "content": "msg6"},
        {"role": "user", "content": "msg7"},
        {"role": "assistant", "content": "msg8"},
        {"role": "user", "content": "msg9"},
        {"role": "assistant", "content": "msg10"},
    ]
    summary_text = "Earlier: user said hello, assistant replied."

    # Mock the summary file read
    with patch("agent.turn_context._read_running_summary", return_value=summary_text):
        result, summary = _window_conversation_history(
            messages, session_id="sess-1", max_verbatim_messages=4
        )

    # Last 4 messages kept verbatim
    assert len(result) == 5  # 1 summary user message + 4 verbatim
    assert result[0]["role"] == "user"
    assert "Earlier in this session" in result[0]["content"]
    assert summary_text in result[0]["content"]
    assert result[1:] == messages[-4:]
    assert summary == summary_text


def test_window_degrade_when_summary_missing(tmp_path):
    """Summary file doesn't exist → degrade to full history (no windowing)."""
    from agent.turn_context import _window_conversation_history
    from unittest.mock import patch

    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
        {"role": "assistant", "content": "f"},
    ]
    with patch("agent.turn_context._read_running_summary", return_value=None):
        result, summary = _window_conversation_history(
            messages, session_id="sess-1", max_verbatim_messages=3
        )
    assert result is messages
    assert summary is None


def test_window_preserves_tool_chain_at_boundary():
    """When the cut point lands on a tool-role result, extend backward
    to include the full tool-call chain (assistant + tool results)."""
    from agent.turn_context import _window_conversation_history
    from unittest.mock import patch

    messages = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "msg2"},
        {"role": "user", "content": "msg3"},
        {"role": "assistant", "content": "msg4", "tool_calls": [
            {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file content"},
        {"role": "assistant", "content": "msg5"},
        {"role": "user", "content": "msg6"},
        {"role": "assistant", "content": "msg7"},
    ]
    summary_text = "Earlier summary."

    with patch("agent.turn_context._read_running_summary", return_value=summary_text):
        result, summary = _window_conversation_history(
            messages, session_id="sess-1", max_verbatim_messages=3
        )

    # max_verbatim=3 would normally keep last 3: [msg6, msg7].
    # But cut lands on msg5 (assistant, no tool_calls) — that's fine.
    # Actually: cut = 8 - 3 = 5 → messages[5] = msg6 (user). No tool-chain issue.
    # Let's verify the window includes the summary + last 3.
    assert len(result) == 4  # summary + 3 verbatim
    assert result[0]["role"] == "user"
    assert "Earlier in this session" in result[0]["content"]
    assert result[1:] == messages[-3:]


def test_window_extends_past_tool_chain():
    """When the cut point lands on a tool-role result, the window extends
    backward to include the assistant that initiated the tool calls."""
    from agent.turn_context import _window_conversation_history
    from unittest.mock import patch

    messages = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "msg2"},
        {"role": "user", "content": "msg3"},
        {"role": "assistant", "content": "msg4", "tool_calls": [
            {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file content"},
        {"role": "assistant", "content": "msg5"},
        {"role": "user", "content": "msg6"},
    ]
    summary_text = "Earlier summary."

    with patch("agent.turn_context._read_running_summary", return_value=summary_text):
        result, summary = _window_conversation_history(
            messages, session_id="sess-1", max_verbatim_messages=2
        )

    # max_verbatim=2: cut = 7 - 2 = 5 → messages[5] = msg5 (assistant, no tool_calls)
    # That's fine — no tool chain to protect.
    # But let's test with max_verbatim=1: cut = 6 → messages[6] = msg6 (user). Fine.
    # The real test: max_verbatim=3: cut = 4 → messages[4] = tool result.
    # Window should extend back to include msg4 (assistant with tool_calls).
    with patch("agent.turn_context._read_running_summary", return_value=summary_text):
        result2, _ = _window_conversation_history(
            messages, session_id="sess-1", max_verbatim_messages=3
        )

    # cut=4 lands on tool result → extends back past assistant with tool_calls
    # → cut becomes 3 (msg3, user). Window = messages[3:] = [msg3, msg4, tool, msg5, msg6]
    assert len(result2) == 6  # summary + 5 verbatim (extended past tool chain)
    assert result2[0]["role"] == "user"
    assert "Earlier in this session" in result2[0]["content"]
    # The verbatim window includes the full tool chain
    assert result2[1]["role"] == "user"  # msg3
    assert result2[2]["role"] == "assistant"  # msg4 with tool_calls
    assert result2[3]["role"] == "tool"  # tool result
    assert result2[4]["role"] == "assistant"  # msg5
    assert result2[5]["role"] == "user"  # msg6


def test_build_turn_context_applies_windowing():
    """Full integration: build_turn_context with max_verbatim_messages set
    windows the conversation history, injecting a summary block."""
    from unittest.mock import patch

    agent = _FakeAgent()
    agent.session_id = "sess-window-test"

    # 10 messages in history + 1 new user message = 11 total
    history = [
        {"role": "user", "content": f"msg{i}"}
        if i % 2 == 0 else
        {"role": "assistant", "content": f"msg{i}"}
        for i in range(10)
    ]
    summary_text = "## 10:00\nUser asked about context windowing."

    with patch(
        "agent.turn_context._read_running_summary", return_value=summary_text
    ), patch(
        "hermes_cli.config.load_config",
        return_value={"context": {"max_verbatim_messages": 4}},
    ):
        ctx = _build(agent, conversation_history=history)

    # With max_verbatim=4, we should have: summary + last 3 history + new user msg
    # = 5 messages total (the new user msg is one of the 4 verbatim)
    assert len(ctx.messages) == 5
    assert ctx.messages[0]["role"] == "user"
    assert "Earlier in this session" in ctx.messages[0]["content"]
    assert summary_text in ctx.messages[0]["content"]
    # Last 4 messages are the verbatim window (3 history + 1 new)
    assert ctx.messages[1:] == history[-3:] + [{"role": "user", "content": "hello"}]
