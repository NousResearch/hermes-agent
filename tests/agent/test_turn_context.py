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
        # No session DB by default — tests that exercise SQLite persistence
        # attach one via ``_FakeAgentWithDB``.
        self._session_db = None
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

    # Inbound user-message persist (#45110) — called by build_turn_context
    # immediately after the user message is appended. The FakeAgent has no
    # real session_db, so this is a no-op that mirrors the production
    # guard (``if not self._session_db: return``). Tests that exercise
    # this method attach a real (in-memory) SessionDB via ``_FakeAgentWithDB``.
    def _persist_inbound_user_message(self, *_a, **_k):
        self._inbound_persist_calls = getattr(self, "_inbound_persist_calls", 0) + 1


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

# =============================================================================
# Inbound user-message persistence (#45110)
# =============================================================================
# The user message must land in state.db on receipt, before any model call,
# so a stall, a streaming client disconnect, or a mid-generation process
# crash does not lose the prompt. The targeted
# ``agent._persist_inbound_user_message`` call in the prologue writes ONLY
# the user message directly to SQLite, independent of the full
# ``_persist_session`` crash-resilience call that runs the rest of the
# flush. These tests pin both halves of the contract.


class _InMemorySessionDB:
    """Minimal in-memory stand-in for ``hermes_state.SessionDB``.

    Records the call args so the test can assert the user message reached
    SQLite (via ``append_message``) on receipt, without spinning up a real
    SQLite file.
    """

    def __init__(self):
        self.rows = []
        self._session_created = False
        self._append_message_calls = []

    def ensure_session(self, *a, **kw):
        self._session_created = True

    def append_message(self, **kwargs):
        self._append_message_calls.append(kwargs)
        self.rows.append(kwargs)
        return len(self.rows)

    # The agent's prologue also calls these for the full _persist_session
    # path; we ignore them here.
    def list_messages(self, *_a, **_k):
        return list(self.rows)


def _FakeAgentWithDB(db):
    """Builds a _FakeAgent that routes ``_ensure_db_session`` and the
    session_db calls to the in-memory stub.

    The inbound user-message persist method is bound to the real
    ``AIAgent._persist_inbound_user_message`` so the test exercises
    production SQLite-write behaviour (via the in-memory stub), not the
    FakeAgent's counter-only stub. The other prologue methods (the
    ``_persist_session`` crash-resilience call etc.) stay stubbed — only
    the new method is wired up because it's the one we're regression-
    testing.
    """
    from run_agent import AIAgent
    agent = _FakeAgent()
    agent._session_db = db
    agent._ensure_db_session = db.ensure_session
    # Bind the real implementation so ``_persist_inbound_user_message``
    # actually hits the in-memory stub via ``self._session_db``.
    agent._persist_inbound_user_message = lambda user_msg, _impl=AIAgent._persist_inbound_user_message: _impl(agent, user_msg)
    return agent


def test_persist_inbound_user_message_writes_user_row_on_receipt():
    """#45110: build_turn_context must persist the user message to
    state.db on receipt, before the first model call.

    A stall, a streaming client disconnect, or a mid-generation process
    crash can prevent the assistant response from completing. Without
    this persist, the user prompt vanishes from the session history as
    if it were never asked.
    """
    db = _InMemorySessionDB()
    agent = _FakeAgentWithDB(db)
    _build(agent, user_message="hello world")

    # At least one append_message call fired with role=user and the
    # exact content. The targeted inbound call is the FIRST one (the
    # crash-resilience ``_persist_session`` call would also append, but
    # the targeted call is the "guarantee" path).
    user_calls = [
        c for c in db._append_message_calls
        if c.get("role") == "user" and c.get("content") == "hello world"
    ]
    assert len(user_calls) >= 1, (
        f"user message never persisted on receipt — calls: "
        f"{db._append_message_calls!r}"
    )
    # First user-role call has role=user (sanity).
    assert db._append_message_calls[0]["role"] == "user"
    assert db._session_created is True


def test_persist_inbound_user_message_handles_no_db():
    """No ``_session_db`` configured → silently skip (test stubs,
    ephemeral flows). Mirrors the same guard in
    ``_flush_messages_to_session_db``.
    """
    agent = _FakeAgent()
    assert agent._session_db is None  # _FakeAgent default
    # Should not raise.
    _build(agent, user_message="hello")


def test_persist_inbound_user_message_persists_only_user_not_assistant():
    """The targeted inbound call must NOT persist assistant messages,
    tool results, or other non-user content. Only the inbound user
    turn. The full _persist_session call later in the prologue handles
    everything else.
    """
    db = _InMemorySessionDB()
    agent = _FakeAgentWithDB(db)
    _build(agent, user_message="hello")

    # All calls from the targeted inbound path are role=user. The
    # crash-resilience _persist_session call may also fire, but at
    # build_turn_context time there are no assistant messages yet
    # (we haven't called any model). So every append_message call in
    # the in-memory stub is a user-role call.
    for c in db._append_message_calls:
        assert c.get("role") == "user", (
            f"non-user role leaked into inbound persist call: {c!r}"
        )


def test_persist_inbound_user_message_skips_non_user_dicts():
    """Defensive guard: only dicts with role=user are persisted. A
    stray tool-result or assistant message accidentally passed in is
    ignored.
    """
    db = _InMemorySessionDB()
    agent = _FakeAgentWithDB(db)
    # Build normally (so session row exists), then call directly with
    # a non-user dict.
    _build(agent, user_message="hello")
    db._append_message_calls.clear()
    agent._persist_inbound_user_message({"role": "assistant", "content": "x"})
    agent._persist_inbound_user_message({"role": "tool", "content": "y"})
    agent._persist_inbound_user_message("not a dict")
    agent._persist_inbound_user_message(None)
    assert db._append_message_calls == []


def test_persist_inbound_user_message_strips_multimodal_to_text_summary():
    """Multimodal user content (list of content parts with text +
    image blocks) is persisted as a text-only summary — base64 images
    bloat the session DB and aren't useful for cross-session replay.
    Mirrors the same logic in ``_flush_messages_to_session_db``.
    """
    db = _InMemorySessionDB()
    agent = _FakeAgentWithDB(db)
    _build(agent)
    db._append_message_calls.clear()
    multimodal = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ],
    }
    agent._persist_inbound_user_message(multimodal)
    assert len(db._append_message_calls) == 1
    assert db._append_message_calls[0]["content"] == "what is in this image?\n[screenshot]"
