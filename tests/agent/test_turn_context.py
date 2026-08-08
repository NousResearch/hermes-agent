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
from agent.turn_context import TurnContext, build_turn_context, prepend_user_note
from hermes_state import SessionDB


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
        # Keep generic turn-prologue tests independent from the process-wide
        # registry. Refresh-specific tests opt in explicitly below.
        self._skip_mcp_refresh = True
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

    def _warn_context_overflow_blocked(
        self, reason, preflight_tokens, threshold_tokens
    ):
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

    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
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
    with patch(
        "agent.auxiliary_client.set_runtime_main", lambda *a, **k: None
    ), patch(
        "hermes_cli.config.read_raw_config", return_value={}
    ):
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


def test_between_turns_refresh_adds_late_tool_when_mcp_module_loaded():
    """R1: a tool that registered since build lands in this turn's snapshot."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = False

    new_def = {
        "type": "function",
        "function": {"name": "mcp_x_tool", "description": "", "parameters": {}},
    }

    import model_tools
    import tools.mcp_tool  # noqa: F401 -- activates the cheap sys.modules gate

    with patch.object(model_tools, "get_tool_definitions", return_value=[new_def]):
        _build(agent)

    assert "mcp_x_tool" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_x_tool" for t in agent.tools)


def test_between_turns_refresh_observes_removal_to_empty_catalog():
    """Once MCP loaded, refresh still runs after the last tool is removed."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = False
    import model_tools
    import tools.mcp_tool  # noqa: F401 -- activates the cheap sys.modules gate

    with patch.object(model_tools, "get_tool_definitions", return_value=[]) as gtd:
        _build(agent)

    gtd.assert_called_once()


def test_between_turns_refresh_skips_synchronized_empty_catalog():
    """An empty catalog does not trigger a full schema rebuild every turn."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = False
    import model_tools
    import tools.mcp_tool  # noqa: F401 -- activates the cheap sys.modules gate
    from tools.registry import registry

    agent._tool_snapshot_generation = registry._generation
    with (
        patch("tools.mcp_tool.has_registered_mcp_tools", return_value=False),
        patch.object(model_tools, "get_tool_definitions") as gtd,
    ):
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_skipped_when_skip_flag_set():
    """Internal forks (background_review) set _skip_mcp_refresh to keep tools[]
    byte-identical to the parent for cache parity — the hook must honor it even
    when MCP servers are registered."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = True
    import model_tools

    with patch.object(model_tools, "get_tool_definitions") as gtd:
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_no_churn_when_unchanged():
    """R2: an unchanged tool set leaves the snapshot object identity intact
    (no needless swap → nothing for the next request prefix to diff against)."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = False
    same = [
        {
            "type": "function",
            "function": {"name": "a", "description": "", "parameters": {}},
        }
    ]
    agent.tools = same
    agent.valid_tool_names = {"a"}

    import model_tools

    with patch.object(
        model_tools,
        "get_tool_definitions",
        return_value=[
            {
                "type": "function",
                "function": {"name": "a", "description": "", "parameters": {}},
            }
        ],
    ):
        _build(agent)

    assert agent.tools is same  # not replaced → no churn


def test_pending_mcp_notice_folds_into_next_real_user_turn_once():
    agent = _FakeAgent()
    agent._skip_mcp_refresh = True
    agent._pending_mcp_catalog_notice = "[MCP CATALOG UPDATED]"

    first = _build(agent)
    assert first.messages[-1]["content"] == "[MCP CATALOG UPDATED]\n\nhello"
    assert first.original_user_message == "hello"
    assert agent._pending_mcp_catalog_notice is None

    second = _build(agent)
    assert second.messages[-1]["content"] == "hello"


def test_initial_tool_catalog_snapshot_folds_into_real_user_turn():
    from tools.registry import registry
    from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig

    name = "mcp_turn_snapshot_initial"
    tool_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Create a turn snapshot.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    registry.register(
        name=name,
        handler=lambda args, **kw: "{}",
        schema=tool_def["function"],
        toolset="mcp-turn-snapshot",
    )
    try:
        agent = _FakeAgent()
        agent._skip_mcp_refresh = True
        agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
        agent.context_compressor.context_length = 200_000
        with (
            patch(
                "model_tools.get_tool_definitions", return_value=[tool_def]
            ) as get_defs,
            patch(
                "tools.tool_search.load_config",
                return_value=ToolSearchConfig.from_raw({"listing": "auto"}),
            ),
        ):
            ctx = _build(agent)
            second = _build(
                agent,
                conversation_history=[
                    {
                        "role": "user",
                        "content": "hello",
                        "api_content": ctx.messages[-1]["content"],
                    },
                    {"role": "assistant", "content": "done"},
                ],
            )

        content = ctx.messages[-1]["content"]
        assert content.endswith("\n\nhello")
        assert "[HERMES TOOL CATALOG SNAPSHOT" in content
        assert name in content
        assert ctx.original_user_message == "hello"
        assert [message["role"] for message in ctx.messages] == ["user"]
        assert second.messages[-1]["content"] == "hello"
        get_defs.assert_called_once()
    finally:
        registry.deregister(name)


def test_resumed_agent_reuses_prior_snapshot_without_reannouncing():
    from tools.registry import registry
    from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig

    name = "mcp_turn_snapshot_resume"
    tool_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Resume a catalog snapshot.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    registry.register(
        name=name,
        handler=lambda args, **kw: "{}",
        schema=tool_def["function"],
        toolset="mcp-turn-snapshot",
    )
    try:
        config = ToolSearchConfig.from_raw({"listing": "auto"})
        first_agent = _FakeAgent()
        first_agent._skip_mcp_refresh = True
        first_agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
        first_agent.context_compressor.context_length = 200_000
        with (
            patch("model_tools.get_tool_definitions", return_value=[tool_def]),
            patch("tools.tool_search.load_config", return_value=config),
        ):
            first = _build(first_agent)

        history = [
            {
                "role": "user",
                "content": "hello",
                "api_content": first.messages[-1]["content"],
            },
            {"role": "assistant", "content": "done"},
        ]
        resumed_agent = _FakeAgent()
        resumed_agent._skip_mcp_refresh = True
        resumed_agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
        resumed_agent.context_compressor.context_length = 200_000
        with (
            patch("model_tools.get_tool_definitions", return_value=[tool_def]),
            patch("tools.tool_search.load_config", return_value=config),
        ):
            resumed = _build(resumed_agent, conversation_history=history)

        assert resumed.messages[-1]["content"] == "hello"
        assert [message["role"] for message in resumed.messages] == [
            "user",
            "assistant",
            "user",
        ]
    finally:
        registry.deregister(name)


def test_same_agent_reannounces_snapshot_when_history_is_absent():
    from tools.registry import registry
    from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig

    name = "mcp_turn_snapshot_stateless"
    tool_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Serve a stateless catalog snapshot.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    registry.register(
        name=name,
        handler=lambda args, **kw: "{}",
        schema=tool_def["function"],
        toolset="mcp-turn-snapshot",
    )
    try:
        agent = _FakeAgent()
        agent._skip_mcp_refresh = True
        agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
        agent.context_compressor.context_length = 200_000
        with (
            patch("model_tools.get_tool_definitions", return_value=[tool_def]),
            patch(
                "tools.tool_search.load_config",
                return_value=ToolSearchConfig.from_raw({"listing": "auto"}),
            ),
        ):
            first = _build(agent)
            second = _build(agent)

        assert "[HERMES TOOL CATALOG SNAPSHOT" in first.messages[-1]["content"]
        assert "[HERMES TOOL CATALOG SNAPSHOT" in second.messages[-1]["content"]
    finally:
        registry.deregister(name)


def test_compression_reanchors_catalog_on_latest_real_user_sidecar_once():
    from agent.conversation_compression import (
        _reanchor_tool_catalog_snapshot_after_compression,
    )
    from tools.tool_search import build_catalog_snapshot

    snapshot = build_catalog_snapshot([])
    agent = types.SimpleNamespace(
        _tool_catalog_snapshot_id=snapshot.snapshot_id,
        _tool_catalog_snapshot_notice=snapshot.notice,
    )
    compressed = [
        {
            "role": "user",
            "content": "older request",
            "api_content": snapshot.notice + "\n\nolder request",
        },
        {"role": "assistant", "content": "compressed summary"},
        {
            "role": "user",
            "content": "current request",
            "api_content": "current request\n\nPLUGIN-CTX",
        },
    ]

    assert _reanchor_tool_catalog_snapshot_after_compression(agent, compressed)
    assert compressed[-1]["content"] == "current request"
    assert compressed[-1]["api_content"] == (
        snapshot.notice + "\n\ncurrent request\n\nPLUGIN-CTX"
    )

    assert not _reanchor_tool_catalog_snapshot_after_compression(agent, compressed)
    assert compressed[-1]["api_content"].count(snapshot.notice) == 1


def test_compression_reanchors_catalog_on_synthetic_user_fallback():
    from agent.conversation_compression import (
        _reanchor_tool_catalog_snapshot_after_compression,
    )
    from tools.tool_search import build_catalog_snapshot

    snapshot = build_catalog_snapshot([])
    agent = types.SimpleNamespace(
        _tool_catalog_snapshot_id=snapshot.snapshot_id,
        _tool_catalog_snapshot_notice=snapshot.notice,
    )
    compressed = [
        {
            "role": "assistant",
            "content": "summary",
            "_compressed_summary": True,
        },
        {
            "role": "user",
            "content": (
                "Continue from the compressed conversation context above. "
                "This marker exists because no human user turn was available."
            ),
            "_empty_recovery_synthetic": True,
        },
    ]

    assert _reanchor_tool_catalog_snapshot_after_compression(agent, compressed)
    assert compressed[-1]["content"].startswith("Continue from")
    assert compressed[-1]["api_content"].startswith(snapshot.notice + "\n\n")


def test_compression_reanchors_catalog_in_multimodal_text_part():
    from agent.conversation_compression import (
        _reanchor_tool_catalog_snapshot_after_compression,
    )
    from tools.tool_search import build_catalog_snapshot

    snapshot = build_catalog_snapshot([])
    agent = types.SimpleNamespace(
        _tool_catalog_snapshot_id=snapshot.snapshot_id,
        _tool_catalog_snapshot_notice=snapshot.notice,
    )
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
    text_part = {"type": "text", "text": "inspect this image"}
    compressed = [
        {
            "role": "user",
            "content": [image_part, text_part],
        }
    ]

    assert _reanchor_tool_catalog_snapshot_after_compression(agent, compressed)
    assert "api_content" not in compressed[-1]
    assert compressed[-1]["content"][0] is image_part
    assert compressed[-1]["content"][1] is not text_part
    assert compressed[-1]["content"][1]["text"] == (
        snapshot.notice + "\n\ninspect this image"
    )


def test_pending_mcp_server_names_respect_status_and_session_scope():
    from agent.turn_context import _pending_mcp_server_names

    raw_config = {
        "mcp_servers": {
            "connecting": {"enabled": True},
            "configured": {},
            "ready": {},
            "failed": {},
            "disabled": {"enabled": False},
        }
    }
    statuses = [
        {"name": "connecting", "status": "connecting"},
        {"name": "configured", "status": "configured"},
        {"name": "ready", "status": "connected"},
        {"name": "failed", "status": "failed"},
    ]
    with patch(
        "hermes_cli.config.read_raw_config", return_value=raw_config
    ), patch(
        "tools.mcp_tool.get_mcp_status", return_value=statuses
    ):
        assert _pending_mcp_server_names(None, None) == (
            "configured",
            "connecting",
        )
        assert _pending_mcp_server_names(["connecting"], None) == ("connecting",)
        assert _pending_mcp_server_names(
            ["connecting", "configured"],
            ["mcp-connecting"],
        ) == ("configured",)


def test_pending_mcp_server_probe_does_not_import_mcp_stack():
    import sys

    from agent.turn_context import _pending_mcp_server_names

    with patch(
        "hermes_cli.config.read_raw_config",
        return_value={"mcp_servers": {"slow-server": {}}},
    ), patch.dict(
        sys.modules,
        {"tools.mcp_tool": None},
    ):
        assert _pending_mcp_server_names(None, None) == ("slow-server",)


def test_pending_mcp_resolution_publishes_superseding_snapshot():
    from tools.tool_search import (
        BRIDGE_TOOL_NAMES,
        ToolSearchConfig,
        catalog_snapshot_id_from_text,
    )

    agent = _FakeAgent()
    agent._skip_mcp_refresh = True
    agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
    agent.context_compressor.context_length = 200_000
    config = ToolSearchConfig.from_raw({"listing": "auto"})

    with patch(
        "agent.turn_context._pending_mcp_server_names",
        side_effect=[("slow-github",), ()],
    ), patch(
        "model_tools.get_tool_definitions", return_value=[]
    ) as get_defs, patch(
        "tools.tool_search.load_config", return_value=config
    ):
        first = _build(agent)
        first_content = first.messages[-1]["content"]
        history = [
            {
                "role": "user",
                "content": "hello",
                "api_content": first_content,
            },
            {"role": "assistant", "content": "done"},
        ]
        second = _build(agent, conversation_history=history)

    second_content = second.messages[-1]["content"]
    assert "MCP servers still initializing (1): `slow-github`" in first_content
    assert "not included in the ready catalog yet" in first_content
    assert "[HERMES TOOL CATALOG SNAPSHOT" in second_content
    assert "MCP servers still initializing" not in second_content
    assert catalog_snapshot_id_from_text(first_content) != (
        catalog_snapshot_id_from_text(second_content)
    )
    assert get_defs.call_count == 2


def test_schema_change_appends_superseding_catalog_snapshot():
    from tools.registry import registry
    from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig

    name = "mcp_turn_snapshot_schema_change"
    old_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Schema-changing tool.",
            "parameters": {
                "type": "object",
                "properties": {"old": {"type": "string"}},
            },
        },
    }
    new_def = {
        "type": "function",
        "function": {
            **old_def["function"],
            "parameters": {
                "type": "object",
                "properties": {"fresh": {"type": "integer"}},
            },
        },
    }
    registry.register(
        name=name,
        handler=lambda args, **kw: "{}",
        schema=old_def["function"],
        toolset="mcp-turn-snapshot",
    )
    try:
        config = ToolSearchConfig.from_raw({"listing": "auto"})
        old_agent = _FakeAgent()
        old_agent._skip_mcp_refresh = True
        old_agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
        old_agent.context_compressor.context_length = 200_000
        with (
            patch("model_tools.get_tool_definitions", return_value=[old_def]),
            patch("tools.tool_search.load_config", return_value=config),
        ):
            old_ctx = _build(old_agent)

        history = [
            {
                "role": "user",
                "content": "hello",
                "api_content": old_ctx.messages[-1]["content"],
            },
            {"role": "assistant", "content": "done"},
        ]
        new_agent = _FakeAgent()
        new_agent._skip_mcp_refresh = True
        new_agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
        new_agent.context_compressor.context_length = 200_000
        new_agent._pending_mcp_catalog_notice = "[TOOL CATALOG UPDATED]"
        with (
            patch("model_tools.get_tool_definitions", return_value=[new_def]),
            patch("tools.tool_search.load_config", return_value=config),
        ):
            new_ctx = _build(new_agent, conversation_history=history)

        current = new_ctx.messages[-1]["content"]
        assert current.startswith("[TOOL CATALOG UPDATED]\n\n")
        assert "[HERMES TOOL CATALOG SNAPSHOT" in current
        assert "supersedes every earlier" in current
        assert current.endswith("\n\nhello")
    finally:
        registry.deregister(name)


def test_listing_off_keeps_stable_bridge_bare_on_initial_turn():
    from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig

    agent = _FakeAgent()
    agent._skip_mcp_refresh = True
    agent.valid_tool_names = set(BRIDGE_TOOL_NAMES)
    with patch(
        "tools.tool_search.load_config",
        return_value=ToolSearchConfig.from_raw({"listing": "off"}),
    ):
        ctx = _build(agent)

    assert ctx.messages[-1]["content"] == "hello"


def test_prepend_user_note_preserves_multimodal_parts():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    parts = [{"type": "text", "text": "caption"}, image]

    result = prepend_user_note(parts, "[NOTICE]")

    assert result == [
        {"type": "text", "text": "[NOTICE]\n\ncaption"},
        image,
    ]
    assert parts[0]["text"] == "caption"  # caller-owned input was not mutated
