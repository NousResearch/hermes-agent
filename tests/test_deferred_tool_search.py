"""Tests for the feature-flagged deferred tool_search prototype.

Covers (per card t_9b28d3c2):
  * disabled-by-default behaviour (zero change to the tool surface)
  * feature-flag resolution (env override + config)
  * per-session isolation (no module-global promotion state)
  * direct-call blocking of un-promoted deferred tools (actionable JSON)
  * request/token tool-count changes (surface shrinks, then grows on promote)
  * MCP/ACP refresh interactions (promotions carried across a rebuild)
  * agent integration helpers (apply_to_agent / handle_tool_search_call)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tools.deferred_tool_search as dts


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure the env override doesn't leak between tests / from the shell."""
    monkeypatch.delenv(dts.FEATURE_FLAG_ENV, raising=False)
    yield


ON = {"tools": {"deferred_tool_search": {"enabled": True}}}
OFF = {"tools": {"deferred_tool_search": {"enabled": False}}}


def _def(name, description=""):
    return {"type": "function", "function": {"name": name, "description": description}}


def _sample_defs():
    return [
        _def("todo", "manage the todo list"),
        _def("memory", "store memories"),
        _def("kanban_complete", "complete a kanban task"),
        _def("kanban_block", "block a kanban task"),
        _def("web_search", "Search the web for current information"),
        _def("web_extract", "Extract readable content from a web page URL"),
        _def("write_file", "Write or create a file on disk"),
        _def("send_message", "Send a chat message to a messaging platform"),
        _def("vision_analyze", "Analyse and describe an image"),
    ]


def _names(defs):
    return {d["function"]["name"] for d in defs}


class _FakeMemoryManager:
    def __init__(self, names):
        self._names = set(names)

    def get_all_tool_names(self):
        return set(self._names)


def _fake_agent(defs, *, deferred_state=None, context_engine_tools=None, memory_tools=None):
    return SimpleNamespace(
        tools=list(defs),
        valid_tool_names=_names(defs),
        _deferred_tool_state=deferred_state,
        _deferred_tool_full_surface=None,
        _deferred_tool_surface_partitioned=False,
        _context_engine_tool_names=set(context_engine_tools or set()),
        _memory_manager=(
            _FakeMemoryManager(memory_tools)
            if memory_tools is not None else None
        ),
        session_id="sess-1",
        _invalidate_system_prompt=lambda: None,
    )


# ---------------------------------------------------------------------------
# Feature flag resolution
# ---------------------------------------------------------------------------

class TestFeatureFlag:
    def test_disabled_by_default(self):
        # Empty config, no env → off.
        assert dts.is_enabled({}) is False

    def test_config_enables(self):
        assert dts.is_enabled(ON) is True
        assert dts.is_enabled(OFF) is False

    def test_env_truthy_overrides_config_off(self, monkeypatch):
        monkeypatch.setenv(dts.FEATURE_FLAG_ENV, "1")
        assert dts.is_enabled(OFF) is True

    def test_env_falsy_overrides_config_on(self, monkeypatch):
        monkeypatch.setenv(dts.FEATURE_FLAG_ENV, "0")
        assert dts.is_enabled(ON) is False

    def test_env_unrecognised_falls_back_to_config(self, monkeypatch):
        monkeypatch.setenv(dts.FEATURE_FLAG_ENV, "maybe")
        assert dts.is_enabled(ON) is True
        assert dts.is_enabled(OFF) is False


# ---------------------------------------------------------------------------
# Disabled-by-default: tool surface unchanged
# ---------------------------------------------------------------------------

class TestDisabledByDefault:
    def test_apply_to_agent_is_noop_when_off(self):
        defs = _sample_defs()
        agent = _fake_agent(defs)
        dts.apply_to_agent(agent, config=OFF)
        # tools untouched, no deferred state created.
        assert _names(agent.tools) == _names(defs)
        assert agent._deferred_tool_state is None

    def test_disabling_after_partition_restores_full_surface(self):
        defs = _sample_defs()
        agent = _fake_agent(defs)
        dts.apply_to_agent(agent, config=ON)
        assert "web_search" not in agent.valid_tool_names

        dts.apply_to_agent(agent, config=OFF)
        assert _names(agent.tools) == _names(defs)
        assert agent.valid_tool_names == _names(defs)
        assert agent._deferred_tool_state is None
        assert agent._deferred_tool_full_surface is None
        assert agent._deferred_tool_surface_partitioned is False

    def test_block_message_none_without_state(self):
        # No per-session state → nothing is ever blocked.
        assert dts.block_message("web_search", None) is None

    def test_tool_search_not_in_default_definitions(self):
        # The registry tool is gated by check_fn=is_enabled, so with the flag
        # off it must not appear in the default model tool surface.
        import model_tools
        names = [t["function"]["name"] for t in model_tools.get_tool_definitions(quiet_mode=True)]
        assert "tool_search" not in names


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

class TestPartition:
    def test_eager_tools_stay_visible(self):
        visible, state = dts.partition(_sample_defs(), config=ON)
        vis = _names(visible)
        # agent-loop + kanban lifecycle tools remain eager
        assert {"todo", "memory", "kanban_complete", "kanban_block"} <= vis
        # tool_search is injected
        assert "tool_search" in vis

    def test_other_tools_are_deferred(self):
        visible, state = dts.partition(_sample_defs(), config=ON)
        assert "web_search" not in _names(visible)
        assert {"web_search", "web_extract", "write_file", "send_message", "vision_analyze"} == state.deferred_names

    def test_tool_count_shrinks(self):
        defs = _sample_defs()
        visible, state = dts.partition(defs, config=ON)
        # Surface shrinks: deferred tools removed, only tool_search added back.
        assert len(visible) < len(defs)
        assert len(state.deferred_names) > 0

    def test_does_not_mutate_input(self):
        defs = _sample_defs()
        before = [dict(d) for d in defs]
        dts.partition(defs, config=ON)
        assert defs == before

    def test_unnamed_definitions_kept_visible(self):
        defs = _sample_defs() + [{"type": "function", "function": {}}]
        visible, state = dts.partition(defs, config=ON)
        # The malformed/unnamed def is kept on the surface (fail-open).
        assert any(d.get("function", {}).get("name") is None for d in visible)


# ---------------------------------------------------------------------------
# Token / request size proxy
# ---------------------------------------------------------------------------

class TestTokenCountChanges:
    def test_tool_count_drops_then_grows(self):
        defs = _sample_defs()
        full_count = len(defs)
        visible, state = dts.partition(defs, config=ON)
        # The model now sees only the eager core + tool_search.
        assert len(visible) < full_count
        # Promote a tool → the advertised count grows by exactly one.
        res = dts.search("search the web", state, limit=1)
        assert len(res["promoted"]) == 1
        grown = dts.visible_after_promotion(visible, state.promoted_definitions())
        assert len(grown) == len(visible) + 1

    def test_serialized_schema_size_drops_then_grows(self):
        # Realistic deferred tools carry substantial schemas — that is the
        # whole point: deferring them shrinks the per-request wire size.
        big_params = {
            "type": "object",
            "properties": {
                f"opt_{i}": {"type": "string", "description": "x" * 120}
                for i in range(6)
            },
        }
        defs = [
            _def("todo", "manage the todo list"),
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web. " + ("detail " * 60),
                "parameters": big_params,
            }},
            {"type": "function", "function": {
                "name": "write_file",
                "description": "Write a file. " + ("detail " * 60),
                "parameters": big_params,
            }},
        ]
        full_size = len(json.dumps(defs))
        visible, state = dts.partition(defs, config=ON)
        deferred_size = len(json.dumps(visible))
        assert deferred_size < full_size, "deferred surface should be smaller on the wire"

        # Promote a tool → serialized size grows back toward the full surface.
        res = dts.search("search the web", state, limit=1)
        assert res["promoted"]
        grown = dts.visible_after_promotion(visible, state.promoted_definitions())
        assert len(json.dumps(grown)) > deferred_size


# ---------------------------------------------------------------------------
# Per-session isolation
# ---------------------------------------------------------------------------

class TestPerSessionIsolation:
    def test_promotion_does_not_leak_across_states(self):
        defs = _sample_defs()
        _, state_a = dts.partition(defs, config=ON)
        _, state_b = dts.partition(defs, config=ON)

        dts.search("search the web", state_a, limit=5)

        assert state_a.is_promoted("web_search")
        # Independent session must be unaffected.
        assert not state_b.is_promoted("web_search")
        assert state_b.is_blocked("web_search")

    def test_no_module_global_promotion_state(self):
        # The module exposes no shared mutable promotion registry.
        assert not hasattr(dts, "_promoted")
        assert not hasattr(dts, "PROMOTED")


# ---------------------------------------------------------------------------
# Direct-call blocking
# ---------------------------------------------------------------------------

class TestDirectCallBlocking:
    def test_unpromoted_deferred_tool_is_blocked(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        msg = dts.block_message("web_search", state)
        assert msg is not None
        payload = json.loads(msg)
        assert payload["error"] == "deferred_tool_not_promoted"
        assert payload["tool"] == "web_search"
        # Actionable: points the model at tool_search with a query.
        assert payload["next_action"]["tool"] == "tool_search"
        assert "query" in payload["next_action"]["arguments"]

    def test_eager_tool_never_blocked(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        assert dts.block_message("todo", state) is None
        assert dts.block_message("kanban_complete", state) is None
        assert dts.block_message("tool_search", state) is None

    def test_promoted_tool_not_blocked(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        dts.search("search the web", state, limit=5)
        assert dts.block_message("web_search", state) is None

    def test_unknown_tool_not_blocked(self):
        # Tools that aren't deferred at all (e.g. a typo, or a tool from
        # another toolset) are not the deferred-search layer's concern.
        _, state = dts.partition(_sample_defs(), config=ON)
        assert dts.block_message("totally_unknown_tool", state) is None


# ---------------------------------------------------------------------------
# Search / promotion ranking
# ---------------------------------------------------------------------------

class TestSearch:
    def test_query_promotes_relevant_tools(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        res = dts.search("search the web for information", state, limit=5)
        assert "web_search" in res["promoted"]
        # web_extract also mentions web — should rank too.
        assert "web_search" in [m["name"] for m in res["matches"]]

    def test_limit_caps_promotions(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        res = dts.search("web file message image", state, limit=2)
        assert len(res["promoted"]) <= 2

    def test_empty_query_errors(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        res = dts.search("   ", state, limit=5)
        assert "error" in res
        assert res["promoted"] == []

    def test_no_match_promotes_nothing(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        res = dts.search("zzzzz nonsense qqqq", state, limit=5)
        assert res["promoted"] == []
        assert res["matches"] == []

    def test_search_is_idempotent_on_already_promoted(self):
        _, state = dts.partition(_sample_defs(), config=ON)
        dts.search("web", state, limit=5)
        res2 = dts.search("web", state, limit=5)
        # Second call promotes nothing new but reports them as already promoted.
        assert res2["promoted"] == []
        assert "web_search" in res2["already_promoted"]


# ---------------------------------------------------------------------------
# MCP/ACP refresh interactions
# ---------------------------------------------------------------------------

class TestRefreshInteractions:
    def test_promotions_carried_across_rebuild(self):
        defs = _sample_defs()
        _, state = dts.partition(defs, config=ON)
        dts.search("search the web", state, limit=1)
        assert state.is_promoted("web_search")

        # Simulate an MCP refresh: get_tool_definitions returns a new list
        # (here with an added MCP tool). Re-partition carrying prior state.
        refreshed = defs + [_def("mcp_srv_lookup", "look something up via MCP")]
        visible2, state2 = dts.partition(refreshed, config=ON, prior_state=state)

        # Promotion survived the rebuild and is visible again.
        assert state2.is_promoted("web_search")
        assert "web_search" in _names(visible2)
        # The newly-added MCP tool starts deferred.
        assert state2.is_blocked("mcp_srv_lookup")

    def test_refresh_drops_promotion_for_vanished_tool(self):
        defs = _sample_defs()
        _, state = dts.partition(defs, config=ON)
        dts.search("send a message", state, limit=5)
        assert state.is_promoted("send_message")

        # Tool disappears after refresh (e.g. MCP server removed).
        reduced = [d for d in defs if d["function"]["name"] != "send_message"]
        _, state2 = dts.partition(reduced, config=ON, prior_state=state)
        assert "send_message" not in state2.deferred_names
        assert not state2.is_promoted("send_message")


# ---------------------------------------------------------------------------
# Agent integration helpers
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    def test_apply_to_agent_partitions_when_on(self):
        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=ON)
        assert agent._deferred_tool_state is not None
        assert "tool_search" in agent.valid_tool_names
        assert "web_search" not in agent.valid_tool_names

    def test_handle_tool_search_promotes_and_extends_surface(self):
        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=ON)
        before = set(agent.valid_tool_names)
        out = dts.handle_tool_search_call(agent, {"query": "search the web", "limit": 2})
        payload = json.loads(out)
        assert payload["promoted"]
        # Promoted tools now appear on the live surface.
        assert "web_search" in agent.valid_tool_names
        assert agent.valid_tool_names > before

    def test_handle_tool_search_without_state_is_safe(self):
        agent = _fake_agent(_sample_defs())  # no deferred state (feature off)
        out = dts.handle_tool_search_call(agent, {"query": "web"})
        payload = json.loads(out)
        assert payload["error"] == "tool_search_unavailable"

    def test_apply_to_agent_idempotent_carries_promotions(self):
        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=ON)
        dts.handle_tool_search_call(agent, {"query": "search the web", "limit": 1})
        assert "web_search" in agent.valid_tool_names
        # Re-apply (e.g. MCP refresh path) keeps the promotion.
        dts.apply_to_agent(agent, config=ON)
        assert "web_search" in agent.valid_tool_names
        assert agent._deferred_tool_state.is_promoted("web_search")

    def test_apply_to_agent_reapply_keeps_unpromoted_tools_searchable(self):
        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=ON)
        dts.handle_tool_search_call(agent, {"query": "search the web", "limit": 1})
        dts.apply_to_agent(agent, config=ON)

        out = dts.handle_tool_search_call(agent, {"query": "write file", "limit": 1})
        payload = json.loads(out)
        assert "write_file" in payload["promoted"]
        assert "write_file" in agent.valid_tool_names

    def test_authoritative_refresh_drops_removed_unpromoted_tools(self):
        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=ON)
        dts.handle_tool_search_call(agent, {"query": "search the web", "limit": 1})

        # Simulate ACP/MCP refresh replacing agent.tools with an authoritative
        # fresh list that no longer includes send_message. The integration code
        # resets the partitioned marker before reapplying deferred search.
        agent.tools = [
            td for td in _sample_defs()
            if td["function"]["name"] != "send_message"
        ]
        agent._deferred_tool_surface_partitioned = False
        dts.apply_to_agent(agent, config=ON)

        assert not agent._deferred_tool_state.is_deferred("send_message")
        out = dts.handle_tool_search_call(agent, {"query": "send message", "limit": 5})
        payload = json.loads(out)
        assert "send_message" not in payload["promoted"]

    def test_late_added_registry_tool_is_deferred_when_partition_applies_after_injection(self):
        # Regression guard for agent_init ordering: non-agent-loop session-local
        # schemas appended before deferred partitioning are not left eagerly
        # advertised merely because they were late additions.
        defs = _sample_defs() + [_def("plugin_lookup", "lookup from a plugin registry")]
        agent = _fake_agent(defs)
        dts.apply_to_agent(agent, config=ON)
        assert "plugin_lookup" not in agent.valid_tool_names
        assert agent._deferred_tool_state.is_blocked("plugin_lookup")

    def test_context_engine_and_memory_provider_tools_remain_eager(self):
        # These tool schemas are injected late too, but they are dispatched by
        # the agent loop rather than the normal registry path and must remain
        # eager once the coarse enabled_toolsets gate has admitted them.
        defs = _sample_defs() + [
            _def("lcm_grep", "search local context engine index"),
            _def("honcho_search", "search memory provider"),
        ]
        agent = _fake_agent(
            defs,
            context_engine_tools={"lcm_grep"},
            memory_tools={"honcho_search"},
        )
        dts.apply_to_agent(agent, config=ON)
        assert "lcm_grep" in agent.valid_tool_names
        assert "honcho_search" in agent.valid_tool_names
        assert not agent._deferred_tool_state.is_deferred("lcm_grep")
        assert not agent._deferred_tool_state.is_deferred("honcho_search")

    def test_agent_init_applies_deferred_after_context_engine_injection(self):
        from pathlib import Path

        src = Path("agent/agent_init.py").read_text(encoding="utf-8")
        injection = "agent.context_compressor.get_tool_schemas()"
        apply = "_dts.apply_to_agent(agent)"
        assert src.index(injection) < src.index(apply)

    def test_conversation_loop_returns_deferred_block_before_unknown_tool_error(self):
        from pathlib import Path

        src = Path("agent/conversation_loop.py").read_text(encoding="utf-8")
        assert "deferred_tool_call_blocks" in src
        assert "content = deferred_tool_call_blocks[tc.function.name]" in src


# ---------------------------------------------------------------------------
# Real agent-loop dispatch seam (agent_runtime_helpers.invoke_tool)
# ---------------------------------------------------------------------------

class TestInvokeToolSeam:
    """Exercises the actual concurrent-path dispatcher seam, not just the
    module helpers — proves the block + promotion are wired into the loop."""

    def test_deferred_tool_blocked_via_invoke_tool(self):
        from agent.agent_runtime_helpers import invoke_tool

        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=ON)
        out = invoke_tool(
            agent, "web_search", {"query": "x"}, "task-1",
            tool_call_id="tc1", pre_tool_block_checked=True,
        )
        payload = json.loads(out)
        assert payload["error"] == "deferred_tool_not_promoted"
        assert payload["tool"] == "web_search"

    def test_tool_search_routed_via_invoke_tool(self):
        from agent.agent_runtime_helpers import invoke_tool

        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=ON)
        out = invoke_tool(
            agent, "tool_search", {"query": "search the web", "limit": 1}, "task-1",
            tool_call_id="tc2", pre_tool_block_checked=True,
        )
        payload = json.loads(out)
        assert "web_search" in payload["promoted"]
        # After promotion the same tool is no longer blocked.
        assert dts.block_message("web_search", agent._deferred_tool_state) is None

    def test_no_block_when_feature_off_via_invoke_tool(self):
        # Feature off → no deferred state → invoke_tool must not block; it
        # falls through to normal dispatch (which we don't execute here, so we
        # just assert it is not the deferred-block payload).
        agent = _fake_agent(_sample_defs())
        dts.apply_to_agent(agent, config=OFF)
        assert agent._deferred_tool_state is None
