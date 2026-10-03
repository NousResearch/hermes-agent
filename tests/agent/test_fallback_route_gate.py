"""Mid-turn route-change gate for side-effecting tools.

Regression tests for #117495: ``try_activate_fallback()`` swaps provider/model/base_url in
place and the tool loop keeps running, so a pending side-effecting tool executes on a route
the caller never selected. ``fallback.halt_on_route_change`` (opt-in) captures the acting
route at turn start and blocks side-effecting tools at dispatch when it changed. Read-only
tools and same-route turns are unaffected; the gate is provenance detection ("the route
changed"), not fallback interpretation.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent import fallback_route_gate
from agent.fallback_route_gate import (
    check_tool_dispatch,
    latch_turn_route,
    capture_route_snapshot,
)
from agent.tool_executor import _dispatch_authorized_once, _ManagedToolResult, _ToolCallRef


def _agent(model="m1", provider="p1", base_url="https://a.example"):
    """Bare namespace stand-in: the gate reads only route attrs; the extra attributes let the
    un-blocked paths run the rest of the dispatch chain without a full AIAgent."""
    agent = SimpleNamespace(model=model, provider=provider, base_url=base_url, quiet_mode=True)
    agent._tool_guardrails = SimpleNamespace(
        before_call=lambda name, args: SimpleNamespace(allows_execution=True))
    agent._touch_activity = lambda label: None
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    return agent


def _dispatch(agent, name, execute=None):
    """Drive one call through the policy funnel the production loop uses."""
    ref = _ToolCallRef(name=name, args={}, task_id="t", call_id="c1", trace=[])
    state = _ManagedToolResult(result=None, args=ref.args, middleware_trace=[], blocked=False, dispatched=False)
    result = _dispatch_authorized_once(
        agent, state, ref,
        execute=execute or (lambda args: "{}"),
        scope_block=None, display_index=None,
        begin_execution=None, authorization_gate=None,
    )
    return ref, state, result


def _config(halt):
    return {"fallback": {"halt_on_route_change": halt}}


# ── Unit: provenance primitives ───────────────────────────────────────────


class TestRouteSnapshot:
    def test_snapshot_keys(self):
        snap = capture_route_snapshot(_agent())
        assert snap == {"model": "m1", "provider": "p1", "base_url": "https://a.example"}

    def test_latch_stores_on_agent(self):
        agent = _agent()
        latch_turn_route(agent)
        assert agent._turn_route_snapshot == {"model": "m1", "provider": "p1", "base_url": "https://a.example"}


# ── Unit: dispatch check ──────────────────────────────────────────────────


class TestCheckToolDispatch:
    def test_no_snapshot_no_block(self):
        agent = _agent()
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=True):
            assert check_tool_dispatch(agent, "terminal") is None

    def test_same_route_no_block(self):
        agent = _agent()
        latch_turn_route(agent)
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=True):
            assert check_tool_dispatch(agent, "terminal") is None

    def test_route_changed_blocks_side_effecting_tool(self):
        agent = _agent()
        latch_turn_route(agent)
        agent.model, agent.provider, agent.base_url = "m2", "p2", "https://b.example"
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=True):
            msg = check_tool_dispatch(agent, "terminal")
        assert msg is not None and "ROUTE_CHANGED_BLOCKED" in msg

    def test_read_only_tool_never_blocked_on_route_change(self):
        agent = _agent()
        latch_turn_route(agent)
        agent.model, agent.provider, agent.base_url = "m2", "p2", "https://b.example"
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=True):
            for name in ("read_file", "search_files", "web_search", "todo_list"):
                assert check_tool_dispatch(agent, name) is None

    def test_disabled_gate_never_blocks(self):
        agent = _agent()
        latch_turn_route(agent)
        agent.model = "m2"
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=False):
            assert check_tool_dispatch(agent, "terminal") is None


# ── Integration: the production policy funnel ────────────────────────────


class TestDispatchAuthorizedOnce:
    def test_blocked_on_route_change_when_enabled(self):
        agent = _agent()
        latch_turn_route(agent)
        agent.model, agent.provider = "m2", "p2"
        executed = []
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=True):
            ref, state, result = _dispatch(agent, "terminal", execute=lambda a: executed.append(a))
        assert executed == []
        assert state.blocked is True
        assert "ROUTE_CHANGED_BLOCKED" in result

    def test_not_blocked_on_route_change_when_disabled(self):
        agent = _agent()
        latch_turn_route(agent)
        agent.model, agent.provider = "m2", "p2"
        executed = []
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=False):
            ref, state, result = _dispatch(agent, "terminal", execute=lambda a: executed.append(a))
        assert executed == [{}]
        assert state.blocked is False

    def test_read_only_tool_dispatches_despite_route_change(self):
        agent = _agent()
        latch_turn_route(agent)
        agent.model = "m2"
        executed = []
        with patch.object(fallback_route_gate, "_gate_enabled", return_value=True):
            ref, state, result = _dispatch(agent, "read_file", execute=lambda a: executed.append(a))
        assert executed == [{}]
        assert state.blocked is False


# ── Contract: config default is opt-in (never freeze the literal value set) ──


class TestConfigDefault:
    def test_default_config_carries_halt_key(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG.get("fallback"), dict)
        assert "halt_on_route_change" in DEFAULT_CONFIG["fallback"]
