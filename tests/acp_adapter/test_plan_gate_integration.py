"""M4·A3 — the plan gate installed by the ACP adapter actually blocks mutating
tools (and only those) via the real pre-tool-call block path."""
from hermes_cli.plugins import get_pre_tool_call_block_message

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState


def _agent_with_session(sid: str, mode: str | None = None) -> tuple[HermesACPAgent, SessionState]:
    mgr = SessionManager(agent_factory=lambda **_: object())
    agent = HermesACPAgent(session_manager=mgr)
    state = SessionState(session_id=sid, agent=object())
    if mode is not None:
        setattr(state, "mode", mode)
    mgr._sessions[sid] = state
    return agent, state


def test_plan_gate_blocks_mutating_allows_readonly_then_clears():
    agent, state = _agent_with_session("s-gate", mode="plan")
    installed = agent._install_plan_gate(state)
    try:
        assert installed is True
        # Mutating tools are blocked with the plan message.
        for t in ("write_file", "patch", "terminal", "process", "execute_code", "delegate_task"):
            msg = get_pre_tool_call_block_message(t, {})
            assert msg is not None and "Plan mode" in msg, t
        # Read-only + planning tools are allowed (no block).
        for t in ("read_file", "search_files", "web_search", "todo"):
            assert get_pre_tool_call_block_message(t, {}) is None, t
    finally:
        agent._clear_plan_gate(installed)
    # After clearing, nothing is gated.
    assert get_pre_tool_call_block_message("write_file", {}) is None


def test_plan_gate_is_noop_outside_plan_mode():
    agent, state = _agent_with_session("s-default")  # no mode => default
    installed = agent._install_plan_gate(state)
    try:
        assert installed is False
        assert get_pre_tool_call_block_message("write_file", {}) is None
    finally:
        agent._clear_plan_gate(installed)
