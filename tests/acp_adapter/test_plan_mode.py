"""M4·A2 — the ``plan`` session mode is offered and selectable."""
import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState


def _agent_with_session(sid: str = "s-plan") -> tuple[HermesACPAgent, SessionState]:
    mgr = SessionManager(agent_factory=lambda **_: object())
    agent = HermesACPAgent(session_manager=mgr)
    state = SessionState(session_id=sid, agent=object())
    mgr._sessions[sid] = state
    return agent, state


def test_plan_mode_is_offered():
    agent, state = _agent_with_session()
    mode_ids = {m.id for m in agent._session_modes(state).available_modes}
    assert "plan" in mode_ids
    # the existing edit-approval modes are preserved
    assert {"default", "accept_edits", "dont_ask"} <= mode_ids


@pytest.mark.asyncio
async def test_plan_mode_is_selectable(monkeypatch):
    agent, state = _agent_with_session("s-sel")
    monkeypatch.setattr(agent.session_manager, "save_session", lambda *_a, **_k: None)
    await agent.set_session_mode(mode_id="plan", session_id="s-sel")
    assert getattr(state, "mode", None) == "plan"
