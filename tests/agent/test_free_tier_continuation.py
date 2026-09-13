"""Exercise turn admission through the real AIAgent with durable temporary auth."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def free_agent(tmp_path, monkeypatch):
    from hermes_cli import auth, auth_nous
    from run_agent import AIAgent
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    guest = {"auth_method": "anonymous", "anon_token": "anon_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    agent = AIAgent(api_key="fixture", provider="nous", model="nous/welcome",
                    base_url="https://welcome-api.nousresearch.com/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent._cached_system_prompt = "unchanged-prefix"
    return agent


def test_crossing_turn_finishes_then_next_turn_refuses_before_preflight(free_agent, monkeypatch):
    from agent import conversation_loop
    from agent.free_tier import record_tool_completion
    from hermes_cli import free_tier_usage as usage
    identity = usage.current_identity()
    for _ in range(9):
        usage.record_completed_tool(identity)
    entered = []
    def turn(agent, text, **kwargs):
        entered.append(text)
        for _ in range(3):
            record_tool_completion(agent)
        return {"final_response": "Finished the task.", "messages": kwargs["conversation_history"] + [
            {"role": "user", "content": text}, {"role": "assistant", "content": "Finished the task."}],
            "completed": True}
    monkeypatch.setattr(conversation_loop, "_run_conversation_turn", turn)
    history = [{"role": "system", "content": "unchanged-prefix"}]
    result = free_agent.run_conversation("first", conversation_history=history)
    assert result["completed"] and result["free_tier"]["tool_calls_used"] == 12
    assert result["final_response"].endswith(usage.LIMIT_NOTICE)
    assert result["messages"][-1]["content"] == "Finished the task."
    before = copy.deepcopy(result["messages"])
    blocked = free_agent.run_conversation("not appended", conversation_history=result["messages"])
    assert blocked["error"] == usage.LIMIT_REASON and blocked["retryable"]
    assert blocked["failed"] is True and blocked["completed"] is False
    assert blocked["failure_retryable"] is False and blocked["continuation_required"] is True
    assert blocked["messages"] == before and entered == ["first"]
    assert free_agent._cached_system_prompt == "unchanged-prefix"
    free_agent.base_url = "http://localhost:11434/v1"
    free_agent.provider = "custom"
    assert free_agent.run_conversation("local", conversation_history=before)["completed"]


def test_children_and_nested_tools_share_admission_but_guide_never_counts(free_agent, monkeypatch):
    from agent import conversation_loop
    from agent.free_tier import record_tool_completion
    from hermes_cli import free_tier_usage as usage
    from hermes_cli.onboarding_profile import mark_onboarding_profile
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
    from model_tools import handle_function_call
    from tools.registry import registry

    identity = usage.current_identity()
    for _ in range(9):
        usage.record_completed_tool(identity)
    seen = []
    child = copy.copy(free_agent)
    child.session_id += "-child"
    import weakref
    child._delegate_parent_ref = weakref.ref(free_agent)
    def turn(agent, text, **kwargs):
        seen.append(text)
        if text == "parent":
            record_tool_completion(agent)
            assert child.run_conversation("child")["completed"]
            # Nested execute_code RPC calls have no model-issued tool_call_id.
            handle_function_call("fixture_nested", {}, task_id="nested")
        else:
            record_tool_completion(agent)
        return {"completed": True, "messages": [], "final_response": text}
    monkeypatch.setattr(conversation_loop, "_run_conversation_turn", turn)
    monkeypatch.setattr(registry, "dispatch", lambda *a, **kw: '{"ok":true}')
    assert free_agent.run_conversation("parent")["completed"]
    assert usage.status()["tool_calls_used"] == 12
    # A profile merely named like the guide remains capped.
    home = get_hermes_home()
    guide = home / "profiles" / "hermes-setup"
    guide.mkdir(parents=True)
    from hermes_cli import auth
    token = set_hermes_home_override(str(guide))
    try:
        auth._save_active_provider_state("nous", {"auth_method": "anonymous", "anon_token": "anon_fixture"})
        assert not free_agent.run_conversation("name-only")["completed"]
        mark_onboarding_profile(guide)
        assert free_agent.run_conversation("guide")["completed"]
        assert usage.status()["tool_calls_used"] == 12
    finally:
        reset_hermes_home_override(token)
