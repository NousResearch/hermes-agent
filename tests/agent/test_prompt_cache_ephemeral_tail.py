"""The second system breakpoint sits before the API-time ephemeral tail.

A subagent's per-task text (and a personality) is appended to the session's system prompt at
API time. With the breakpoint after it, sessions that differ only in that tail re-wrote the
skills index on every first call. These tests pin where the breakpoints land and that the
request still carries the same bytes.
"""

from agent.prompt_caching import apply_anthropic_cache_control, strip_anthropic_cache_control

STATIC = "You are Hermes.\n"
SESSION = STATIC + "## Skills\n- alpha: does a thing\n"
SYSTEM = SESSION + "\n\nYou are a focused subagent.\nCONTEXT:\ntask-specific text"


def _marked(messages):
    """(message index, part index) of every cache breakpoint in a planned request."""
    return [(i, j) for i, m in enumerate(messages) if isinstance(m.get("content"), list)
            for j, part in enumerate(m["content"]) if "cache_control" in part]


def _plan(history):
    return apply_anthropic_cache_control(
        [{"role": "system", "content": SYSTEM}] + history,
        static_system_prefix=STATIC, session_system_prompt=SESSION)


def test_first_call_marks_session_prompt_end_and_keeps_the_tail_breakpoint():
    planned = _plan([{"role": "user", "content": "goal"}])

    parts = planned[0]["content"]
    assert [p["text"] for p in parts] == [STATIC, SESSION[len(STATIC):], SYSTEM[len(SESSION):]]
    # static tier, session prompt end, ephemeral tail, the one message: the provider's limit of 4.
    assert _marked(planned) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    # Same bytes on the wire, and a failover re-plan gets the original string back.
    assert strip_anthropic_cache_control(planned)[0]["content"] == SYSTEM

    # A session with no ephemeral tail is planned exactly as before.
    history = [{"role": "user", "content": "goal"}, {"role": "assistant", "content": "working"}]
    plain = apply_anthropic_cache_control(
        [{"role": "system", "content": SESSION}] + history, static_system_prefix=STATIC, session_system_prompt=SESSION)
    assert plain == apply_anthropic_cache_control(
        [{"role": "system", "content": SESSION}] + history, static_system_prefix=STATIC)


def test_real_agent_keeps_one_split_and_moves_the_tail_breakpoint_to_the_conversation(tmp_path):
    """Through a real AIAgent with an ephemeral prompt: every request sends the same three system
    parts ending at the session prompt, so nothing is re-written, and only the first request spends
    a breakpoint on the tail."""
    import os
    from pathlib import Path

    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.agent.test_prompt_cache_prefix_contract import _prefix_breaks
    from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer([]) as srv:
        route = (f"custom_providers:\n  - name: cacherelay\n    base_url: {srv.base_url}\n    api_key: sk-fake-e2e\n"
                 "    models:\n      fake-model:\n        prompt_caching: true\n")
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + route)
        srv.push(ToolCall("terminal", {"command": "echo one"}), ToolCall("terminal", {"command": "echo two"}), Text("done"))
        agent = AIAgent(provider="custom:cacherelay", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                        session_db=SessionDB(db_path=home / "state.db"), session_id="tail", quiet_mode=True,
                        platform="cli", ephemeral_system_prompt="You are a focused subagent.\nCONTEXT:\ntask text")
        agent.run_conversation("go", conversation_history=[], task_id="t")
        session_prompt = agent._cached_system_prompt
        agent.close()
        requests = srv.main_requests()

    assert len(requests) == 3
    assert not _prefix_breaks(requests), _prefix_breaks(requests)
    systems = [r["messages"][0]["content"] for r in requests]
    for parts in systems:
        assert len(parts) == 3 and parts[0]["text"] + parts[1]["text"] == session_prompt
        assert "task text" in parts[2]["text"] and "cache_control" in parts[0] and "cache_control" in parts[1]
    assert ["cache_control" in parts[2] for parts in systems] == [True, False, False]
    assert all(str(r).count("cache_control") <= 4 for r in requests)
