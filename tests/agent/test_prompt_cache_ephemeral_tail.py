"""A first call also marks the end of the session prompt, before the API-time ephemeral tail.

Text appended to the session's system prompt at API time (a personality, a task brief) used to
sit ahead of the only breakpoint after the static tier, so sessions that differ only in that tail
re-wrote the skills index on every first call. These tests pin where the breakpoints land and
that the request still carries the same bytes.
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


def test_a_session_that_starts_with_history_still_marks_the_end_of_the_system_prompt():
    """A replayed or edited history misses the conversation breakpoints; the tail must not go with it."""
    history = [{"role": "user", "content": "goal"}, {"role": "assistant", "content": "working"},
               {"role": "user", "content": "more"}]
    assert _marked(_plan(history)) == [(0, 0), (0, 2), (2, 0), (3, 0)]


def test_a_blank_session_rest_never_becomes_its_own_part():
    """A whitespace-only text block is rejected by Anthropic (400): fall back to the two-part split."""
    planned = apply_anthropic_cache_control(
        [{"role": "system", "content": SYSTEM.replace(SESSION, STATIC + "  ")}, {"role": "user", "content": "goal"}],
        static_system_prefix=STATIC, session_system_prompt=STATIC + "  ")
    assert len(planned[0]["content"]) == 2 and all(p["text"].strip() for p in planned[0]["content"])


def test_strip_keeps_a_three_part_system_message_the_planner_did_not_split():
    """A context engine may send its own three plain parts; only the planner's split (part 0 marked) is joined."""
    own = [{"type": "text", "text": t} for t in ("alpha\n", "beta\n", "gamma")]
    assert strip_anthropic_cache_control([{"role": "system", "content": list(own)}])[0]["content"] == own


def test_three_part_failover_rewrite_keeps_the_unchanged_parts():
    from agent.conversation_loop import _rewrite_system_content_blocks

    parts = [{"type": "text", "text": x, "cache_control": {"type": "ephemeral"}} for x in ["core", "skills", "old tail"]]
    assert _rewrite_system_content_blocks({"role": "system", "content": parts}, "coreskillsnew tail")
    assert [p["text"] for p in parts] == ["core", "skills", "new tail"]
    assert all("cache_control" in p for p in parts)


def test_real_agent_keeps_one_split_and_marks_the_session_prompt_on_the_first_call_only(tmp_path):
    """Through a real AIAgent with an ephemeral prompt: every request sends the same three system
    parts with the end of the whole prompt marked, so nothing is re-written, and only the first
    request spends a breakpoint on the session-prompt boundary."""
    import os
    from pathlib import Path

    from hermes_agent.hermes_state import SessionDB
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
        assert "task text" in parts[2]["text"] and "cache_control" in parts[0] and "cache_control" in parts[2]
    assert ["cache_control" in parts[1] for parts in systems] == [True, False, False]
    assert all(str(r).count("cache_control") <= 4 for r in requests)
