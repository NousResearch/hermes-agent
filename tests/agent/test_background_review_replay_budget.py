"""The default automatic replay must leave the review fork a read -> write cycle.

The aggregate input budget is charged the FULL prompt on every provider request — cache reads
included (``conversation_loop._review_input_tokens_consumed``) — and every request replays the
snapshot. A replay sized for the first request alone therefore leaves room for exactly one
request on the default budget once a long session fills the replay ceiling: the fork's reads
execute, its second request is refused, and no read -> write cycle is possible. The composed
spawn path is asserted here, with the loop's own reservation accounting walked request by
request; the budget-formula unit tests and the admission contracts live in
``test_background_review_compute_overlap.py``, whose fixtures this file shares.
"""

from __future__ import annotations

import types

import run_agent as run_agent_module
from agent import review_admission
from agent.model_metadata import estimate_messages_tokens_rough
from run_agent import AIAgent
from tests.agent.test_background_review_compute_overlap import (  # fixtures
    _bare_agent,
    _config,
    _patch_config,
    _snapshot,
    clear_review_admission_state,
    review_forks,
)


def test_default_replay_leaves_room_for_a_read_write_cycle(
    review_forks,  # health: allow F811 -- pytest injects the imported fixture by parameter name
    monkeypatch,
):
    """A session past the replay ceiling on a 200k window (150k aggregate) with a gateway-sized
    system prompt and tools[]: the fork must get a read, a write and a closing response (the
    review prompt enforces read-before-write) before the aggregate is exhausted."""
    from agent.conversation_loop import _reserve_review_input_request
    from agent.model_metadata import _estimate_tools_tokens_rough

    _patch_config(monkeypatch, _config())
    window = 200_000
    fake_fork = run_agent_module.AIAgent  # the review_forks fixture's recorder

    class WindowedFork(fake_fork):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.context_compressor = types.SimpleNamespace(context_length=window)

    monkeypatch.setattr(run_agent_module, "AIAgent", WindowedFork)
    agent = _bare_agent()
    agent.context_compressor = types.SimpleNamespace(context_length=window)
    agent._cached_system_prompt = "system prompt " + "s" * 20_000  # ~5k tokens
    agent.tools = [  # ~12k tokens of advertised schemas (a gateway surface)
        {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": "d" * 400,
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string", "description": "e" * 500}},
                },
            },
        }
        for index in range(50)
    ]
    # A session past the replay ceiling: the normal long-session case.
    snapshot = _snapshot(pairs=80, filler_chars=8_000)
    assert (
        estimate_messages_tokens_rough(snapshot)
        > review_admission.MAX_REPLAY_TOKENS_DEFAULT
    )

    AIAgent._spawn_background_review(
        agent, messages_snapshot=snapshot, review_memory=True
    )

    assert len(review_forks) == 1
    record = review_forks[0]
    history = record["history"]
    assert history
    fork = types.SimpleNamespace(
        _review_input_token_budget=record["attrs"]["_review_input_token_budget"],
        session_prompt_tokens=0,
        _review_input_tokens_reserved=0,
    )
    tools_tokens = _estimate_tools_tokens_rough(record["attrs"]["tools"])
    messages = (
        [{"role": "system", "content": record["attrs"]["_cached_system_prompt"]}]
        + list(history)
        + [{"role": "user", "content": record["user_message"]}]
    )
    admitted = []
    # Request #1 reads (a skill_view / memory search), #2 writes, #3 closes; each tool
    # result (~1k tokens) grows every later request, as the loop replays it.
    for step, tool_result in (
        ("read", "r" * 4_000),
        ("write", "w" * 4_000),
        ("close", None),
    ):
        projected = estimate_messages_tokens_rough(messages) + tools_tokens
        if not _reserve_review_input_request(fork, projected):
            break
        admitted.append(step)
        fork.session_prompt_tokens += projected  # the provider bills the whole prompt
        if tool_result is not None:
            messages += [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": step,
                            "type": "function",
                            "function": {"name": "memory", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": step, "content": tool_result},
            ]
    assert admitted == ["read", "write", "close"], (
        f"the default replay left room for {admitted} only "
        f"(budget={fork._review_input_token_budget}, replayed="
        f"{estimate_messages_tokens_rough(history)})"
    )
