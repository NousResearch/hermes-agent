"""Agent-as-provider transcript projection + skill-nudge tick.

A provider that IS an agent executes its own tools inside its own session. Those
calls never come back as pending ``tool_calls`` (Hermes would re-run finished
work), so two subsystems would otherwise be blind to them:

  * the self-improvement loop, which distils skills/memories from ``messages``;
  * the skill-review nudge, whose counter only moves on Hermes tool iterations.

``splice_provider_projection`` closes both gaps. The helper is unit tested here,
and the wiring is exercised for real: the last tests drive a whole
``AIAgent.run_conversation`` turn against an in-process fake client and assert on
the resulting transcript and counters, so they fail if the loop ever stops
applying the projection.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.provider_projection import splice_provider_projection  # noqa: E402

_PROJECTED = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "acp_s1_t1",
                "type": "function",
                "function": {"name": "acpagent_edit", "arguments": '{"path": "main.py"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "acp_s1_t1",
        "name": "acpagent_edit",
        "content": "1 file changed",
    },
]


def _projected_rows():
    """Fresh copies — the splice stamps a timestamp onto the dicts it appends."""
    return [dict(row) for row in _PROJECTED]


def _agent(iters: int = 0) -> SimpleNamespace:
    return SimpleNamespace(provider="acp-agent", _iters_since_skill=iters)


def _response(**attrs):
    return SimpleNamespace(**attrs)


# ── unit ─────────────────────────────────────────────────────────────────────


def test_projected_rows_are_appended_and_the_nudge_ticks():
    agent = _agent()
    messages = [{"role": "user", "content": "edit main.py"}]
    spliced = splice_provider_projection(
        agent,
        _response(hermes_projected_messages=_projected_rows(), hermes_provider_tool_iterations=1),
        messages,
    )
    assert spliced == 2
    assert messages[1]["tool_calls"][0]["function"]["name"] == "acpagent_edit"
    assert messages[2]["content"] == "1 file changed"
    assert agent._iters_since_skill == 1


def test_rows_are_stamped_like_every_other_live_transcript_append():
    """They go through ``append_message``; an unstamped row persists differently
    from the ones the loop appends itself."""
    messages: list = []
    splice_provider_projection(
        _agent(), _response(hermes_projected_messages=_projected_rows()), messages
    )
    assert all(isinstance(m.get("timestamp"), float) for m in messages)


def test_iterations_accumulate_across_calls():
    agent = _agent(iters=2)
    splice_provider_projection(agent, _response(hermes_provider_tool_iterations=3), [])
    assert agent._iters_since_skill == 5


def test_ordinary_provider_response_is_a_no_op():
    agent = _agent(iters=1)
    messages = [{"role": "user", "content": "hi"}]
    # A normal OpenAI completion carries neither attribute.
    assert splice_provider_projection(agent, SimpleNamespace(choices=[]), messages) == 0
    assert messages == [{"role": "user", "content": "hi"}]
    assert agent._iters_since_skill == 1


def test_garbage_attributes_cannot_break_the_turn():
    agent = _agent()
    messages: list = []
    assert splice_provider_projection(
        agent,
        _response(
            hermes_projected_messages="not-a-list",
            hermes_provider_tool_iterations="lots",
        ),
        messages,
    ) == 0
    assert messages == []
    assert agent._iters_since_skill == 0

    # A list with non-dict entries keeps only the usable rows.
    assert splice_provider_projection(
        agent,
        _response(hermes_projected_messages=[{"role": "tool", "content": "ok"}, "junk", None]),
        messages,
    ) == 1
    assert [m["role"] for m in messages] == ["tool"]


# ── wired into the real conversation loop ────────────────────────────────────


class _FakeAgentProviderCompletions:
    """One canned completion, shaped like what an ACP client returns."""

    def __init__(self, projected, iterations):
        self._projected = projected
        self._iterations = iterations

    def create(self, **_kwargs):
        message = SimpleNamespace(content="Edited main.py.", tool_calls=[], reasoning=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
            hermes_projected_messages=self._projected,
            hermes_provider_tool_iterations=self._iterations,
        )


class _FakeAgentProviderClient:
    def __init__(self, projected, iterations):
        self.chat = SimpleNamespace(
            completions=_FakeAgentProviderCompletions(projected, iterations)
        )


def _run_turn(monkeypatch, *, projected, iterations):
    """Drive one real ``run_conversation`` turn against the fake client."""
    from run_agent import AIAgent

    monkeypatch.setattr(
        "run_agent.OpenAI",
        lambda **_kw: _FakeAgentProviderClient(projected, iterations),
    )
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda *a, **k: [])

    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=3,
        quiet_mode=True,
        skip_memory=True,
    )
    agent._disable_streaming = True
    result = agent.run_conversation("edit main.py")
    return agent, result


def test_provider_work_lands_in_the_transcript_through_the_real_loop(monkeypatch):
    _agent_, result = _run_turn(monkeypatch, projected=_projected_rows(), iterations=1)

    messages = result["messages"]
    tool_rows = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("name") == "acpagent_edit"
    ]
    assert tool_rows, messages
    assert "1 file changed" in tool_rows[0]["content"]

    # The projected call precedes its result, and both precede the final answer.
    idx_call = next(
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    idx_result = messages.index(tool_rows[0])
    assert idx_call < idx_result < len(messages) - 1


def test_provider_iterations_tick_the_skill_nudge_through_the_real_loop(monkeypatch):
    """Isolated from the loop's own per-iteration bump by running the same turn
    with and without provider iterations."""
    agent_with, _ = _run_turn(monkeypatch, projected=_projected_rows(), iterations=1)
    agent_without, _ = _run_turn(monkeypatch, projected=[], iterations=0)
    assert agent_with._iters_since_skill - agent_without._iters_since_skill == 1


def test_ordinary_provider_turn_is_unchanged(monkeypatch):
    """A completion without the attributes must not gain rows or counter ticks."""
    _agent_, result = _run_turn(monkeypatch, projected=[], iterations=0)
    assert not [
        m for m in result["messages"]
        if isinstance(m, dict) and str(m.get("name") or "").startswith("acpagent_")
    ]
