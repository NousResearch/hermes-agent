"""Regression: a stale reasoning belief must not outlive the turn it was formed in.

Finding (group bot, 2026-11-06): ``reapply_reasoning_echo_for_provider`` replayed every
assistant row's stored ``reasoning_content`` verbatim — including rows from earlier turns.
A session that once reasoned "I can't create a file / the image isn't arriving" kept obeying
that conclusion after ``render_chart`` returned to the request:

    reasoning echoed (live shape)  -> calls=['vision_analyze']
    reasoning dropped             -> calls=['render_chart']

Same 23-tool payload, same question. The provider (``deepseek``) *does* require a non-empty
``reasoning_content`` on tool-call turns, so the field must stay — what must not be replayed
is the earlier turn's conclusion. After the fix the field is padded (``" "``) for history and
carries real text only for the turn in flight, which is also what DeepSeek documents: pass the
reasoning of the current round back, drop the other rounds'.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from run_agent import AIAgent

RC = "reasoning_content"

BELIEF = "I can't create a file and the image isn't arriving, so there is no chart tool."
BELIEF_ANSWER = "Не можу: у мене немає інструмента для зображень."
CURRENT_TURN_REASONING = "render_chart is in the tool list — call it with the series."

RENDER_CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "render_chart",
        "description": "Render a chart from data points to a PNG image and attach it to the reply.",
        "parameters": {
            "type": "object",
            "properties": {"series": {"type": "array", "items": {"type": "number"}}},
            "required": ["series"],
        },
    },
}


def _make_agent(provider: str = "deepseek", model: str = "deepseek-flash",
                base_url: str = "https://api.deepseek.com") -> AIAgent:
    """A bare agent on a chat-completions route (the shape the echo policy reads)."""
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.api_mode = "chat_completions"
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def _history_with_false_belief() -> list[dict]:
    """Two turns: last turn reasoned itself into "no chart tool", this turn was handed one."""
    return [
        {"role": "system", "content": "Ти — бот приватної групи. Відповідай українською."},
        {"role": "user", "content": "намалюй графік цін за тиждень"},
        {"role": "assistant", "content": BELIEF_ANSWER, RC: BELIEF},
        {"role": "user", "content": "то намалюй графік, інструмент є"},
        {"role": "assistant", "content": "", RC: CURRENT_TURN_REASONING,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "render_chart", "arguments": "{\"series\": [1, 2]}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "chart.png"},
    ]


def _reasoning_texts(messages: list[dict]) -> list[str]:
    return [m[RC] for m in messages if m.get("role") == "assistant" and isinstance(m.get(RC), str)]


def test_false_capability_belief_is_not_replayed_while_the_tool_is_in_the_request():
    """The regression: the model's own stale conclusion must not ride into the request."""
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    agent = _make_agent()
    wire = copy.deepcopy(_history_with_false_belief())
    reapply_reasoning_echo_for_provider(agent, wire)

    replayed = _reasoning_texts(wire)
    assert BELIEF not in json.dumps(wire, ensure_ascii=False), (
        "a stale capability belief from an earlier turn was echoed back to the model")
    assert all(text.strip() != BELIEF for text in replayed)
    # DeepSeek still gets a non-empty reasoning_content on every assistant turn (no HTTP 400).
    assert len(replayed) == 2 and all(text != "" for text in replayed)
    assert wire[2][RC] == " ", "earlier turns keep the pad, never their conclusion"


def test_current_turn_reasoning_still_replays():
    """No over-redaction: the reasoning of the turn in flight is what the model needs."""
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    wire = copy.deepcopy(_history_with_false_belief())
    reapply_reasoning_echo_for_provider(_make_agent(), wire)

    assert wire[4][RC] == CURRENT_TURN_REASONING


def test_history_text_is_dropped_on_routes_that_do_not_require_the_field():
    """A non-must-echo chat route keeps no reasoning key at all for earlier turns."""
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    wire = copy.deepcopy(_history_with_false_belief())
    # A model outside the must-echo families: the field is optional there.
    openrouter = _make_agent(provider="openrouter", model="anthropic/claude-sonnet-5.5",
                             base_url="https://openrouter.ai/api/v1")
    reapply_reasoning_echo_for_provider(openrouter, wire)

    assert RC not in wire[2] and "reasoning" not in wire[2]
    assert wire[4][RC] == CURRENT_TURN_REASONING  # the turn in flight keeps its own


def test_idempotent():
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    agent = _make_agent()
    wire = copy.deepcopy(_history_with_false_belief())
    assert reapply_reasoning_echo_for_provider(agent, wire) > 0
    assert reapply_reasoning_echo_for_provider(agent, wire) == 0


def test_turn_anchor():
    from agent.message_sanitization import current_turn_start

    msgs = _history_with_false_belief()
    assert current_turn_start(msgs) == 4  # the row after the last user message (index 3)
    assert current_turn_start([{"role": "system", "content": "s"}]) == 0  # no anchor: fail open


def test_no_user_turn_keeps_history_untouched():
    """A replay with no user row has no turn anchor: scope nothing rather than guess."""
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    wire = [{"role": "system", "content": "s"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
             RC: BELIEF}]
    reapply_reasoning_echo_for_provider(_make_agent(), wire)
    assert wire[1][RC] == BELIEF


def test_wire_payload_carries_the_tool_and_no_stale_belief():
    """End-to-end shape of the failing live request: tools present, belief absent."""
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    wire = copy.deepcopy(_history_with_false_belief())
    reapply_reasoning_echo_for_provider(_make_agent(), wire)
    payload = {"model": "deepseek-flash", "messages": wire, "tools": [RENDER_CHART_TOOL]}

    dumped = json.dumps(payload, ensure_ascii=False)
    assert '"name": "render_chart"' in dumped
    assert BELIEF not in dumped
    assert BELIEF_ANSWER in dumped  # visible answer stays; only the reasoning text is scoped
