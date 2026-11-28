"""``literal_think_tags``: a route that returns reasoning out of band keeps think-tag text.

The reasoning-tag scrubbers delete every ``<think>``-style tag from ``content`` because some
models inline their reasoning there. On a route whose server returns reasoning in its own field
and ends thinking by token id, the tag text in ``content`` is the answer, and deleting it loses
it: asked to reply with exactly ``</think>``, the model did, and the turn came back empty.

Invariants pinned here:
- Opted-in route: lone, paired, orphan, fenced and line-start tag text survives the storage /
  final scrub; tool-call XML is still scrubbed; reasoning stays in its own field.
- No opt-in, another model on the provider, or a fallback route: stripping as before.
- A literal-tag reply is a real reply: one API call, no empty-response recovery.
- Streaming: split-delta tag text reaches the stream callbacks on an opted-in route and is still
  scrubbed on a default one.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.agent_runtime_helpers import strip_think_blocks
from agent.route_capabilities import literal_think_tags_capability

_ROUTE = "https://llm.example.invalid/v1"
_MODEL = "deepseek-v4.1-flash"
_OPTED_IN = [{"name": "local-llm", "base_url": _ROUTE, "models": {_MODEL: {"literal_think_tags": True}}}]

# Texts whose think tags are answer text, not reasoning.
_LITERAL_REPLIES = [
    "</think>",
    "The `<think>…</think>` pair wraps the model's reasoning.",
    "Close the block with </think> on its own line.",
    "```\n<think>\nplan the answer\n</think>\nanswer\n```",
    "<think>\nThis line-start block is an example, not reasoning.\n</think>\nDone.",
]


def _route(model=_MODEL, base_url=_ROUTE, custom_providers=_OPTED_IN, provider="custom", requested=""):
    return SimpleNamespace(model=model, base_url=base_url, provider=provider, requested_provider=requested,
                           _custom_providers=custom_providers)


_A_ON = {"name": "a", "base_url": _ROUTE, "models": {_MODEL: {"literal_think_tags": True}}}
_A_PLAIN = {"name": "a", "base_url": _ROUTE}
_B_OFF = {"name": "b", "base_url": _ROUTE, "capabilities": {"literal_think_tags": False}}


@pytest.mark.parametrize("agent, expected", [
    (_route(), True),
    (_route(model="other-model"), False),  # the per-model key names one model
    (_route(base_url="https://openrouter.ai/api/v1"), False),  # another route never inherits it
    (_route(custom_providers=[{"name": "local-llm", "base_url": _ROUTE}]), False),  # default off
    (_route(custom_providers=[{"name": "local-llm", "base_url": _ROUTE,
                               "capabilities": {"literal_think_tags": True}}]), True),
    (_route(custom_providers=[{"name": "local-llm", "base_url": _ROUTE,
                               "models": {_MODEL: {"literal_think_tags": 1}}}]), False),  # not a bool
    (None, False),
    (_route(custom_providers=[_A_ON, _B_OFF]), False),  # unnamed route: an explicit false wins
    (_route(custom_providers=[_A_ON, _B_OFF], requested="custom:a"), True),  # the live entry decides
    (_route(custom_providers=[_A_ON, _B_OFF], requested="custom:gone"), False),  # identity not here
    (_route(custom_providers=[_A_ON, _A_PLAIN], requested="a"), False),  # ambiguous name: no consensus
])
def test_capability_is_an_explicit_per_route_opt_in(agent, expected):
    assert literal_think_tags_capability(agent) is expected


@pytest.mark.parametrize("text", _LITERAL_REPLIES)
def test_opted_in_route_keeps_think_tag_text(text):
    assert strip_think_blocks(_route(), text) == text
    # The same text on a default route is scrubbed, which is the behaviour the opt-in exists for.
    assert strip_think_blocks(_route(custom_providers=[]), text) != text


def test_opted_in_route_still_scrubs_tool_call_xml():
    text = 'Answer.\n<tool_call>{"name": "terminal", "arguments": {}}</tool_call>\n</tool_call>'
    kept = strip_think_blocks(_route(), text)
    assert kept == strip_think_blocks(None, text)
    assert kept.strip() == "Answer."


def test_fallback_route_gets_default_stripping_back():
    """The opt-in follows the live route: failing over to another provider restores stripping,
    and restoring the primary brings the literal handling back."""
    agent = _route()
    assert strip_think_blocks(agent, "</think>") == "</think>"
    agent.base_url, agent.model = "https://openrouter.ai/api/v1", "deepseek/deepseek-v4.1"
    assert strip_think_blocks(agent, "</think>") == ""
    agent.base_url, agent.model = _ROUTE, _MODEL
    assert strip_think_blocks(agent, "</think>") == "</think>"


def _build_agent(tmp_path, monkeypatch, *, opted_in: bool):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    capability = "        literal_think_tags: true\n" if opted_in else "        context_length: 1000000\n"
    (tmp_path / "config.yaml").write_text(
        f"providers:\n  local-llm:\n    api: {_ROUTE}\n    models:\n      {_MODEL}:\n{capability}",
        encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(model=_MODEL, api_key="sk-dummy", base_url=_ROUTE, quiet_mode=True,
                    skip_context_files=True, skip_memory=True, platform="cli")
    # Non-streaming path, so the monkeypatched responses below are what the loop consumes.
    agent._disable_streaming = True
    return agent


def _literal_tag_reply():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="</think>", reasoning=None,
                                    reasoning_content="The user wants the closing tag, verbatim.",
                                    reasoning_details=None, tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
        model=_MODEL,
    )


def test_literal_tag_reply_is_delivered_on_the_first_call(tmp_path, monkeypatch):
    agent = _build_agent(tmp_path, monkeypatch, opted_in=True)
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: _literal_tag_reply())

    result = agent.run_conversation("Reply with exactly this string and nothing else: </think>")

    assert result["final_response"] == "</think>"
    assert result["api_calls"] == 1
    row = result["messages"][-1]
    assert row["content"] == "</think>"
    assert row["reasoning_content"] == "The user wants the closing tag, verbatim."


def test_without_the_opt_in_the_literal_tag_reply_is_still_stripped(tmp_path, monkeypatch):
    agent = _build_agent(tmp_path, monkeypatch, opted_in=False)
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: _literal_tag_reply())

    result = agent.run_conversation("Reply with exactly this string and nothing else: </think>")

    assert result["final_response"] != "</think>"


def test_tool_call_turn_keeps_literal_tag_and_reasoning_field(tmp_path, monkeypatch):
    """A tool-call turn is stored with its tag text intact; reasoning stays in its own field,
    which is what a replaying thinking-mode endpoint reads back."""
    agent = _build_agent(tmp_path, monkeypatch, opted_in=True)
    message = SimpleNamespace(
        content="Checking how `</think>` renders first.", reasoning=None,
        reasoning_content="private plan", reasoning_details=None,
        tool_calls=[SimpleNamespace(id="call_1", call_id=None, type="function",
                                    function=SimpleNamespace(name="terminal", arguments="{}"))],
    )

    row = agent._build_assistant_message(message, "tool_calls")

    assert row["content"] == "Checking how `</think>` renders first."
    assert row["reasoning_content"] == "private plan"


@pytest.mark.parametrize("opted_in, expected", [
    (True, "Close it with </think>, or write <think>x</think> inline."),
    (False, "Close it with , or write  inline."),
])
def test_streamed_split_tags_follow_the_route(tmp_path, monkeypatch, opted_in, expected):
    agent = _build_agent(tmp_path, monkeypatch, opted_in=opted_in)
    shown: list[str] = []
    agent.stream_delta_callback = shown.append
    agent._reset_stream_delivery_tracking()

    for delta in ["Close it with </th", "ink>, or write <thi", "nk>x</think> inline."]:
        agent._fire_stream_delta(delta)
    agent._reset_stream_delivery_tracking()  # end of response: flushes any held partial tag

    assert "".join(shown) == expected


def test_failover_to_another_route_restores_stripping_for_the_next_response(tmp_path, monkeypatch):
    """Primary opted in; after failing over, the next model response (a fresh stream reset)
    re-resolves the route, so neither the final scrub nor the stream keeps the tags."""
    agent = _build_agent(tmp_path, monkeypatch, opted_in=True)
    shown: list[str] = []
    agent.stream_delta_callback = shown.append
    agent._reset_stream_delivery_tracking()
    agent._fire_stream_delta("</think>")
    assert shown == ["</think>"] and agent._strip_think_blocks("</think>") == "</think>"

    agent.provider, agent.base_url, agent.model = "openrouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4.1"
    agent._reset_stream_delivery_tracking()
    shown.clear()
    agent._fire_stream_delta("Answer </think>")
    agent._reset_stream_delivery_tracking()

    assert "".join(shown) == "Answer "
    assert agent._strip_think_blocks("</think>") == ""


@pytest.mark.parametrize("opted_in, expected", [(True, ""), (False, "literal sample")])
def test_reasoning_preview_never_presents_answer_tags_as_reasoning(tmp_path, monkeypatch, opted_in, expected):
    """The ``reasoning.available`` preview falls back to inline tags only where they are reasoning."""
    from agent.turn_response_intake import _reasoning_preview

    agent = _build_agent(tmp_path, monkeypatch, opted_in=opted_in)
    message = SimpleNamespace(content="Example: <think>literal sample</think>.", reasoning=None,
                              reasoning_content=None, reasoning_details=None, tool_calls=None)
    assert _reasoning_preview(agent, message.content, message) == expected
