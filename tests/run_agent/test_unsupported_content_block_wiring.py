"""Outbound-payload tests for the unsupported-content-block strip.

Companion to ``test_unsupported_content_block_strip.py``, which covers the
helper in isolation. These tests exercise the two WIRING sites by driving a
real ``run_conversation`` / summary call and inspecting the payload the
provider actually receives:

- the main conversation loop (``agent/conversation_loop.py``)
- the bounded-response summary path (``agent/chat_completion_helpers.py``)

They assert on the intercepted ``chat.completions.create(**kwargs)`` messages
rather than on the helper's return value, so a regression that removes either
call site fails here even though the helper still works.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch



from run_agent import AIAgent

COPILOT = "https://api.githubcopilot.com"


def _mock_response(content="done", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=None, reasoning=None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _make_agent(model: str, base_url: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url=base_url,
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def _mixed_history():
    """A Claude-style turn: signed thinking block next to visible text."""
    return [
        {"role": "user", "content": "first question"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal reasoning", "signature": "sig"},
                {"type": "text", "text": "visible answer"},
            ],
        },
    ]


def _run_turn(agent, history):
    """Drive one turn and return the messages sent to the provider."""
    captured = {}

    def _capture(*_args, **kwargs):
        captured.setdefault("messages", copy.deepcopy(kwargs.get("messages")))
        return _mock_response()

    agent.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("second question", conversation_history=history)
    return captured.get("messages")


def _block_types(messages):
    out = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            out.extend(
                str(b.get("type")) for b in c if isinstance(b, dict) and b.get("type")
            )
    return out


def _sent_text(messages):
    """All visible text in the payload, regardless of content representation.

    Downstream normalization may flatten a content list into a plain string
    before the transport, so assertions about surviving *text* must not depend
    on the blocks still being a list. Reasoning text must never appear here.
    """
    parts = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    parts.append(b["text"])
    return "\n".join(parts)


def _has_reasoning_leak(messages):
    """True if any disallowed reasoning block survived, in either representation."""
    if "thinking" in _block_types(messages):
        return True
    # If content was flattened to a string, the reasoning body would ride along
    # as plain text — catch that too.
    return "internal reasoning" in _sent_text(messages)


# --------------------------------------------------------------------------
# Main conversation loop wiring.
# --------------------------------------------------------------------------


def test_main_loop_strips_thinking_block_for_copilot_gemini():
    agent = _make_agent("gemini-3.6-flash", COPILOT)
    sent = _run_turn(agent, _mixed_history())

    assert sent is not None, "provider was never called"
    assert not _has_reasoning_leak(sent), f"reasoning reached the wire: {sent}"
    # The visible half of the same message must survive the strip.
    assert "visible answer" in _sent_text(sent)


def test_main_loop_preserves_thinking_block_for_claude():
    """The same history on Claude must go out untouched."""
    agent = _make_agent("claude-opus-4.8", COPILOT)
    sent = _run_turn(agent, _mixed_history())

    assert sent is not None, "provider was never called"
    assert _has_reasoning_leak(sent), "reasoning was stripped for a non-Gemini model"


def test_gate_does_not_fire_on_empty_base_url():
    """An unknown/empty endpoint is NOT Copilot — do not strip on a guess.

    Regression guard: the gate previously matched any bare ``gemini*`` model
    when no base URL was known, which would strip reasoning blocks for an
    unverified endpoint. Asserted at the gate rather than through a live turn
    because ``AIAgent`` refuses to construct without a resolvable provider.
    """
    from agent.agent_runtime_helpers import _model_requires_text_image_blocks_only

    assert _model_requires_text_image_blocks_only("gemini-3.6-flash", "") is False
    assert _model_requires_text_image_blocks_only("gemini-3.6-flash", None) is False
    # ...while affirmative Copilot evidence still matches.
    assert _model_requires_text_image_blocks_only("gemini-3.6-flash", COPILOT) is True


def test_main_loop_does_not_mutate_stored_history():
    """Stored history keeps the reasoning trace for the UI and persistence."""
    agent = _make_agent("gemini-3.6-flash", COPILOT)
    history = _mixed_history()
    snapshot = copy.deepcopy(history)

    _run_turn(agent, history)

    assert history == snapshot


def test_main_loop_keeps_allowed_block_types_intact():
    """Only disallowed types are dropped; allowed ones pass through untouched.

    Uses two text blocks plus a disallowed one rather than a real image, since
    an image payload is rewritten by vision preprocessing before it reaches the
    transport, which would test that pipeline instead of this strip.
    """
    agent = _make_agent("gemini-3.6-flash", COPILOT)
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "part one"},
                {"type": "text", "text": "part two"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "answer"},
            ],
        },
    ]
    sent = _run_turn(agent, history)

    assert sent is not None, "provider was never called"
    assert "redacted_thinking" not in _block_types(sent)
    assert "opaque" not in _sent_text(sent), "redacted reasoning reached the wire"
    # Every allowed block survives — both user text parts and the answer.
    text = _sent_text(sent)
    for expected in ("part one", "part two", "answer"):
        assert expected in text, f"{expected!r} missing from payload: {text!r}"


# --------------------------------------------------------------------------
# Summary path wiring (chat_completion_helpers).
# --------------------------------------------------------------------------


def _run_summary(agent, history):
    """Drive the bounded-response summary path; return the sent messages."""
    from agent.chat_completion_helpers import handle_max_iterations

    captured = {}

    def _capture(*_args, **kwargs):
        captured.setdefault("messages", copy.deepcopy(kwargs.get("messages")))
        return _mock_response("summary text")

    agent.client.chat.completions.create.side_effect = _capture
    messages = list(history)
    agent.messages = messages
    handle_max_iterations(agent, messages, api_call_count=90)
    return captured.get("messages")


def test_summary_path_strips_thinking_block_for_copilot_gemini():
    agent = _make_agent("gemini-3.6-flash", COPILOT)
    sent = _run_summary(agent, _mixed_history())

    assert sent is not None, "summary path never reached the provider"
    assert not _has_reasoning_leak(sent), f"reasoning reached the wire: {sent}"


def test_summary_path_preserves_thinking_block_for_claude():
    agent = _make_agent("claude-opus-4.8", COPILOT)
    sent = _run_summary(agent, _mixed_history())

    assert sent is not None, "summary path never reached the provider"
    assert _has_reasoning_leak(sent), "reasoning was stripped for a non-Gemini model"
