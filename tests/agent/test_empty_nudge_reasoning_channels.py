"""Two gaps in the empty-response ladder's reasoning detection.

The post-tool empty-response ladder decides whether a response was SILENT. The
model's reasoning can arrive in two shapes, and the ladder treats them
differently — deliberately:

* **Inline** (a think tag left in the content) means the model was mid-thought
  when it stopped. The post-tool nudge is skipped and the thinking-only
  reminder takes over.
* **A separate channel** (``reasoning_content`` / ``reasoning`` /
  ``reasoning_details`` / ``thinking``) is the parser's own split. The model
  still owes an answer, so the nudge is CORRECT there: suppressing it would
  let a reasoning model end its turn with empty output and call it done
  (upstream's decision on #21811 / #22799).

Two real gaps are pinned here:

1. **The inline detector missed the closing tag.** MiniMax M2 is an
   interleaved-thinking model whose chat template PREFILLS the opening
   ``<think>``, so the completion arrives as ``…reasoning…</think>`` — a bare
   closer with no opener. The old regex matched opening tags only, so this
   looked like silence and the nudge fired on a model that had reasoned in
   full (#21811 residual). The detector now matches both forms, in every
   canonical tag name.

2. **The prefill branch read fewer channels than the helper.** ``thinking``
   and the OpenAI SDK's ``model_extra`` fall-through were invisible to the
   branch that routes a reasoning-only reply to the completion reminder, so
   such a reply fell through the empty-response retry as if the model had said
   nothing. Both now reach the branch.

The detection lives in ``agent.reasoning_channels`` so every caller agrees on
the channel set and the tag names (``agent.think_scrubber.THINK_TAG_NAMES`` is
the one list every reasoning-hiding surface binds to).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import turn_empty_response as ter
from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE, _EMPTY_TOOL_RESPONSE_NUDGE
from agent.reasoning_channels import (
    has_inline_reasoning_tag,
    has_separate_reasoning_channel,
)


# --------------------------------------------------------------------------- #
# has_separate_reasoning_channel — the channel shapes                            #
# --------------------------------------------------------------------------- #


def _msg(**fields):
    """An assistant message with the four reasoning fields all absent."""
    base = {"reasoning": None, "reasoning_content": None, "reasoning_details": None}
    base.update(fields)
    return SimpleNamespace(**base)


def test_reasoning_content_attribute():
    assert has_separate_reasoning_channel(_msg(reasoning_content="thinking it through")) is True


def test_reasoning_attribute():
    assert has_separate_reasoning_channel(_msg(reasoning="thinking it through")) is True


def test_reasoning_details_attribute():
    """OpenRouter unified format and MiniMax M2's native array."""
    msg = _msg(reasoning_details=[{"type": "reasoning.text", "text": "step by step"}])
    assert has_separate_reasoning_channel(msg) is True


def test_thinking_attribute():
    """Ollama's native /api/chat field."""
    assert has_separate_reasoning_channel(_msg(thinking="native thinking")) is True


def test_model_extra_fallthrough():
    """The OpenAI SDK hides provider-extra fields under model_extra."""
    msg = _msg()
    msg.model_extra = {"reasoning_content": "hidden channel"}
    assert has_separate_reasoning_channel(msg) is True


def test_thinking_in_model_extra():
    """Ollama's ``thinking`` arrives under model_extra on the OpenAI SDK."""
    msg = _msg()
    msg.model_extra = {"thinking": "hidden thinking"}
    assert has_separate_reasoning_channel(msg) is True


def test_dict_payload():
    assert has_separate_reasoning_channel({"reasoning_content": "raw payload"}) is True


def test_dict_payload_model_extra():
    assert has_separate_reasoning_channel(
        {"model_extra": {"thinking": "raw payload"}}
    ) is True


def test_empty_reasoning_details_is_falsy():
    """An empty list carries no reasoning."""
    assert has_separate_reasoning_channel(_msg(reasoning_details=[])) is False


def test_all_empty_is_falsy():
    assert has_separate_reasoning_channel(_msg()) is False


def test_none_message_is_falsy():
    assert has_separate_reasoning_channel(None) is False


# --------------------------------------------------------------------------- #
# has_inline_reasoning_tag — the inline shapes                                 #
# --------------------------------------------------------------------------- #


def test_inline_opening_tag():
    assert has_inline_reasoning_tag("<think>reasoning</think>") is True


def test_inline_closing_tag_only():
    """MiniMax M2 prefills the opener, so the completion carries only the closer."""
    assert has_inline_reasoning_tag("some reasoning</think>") is True


def test_inline_bare_closing_tag():
    assert has_inline_reasoning_tag("</think>") is True


@pytest.mark.parametrize(
    "tag", ["think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD", "mm:think"]
)
def test_every_canonical_tag_name_is_matched(tag):
    assert has_inline_reasoning_tag(f"<{tag}>reasoning</{tag}>") is True
    assert has_inline_reasoning_tag(f"reasoning</{tag}>") is True


def test_prose_without_a_tag_is_not_inline_reasoning():
    assert has_inline_reasoning_tag("just an answer") is False


def test_empty_content_is_not_inline_reasoning():
    assert has_inline_reasoning_tag("") is False
    assert has_inline_reasoning_tag(None) is False


# --------------------------------------------------------------------------- #
# The ladder                                                                   #
# --------------------------------------------------------------------------- #


def _agent():
    a = MagicMock()
    a.model = "test/model"
    a.provider = "test"
    a._current_streamed_assistant_text = ""
    a._has_content_after_think_block.return_value = False
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._last_content_with_tools = None
    a._thinking_prefill_retries = 0
    a._fallback_chain = None
    a._empty_content_retries = 0
    a._emit_diagnostic_status = MagicMock()
    a._buffer_diagnostic_status = MagicMock()
    a._flush_status_buffer = MagicMock()
    a._build_assistant_message = MagicMock(return_value={"role": "assistant", "content": ""})
    # MagicMock auto-creates truthy child attributes; the real AIAgent does not.
    a._post_tool_empty_retried = False
    a._response_was_previewed = False
    return a


def _recover(agent, assistant_message, final_response, messages):
    return ter.recover_empty_response(
        agent,
        assistant_message,
        None,
        "stop",
        final_response=final_response,
        messages=messages,
        api_messages=[],
        conversation_history=[],
        active_system_prompt="sys",
        api_call_count=1,
        turn_exit_reason=None,
        preflight_compression_blocked=False,
    )


_TOOL_THEN_USER = [
    {"role": "user", "content": "run it"},
    {"role": "tool", "content": "done"},
]


def test_orphan_closing_tag_after_a_tool_call_does_not_nudge():
    """#21811 residual: MiniMax M2's bare closing tag is not silence.

    The old detector matched opening tags only, so this looked like an empty
    reply after a tool call and the nudge fired on a model that had reasoned in
    full.
    """
    agent = _agent()
    verdict = _recover(agent, _msg(), "some reasoning</think>", list(_TOOL_THEN_USER))
    assert agent._post_tool_empty_retried is False
    assert verdict.action == "continue"
    assert agent._empty_response_retry_hint == _CODEX_INCOMPLETE_NUDGE


def test_inline_opening_tag_after_a_tool_call_does_not_nudge():
    agent = _agent()
    verdict = _recover(agent, _msg(), "<think>reasoning</think>", list(_TOOL_THEN_USER))
    assert agent._post_tool_empty_retried is False
    assert verdict.action == "continue"
    assert agent._empty_response_retry_hint == _CODEX_INCOMPLETE_NUDGE


def test_separate_reasoning_after_a_tool_call_still_nudges():
    """Upstream's decision on #21811 / #22799: the nudge is correct here.

    A parser-split reasoning channel still means the model owes an answer.
    Suppressing the nudge would let a reasoning model end its turn with empty
    output and call it done. (PRs #21847 and #22799 proposed suppressing it;
    the change was not accepted, and upstream ``main`` pins this routing.)
    """
    agent = _agent()
    verdict = _recover(
        agent, _msg(reasoning_content="I should look this up"), "", list(_TOOL_THEN_USER)
    )
    assert agent._post_tool_empty_retried is True
    assert verdict.action == "continue"
    assert agent._empty_response_retry_hint == _EMPTY_TOOL_RESPONSE_NUDGE


def test_a_genuinely_empty_response_after_a_tool_call_still_nudges():
    """No reasoning anywhere: the nudge is the correct recovery."""
    agent = _agent()
    verdict = _recover(agent, _msg(), "", list(_TOOL_THEN_USER))
    assert agent._post_tool_empty_retried is True
    assert verdict.action == "continue"
    assert agent._empty_response_retry_hint == _EMPTY_TOOL_RESPONSE_NUDGE


@pytest.mark.parametrize(
    "assistant_message",
    [
        _msg(reasoning_content="x"),
        _msg(reasoning="x"),
        _msg(reasoning_details=[{"type": "reasoning.text", "text": "x"}]),
        _msg(thinking="x"),
    ],
    ids=["reasoning_content", "reasoning", "reasoning_details", "thinking"],
)
def test_every_separate_channel_reaches_the_prefill_branch(assistant_message):
    """The prefill branch must read every channel the helper reads.

    ``thinking`` and the SDK's ``model_extra`` fall-through were invisible to
    it, so such a reply fell through the empty-response retry as if the model
    had said nothing. This is the shape teknium1 asked for on #21847.
    """
    agent = _agent()
    # No prior tool call, so the post-tool nudge does not apply and the
    # prefill branch is the one under test.
    verdict = _recover(agent, assistant_message, "", [{"role": "user", "content": "hi"}])
    assert verdict.action == "continue"
    assert agent._empty_response_retry_hint == _CODEX_INCOMPLETE_NUDGE
    assert agent._thinking_prefill_retries == 1


def test_model_extra_reasoning_reaches_the_prefill_branch():
    """The SDK hides provider-extra fields under model_extra."""
    msg = _msg()
    msg.model_extra = {"thinking": "hidden thinking"}
    agent = _agent()
    verdict = _recover(agent, msg, "", [{"role": "user", "content": "hi"}])
    assert verdict.action == "continue"
    assert agent._empty_response_retry_hint == _CODEX_INCOMPLETE_NUDGE
