"""Regression tests for issue #21811.

The post-tool empty-response nudge previously fired even when the upstream
provider had split the model's chain-of-thought into a separate
``reasoning_content`` / ``reasoning`` channel (Ollama qwen3.x PARSER,
DeepSeek-R1, Moonshot, Novita, etc.). In that case ``content`` is empty
and there is no inline ``<think>`` tag (the parser already stripped it),
so the old detection wrongly classified the response as silent and
triggered a wasteful retry round-trip.

These tests pin down the new behaviour:

* ``_has_separate_reasoning`` recognises reasoning_content / reasoning on
  both pydantic-style attribute objects and plain dict payloads, on
  ``model_extra`` fall-throughs, and ignores empty values.
* Genuinely empty messages (no inline tag, no separate reasoning) still
  trigger the nudge — no regression.
* Inline ``<think>`` blocks in content keep their existing detection
  path (``_has_inline_thinking``).
* When inline thinking AND separate reasoning are both present, both
  guards are true (and the nudge is skipped).
"""

from types import SimpleNamespace

import pytest

import run_agent
from run_agent import _has_separate_reasoning


# --------------------------------------------------------------------------- #
# _has_separate_reasoning unit tests                                          #
# --------------------------------------------------------------------------- #


class _PydanticLikeMessage(SimpleNamespace):
    """Stand-in for openai.types.chat.ChatCompletionMessage which exposes
    provider-extra fields (e.g. ``reasoning_content``) via attributes and
    via ``model_extra``."""


def test_has_separate_reasoning_attr_reasoning_content():
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content="Thought: the user wants me to save…",
        reasoning=None,
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_attr_reasoning():
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content=None,
        reasoning="thinking through the steps",
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_attr_reasoning_details():
    """OpenRouter unified format and MiniMax M2's native OpenAI-compatible
    API return chain-of-thought as a structured ``reasoning_details`` array.
    This channel was previously omitted from the guard, which let the nudge
    fire on MiniMax M2 (issue #21811 residual reported on Ollama cloud)."""
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content=None,
        reasoning=None,
        reasoning_details=[{"type": "reasoning.text", "text": "step-by-step"}],
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_empty_reasoning_details_is_falsy():
    """An empty ``reasoning_details`` list carries no reasoning — must not
    suppress the nudge (matches ``extract_reasoning``'s truthiness guard)."""
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content=None,
        reasoning=None,
        reasoning_details=[],
    )
    assert _has_separate_reasoning(msg) is False


def test_has_separate_reasoning_attr_thinking():
    """Ollama's native /api/chat field and some OpenAI-compat proxies expose
    reasoning under a top-level ``thinking`` key."""
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content=None,
        reasoning=None,
        thinking="let me work through this",
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_reasoning_details_via_model_extra():
    """``reasoning_details`` hidden under the OpenAI SDK's ``model_extra``
    must still be detected."""
    msg = SimpleNamespace(
        content="",
        model_extra={"reasoning_details": [{"text": "hidden array"}]},
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_dict_reasoning_details():
    msg = {"content": "", "reasoning_details": [{"text": "raw dict array"}]}
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_model_extra_fallback():
    """OpenAI SDK hides unknown provider fields under ``model_extra`` —
    we still need to detect them."""
    msg = SimpleNamespace(
        content="",
        model_extra={"reasoning_content": "hidden in model_extra"},
    )
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_dict_payload():
    msg = {"content": "", "reasoning_content": "raw dict reasoning"}
    assert _has_separate_reasoning(msg) is True


def test_has_separate_reasoning_empty_string_is_falsy():
    msg = _PydanticLikeMessage(
        content="",
        reasoning_content="",
        reasoning="",
    )
    assert _has_separate_reasoning(msg) is False


def test_has_separate_reasoning_none_message():
    assert _has_separate_reasoning(None) is False


def test_has_separate_reasoning_no_reasoning_fields():
    msg = _PydanticLikeMessage(content="actual answer")
    assert _has_separate_reasoning(msg) is False


def test_has_separate_reasoning_inline_think_only_does_not_count():
    """Inline ``<think>`` lives in ``content``; this helper only checks
    the separate-channel fields. Inline thinking has its own detector
    (``_has_inline_thinking``)."""
    msg = _PydanticLikeMessage(
        content="<think>reasoning here</think>",
        reasoning_content=None,
        reasoning=None,
    )
    assert _has_separate_reasoning(msg) is False


def test_has_separate_reasoning_both_inline_and_separate():
    """When inline thinking AND separate reasoning coexist (rare, but
    possible when a parser only partially splits), the helper still
    reports the separate channel — and the nudge guard short-circuits
    on either signal."""
    msg = _PydanticLikeMessage(
        content="<think>some inline</think>",
        reasoning_content="and also separate",
    )
    assert _has_separate_reasoning(msg) is True


# --------------------------------------------------------------------------- #
# Source-level guard test                                                     #
#                                                                             #
# The nudge decision lives deep in the agent loop. Rather than spin up a      #
# full integration harness, pin the guard wiring at the source level so a    #
# future refactor can't silently drop it.                                     #
# --------------------------------------------------------------------------- #


def _read_agent_source() -> str:
    """Return the source of the module that owns the post-tool nudge.

    Historically this lived in ``run_agent.py``. After the
    ``run_conversation`` refactor it moved to
    ``agent.conversation_loop``. We concatenate both so the source-level
    pins keep working regardless of where the guard is wired.
    """
    import inspect
    from agent import conversation_loop
    return inspect.getsource(run_agent) + "\n" + inspect.getsource(conversation_loop)


def test_nudge_guard_uses_has_separate_reasoning():
    src = _read_agent_source()
    # Helper must exist and be the one we wired in.
    assert "def _has_separate_reasoning(" in src
    # The post-tool empty nudge condition must consult it.
    # We look for the conjunction with `_has_separate_reasoning_channel`
    # — the local variable bound from the helper inside the loop. After
    # the conversation-loop refactor the call is routed through ``_ra()``
    # so callers can still monkey-patch ``run_agent._has_separate_reasoning``.
    assert (
        "_has_separate_reasoning_channel = _has_separate_reasoning(" in src
        or "_has_separate_reasoning_channel = _ra()._has_separate_reasoning(" in src
    )
    assert "and not _has_separate_reasoning_channel" in src


def test_nudge_guard_keeps_inline_thinking_check():
    """Regression: don't accidentally remove the existing inline-think
    guard while adding the new one."""
    src = _read_agent_source()
    assert "_has_inline_thinking = bool(" in src
    assert "and not _has_inline_thinking" in src


def test_inline_thinking_regex_matches_orphan_closing_tag():
    """MiniMax M2 prefills the opening ``<think>`` and emits only the closing
    ``</think>``. The inline-thinking guard must match a bare closing tag so
    the nudge is skipped when the parser leaves an orphan ``</think>`` in
    content (issue #21811 residual)."""
    import re

    src = _read_agent_source()
    # Pull the regex the loop uses for inline-thinking detection.
    m = re.search(
        r"_has_inline_thinking = bool\(\s*re\.search\(\s*r'([^']+)'",
        src,
    )
    assert m, "could not locate the _has_inline_thinking regex in source"
    pattern = m.group(1)
    # The broadened pattern must match an orphan closing tag...
    assert re.search(pattern, "</think>", re.IGNORECASE)
    # ...while still matching the classic opening-tag form.
    assert re.search(pattern, "<think>reasoning", re.IGNORECASE)


def test_nudge_guard_emits_status_string_unchanged():
    """The user-visible nudge status string is documented in the issue
    and external tooling may grep for it. Pin it."""
    src = _read_agent_source()
    assert "Model returned empty after tool calls" in src
