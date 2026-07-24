"""Structured message content projection regressions.

Canonical owner: ``agent.message_content.flatten_message_text`` — string
pass-through, ordered concatenation of confirmed textual parts
(text / input_text / output_text) without inserted separators, and no
str()/repr()/json.dumps() fallback for images, base64, reasoning metadata,
provider state, or unknown objects. Assistant and tool-role list content
must never reach a string-only sink unnormalized.

The length-continuation assertion follows the baseline concatenation
semantics ("partial" + "finished" == "partialfinished").
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Adapted for upstream main: the canonical projector in agent.message_content
# joins typed parts without a separator on the scrub path (sep="").
from functools import partial as _partial
from agent.message_content import flatten_message_text as _flatten_message_text
visible_text_from_content = _partial(_flatten_message_text, sep="")
from run_agent import AIAgent


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, call_id: str = "call-structured-content-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(content, *, finish_reason: str = "stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, interim_callback=None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="test-model",
            api_key="test-key-not-a-secret",
            base_url="https://test.invalid/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            interim_assistant_callback=interim_callback,
        )
    agent._cached_system_prompt = "You are a test double."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = set(tool_names)
    agent.max_iterations = 4
    return agent


def _run(agent: AIAgent, request, *, extra_patches=()):
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", request))
        save = stack.enter_context(patch.object(agent, "_save_trajectory"))
        cleanup = stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        persist = stack.enter_context(patch.object(agent, "_persist_session"))
        for extra_patch in extra_patches:
            stack.enter_context(extra_patch)
        result = agent.run_conversation("structured content regression")
    return result, save, cleanup, persist


class _TextPart:
    type = "output_text"
    text = "object text"


class _TextPartWithExplosiveLegacyContent:
    type = "output_text"
    text = "safe object text"

    @property
    def content(self):
        raise RuntimeError("legacy content must not be read")


class _Opaque:
    def __str__(self) -> str:
        return "MUST_NOT_LEAK"


class _ExplosiveContentObject:
    @property
    def type(self):
        raise RuntimeError("must be ignored")




@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, ""),
        ("plain", "plain"),
        ({"type": "output_text", "text": "typed dict"}, "typed dict"),
        ({"type": " OUTPUT_TEXT ", "text": "normalized type"}, "normalized type"),
        ({"text": "legacy text"}, "legacy text"),
        ({"content": "legacy content"}, "legacy content"),
        ({"type": "input_image", "text": "MUST_NOT_LEAK"}, ""),
        (_TextPart(), "object text"),
        (_TextPartWithExplosiveLegacyContent(), "safe object text"),
        (_Opaque(), ""),
        (_ExplosiveContentObject(), ""),
        (b"MUST_NOT_LEAK", ""),
        (7, ""),
    ],
)
def test_visible_text_scalar_and_wrapper_contract(content, expected):
    assert visible_text_from_content(content) == expected


def test_visible_text_list_preserves_order_boundaries_and_ignores_non_text():
    content = [
        "hel",
        {"type": "text", "text": "lo"},
        {"type": "input_text", "text": " "},
        {"type": "output_text", "text": "world"},
        SimpleNamespace(type="output_text", text="!"),
        {"type": "input_image", "image_url": "data:image/png;base64,MUST_NOT_LEAK"},
        {"type": "reasoning", "encrypted_content": "MUST_NOT_LEAK"},
        {"type": "unknown", "text": "MUST_NOT_LEAK"},
        {"text": "legacy list text"},
        {"content": "legacy list content"},
        {"type": "text", "text": {"nested": "MUST_NOT_LEAK"}},
        _Opaque(),
    ]
    # Untyped legacy dict wrappers ({"text": str} / {"content": str}) are
    # accepted in any position — same contract as upstream's
    # test_message_content.py. Typed-unknown parts and unknown objects
    # still contribute nothing.
    assert visible_text_from_content(content) == "hello world!legacy list textlegacy list content"


def test_cross_part_reasoning_tag_is_reassembled_before_scrubbing():
    agent = _make_agent()
    content = [
        {"type": "text", "text": "<thi"},
        {"type": "text", "text": "nk>secret</think>visible"},
    ]
    assert agent._strip_think_blocks(content) == "visible"


def test_build_assistant_message_normalizes_visible_text_after_reasoning_capture():
    agent = _make_agent()
    message = _response(
        [
            {"type": "thinking", "thinking": "private plan"},
            {"type": "output_text", "text": "visible answer"},
        ]
    ).choices[0].message

    built = agent._build_assistant_message(message, "stop")

    assert built["content"] == "visible answer"
    assert built["reasoning"] == "private plan"


def test_normal_loop_preserves_structured_reasoning_while_projecting_visible_text():
    agent = _make_agent()
    request = MagicMock(
        return_value=_response(
            [
                {"type": "thinking", "thinking": "private plan"},
                {"type": "output_text", "text": "visible answer"},
            ]
        )
    )

    result, _, _, _ = _run(agent, request)

    assert request.call_count == 1
    assert result["final_response"] == "visible answer"
    assert result["last_reasoning"] == "private plan"
    assistant_rows = [
        row for row in result["messages"] if row.get("role") == "assistant"
    ]
    assert assistant_rows[-1]["content"] == "visible answer"
    assert assistant_rows[-1]["reasoning"] == "private plan"


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "output_text", "text": "list visible"}],
        {"type": "output_text", "text": "dict visible"},
        {"text": "legacy visible"},
    ],
)
def test_structured_normal_response_completes_after_one_provider_request(content):
    agent = _make_agent()
    request = MagicMock(return_value=_response(content))
    result, _, cleanup, persist = _run(agent, request)
    assert request.call_count == 1
    assert result["api_calls"] == 1
    assert result["final_response"].endswith("visible")
    assert result["completed"] is True
    assert result["failed"] is False
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert cleanup.call_count == 1
    assert persist.call_count >= 1


def test_length_list_uses_continuation_instead_of_provider_error_retry():
    agent = _make_agent()
    replies = [
        _response(
            [
                {"type": "thinking", "thinking": "private continuation plan"},
                {"type": "output_text", "text": "partial"},
            ],
            finish_reason="length",
        ),
        _response("finished"),
    ]
    captured = []

    def fake_request(api_kwargs):
        captured.append(copy.deepcopy(api_kwargs))
        return replies.pop(0)

    request = MagicMock(side_effect=fake_request)
    result, _, _, _ = _run(
        agent,
        request,
        extra_patches=(
            patch("agent.conversation_loop.jittered_backoff", return_value=0),
            patch("agent.conversation_loop.time.sleep"),
        ),
    )

    second_messages = captured[1].get("messages", [])
    roles = [m.get("role") for m in second_messages if isinstance(m, dict)]
    assert request.call_count == 2
    assert result["api_calls"] == 2
    assert "assistant" in roles and roles[-1] == "user"
    # Baseline concatenation semantics (conversation_loop.py:
    # final_response = "".join(truncated_response_parts) + final_response):
    # the truncated fragment is preserved and prepended to the continuation.
    assert result["final_response"] == "partialfinished"
    assert result["completed"] is True
    assert result["failed"] is False
    partial_rows = [
        row
        for row in result["messages"]
        if row.get("role") == "assistant"
        and row.get("reasoning") == "private continuation plan"
    ]
    assert partial_rows
    assert partial_rows[-1]["content"] == "partial"


def test_content_filter_list_is_terminal_and_not_retried():
    agent = _make_agent()
    request = MagicMock(
        side_effect=[
            _response(
                [{"type": "output_text", "text": "declined"}],
                finish_reason="content_filter",
            ),
            _response("must not be requested"),
        ]
    )
    result, _, _, _ = _run(
        agent,
        request,
        extra_patches=(
            patch("agent.conversation_loop.jittered_backoff", return_value=0),
            patch("agent.conversation_loop.time.sleep"),
        ),
    )

    assert request.call_count == 1
    assert result["api_calls"] == 1
    assert "declined" in (result["final_response"] or "")
    assert "must not be requested" not in (result["final_response"] or "")
    assert result["completed"] is False


def test_content_filter_reasoning_only_explanation_is_preserved():
    agent = _make_agent()
    request = MagicMock(
        return_value=_response(
            [{"type": "thinking", "thinking": "policy explanation"}],
            finish_reason="content_filter",
        )
    )

    result, _, _, _ = _run(agent, request)

    assert request.call_count == 1
    assert result["api_calls"] == 1
    assert "policy explanation" in (result["final_response"] or "")
    assert result["completed"] is False


def test_tool_role_vision_result_reaches_interim_processing_without_crash():
    """Issue #66267: a tool-role message whose content is a vision list must
    not crash interim visible-text processing (the role gate historically
    came after the text extraction)."""
    agent = _make_agent()
    vision_result = [
        {"type": "image_url", "image_url": {"url": "https://img.invalid/x.png"}},
        {"type": "text", "text": "tool saw a diagram"},
    ]
    # Direct hit on the exact crash point from the issue traceback — any role,
    # including non-assistant roles, must be safe.
    out = agent._interim_assistant_visible_text({"role": "tool", "content": vision_result})
    assert "tool saw a diagram" in out
    assert "img.invalid" not in out


def test_tool_role_list_content_full_turn_completes_without_assistant_interim_probe():
    agent = _make_agent("vision_tool")
    request = MagicMock(
        side_effect=[
            _response(None, tool_calls=[_tool_call("vision_tool")]),
            _response("done after vision tool"),
        ]
    )
    original = agent._interim_assistant_visible_text
    probed_roles = []

    def _record_interim_probe(message):
        probed_roles.append(message.get("role") if isinstance(message, dict) else None)
        return original(message)

    with (
        patch(
            "run_agent.handle_function_call",
            return_value=[
                {"type": "image_url", "image_url": {"url": "https://img.invalid/y.png"}},
                {"type": "text", "text": "pixels"},
            ],
        ),
        patch.object(
            agent,
            "_interim_assistant_visible_text",
            side_effect=_record_interim_probe,
        ),
    ):
        result, _, _, _ = _run(agent, request)
    assert request.call_count == 2
    assert result["completed"] is True
    assert result["final_response"] == "done after vision tool"
    assert "tool" not in probed_roles


def test_image_only_list_produces_no_visible_text_and_no_crash():
    """An image-only assistant response normalizes to empty visible text.

    The turn must not crash with a TypeError and no URL/base64 may leak; the
    loop's pre-existing empty-response retry budget applies exactly as it
    would for a string-empty response (baseline-consistent behavior).
    """
    agent = _make_agent()
    request = MagicMock(
        return_value=_response(
            [{"type": "input_image", "image_url": "https://img.invalid/z.png"}]
        )
    )
    result, _, _, _ = _run(agent, request)
    assert request.call_count >= 2  # empty-response retry budget, not a crash loop
    assert result.get("turn_exit_reason") == "empty_response_exhausted"
    assert "img.invalid" not in (result["final_response"] or "")


# ── unknown objects must never be read reflectively into text ─────────────

class _UnknownWithText:
    def __init__(self):
        self.text = "ATTR-TEXT-MUST-NOT-LEAK"


class _UnknownWithContent:
    def __init__(self):
        self.content = "ATTR-CONTENT-MUST-NOT-LEAK"


class _UnknownWithStr:
    def __str__(self):
        return "STR-REPR-MUST-NOT-LEAK"


class _UnknownNested:
    def __init__(self):
        self.inner = _UnknownWithText()
        self.payload = {"type": "output_text", "text": "NESTED-MUST-NOT-LEAK"}


@pytest.mark.parametrize(
    "content",
    [
        _UnknownWithText(),
        _UnknownWithContent(),
        _UnknownWithStr(),
        _UnknownNested(),
        [_UnknownWithText(), _UnknownWithContent(), _UnknownWithStr()],
        {"outer": _UnknownNested()},
    ],
)
def test_unknown_objects_return_empty_text_and_never_leak(content):
    out = visible_text_from_content(content)
    assert out == ""
    assert "LEAK" not in out


# ── hostile Mappings: accessor failures must yield "", never propagate ──────

class _HostileGetMapping(dict):
    """Mapping whose .get() always raises."""

    def get(self, key, default=None):
        raise RuntimeError("hostile get")


class _HostileKeysMapping(dict):
    """Mapping whose keys()/iteration always raises."""

    def keys(self):
        raise RuntimeError("hostile keys")

    def __iter__(self):
        raise RuntimeError("hostile iter")


@pytest.mark.parametrize(
    "mapping",
    [
        _HostileGetMapping({"text": "SECRET"}),
        _HostileGetMapping({"content": "SECRET"}),
        _HostileKeysMapping({"text": "SECRET"}),
        _HostileKeysMapping({"content": "SECRET"}),
    ],
)
def test_hostile_mapping_top_level_returns_empty_without_propagating(mapping):
    out = visible_text_from_content(mapping)
    assert out == ""
    assert "SECRET" not in out


@pytest.mark.parametrize(
    "mapping",
    [
        _HostileGetMapping({"text": "SECRET"}),
        _HostileKeysMapping({"text": "SECRET"}),
    ],
)
def test_hostile_mapping_inside_list_returns_empty_without_propagating(mapping):
    out = visible_text_from_content([mapping])
    assert out == ""
    assert "SECRET" not in out
    # A hostile item must not poison its neighbours either.
    out = visible_text_from_content([{"type": "text", "text": "ok"}, mapping])
    assert out == "ok"
