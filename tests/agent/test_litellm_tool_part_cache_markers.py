"""Regression tests for #89886: part-level cache_control on role:tool messages
must be suppressed on LiteLLM-style envelope routes.

LiteLLM's OpenAI→Anthropic translation copies tool-message content parts
verbatim, so a part-level ``cache_control`` lands at ``tool_result.content[0]``
— a placement the Anthropic Messages schema rejects with a non-retryable
HTTP 400 that kills the whole turn. OpenRouter relocates the marker correctly,
so the part-level form stays on for it.
"""

import copy

from agent.prompt_caching import (
    apply_anthropic_cache_control,
    build_prompt_cache_plan,
    envelope_tool_part_cache_markers_supported,
)


def _tool_history():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "run the tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "tool output"},
    ]


def _tool_messages(msgs):
    return [m for m in msgs if m.get("role") == "tool"]


def _has_part_marker(msg):
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(p, dict) and "cache_control" in p for p in content
        )
    return False


class TestEnvelopeToolPartMarkerSuppression:
    def test_litellm_route_detected_as_unsupported(self):
        assert not envelope_tool_part_cache_markers_supported(
            "litellm", "https://litellm.internal.example/v1"
        )
        assert not envelope_tool_part_cache_markers_supported(
            "custom:litellm", "https://gw.example.com/v1"
        )
        assert not envelope_tool_part_cache_markers_supported(
            "custom", "https://litellm-proxy.example.com/v1"
        )

    def test_openrouter_route_stays_supported(self):
        assert envelope_tool_part_cache_markers_supported(
            "openrouter", "https://openrouter.ai/api/v1"
        )
        # Lookalike names must not be swept in.
        assert envelope_tool_part_cache_markers_supported(
            "custom:notlitellm", "https://notlitellm.example.com/v1"
        )

    def test_tool_message_not_marked_when_part_markers_off(self):
        decorated = apply_anthropic_cache_control(
            copy.deepcopy(_tool_history()),
            native_anthropic=False,
            tool_part_markers=False,
        )
        for tool_msg in _tool_messages(decorated):
            assert "cache_control" not in tool_msg
            assert not _has_part_marker(tool_msg)
            # Content must stay a plain string — no decoration-produced
            # part list that LiteLLM would forward into tool_result.content.
            assert isinstance(tool_msg.get("content"), str)

    def test_breakpoint_reallocates_to_non_tool_message(self):
        decorated = apply_anthropic_cache_control(
            copy.deepcopy(_tool_history()),
            native_anthropic=False,
            tool_part_markers=False,
        )
        non_tool_marked = [
            m
            for m in decorated
            if m.get("role") != "tool"
            and ("cache_control" in m or _has_part_marker(m))
        ]
        assert non_tool_marked, "budget must reallocate, not vanish"

    def test_openrouter_default_keeps_tool_part_marker(self):
        decorated = apply_anthropic_cache_control(
            copy.deepcopy(_tool_history()), native_anthropic=False
        )
        assert any(_has_part_marker(m) for m in _tool_messages(decorated))

    def test_build_prompt_cache_plan_threads_flag(self):
        plan = build_prompt_cache_plan(
            _tool_history(),
            [],
            native_anthropic=False,
            tool_part_markers=False,
        )
        for tool_msg in _tool_messages(plan.messages):
            assert "cache_control" not in tool_msg
            assert not _has_part_marker(tool_msg)

    def test_native_anthropic_layout_unaffected(self):
        # Native layout relies on the adapter to relocate a top-level marker
        # onto the tool_result block; the flag must not interfere.
        decorated = apply_anthropic_cache_control(
            copy.deepcopy(_tool_history()),
            native_anthropic=True,
            tool_part_markers=False,
        )
        assert any(
            "cache_control" in m for m in _tool_messages(decorated)
        )
