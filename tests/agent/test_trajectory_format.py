"""Regression tests for ``convert_to_trajectory_format`` multimodal handling.

``_trajectory_normalize_msg`` keeps multimodal message content as a *list* of
parts (image parts replaced by ``[screenshot]`` text parts), but the text-only
ShareGPT trajectory format expects ``content`` to be a string. Before the fix,
any conversation containing an image (vision / computer-use / screenshot tools)
either crashed during trajectory saving or silently wrote a Python list as a
record ``value``. These tests exercise all three str-assuming sites.
"""

from agent.agent_runtime_helpers import convert_to_trajectory_format


class _FakeAgent:
    """Minimal stand-in exposing only what the converter touches."""

    def _format_tools_for_system_message(self) -> str:
        return ""


def _multimodal_user_content():
    return [
        {"type": "text", "text": "what is in this screenshot?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


def test_multimodal_user_turn_value_is_str():
    """A multimodal user turn must serialize to a string ShareGPT value."""
    messages = [
        {"role": "user", "content": "first prompt"},
        {"role": "user", "content": _multimodal_user_content()},
    ]
    trajectory = convert_to_trajectory_format(
        _FakeAgent(), messages, user_query="first prompt", completed=True
    )
    human_turns = [t for t in trajectory if t["from"] == "human"]
    # The second human turn corresponds to the multimodal user message.
    multimodal_turn = human_turns[-1]
    assert isinstance(multimodal_turn["value"], str)
    assert "what is in this screenshot?" in multimodal_turn["value"]
    # The image part was normalized to a [screenshot] placeholder upstream and
    # must survive flattening rather than being dropped or crashing.
    assert "[screenshot]" in multimodal_turn["value"]


def test_multimodal_assistant_turn_no_crash():
    """A regular assistant turn with list content must not raise (TypeError)."""
    messages = [
        {"role": "user", "content": "look at this"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I see a cat"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
            ],
        },
    ]
    trajectory = convert_to_trajectory_format(
        _FakeAgent(), messages, user_query="look at this", completed=True
    )
    gpt_turns = [t for t in trajectory if t["from"] == "gpt"]
    assert gpt_turns, "expected an assistant (gpt) turn"
    assert isinstance(gpt_turns[-1]["value"], str)
    assert "I see a cat" in gpt_turns[-1]["value"]


def test_assistant_tool_calls_list_content_no_crash():
    """Assistant turn with tool_calls AND list content must not raise (AttributeError)."""
    messages = [
        {"role": "user", "content": "click the button"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "clicking now"}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "click", "arguments": '{"x": 1, "y": 2}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\": true}"},
    ]
    trajectory = convert_to_trajectory_format(
        _FakeAgent(), messages, user_query="click the button", completed=True
    )
    gpt_turns = [t for t in trajectory if t["from"] == "gpt"]
    assert gpt_turns, "expected an assistant (gpt) turn"
    value = gpt_turns[-1]["value"]
    assert isinstance(value, str)
    assert "clicking now" in value
    assert "<tool_call>" in value
