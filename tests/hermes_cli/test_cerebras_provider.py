"""Cerebras provider profile contracts."""

from providers import get_provider_profile


def test_cerebras_strips_internal_reasoning_fields_and_preserves_reasoning():
    messages = [
        {
            "role": "assistant",
            "content": "done",
            "reasoning_content": "plan",
            "reasoning_details": [{"type": "summary_text", "text": "plan"}],
        }
    ]

    profile = get_provider_profile("cerebras")
    assert profile is not None
    prepared = profile.prepare_messages(messages)

    assert prepared == [{"role": "assistant", "content": "done", "reasoning": "plan"}]
    assert messages[0]["reasoning_content"] == "plan"


def test_cerebras_reasoning_effort_maps_to_supported_wire_values():
    profile = get_provider_profile("cerebras")
    assert profile is not None

    assert profile.build_api_kwargs_extras(
        reasoning_config={"effort": "xhigh"}, model="gpt-oss-120b"
    )[1] == {"reasoning_effort": "high"}
    assert profile.build_api_kwargs_extras(
        reasoning_config={"effort": "none"}, model="gpt-oss-120b"
    ) == ({}, {})
