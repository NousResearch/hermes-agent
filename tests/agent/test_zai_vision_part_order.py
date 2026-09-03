"""Regression coverage for Z.ai vision content-part ordering."""

import pytest

from agent.auxiliary_client import _build_call_kwargs


def _text_then_image_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read every visible value."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                },
            ],
        }
    ]


@pytest.mark.parametrize("provider", ["zai", "custom"])
def test_zai_vision_sends_image_before_text_without_mutating_history(provider):
    """Z.ai rejects the generic text-first vision payload with error 1210."""
    messages = _text_then_image_messages()

    kwargs = _build_call_kwargs(
        provider,
        "glm-4.6v-flash",
        messages,
        base_url="https://api.z.ai/api/paas/v4",
        task="vision",
    )

    assert [part["type"] for part in kwargs["messages"][0]["content"]] == [
        "image_url",
        "text",
    ]
    assert [part["type"] for part in messages[0]["content"]] == [
        "text",
        "image_url",
    ]


def test_non_zai_vision_preserves_caller_part_order():
    messages = _text_then_image_messages()

    kwargs = _build_call_kwargs(
        "openrouter",
        "google/gemini-2.5-flash",
        messages,
        base_url="https://openrouter.ai/api/v1",
        task="vision",
    )

    assert kwargs["messages"] is messages
    assert [part["type"] for part in kwargs["messages"][0]["content"]] == [
        "text",
        "image_url",
    ]


def test_zai_vision_moves_only_first_image_and_preserves_remaining_part_order():
    """Starting with an image must not detach later text from later images."""
    content = [
        {"type": "text", "text": "Describe image A."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}},
        {"type": "text", "text": "Compare image B with image A."},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,B"}},
    ]
    messages = [{"role": "user", "content": content}]

    kwargs = _build_call_kwargs(
        "custom",
        "glm-4.6v-flash",
        messages,
        base_url="https://api.z.ai/api/paas/v4",
        task="vision",
    )

    assert kwargs["messages"][0]["content"] == [
        content[1],
        content[0],
        content[2],
        content[3],
        content[4],
    ]
    assert messages[0]["content"] == content
