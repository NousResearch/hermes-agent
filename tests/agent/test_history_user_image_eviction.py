"""Behavior tests for request-time historical user-image eviction."""

from __future__ import annotations

from agent.context_compressor import (
    HISTORICAL_USER_IMAGE_PLACEHOLDER,
    _content_has_images,
    _evict_historical_user_images,
    _is_image_part,
)


def _image(label: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{label}"},
    }


def _image_labels(message: dict) -> list[str]:
    labels = []
    for part in message.get("content", []):
        if _is_image_part(part):
            labels.append(part["image_url"]["url"].rsplit(",", 1)[-1])
    return labels


def test_non_positive_limit_preserves_current_behavior_by_identity():
    messages = [{"role": "user", "content": [_image("old")]}]

    assert _evict_historical_user_images(messages, max_keep=0) is messages
    assert _evict_historical_user_images(messages, max_keep=-1) is messages


def test_current_user_image_batch_is_never_capped_before_first_send():
    old = {"role": "user", "content": [_image("old-a"), _image("old-b")]}
    assistant = {"role": "assistant", "content": "send another batch"}
    current = {"role": "user", "content": [_image("new-a"), _image("new-b")]}
    messages = [old, assistant, current]

    output = _evict_historical_user_images(messages, max_keep=1)

    assert _image_labels(output[0]) == ["old-b"]
    assert _image_labels(output[2]) == ["new-a", "new-b"]
    assert output[2] is current
    assert _image_labels(old) == ["old-a", "old-b"]


def test_text_followup_makes_previous_image_batch_historical():
    image_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": "compare these"},
            _image("a"),
            _image("b"),
            _image("c"),
        ],
    }
    messages = [
        image_turn,
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
    ]

    output = _evict_historical_user_images(messages, max_keep=2)

    assert _image_labels(output[0]) == ["b", "c"]
    placeholders = [
        part
        for part in output[0]["content"]
        if part == {"type": "text", "text": HISTORICAL_USER_IMAGE_PLACEHOLDER}
    ]
    assert len(placeholders) == 1
    assert _image_labels(image_turn) == ["a", "b", "c"]


def test_limit_counts_individual_images_across_historical_user_turns():
    messages = [
        {"role": "user", "content": [_image("oldest")]},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": [_image("middle-a"), _image("middle-b")]},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "latest text"},
    ]

    output = _evict_historical_user_images(messages, max_keep=2)

    assert not _content_has_images(output[0]["content"])
    assert _image_labels(output[2]) == ["middle-a", "middle-b"]


def test_responses_image_uses_input_text_placeholder():
    responses_image = {
        "type": "input_image",
        "image_url": "data:image/png;base64,b2xk",
    }
    messages = [
        {"role": "user", "content": [responses_image, _image("newer")]},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
    ]

    output = _evict_historical_user_images(messages, max_keep=1)

    assert output[0]["content"][0] == {
        "type": "input_text",
        "text": HISTORICAL_USER_IMAGE_PLACEHOLDER,
    }


def test_non_user_images_and_untouched_messages_pass_through():
    tool = {"role": "tool", "content": [_image("tool-shot")]}
    assistant = {"role": "assistant", "content": "ok"}
    messages = [
        {"role": "user", "content": [_image("old")]},
        tool,
        assistant,
        {"role": "user", "content": "continue"},
    ]

    output = _evict_historical_user_images(messages, max_keep=1)

    assert output is messages
    assert output[1] is tool
    assert _image_labels(output[1]) == ["tool-shot"]
    assert output[2] is assistant
