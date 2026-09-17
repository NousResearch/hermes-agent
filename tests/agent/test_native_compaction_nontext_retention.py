"""Regression coverage for image-only user content across native compaction."""

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.native_compaction import _extract_item_text, prune_pre_checkpoint_items


_IMAGE_URL = "data:image/png;base64,AAAA"
_CHAT_IMAGE_PART = {"type": "image_url", "image_url": {"url": _IMAGE_URL}}
_RESPONSES_IMAGE_PART = {"type": "input_image", "image_url": _IMAGE_URL}
_CHECKPOINT = {"type": "compaction", "encrypted_content": "blob_cp"}


def test_extract_item_text_remains_text_only_for_image_content():
    image_only = {"content": [_RESPONSES_IMAGE_PART]}

    assert _extract_item_text(image_only) is None
    assert _extract_item_text({"content": []}) is None


def test_image_only_user_message_survives_pre_checkpoint_pruning_verbatim():
    image_only = {"role": "user", "content": [_RESPONSES_IMAGE_PART]}
    items = [
        image_only,
        _CHECKPOINT,
        {"role": "user", "content": "after checkpoint"},
    ]

    pruned = prune_pre_checkpoint_items(items, retained_user_token_budget=1)

    assert pruned == [
        _CHECKPOINT,
        image_only,
        {"role": "user", "content": "after checkpoint"},
    ]
    assert pruned[1] is image_only


def test_image_only_user_message_still_obeys_retention_budget():
    image_only = {"role": "user", "content": [_RESPONSES_IMAGE_PART]}

    assert prune_pre_checkpoint_items(
        [image_only, _CHECKPOINT],
        retained_user_token_budget=0,
    ) == [_CHECKPOINT]


def test_malformed_or_unknown_multipart_is_not_retained():
    invalid_contents = [
        [{}],
        [{"type": "input_image", "image_url": ""}],
        [{"type": "unknown", "value": "not an attachment"}],
    ]

    for content in invalid_contents:
        malformed = {"role": "user", "content": content}
        assert prune_pre_checkpoint_items(
            [malformed, _CHECKPOINT],
            retained_user_token_budget=1,
        ) == [_CHECKPOINT]


def test_adapter_preserves_image_only_user_message_across_checkpoint():
    messages = [
        {"role": "user", "content": [_CHAT_IMAGE_PART]},
        {
            "role": "assistant",
            "content": "checkpoint turn",
            "codex_reasoning_items": [_CHECKPOINT],
        },
        {"role": "user", "content": "after checkpoint"},
    ]

    converted = _chat_messages_to_responses_input(
        messages,
        native_compaction_eligible=True,
    )

    assert converted == [
        _CHECKPOINT,
        {"role": "user", "content": [_RESPONSES_IMAGE_PART]},
        {"role": "assistant", "content": "checkpoint turn"},
        {"role": "user", "content": "after checkpoint"},
    ]


def test_image_only_user_message_is_retained_with_interleaved_assistant():
    image_user = {
        "role": "user",
        "content": [_RESPONSES_IMAGE_PART],
    }
    items = [
        image_user,
        {"role": "assistant", "content": "I see the screenshot."},
        _CHECKPOINT,
        {"role": "user", "content": "what was in that screenshot?"},
    ]
    out = prune_pre_checkpoint_items(items)
    assert out[0] == _CHECKPOINT
    assert out[1] == image_user
    assert out[2] == {"role": "user", "content": "what was in that screenshot?"}
