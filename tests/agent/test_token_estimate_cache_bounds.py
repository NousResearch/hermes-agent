"""Token-estimate memo must not retain stripped or unbounded payload data."""

from __future__ import annotations

import pytest

from agent import model_metadata as metadata


@pytest.fixture(autouse=True)
def _clear_message_token_cache():
    metadata._MSG_TOKENS_CACHE.clear()
    yield
    metadata._MSG_TOKENS_CACHE.clear()


def _image_message(payload: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "inspect this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            },
        ],
    }


def test_image_payload_is_never_pinned_by_token_cache():
    payload = "unique-base64-payload-" + ("A" * 2_000_000)
    message = _image_message(payload)

    actual = metadata.estimate_messages_tokens_rough([message])
    expected = (
        metadata._estimate_message_tokens_without_images(message)
        + metadata._count_image_tokens(message, 1500)
    )

    assert actual == expected
    assert len(metadata._MSG_TOKENS_CACHE) == 1
    pins = next(iter(metadata._MSG_TOKENS_CACHE.values()))[0]
    assert all(payload not in pin for pin in pins)


def test_equivalent_image_shadows_share_cache_entry():
    first = _image_message("A" * 1000)
    second = _image_message("B" * 1000)

    assert metadata.estimate_messages_tokens_rough([first]) == (
        metadata.estimate_messages_tokens_rough([second])
    )
    assert len(metadata._MSG_TOKENS_CACHE) == 1


def test_oversized_text_entry_bypasses_cache(monkeypatch):
    monkeypatch.setattr(metadata, "_MSG_TOKENS_CACHE_ENTRY_MAX_BYTES", 1024)
    message = {"role": "user", "content": "x" * 4096}

    actual = metadata.estimate_messages_tokens_rough([message])

    assert actual == metadata._estimate_message_tokens_without_images(message)
    assert metadata._MSG_TOKENS_CACHE == {}


def test_cache_evicts_to_aggregate_retained_byte_budget(monkeypatch):
    monkeypatch.setattr(metadata, "_MSG_TOKENS_CACHE_ENTRY_MAX_BYTES", 4096)
    monkeypatch.setattr(metadata, "_MSG_TOKENS_CACHE_MAX_BYTES", 1500)

    for index in range(6):
        content = f"message-{index}-" + (chr(65 + index) * 500)
        metadata.estimate_messages_tokens_rough(
            [{"role": "user", "content": content}]
        )

    retained = sum(entry[2] for entry in metadata._MSG_TOKENS_CACHE.values())
    assert retained <= metadata._MSG_TOKENS_CACHE_MAX_BYTES
    assert len(metadata._MSG_TOKENS_CACHE) < 6
