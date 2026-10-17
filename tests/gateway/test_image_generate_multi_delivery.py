"""Every image an ``image_generate`` call returns reaches the messaging gateway, not only the first.

A multi-image backend (OpenRouter ``n``) puts the first image in ``image`` and the rest in
``additional_images``; the auto-append collector used to stop at the primary field.
"""

import json

from gateway.run import _collect_auto_append_media_tags, _collect_history_media_paths


def _turn(payload: dict) -> list[dict]:
    return [
        {"role": "user", "content": "four variations please"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "image_generate"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(payload)},
        {"role": "assistant", "content": "Here they are."},
    ]


_PAYLOAD = {
    "success": True,
    "image": "/tmp/gen/a.png",
    "additional_images": ["/tmp/gen/b.png", "/tmp/gen/c.webp"],
}


def test_auto_append_delivers_every_generated_image():
    tags, _voice = _collect_auto_append_media_tags(_turn(_PAYLOAD), history_offset=0)

    assert tags == ["MEDIA:/tmp/gen/a.png", "MEDIA:/tmp/gen/b.png", "MEDIA:/tmp/gen/c.webp"]


def test_already_delivered_extras_are_not_resent():
    tags, _voice = _collect_auto_append_media_tags(
        _turn(_PAYLOAD), history_offset=0, history_media_paths={"/tmp/gen/b.png"})

    assert tags == ["MEDIA:/tmp/gen/a.png", "MEDIA:/tmp/gen/c.webp"]


def test_history_dedup_set_covers_every_generated_image():
    paths = _collect_history_media_paths(_turn(_PAYLOAD))

    assert {"/tmp/gen/a.png", "/tmp/gen/b.png", "/tmp/gen/c.webp"} <= paths


def test_failed_generation_delivers_nothing():
    tags, _voice = _collect_auto_append_media_tags(
        _turn({**_PAYLOAD, "success": False}), history_offset=0)

    assert tags == []
