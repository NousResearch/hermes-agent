"""Request-time eviction of historical multimodal image payloads."""

from __future__ import annotations

import copy

from agent.context_compressor import (
    _content_has_images,
    _strip_old_image_parts,
)


def _image(label: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + label * 100},
    }


def _image_message(role: str, label: str, *, sidecar: bool = False) -> dict:
    message = {
        "role": role,
        "content": [
            {"type": "text", "text": f"caption {label}"},
            _image(label),
        ],
    }
    if role == "tool":
        message["tool_call_id"] = f"call_{label}"
    if sidecar:
        message["api_content"] = "stale bytes"
    return message


class TestStripOldImageParts:
    def test_keeps_three_newest_image_bearing_messages_across_roles(self):
        messages = [
            _image_message("user", "old-user"),
            {"role": "assistant", "content": "one"},
            _image_message("tool", "old-tool"),
            {"role": "assistant", "content": "two"},
            _image_message("user", "new-user"),
            _image_message("tool", "new-tool-a"),
            _image_message("tool", "new-tool-b"),
        ]

        replaced = _strip_old_image_parts(messages, keep_recent=3)

        assert replaced == 2
        assert not _content_has_images(messages[0]["content"])
        assert not _content_has_images(messages[2]["content"])
        for index in (4, 5, 6):
            assert _content_has_images(messages[index]["content"])

    def test_keep_window_counts_messages_not_individual_image_parts(self):
        old = _image_message("tool", "old")
        newest = {
            "role": "tool",
            "tool_call_id": "call_new",
            "content": [_image("new-a"), _image("new-b")],
        }
        messages = [old, newest]

        replaced = _strip_old_image_parts(messages, keep_recent=1)

        assert replaced == 1
        assert not _content_has_images(old["content"])
        assert sum(1 for part in newest["content"] if part["type"] == "image_url") == 2

    def test_image_only_tool_result_keeps_tool_pair_slot(self):
        message = {
            "role": "tool",
            "tool_call_id": "call_old",
            "content": [_image("only")],
        }

        replaced = _strip_old_image_parts([message], keep_recent=0)

        assert replaced == 1
        assert message["tool_call_id"] == "call_old"
        assert message["content"] == [
            {
                "type": "text",
                "text": "[Attached image — stripped from older request context]",
            }
        ]

    def test_rewrite_drops_stale_api_content_sidecar(self):
        old = _image_message("user", "old", sidecar=True)
        messages = [old, _image_message("tool", "new")]

        _strip_old_image_parts(messages, keep_recent=1)

        assert "api_content" not in old

    def test_replaces_all_supported_image_part_shapes(self):
        message = {
            "role": "tool",
            "tool_call_id": "call_formats",
            "content": [
                _image("chat"),
                {"type": "input_image", "image_url": "data:image/png;base64,responses"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "anthropic",
                    },
                },
            ],
        }

        replaced = _strip_old_image_parts([message], keep_recent=0)

        assert replaced == 3
        assert not _content_has_images(message["content"])
        assert all(part["type"] == "text" for part in message["content"])

    def test_no_images_or_images_within_window_are_noops(self):
        text_only = [{"role": "user", "content": "hello"}]
        within_window = [_image_message("user", "one"), _image_message("tool", "two")]

        assert _strip_old_image_parts(text_only, keep_recent=3) == 0
        assert _strip_old_image_parts(within_window, keep_recent=3) == 0
        assert all(_content_has_images(message["content"]) for message in within_window)

    def test_projection_is_deterministic_and_each_image_ages_out_once(self):
        """A rolled-out image changes once, then its placeholder stays stable.

        Bounding a historical payload necessarily changes the request prefix
        when an image first leaves the keep window. The important cache
        invariant is that the projection is deterministic: repeated requests
        over the same history are byte-identical, and adding one image changes
        only the one additional message that just aged out.
        """
        canonical = [_image_message("tool", str(index)) for index in range(4)]
        first = copy.deepcopy(canonical)
        second = copy.deepcopy(canonical)

        assert _strip_old_image_parts(first, keep_recent=3) == 1
        assert _strip_old_image_parts(second, keep_recent=3) == 1
        assert first == second
        assert all(_content_has_images(message["content"]) for message in canonical)

        canonical.append(_image_message("tool", "4"))
        rolled = copy.deepcopy(canonical)
        assert _strip_old_image_parts(rolled, keep_recent=3) == 2

        # Message 0 was already a placeholder and remains byte-identical.
        assert rolled[0] == first[0]
        # Only message 1 newly crossed the keep-window boundary.
        assert _content_has_images(first[1]["content"])
        assert not _content_has_images(rolled[1]["content"])
