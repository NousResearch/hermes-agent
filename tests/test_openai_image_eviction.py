"""Tests for tool-result image eviction on the OpenAI-compatible path.

Background
----------
Tool-result images (browser_vision / vision_analyze screenshots)
accumulate in conversation history on the chat_completions wire format
and are re-sent every turn, causing token bloat and local-model OOM.

_evict_old_screenshots_openai (agent/context_compressor.py) keeps the
most recent N tool-result images and replaces older ones with a text
placeholder.  It only touches the per-call api_messages copy — stored
history is never mutated.

See: GitHub issue #63849
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.context_compressor import (
    _evict_old_screenshots_openai,
    _is_image_part,
    _strip_historical_media,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_part(url="data:image/png;base64,AAAA"):
    """OpenAI chat.completions image_url part."""
    return {"type": "image_url", "image_url": {"url": url}}


def _text_part(text="hello"):
    return {"type": "text", "text": text}


def _tool_msg(content, tool_call_id="call_1"):
    """Tool-role message with multipart content."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _user_msg(content):
    return {"role": "user", "content": content}


def _assistant_msg(content="ok"):
    return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# _evict_old_screenshots_openai — unit tests
# ---------------------------------------------------------------------------

class TestEvictOldScreenshotsOpenAI:
    """Pure-function tests; no I/O required."""

    def test_keeps_most_recent_n_images(self):
        """With default max_keep=3, only the last 3 images survive."""
        msgs = [
            _tool_msg([_img_part("img_1")], "c1"),
            _tool_msg([_img_part("img_2")], "c2"),
            _tool_msg([_img_part("img_3")], "c3"),
            _tool_msg([_img_part("img_4")], "c4"),
            _tool_msg([_img_part("img_5")], "c5"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=3)

        # First two images evicted
        assert msgs[0]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        assert msgs[1]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        # Last three kept
        assert msgs[2]["content"][0]["type"] == "image_url"
        assert msgs[3]["content"][0]["type"] == "image_url"
        assert msgs[4]["content"][0]["type"] == "image_url"

    def test_preserves_text_parts_in_tool_messages(self):
        """Non-image parts in tool messages are never touched."""
        msgs = [
            _tool_msg([_img_part("img_1"), _text_part("result")], "c1"),
            _tool_msg([_img_part("img_2"), _text_part("result")], "c2"),
            _tool_msg([_img_part("img_3"), _text_part("result")], "c3"),
            _tool_msg([_img_part("img_4"), _text_part("result")], "c4"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=2)

        # First tool msg: image evicted, text preserved
        assert msgs[0]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        assert msgs[0]["content"][1] == {"type": "text", "text": "result"}
        # Second tool msg: image evicted, text preserved
        assert msgs[1]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        assert msgs[1]["content"][1] == {"type": "text", "text": "result"}
        # Last two: both kept
        assert msgs[2]["content"][0]["type"] == "image_url"
        assert msgs[3]["content"][0]["type"] == "image_url"

    def test_ignores_non_tool_messages(self):
        """Only tool-role messages are walked; user/assistant messages are skipped."""
        msgs = [
            _user_msg([_img_part("user_img")]),
            _assistant_msg("saw screenshot"),
            _tool_msg([_img_part("tool_img_1")], "c1"),
            _tool_msg([_img_part("tool_img_2")], "c2"),
            _tool_msg([_img_part("tool_img_3")], "c3"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=2)

        # User message untouched
        assert msgs[0]["content"][0]["type"] == "image_url"
        # First tool evicted
        assert msgs[2]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        # Last two tools kept
        assert msgs[3]["content"][0]["type"] == "image_url"
        assert msgs[4]["content"][0]["type"] == "image_url"

    def test_no_images_noop(self):
        """Messages with no images are returned unchanged."""
        msgs = [
            _user_msg("hello"),
            _assistant_msg("hi"),
            _tool_msg([_text_part("result")], "c1"),
        ]
        original = [m.copy() for m in msgs]
        _evict_old_screenshots_openai(msgs, max_keep=3)
        assert msgs == original

    def test_empty_list_noop(self):
        """Empty message list is returned unchanged."""
        msgs = []
        _evict_old_screenshots_openai(msgs, max_keep=3)
        assert msgs == []

    def test_string_content_skipped(self):
        """Tool messages with string content (not list) are skipped."""
        msgs = [
            _tool_msg("plain text result", "c1"),
            _tool_msg([_img_part("img_1")], "c2"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=1)
        # String content untouched
        assert msgs[0]["content"] == "plain text result"
        # Only one image, within limit
        assert msgs[1]["content"][0]["type"] == "image_url"

    def test_max_keep_zero_disables_eviction(self):
        """Setting max_keep=0 means nothing is evicted (all images kept)."""
        msgs = [
            _tool_msg([_img_part("img_1")], "c1"),
            _tool_msg([_img_part("img_2")], "c2"),
            _tool_msg([_img_part("img_3")], "c3"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=0)
        # All images kept (0 means no limit applied — the > max_keep gate never fires)
        assert msgs[0]["content"][0]["type"] == "image_url"
        assert msgs[1]["content"][0]["type"] == "image_url"
        assert msgs[2]["content"][0]["type"] == "image_url"

    def test_input_image_type_recognized(self):
        """input_image type (OpenAI Responses API) is recognized and evicted."""
        msgs = [
            _tool_msg([{"type": "input_image", "image_url": "img_1"}], "c1"),
            _tool_msg([{"type": "input_image", "image_url": "img_2"}], "c2"),
            _tool_msg([{"type": "input_image", "image_url": "img_3"}], "c3"),
            _tool_msg([{"type": "input_image", "image_url": "img_4"}], "c4"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=2)
        assert msgs[0]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        assert msgs[1]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        assert msgs[2]["content"][0]["type"] == "input_image"
        assert msgs[3]["content"][0]["type"] == "input_image"

    def test_mixed_image_types(self):
        """Mix of image_url and input_image parts are all counted together."""
        msgs = [
            _tool_msg([_img_part("img_1")], "c1"),
            _tool_msg([{"type": "input_image", "image_url": "img_2"}], "c2"),
            _tool_msg([_img_part("img_3")], "c3"),
            _tool_msg([{"type": "input_image", "image_url": "img_4"}], "c4"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=2)
        assert msgs[0]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        assert msgs[1]["content"][0] == {"type": "text", "text": "[screenshot removed to save context]"}
        assert msgs[2]["content"][0]["type"] == "image_url"
        assert msgs[3]["content"][0]["type"] == "input_image"

    def test_exact_max_keep_boundary(self):
        """Exactly max_keep images means none are evicted."""
        msgs = [
            _tool_msg([_img_part("img_1")], "c1"),
            _tool_msg([_img_part("img_2")], "c2"),
            _tool_msg([_img_part("img_3")], "c3"),
        ]
        _evict_old_screenshots_openai(msgs, max_keep=3)
        assert msgs[0]["content"][0]["type"] == "image_url"
        assert msgs[1]["content"][0]["type"] == "image_url"
        assert msgs[2]["content"][0]["type"] == "image_url"


# ---------------------------------------------------------------------------
# _strip_historical_media — anchor expansion tests
# ---------------------------------------------------------------------------

class TestStripHistoricalMediaAnchor:
    """Verify that _strip_historical_media now anchors on tool-role images."""

    def test_tool_role_image_used_as_anchor(self):
        """A tool-role image anchors the strip — images before it are replaced."""
        msgs = [
            _user_msg([_img_part("old_user_img")]),
            _assistant_msg("let me check"),
            _tool_msg([_img_part("tool_screenshot")], "c1"),
            _user_msg("what do you see?"),
        ]
        result = _strip_historical_media(msgs)

        # The tool message at index 2 is the anchor.
        # Messages before it with images get stripped.
        # The user message at index 0 has an image before the anchor -> stripped.
        assert result[0]["content"][0]["type"] == "text"
        assert "stripped after compression" in result[0]["content"][0]["text"]
        # Tool message image kept (it IS the anchor)
        assert result[2]["content"][0]["type"] == "image_url"
        # User message after anchor untouched
        assert result[3]["content"] == "what do you see?"

    def test_only_tool_images_no_user_images(self):
        """Typical browser_vision flow: user text -> tool screenshots.
        The last tool screenshot anchors; earlier ones are stripped."""
        msgs = [
            _user_msg("check this page"),
            _assistant_msg("looking"),
            _tool_msg([_img_part("screenshot_1")], "c1"),
            _tool_msg([_img_part("screenshot_2")], "c2"),
            _assistant_msg("I see..."),
        ]
        result = _strip_historical_media(msgs)

        # Anchor is the last tool image (index 3).
        # Index 2 is before anchor -> stripped.
        assert "stripped" in result[2]["content"][0]["text"]
        # Index 3 IS the anchor -> kept
        assert result[3]["content"][0]["type"] == "image_url"

    def test_user_image_before_tool_anchor(self):
        """User image before tool anchor gets stripped."""
        msgs = [
            _user_msg([_img_part("user_img")]),
            _assistant_msg("let me look"),
            _tool_msg([_img_part("tool_screenshot")], "c1"),
        ]
        result = _strip_historical_media(msgs)

        # Anchor is tool message at index 2.
        # User image at index 0 is before anchor -> stripped.
        assert result[0]["content"][0]["type"] == "text"
        assert "stripped after compression" in result[0]["content"][0]["text"]
