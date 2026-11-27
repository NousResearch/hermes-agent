"""Usage-anchor fingerprints survive the durable projection of multimodal messages (#159070, site 1).

``capture_usage_anchor`` fingerprints the live message list, where a user turn with an image is a
list of parts. The session row stores the text projection instead (``session_rules.turns.durable_content``:
text parts kept, each image folded to ``[screenshot]`` / ``[screenshot detail=...]``), so after a
reload the last priced row no longer hashed the same, the anchor failed closed, and the context
meter fell back to the rough ``bytes/4`` estimate.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor, message_fingerprint
from session_rules.turns import durable_content

_IMAGE_PARTS = [
    [{"type": "text", "text": "what is on screen?"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}],
    [{"type": "text", "text": "compare"},
     {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
     {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}, "detail": "high"}],
    [{"type": "input_image", "image_url": "data:image/png;base64,CCCC"}],
]


def _reloaded(messages):
    """The transcript as a resumed session reads it back: each row's content through the durable projection."""
    return [{**m, "content": durable_content(m.get("content"))} for m in messages]


@pytest.mark.parametrize("parts", _IMAGE_PARTS)
def test_anchor_captured_on_a_live_multimodal_turn_still_matches_after_reload(parts):
    live = [{"role": "system", "content": "sys"}, {"role": "user", "content": parts}]
    anchor = capture_usage_anchor(850_000, 500, live)

    assert message_fingerprint(live[-1]) == message_fingerprint(_reloaded(live)[-1])
    assert anchored_context_tokens(_reloaded(live), anchor) == 850_500


def test_multimodal_tool_result_envelope_matches_its_reloaded_summary():
    live = [{"role": "user", "content": "take a screenshot"},
            {"role": "tool", "tool_call_id": "c1", "content": {
                "_multimodal": True, "text_summary": "Screenshot captured",
                "content": [{"type": "text", "text": "Screenshot captured"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,DDDD"}}]}}]
    anchor = capture_usage_anchor(120_000, 40, live)
    assert anchored_context_tokens(_reloaded(live), anchor) == 120_040


def test_plain_string_fingerprints_are_unchanged_so_persisted_anchors_stay_valid():
    msg = {"role": "user", "content": "hello", "tool_call_id": None}
    legacy = hashlib.sha256(json.dumps({"role": "user", "content": "hello"}, sort_keys=True, default=str,
                                       ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    assert message_fingerprint(msg) == legacy


def test_a_different_image_detail_is_still_a_different_message():
    low = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x", "detail": "low"}}]}
    high = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x", "detail": "high"}}]}
    assert message_fingerprint(low) != message_fingerprint(high)
