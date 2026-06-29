"""Regression: output-only SDK fields must not leak into Anthropic request input.

Reproduces HTTP 400 `messages.N.content.M.text.parsed_output: Extra inputs are
not permitted`. Anthropic SDK response blocks carry output-only attributes
(text blocks: `parsed_output`, `citations=None`; tool_use blocks: `caller`)
that the Messages *input* schema forbids. normalize_response captured blocks
verbatim via _to_plain_data and replayed them as input → 400.

Fix: whitelist input-permitted fields per block type at three points —
normalize_response capture, _sanitize_replay_block (ordered-blocks replay), and
_convert_content_part_to_anthropic (content-list replay).
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

import pytest
from agent.providers.anthropic_adapter import (
    _sanitize_replay_block,
    _convert_content_part_to_anthropic,
    _convert_assistant_message,
)

FORBIDDEN = {"parsed_output", "caller"}


def _assert_clean(block):
    """No forbidden output-only key, and no null citations, anywhere."""
    assert isinstance(block, dict)
    for k in FORBIDDEN:
        assert k not in block, f"forbidden field {k!r} survived: {block}"
    if "citations" in block:
        assert isinstance(block["citations"], list) and block["citations"], \
            "citations must be a non-empty list if present (None/[] is input-invalid)"


class TestSanitizeReplayBlock:
    def test_text_block_strips_parsed_output_and_null_citations(self):
        poisoned = {"type": "text", "text": "hi", "parsed_output": None, "citations": None}
        out = _sanitize_replay_block(poisoned)
        _assert_clean(out)
        assert out == {"type": "text", "text": "hi"}

    def test_tool_use_strips_caller(self):
        poisoned = {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                    "input": {"path": "a"}, "caller": {"type": "agent"}}
        out = _sanitize_replay_block(poisoned)
        _assert_clean(out)
        assert out["name"] == "read_file" and out["input"] == {"path": "a"}

    def test_thinking_preserves_signature(self):
        b = {"type": "thinking", "thinking": "x", "signature": "sig-AAA"}
        out = _sanitize_replay_block(b)
        assert out == {"type": "thinking", "thinking": "x", "signature": "sig-AAA"}

    def test_text_keeps_real_citations(self):
        real = [{"type": "char_location", "cited_text": "q"}]
        out = _sanitize_replay_block({"type": "text", "text": "t", "citations": real})
        assert out["citations"] == real

    def test_unknown_type_dropped(self):
        assert _sanitize_replay_block({"type": "server_tool_use", "foo": 1}) is None


class TestContentPartConversion:
    def test_stored_text_block_with_parsed_output_cleaned(self):
        # The exact content.N.text.parsed_output failure shape.
        part = {"type": "text", "text": "hello", "parsed_output": None, "citations": None}
        out = _convert_content_part_to_anthropic(part)
        _assert_clean(out)


class TestAssistantReplay:
    def test_interleaved_blocks_replayed_clean_and_ordered(self):
        m = {
            "role": "assistant",
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": "plan", "signature": "s1"},
                {"type": "text", "text": "doing it", "parsed_output": None, "citations": None},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                 "input": {"path": "a"}, "caller": {"type": "agent"}},
            ],
        }
        out = _convert_assistant_message(m)
        blocks = out["content"]
        # order preserved
        assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use"]
        # every block clean
        for b in blocks:
            _assert_clean(b)
        # signature + tool fields intact
        assert blocks[0]["signature"] == "s1"
        assert blocks[2]["name"] == "read_file"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
