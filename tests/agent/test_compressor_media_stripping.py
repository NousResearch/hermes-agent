"""Tests for MEDIA directive stripping in context compaction (#14665).

MEDIA directives in assistant messages must not leak into compaction
summaries — if they do, the downstream model re-emits them as active
directives on the next turn.
"""
import pytest
from unittest.mock import patch
from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )


class TestMediaDirectiveStripping:
    """MEDIA directives must be stripped before summarization (#14665)."""

    def test_media_directive_stripped_from_assistant(self, compressor):
        turns = [
            {"role": "assistant", "content": "Here is the audio MEDIA:/tmp/voice.ogg done."},
        ]
        result = compressor._serialize_for_summary(turns)
        assert "MEDIA:/tmp/voice.ogg" not in result
        assert "[media attachment]" in result

    def test_media_directive_stripped_from_tool_result(self, compressor):
        turns = [
            {"role": "tool", "tool_call_id": "t1", "content": "Generated MEDIA:/tmp/out.mp3 successfully"},
        ]
        result = compressor._serialize_for_summary(turns)
        assert "MEDIA:/tmp/out.mp3" not in result
        assert "[media attachment]" in result

    def test_non_media_content_preserved(self, compressor):
        turns = [
            {"role": "assistant", "content": "The file path is /tmp/test.txt and it works."},
        ]
        result = compressor._serialize_for_summary(turns)
        assert "/tmp/test.txt" in result

    def test_multiple_media_directives(self, compressor):
        turns = [
            {"role": "assistant", "content": "MEDIA:/a.ogg and MEDIA:/b.mp3"},
        ]
        result = compressor._serialize_for_summary(turns)
        assert "MEDIA:" not in result
        assert result.count("[media attachment]") == 2

    def test_multimodal_list_content_does_not_crash(self, compressor):
        """content as a list (multimodal) must not crash redact_sensitive_text.

        Regression: msg.get('content') can return a list of parts for
        multimodal messages (images + text).  The old code passed this
        list directly to redact_sensitive_text(str), which raised
        AttributeError on list.replace().
        """
        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            },
        ]
        # Must not raise
        result = compressor._serialize_for_summary(turns)
        assert "What is in this image?" in result
        assert "[image]" in result

    def test_multimodal_list_text_parts_extracted(self, compressor):
        """Text parts from multimodal list content are preserved in output."""
        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first part"},
                    {"type": "text", "text": "second part"},
                ],
            },
        ]
        result = compressor._serialize_for_summary(turns)
        assert "first part" in result
        assert "second part" in result

    def test_multimodal_list_bare_strings_handled(self, compressor):
        """Bare strings inside a content list are joined."""
        turns = [
            {"role": "user", "content": ["hello", "world"]},
        ]
        result = compressor._serialize_for_summary(turns)
        assert "hello" in result
        assert "world" in result
