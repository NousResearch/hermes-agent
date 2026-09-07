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
        """content as a list (multimodal) must be flattened to clean text.

        Without flattening, str() coercion in redact_sensitive_text dumps
        the raw part-dict repr — including base64 image data — into the
        summarizer input, burning summary budget on noise.
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
        result = compressor._serialize_for_summary(turns)
        assert "What is in this image?" in result
        assert "[image]" in result
        assert "base64" not in result




