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
