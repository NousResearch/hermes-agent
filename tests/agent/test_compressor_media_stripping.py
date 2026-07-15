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


class TestFallbackSummaryMediaStripping:
    """MEDIA directives must also be stripped from the deterministic fallback
    summary (the path taken when the LLM summarizer is unavailable). That
    summary is *persisted*, so a leaked MEDIA:<path> would re-trigger media
    reinjection on resume — the same bug #44708 fixed for the LLM path (#14665).
    """

    def test_media_stripped_from_fallback_assistant_content(self, compressor):
        turns = [
            {"role": "user", "content": "Please play MEDIA:/tmp/voice.ogg for me."},
            {"role": "assistant", "content": "Sent the audio MEDIA:/tmp/voice.ogg."},
        ]
        result = compressor._build_static_fallback_summary(turns)
        assert "MEDIA:/tmp/voice.ogg" not in result
        assert "[media attachment]" in result

    def test_media_stripped_from_fallback_tool_result(self, compressor):
        turns = [
            {"role": "tool", "tool_call_id": "t1", "content": "Generated MEDIA:/tmp/out.mp3 ok"},
        ]
        result = compressor._build_static_fallback_summary(turns)
        assert "MEDIA:/tmp/out.mp3" not in result

    def test_media_stripped_from_fallback_tool_args(self, compressor):
        # A MEDIA: directive can reach the body through tool-call arguments
        # (embedded by _summarize_tool_result and collected into relevant
        # files), which bypass the per-turn text path — this is covered only by
        # the final body scrub. The tool-call arg is the ONLY MEDIA source here,
        # so "[media attachment]" appearing proves the arg actually reached the
        # body and was scrubbed (positive control for the final guard); without
        # the final scrub the raw directive would survive instead.
        turns = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "t1", "function": {"name": "read_file", "arguments": '{"path": "MEDIA:/tmp/render.png"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        ]
        result = compressor._build_static_fallback_summary(turns)
        assert "MEDIA:/tmp/render.png" not in result
        assert "[media attachment]" in result

    def test_non_media_path_preserved_in_fallback(self, compressor):
        turns = [
            {"role": "assistant", "content": "Edited /tmp/app.py and the tests pass."},
        ]
        result = compressor._build_static_fallback_summary(turns)
        assert "/tmp/app.py" in result
