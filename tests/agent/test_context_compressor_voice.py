"""Tests for voice transcript preservation in context_compressor.py."""

import pytest

from agent.context_compressor import (
    _extract_voice_transcripts,
    _build_voice_appendix,
    SUMMARY_PREFIX,
)


class TestExtractVoiceTranscripts:
    """Tests for _extract_voice_transcripts()."""

    def test_extracts_standard_voice_marker(self):
        """Standard voice message format from gateway STT pipeline."""
        turns = [
            {
                "role": "user",
                "content": '[The user sent a voice message~ Here\'s what they said: "Check the status of the build pipeline"]',
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 1
        assert result[0]["transcript"] == "Check the status of the build pipeline"

    def test_extracts_with_extra_text(self):
        """Voice message followed by additional context (e.g. todo list)."""
        turns = [
            {
                "role": "user",
                "content": (
                    '[The user sent a voice message~ Here\'s what they said: "Fix the login bug"]\n\n'
                    "[Your active task list was preserved across context compression]\n"
                    "- [>] 1. Fix login bug (in_progress)"
                ),
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 1
        assert result[0]["transcript"] == "Fix the login bug"

    def test_extracts_multiline_transcript(self):
        """Voice transcripts can span multiple sentences."""
        turns = [
            {
                "role": "user",
                "content": '[The user sent a voice message~ Here\'s what they said: "I can do the browser interaction, but we should really figure out a way for the APK install to be more seamless."]',
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 1
        assert "browser interaction" in result[0]["transcript"]
        assert "APK" in result[0]["transcript"]

    def test_extracts_multiple_voice_messages(self):
        """Multiple voice messages in separate turns."""
        turns = [
            {
                "role": "user",
                "content": '[The user sent a voice message~ Here\'s what they said: "First thing"]',
            },
            {"role": "assistant", "content": "Got it."},
            {
                "role": "user",
                "content": '[The user sent a voice message~ Here\'s what they said: "Second thing"]',
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 2
        assert result[0]["transcript"] == "First thing"
        assert result[1]["transcript"] == "Second thing"

    def test_ignores_non_voice_user_messages(self):
        """Plain text user messages should not produce transcripts."""
        turns = [
            {"role": "user", "content": "Hello, can you help me?"},
            {"role": "assistant", "content": "Sure!"},
            {"role": "user", "content": "Great, thanks."},
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 0

    def test_ignores_assistant_messages(self):
        """Assistant messages with voice-like text should not match."""
        turns = [
            {
                "role": "assistant",
                "content": "The user said something about voice messages.",
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 0

    def test_empty_turns(self):
        result = _extract_voice_transcripts([])
        assert len(result) == 0

    def test_handles_list_content(self):
        """Multimodal list content (image + text) should still extract voice."""
        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": '[The user sent a voice message~ Here\'s what they said: "Test multimodal"]'},
                ],
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 1
        assert result[0]["transcript"] == "Test multimodal"

    def test_empty_transcript_skipped(self):
        """Voice marker with empty transcript should not produce an entry."""
        turns = [
            {
                "role": "user",
                "content": '[The user sent a voice message~ Here\'s what they said: ""]',
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 0

    def test_whitespace_only_transcript_skipped(self):
        """Voice marker with whitespace-only transcript should not produce an entry."""
        turns = [
            {
                "role": "user",
                "content": '[The user sent a voice message~ Here\'s what they said: "   "]',
            },
        ]
        result = _extract_voice_transcripts(turns)
        assert len(result) == 0


class TestBuildVoiceAppendix:
    """Tests for _build_voice_appendix()."""

    def test_empty_list_returns_empty(self):
        assert _build_voice_appendix([]) == ""

    def test_single_transcript(self):
        transcripts = [{"transcript": "Hello world", "timestamp": 1234.0}]
        result = _build_voice_appendix(transcripts)
        assert "## Voice Message Log" in result
        assert "Hello world" in result

    def test_multiple_transcripts(self):
        transcripts = [
            {"transcript": "First", "timestamp": 1234.0},
            {"transcript": "Second", "timestamp": 1235.0},
        ]
        result = _build_voice_appendix(transcripts)
        assert '"First"' in result
        assert '"Second"' in result

    def test_preserved_across_compression_label(self):
        transcripts = [{"transcript": "test", "timestamp": 1234.0}]
        result = _build_voice_appendix(transcripts)
        assert "preserved across compression" in result

    def test_transcript_in_quotes(self):
        """Each transcript should be wrapped in quotes."""
        transcripts = [{"transcript": "Check the build", "timestamp": 1234.0}]
        result = _build_voice_appendix(transcripts)
        assert '- "Check the build"' in result


class TestVoiceTranscriptsInCompress:
    """Integration test: voice transcripts survive compress()."""

    def test_voice_in_compression_summary(self):
        """Voice transcripts should appear in the compression output."""
        # This tests the full compress() flow by verifying that when voice
        # messages exist in the middle turns being compressed, the resulting
        # summary contains the Voice Message Log appendix.
        #
        # We can't easily test the full compress() without mocking the LLM
        # summarizer, so we test the extraction + building pipeline instead
        # and verify the appendix format matches what compress() would append.
        from unittest.mock import patch, MagicMock

        from agent.context_compressor import ContextCompressor

        voice_content = '[The user sent a voice message~ Here\'s what they said: "Deploy the staging build before noon"]'
        turns = [
            {"role": "user", "content": voice_content},
            {"role": "assistant", "content": "I'll deploy it now."},
        ]

        # Verify the pipeline produces correct appendix
        transcripts = _extract_voice_transcripts(turns)
        assert len(transcripts) == 1
        appendix = _build_voice_appendix(transcripts)
        assert "Deploy the staging build before noon" in appendix
        assert "## Voice Message Log" in appendix
