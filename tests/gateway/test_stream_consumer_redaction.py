"""Tests for GatewayStreamConsumer._clean_for_display — secret redaction.

Regression tests for the streaming-path secret redaction gap.
The streaming path must redact secrets and strip tool-trace banners
in every chunk, including finalized split-message chunks that are
never edited again.
"""

from __future__ import annotations

from gateway.stream_consumer import GatewayStreamConsumer


class TestCleanForDisplaySecretRedaction:
    """Verify _clean_for_display redacts secrets and strips banners."""

    def test_media_tags_still_stripped(self):
        """Existing behavior: MEDIA: tags are removed."""
        text = "Hello MEDIA:/path/to/file.png world"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "MEDIA:" not in result
        assert "Hello" in result
        assert "world" in result

    def test_audio_as_voice_still_stripped(self):
        """Existing behavior: [[audio_as_voice]] directives are removed."""
        text = "Hello [[audio_as_voice]] world"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert "[[audio_as_voice]]" not in result

    def test_normal_text_preserved(self):
        """Normal text without secrets passes through unchanged."""
        text = "Hello world, this is a normal response."
        result = GatewayStreamConsumer._clean_for_display(text)
        assert result == text

    def test_api_key_redacted(self):
        """API keys in streamed text must be redacted."""
        text = "Here is your key: sk-abc123def456ghi789jkl012mno345pqr678stu"
        result = GatewayStreamConsumer._clean_for_display(text)
        # The redactor preserves first 6 + last 4 chars for long tokens
        assert "sk-abc123def456" not in result
        assert "sk-abc" in result  # prefix preserved
        assert "8stu" in result  # suffix preserved

    def test_tool_trace_banner_stripped(self):
        """Tool-trace banners in streamed text must be stripped."""
        text = "Done.\n⚠️ 🛠️ `search repos (agent)` failed"
        result = GatewayStreamConsumer._clean_for_display(text)
        assert result == "Done."
        assert "failed" not in result

    def test_empty_text_returns_empty(self):
        """Empty text returns empty string."""
        result = GatewayStreamConsumer._clean_for_display("")
        assert result == ""

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only text returns empty string after rstrip."""
        result = GatewayStreamConsumer._clean_for_display("   \n\n  ")
        assert result == ""
