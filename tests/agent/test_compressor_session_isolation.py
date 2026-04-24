"""Tests for ContextCompressor cross-session state isolation (#14603).

_previous_summary leaks across sessions when the same cached AIAgent is
reused in gateway/cron, causing old task context to bleed into new session
compression summaries.
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


class TestSessionIsolation:
    """_previous_summary must not leak across sessions (#14603)."""

    def test_on_session_start_clears_previous_summary(self, compressor):
        compressor._previous_summary = "## Active Task\nDeploy the web server"
        compressor._ineffective_compression_count = 3
        compressor.on_session_start("new-session-123")
        assert compressor._previous_summary is None
        assert compressor._ineffective_compression_count == 0

    def test_on_session_reset_still_works(self, compressor):
        compressor._previous_summary = "old summary"
        compressor._context_probed = True
        compressor.on_session_reset()
        assert compressor._previous_summary is None
        assert compressor._context_probed is False

    def test_summary_does_not_leak_between_sessions(self, compressor):
        compressor._previous_summary = "## Active Task\nMigrate database schema"
        compressor.on_session_start("session-2")
        assert compressor._previous_summary is None
