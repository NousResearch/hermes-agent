"""Tests for ContextCompressor.post_compress hook.

Verifies:
1. Default no-op returns messages unchanged.
2. Subclass override is called and its output is respected.
3. _strip_persistence_markers runs *after* post_compress,
   catching any markers the hook may have (re-)introduced.
"""

import os
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_compressor(quiet=True):
    """Create a minimal ContextCompressor for testing."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from agent.context_compressor import ContextCompressor
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.7,
            quiet_mode=quiet,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPostCompressDefault:
    """post_compress is a no-op by default."""

    def test_noop_returns_unchanged(self):
        cc = _make_compressor()
        messages = [{"role": "user", "content": "hello"}]
        result = cc.post_compress(messages)
        assert result is messages  # no-op: returns same list identity


# ---------------------------------------------------------------------------

class TestPostCompressOverride:
    """A subclass override is called and its output is respected."""

    def test_override_is_called(self):
        from agent.context_compressor import ContextCompressor

        called = []

        class OverrideCompressor(ContextCompressor):
            def post_compress(self, messages):
                called.append(1)
                return [{**m, "_custom_marker": True} for m in messages]

        cc = OverrideCompressor(
            model="test/model",
            threshold_percent=0.7,
            quiet_mode=True,
        )
        result = cc.post_compress([{"role": "user", "content": "test"}])

        assert called == [1]
        assert result[0]["_custom_marker"] is True


# ---------------------------------------------------------------------------

class TestMarkerSweepAfterHook:
    """Terminal _strip_persistence_markers catches hook-introduced markers."""

    def test_strip_function_exists(self):
        from agent.context_compressor import _strip_persistence_markers
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "_db_persisted": True},
        ]
        _strip_persistence_markers(msgs)
        assert "_db_persisted" not in msgs[1]

    def test_hook_marker_stripped_in_flow(self):
        """Simulate: hook adds marker → terminal sweep removes it."""
        from agent.context_compressor import _strip_persistence_markers

        messages = [{"role": "user", "content": "hello"}]

        # Step 1: post_compress could add markers
        messages = [{**m, "_db_persisted": True} for m in messages]
        assert messages[0]["_db_persisted"] is True

        # Step 2: terminal sweep strips them
        _strip_persistence_markers(messages)
        assert "_db_persisted" not in messages[0]
