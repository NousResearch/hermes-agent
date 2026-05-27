"""Tests for progressive chunked summarization in ContextCompressor.

Feature 3: When compression runs, discarded conversation turns are archived
to chunk-N.md files and a reference is injected into the summary.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _append_text_to_content,
    _content_text_for_contains,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_compressor(chunk_enabled=True, max_chunks=20, quiet=True):
    """Create a ContextCompressor with chunk archiving configured."""
    cc = ContextCompressor(
        model="gpt-4o",
        quiet_mode=quiet,
        config_context_length=128000,
    )
    cc._chunk_archiving_enabled = chunk_enabled
    cc._max_chunks = max_chunks
    cc._chunk_index = 0
    cc._last_discarded_messages = []
    cc._last_discarded_topics = ""
    return cc


def _make_messages(count=10):
    """Build a list of simple user/assistant/tool messages."""
    msgs = []
    for i in range(count):
        msgs.append({"role": "user", "content": f"user message {i}"})
        msgs.append({"role": "assistant", "content": f"assistant response {i}"})
        if i % 2 == 0:
            msgs.append({"role": "tool", "content": f"tool result {i}"})
    return msgs


# ── _save_chunk ─────────────────────────────────────────────────────

class TestSaveChunk:
    def test_returns_none_when_disabled(self, tmp_path):
        cc = _make_compressor(chunk_enabled=False)
        msgs = _make_messages(5)
        result = cc._save_chunk(msgs, "test-session", 1)
        assert result is None

    def test_returns_none_when_no_session_id(self, tmp_path):
        cc = _make_compressor()
        msgs = _make_messages(5)
        result = cc._save_chunk(msgs, None, 1)
        assert result is None

    def test_returns_none_when_max_chunks_exceeded(self, tmp_path):
        cc = _make_compressor(max_chunks=3)
        msgs = _make_messages(5)
        result = cc._save_chunk(msgs, "test-session", 5)
        assert result is None

    def test_saves_chunk_file(self, tmp_path):
        cc = _make_compressor()
        cc._get_sessions_dir = lambda: tmp_path
        msgs = _make_messages(5)
        result = cc._save_chunk(msgs, "test-session", 1)
        assert result is not None
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "session_id: test-session" in content
        assert "chunk: 1" in content
        assert "Session Chunk 1" in content
        assert "user message 0" in content

    def test_chunk_includes_multimodal_fallback(self, tmp_path):
        """Multimodal content should extract text parts only."""
        cc = _make_compressor()
        cc._get_sessions_dir = lambda: tmp_path
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        result = cc._save_chunk(msgs, "test-session", 1)
        assert result is not None
        content = Path(result).read_text()
        assert "hello" in content
        assert "image_url" not in content  # stripped

    def test_chunk_front_matter_includes_topics(self, tmp_path):
        cc = _make_compressor()
        cc._get_sessions_dir = lambda: tmp_path
        cc._previous_summary = (
            f"{SUMMARY_PREFIX}\n## Goal\n[Build a FastAPI auth service]\n"
            "## Active Task\n[Implement JWT middleware]\n"
        )
        msgs = _make_messages(3)
        result = cc._save_chunk(msgs, "test-session", 1)
        content = Path(result).read_text()
        assert "Build a FastAPI auth service" in content


# ── _build_chunk_reference ───────────────────────────────────────────

class TestBuildChunkReference:
    def test_basic_reference(self):
        cc = _make_compressor()
        ref = cc._build_chunk_reference("/tmp/chunk-1.md", 1, "auth, FastAPI")
        assert "Chunk 1" in ref
        assert "/tmp/chunk-1.md" in ref
        assert "file_reader" in ref
        assert "auth, FastAPI" in ref

    def test_reference_without_topics(self):
        cc = _make_compressor()
        ref = cc._build_chunk_reference("/tmp/chunk-2.md", 2)
        assert "Chunk 2" in ref
        assert "Topics" not in ref

    def test_reference_is_markdown(self):
        cc = _make_compressor()
        ref = cc._build_chunk_reference("/tmp/c.md", 3, "test")
        assert ref.startswith("\n\n## Archived Context")


# ── _extract_topics_from_text ───────────────────────────────────────

class TestExtractTopics:
    def test_extracts_from_goal_section(self):
        text = "## Goal\n[Implement user authentication with JWT]"
        result = ContextCompressor._extract_topics_from_text(text)
        assert "Implement user authentication with JWT" in result

    def test_extracts_from_active_task_fallback(self):
        text = "## Active Task\n[Debug database connection pool]"
        result = ContextCompressor._extract_topics_from_text(text)
        assert "Debug database connection pool" in result

    def test_truncates_long_topics(self):
        text = "## Goal\n[" + "A" * 200 + "]"
        result = ContextCompressor._extract_topics_from_text(text)
        assert len(result) <= 120
        assert result.endswith("...")

    def test_empty_text_returns_empty(self):
        assert ContextCompressor._extract_topics_from_text("") == ""

    def test_no_goal_or_task_returns_empty(self):
        text = "Just some random text without goal section"
        assert ContextCompressor._extract_topics_from_text(text) == ""


# ── Chunk reference injection ───────────────────────────────────────

class TestChunkInjection:
    """Test the chunk reference injection logic in compress_context."""

    def test_chunk_ref_injected_into_summary_message(self):
        """When compression succeeds, chunk reference should be appended."""
        cc = _make_compressor()
        cc._previous_summary = f"{SUMMARY_PREFIX}\n## Goal\n[Test project]\n"

        # Simulate the injection logic from compress_context
        compressed = [{"role": "user", "content": f"{SUMMARY_PREFIX}\nSummary here\n"}]
        chunk_ref = cc._build_chunk_reference("/tmp/chunk-1.md", 1, "Test project")

        for msg in compressed:
            content = msg.get("content", "")
            if isinstance(content, str) and SUMMARY_PREFIX in content:
                msg["content"] = _append_text_to_content(content, chunk_ref)

        assert "Archived Context" in compressed[0]["content"]
        assert "Chunk 1" in compressed[0]["content"]

    def test_no_duplicate_injection(self):
        """Chunk reference should not be injected twice."""
        cc = _make_compressor()
        chunk_ref = cc._build_chunk_reference("/tmp/chunk-1.md", 1, "test")

        compressed = [{"role": "user", "content": f"{SUMMARY_PREFIX}\nSummary{chunk_ref}"}]

        for msg in compressed:
            content = msg.get("content", "")
            if isinstance(content, str) and SUMMARY_PREFIX in content:
                if chunk_ref not in _content_text_for_contains(content):
                    msg["content"] = _append_text_to_content(content, chunk_ref)

        # Should still have only ONE chunk reference
        count = compressed[0]["content"].count("Archived Context")
        assert count == 1

    def test_no_injection_when_no_summary_prefix(self):
        """Messages without SUMMARY_PREFIX should not get chunk reference."""
        compressed = [{"role": "user", "content": "Just a regular message"}]
        chunk_ref = "## Archived Context\n..."

        for msg in compressed:
            content = msg.get("content", "")
            if isinstance(content, str) and SUMMARY_PREFIX in content:
                msg["content"] = _append_text_to_content(content, chunk_ref)

        assert "Archived Context" not in compressed[0]["content"]


# ── _get_sessions_dir ───────────────────────────────────────────────

class TestGetSessionsDir:
    def test_uses_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        result = ContextCompressor._get_sessions_dir()
        assert result.name == "sessions"
        assert ".hermes" in str(result)

    def test_uses_hermes_home_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = ContextCompressor._get_sessions_dir()
        assert result == tmp_path / "sessions"


# ── configure_chunk_archiving ───────────────────────────────────────

class TestConfigureChunkArchiving:
    def test_defaults(self):
        cc = _make_compressor()
        assert cc._chunk_archiving_enabled is True
        assert cc._max_chunks == 20

    def test_disable(self):
        cc = _make_compressor()
        cc.configure_chunk_archiving(enabled=False, max_chunks=10)
        assert cc._chunk_archiving_enabled is False
        assert cc._max_chunks == 10

    def test_custom_max(self):
        cc = _make_compressor()
        cc.configure_chunk_archiving(enabled=True, max_chunks=50)
        assert cc._max_chunks == 50


# ── on_session_reset ────────────────────────────────────────────────

class TestSessionReset:
    def test_reset_clears_chunk_state(self):
        cc = _make_compressor()
        cc._chunk_index = 5
        cc._last_discarded_messages = [{"role": "user", "content": "test"}]
        cc._last_discarded_topics = "some topics"

        cc.on_session_reset()

        assert cc._chunk_index == 0
        assert cc._last_discarded_messages == []
        assert cc._last_discarded_topics == ""
