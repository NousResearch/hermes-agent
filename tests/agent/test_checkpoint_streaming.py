"""Tests for agent checkpoint strategy and streaming result heuristics."""

from __future__ import annotations

import pytest

from agent.checkpoint_strategy import CheckpointManager, CheckpointStrategy
from agent.streaming_results import StreamedToolResult, should_stream_tool


# ---------------------------------------------------------------------------
# CheckpointStrategy
# ---------------------------------------------------------------------------


class TestCheckpointStrategyNever:
    def setup_method(self):
        self.mgr = CheckpointManager(strategy=CheckpointStrategy.NEVER)

    def test_never_checkpoints_after_write(self):
        assert self.mgr.should_checkpoint("write_file", "ok") is False

    def test_never_checkpoints_after_error(self):
        assert self.mgr.should_checkpoint("read_file", {"error": "not found"}) is False

    def test_never_checkpoints_after_read(self):
        assert self.mgr.should_checkpoint("read_file", "content") is False


class TestCheckpointStrategyAll:
    def setup_method(self):
        self.mgr = CheckpointManager(strategy=CheckpointStrategy.ALL)

    def test_checkpoints_after_every_tool(self):
        assert self.mgr.should_checkpoint("read_file", "content") is True
        assert self.mgr.should_checkpoint("web_search", []) is True
        assert self.mgr.should_checkpoint("some_unknown_tool", None) is True


class TestCheckpointStrategySmart:
    def setup_method(self):
        self.mgr = CheckpointManager(strategy=CheckpointStrategy.SMART)

    def test_checkpoints_after_write_file(self):
        assert self.mgr.should_checkpoint("write_file", "42 bytes written") is True

    def test_checkpoints_after_patch(self):
        assert self.mgr.should_checkpoint("patch", "patched") is True

    def test_does_not_checkpoint_after_read(self):
        assert self.mgr.should_checkpoint("read_file", "content") is False

    def test_does_not_checkpoint_after_search(self):
        assert self.mgr.should_checkpoint("search_files", "src/x.py:1") is False

    def test_checkpoints_after_api_error(self):
        # Any result that looks like an error triggers a checkpoint so state is
        # preserved before a potential retry loop.
        assert self.mgr.should_checkpoint("web_search", {"error": "rate limit"}) is True

    def test_does_not_checkpoint_after_successful_read(self):
        assert self.mgr.should_checkpoint("web_search", [{"title": "T"}]) is False


class TestCheckpointStrategyRisky:
    def setup_method(self):
        self.mgr = CheckpointManager(strategy=CheckpointStrategy.RISKY)

    def test_checkpoints_after_write(self):
        assert self.mgr.should_checkpoint("write_file", "ok") is True

    def test_does_not_checkpoint_after_read(self):
        assert self.mgr.should_checkpoint("read_file", "content") is False


class TestCheckpointManagerHistory:
    def setup_method(self):
        self.mgr = CheckpointManager(strategy=CheckpointStrategy.SMART)

    def test_record_and_summary(self):
        self.mgr.record_checkpoint("write_file", "ckpt_abc")
        summary = self.mgr.get_summary()
        assert summary["checkpoints_taken"] == 1
        assert summary["strategy"] == CheckpointStrategy.SMART.value

    def test_multiple_records(self):
        self.mgr.record_checkpoint("write_file", "ckpt_1")
        self.mgr.record_checkpoint("patch", "ckpt_2")
        summary = self.mgr.get_summary()
        assert summary["checkpoints_taken"] == 2

    def test_summary_zero_when_empty(self):
        summary = self.mgr.get_summary()
        assert summary["checkpoints_taken"] == 0


# ---------------------------------------------------------------------------
# StreamedToolResult / should_stream_tool
# ---------------------------------------------------------------------------


class TestShouldStreamTool:
    def test_large_terminal_output_should_stream(self):
        assert should_stream_tool("terminal", "x" * 2000) is True

    def test_small_terminal_output_should_not_stream(self):
        assert should_stream_tool("terminal", "ok") is False

    def test_multiitem_list_should_stream(self):
        assert should_stream_tool("web_search", [{"a": 1}, {"b": 2}]) is True

    def test_single_item_list_should_not_stream(self):
        assert should_stream_tool("web_search", [{"a": 1}]) is False

    def test_large_file_content_should_stream(self):
        assert should_stream_tool("read_file", "line\n" * 500) is True

    def test_small_file_content_should_not_stream(self):
        assert should_stream_tool("read_file", "short") is False

    def test_none_result_should_not_stream(self):
        assert should_stream_tool("read_file", None) is False

    def test_unknown_tool_does_not_stream(self):
        # Without a known result type we can't decide — default to no streaming
        assert should_stream_tool("mystery_tool", "x" * 5000) is False


class TestStreamedToolResult:
    def test_chunks_large_text(self):
        content = "word " * 300  # ~1500 chars
        streamed = StreamedToolResult("read_file", content, chunk_size=256)
        chunks = list(streamed.stream_text())
        assert len(chunks) > 1
        assert "".join(chunks) == content

    def test_single_chunk_for_small_text(self):
        content = "hello"
        streamed = StreamedToolResult("terminal", content, chunk_size=256)
        chunks = list(streamed.stream_text())
        assert chunks == ["hello"]

    def test_empty_string_yields_empty_string(self):
        streamed = StreamedToolResult("terminal", "", chunk_size=256)
        chunks = list(streamed.stream_text())
        assert chunks == [""]

    def test_non_string_result_coerced_to_string(self):
        # Non-string results (e.g. list) are serialised so the caller gets text
        streamed = StreamedToolResult("web_search", [{"title": "x"}], chunk_size=256)
        chunks = list(streamed.stream_text())
        joined = "".join(chunks)
        assert "title" in joined or "x" in joined
