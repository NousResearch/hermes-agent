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
        # New behaviour: streaming is gated on result size, not tool name.
        # A large result from any tool — including unknown ones — is worth
        # streaming; the old name-gate was the anti-pattern the reviewer called out.
        assert should_stream_tool("mystery_tool", "x" * 5000) is True

    def test_unknown_tool_small_does_not_stream(self):
        # Small results are never worth streaming regardless of tool name.
        assert should_stream_tool("mystery_tool", "short") is False


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


# ---------------------------------------------------------------------------
# async delegation: stream_text_async must match stream_text exactly
# ---------------------------------------------------------------------------


class TestStreamedToolResultAsync:
    """stream_text_async must yield exactly the same chunks as stream_text."""

    def _collect_async(self, streamed: StreamedToolResult) -> list:
        import asyncio

        async def _run():
            return [chunk async for chunk in streamed.stream_text_async()]

        return asyncio.get_event_loop().run_until_complete(_run())

    def test_async_matches_sync_large_text(self):
        content = "abc " * 400
        streamed = StreamedToolResult("read_file", content, chunk_size=256)
        assert list(streamed.stream_text()) == self._collect_async(streamed)

    def test_async_matches_sync_empty_string(self):
        streamed = StreamedToolResult("terminal", "", chunk_size=256)
        assert list(streamed.stream_text()) == self._collect_async(streamed)

    def test_async_matches_sync_non_string(self):
        streamed = StreamedToolResult("web_search", {"key": "val"}, chunk_size=256)
        assert list(streamed.stream_text()) == self._collect_async(streamed)


# ---------------------------------------------------------------------------
# MCP tool label fix: mcp_<server>_<tool> must produce a sensible label
# ---------------------------------------------------------------------------


class TestMcpCheckpointLabel:
    def test_mcp_tool_label(self):
        from agent.checkpoint_strategy import get_checkpoint_label

        label = get_checkpoint_label("mcp_github_list_prs")
        assert label.startswith("after_mcp_")
        assert "github" in label

    def test_mcp_tool_should_not_checkpoint_on_success(self):
        from agent.checkpoint_strategy import should_checkpoint, CheckpointStrategy

        # Successful MCP tool call → no checkpoint under SMART strategy
        assert should_checkpoint("mcp_github_list_prs", [{"pr": 1}], CheckpointStrategy.SMART) is False

    def test_mcp_tool_checkpoints_on_error_result(self):
        from agent.checkpoint_strategy import should_checkpoint, CheckpointStrategy

        # MCP tool returning an error dict → checkpoint
        assert should_checkpoint("mcp_github_list_prs", {"error": "not found"}, CheckpointStrategy.SMART) is True

