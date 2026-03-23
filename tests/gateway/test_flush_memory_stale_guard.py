"""Tests for memory flush stale-overwrite prevention (#2670).

Verifies that:
1. Cron sessions are skipped (no flush for headless cron runs)
2. Current memory state is injected into the flush prompt so the
   flush agent can see what's already saved and avoid overwrites
3. The flush still works normally when memory files don't exist
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.session_store = MagicMock()
    return runner


_TRANSCRIPT_4_MSGS = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "remember my name is Alice"},
    {"role": "assistant", "content": "Got it, Alice!"},
]


class TestCronSessionBypass:
    """Cron sessions should never trigger a memory flush."""

    def test_cron_session_skipped(self):
        runner = _make_runner()
        runner._flush_memories_for_session("cron_job123_20260323_120000")
        # session_store.load_transcript should never be called
        runner.session_store.load_transcript.assert_not_called()

    def test_cron_session_with_honcho_key_skipped(self):
        runner = _make_runner()
        runner._flush_memories_for_session("cron_daily_20260323", "some-honcho-key")
        runner.session_store.load_transcript.assert_not_called()

    def test_non_cron_session_proceeds(self):
        """Non-cron sessions should still attempt the flush."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = []
        runner._flush_memories_for_session("session_abc123")
        runner.session_store.load_transcript.assert_called_once_with("session_abc123")


class TestMemoryInjection:
    """The flush prompt should include current memory state from disk."""

    def test_memory_content_injected_into_flush_prompt(self, tmp_path):
        """When memory files exist, their content appears in the flush prompt."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS

        tmp_agent = MagicMock()
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("Agent knows Python\n§\nUser prefers dark mode")
        (memory_dir / "USER.md").write_text("Name: Alice\n§\nTimezone: PST")

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("run_agent.AIAgent", return_value=tmp_agent),
            # Intercept `from tools.memory_tool import MEMORY_DIR` inside the function
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(MEMORY_DIR=memory_dir)}),
        ):
            runner._flush_memories_for_session("session_123")

        tmp_agent.run_conversation.assert_called_once()
        call_kwargs = tmp_agent.run_conversation.call_args.kwargs
        flush_prompt = call_kwargs.get("user_message", "")
        
        # Verify both memory sections appear in the prompt
        assert "Agent knows Python" in flush_prompt
        assert "User prefers dark mode" in flush_prompt
        assert "Name: Alice" in flush_prompt
        assert "Timezone: PST" in flush_prompt
        # Verify the stale-overwrite warning is present
        assert "Do NOT overwrite or remove entries" in flush_prompt
        assert "current live state of memory" in flush_prompt

    def test_flush_works_without_memory_files(self, tmp_path):
        """When no memory files exist, flush still runs without the guard."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS

        tmp_agent = MagicMock()
        empty_dir = tmp_path / "no_memories"
        empty_dir.mkdir()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("run_agent.AIAgent", return_value=tmp_agent),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(MEMORY_DIR=empty_dir)}),
        ):
            runner._flush_memories_for_session("session_456")

        # Should still run, just without the memory guard section
        tmp_agent.run_conversation.assert_called_once()
        flush_prompt = tmp_agent.run_conversation.call_args.kwargs.get("user_message", "")
        assert "Do NOT overwrite or remove entries" not in flush_prompt
        assert "Review the conversation above" in flush_prompt

    def test_empty_memory_files_no_injection(self, tmp_path):
        """Empty memory files should not trigger the guard section."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS

        tmp_agent = MagicMock()
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("")
        (memory_dir / "USER.md").write_text("  \n  ")  # whitespace only

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("run_agent.AIAgent", return_value=tmp_agent),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(MEMORY_DIR=memory_dir)}),
        ):
            runner._flush_memories_for_session("session_789")

        tmp_agent.run_conversation.assert_called_once()
        flush_prompt = tmp_agent.run_conversation.call_args.kwargs.get("user_message", "")
        # No memory content → no guard section
        assert "current live state of memory" not in flush_prompt


class TestFlushPromptStructure:
    """Verify the flush prompt retains its core instructions."""

    def test_core_instructions_present(self):
        """The flush prompt should still contain the original guidance."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS

        tmp_agent = MagicMock()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("run_agent.AIAgent", return_value=tmp_agent),
            # Make the import fail gracefully so we test without memory files
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(MEMORY_DIR=Path("/nonexistent"))}),
        ):
            runner._flush_memories_for_session("session_struct")

        flush_prompt = tmp_agent.run_conversation.call_args.kwargs.get("user_message", "")
        assert "automatically reset" in flush_prompt
        assert "Save any important facts" in flush_prompt
        assert "consider saving it as a skill" in flush_prompt
        assert "Do NOT respond to the user" in flush_prompt
