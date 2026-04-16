"""Tests for memory flush stale-overwrite prevention (#2670).

Verifies that:
1. Cron sessions are skipped (no flush for headless cron runs)
2. Current memory state is injected into the flush prompt so the
   flush agent can see what's already saved and avoid overwrites
3. The flush still works normally when memory files don't exist
"""

import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    """gateway.run imports dotenv at module level; stub it so tests run without the package."""
    fake = types.ModuleType("dotenv")
    fake.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake)


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

    def test_cron_session_with_prefix_skipped(self):
        """Cron sessions with different prefixes are still skipped."""
        runner = _make_runner()
        runner._flush_memories_for_session("cron_daily_20260323")
        runner.session_store.load_transcript.assert_not_called()

    def test_non_cron_session_proceeds(self):
        """Non-cron sessions should still attempt the flush."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = []
        runner._flush_memories_for_session("session_abc123")
        runner.session_store.load_transcript.assert_called_once_with("session_abc123")


def _make_flush_context(monkeypatch, memory_dir=None):
    """Return (runner, tmp_agent, fake_run_agent) with run_agent mocked in sys.modules."""
    tmp_agent = MagicMock()
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = MagicMock(return_value=tmp_agent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    runner = _make_runner()
    runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS
    return runner, tmp_agent, memory_dir


class TestMemoryInjection:
    """The flush prompt should include current memory state from disk."""

    def test_memory_content_injected_into_flush_prompt(self, tmp_path, monkeypatch):
        """When memory files exist, their content appears in the flush prompt."""
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("Agent knows Python\n§\nUser prefers dark mode")
        (memory_dir / "USER.md").write_text("Name: Alice\n§\nTimezone: PST")

        runner, tmp_agent, _ = _make_flush_context(monkeypatch, memory_dir)

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: memory_dir)}),
        ):
            runner._flush_memories_for_session("session_123")

        tmp_agent.run_conversation.assert_called_once()
        flush_prompt = tmp_agent.run_conversation.call_args.kwargs.get("user_message", "")

        assert "Agent knows Python" in flush_prompt
        assert "User prefers dark mode" in flush_prompt
        assert "Name: Alice" in flush_prompt
        assert "Timezone: PST" in flush_prompt
        assert "Do NOT overwrite or remove entries" in flush_prompt
        assert "current live state of memory" in flush_prompt

    def test_flush_works_without_memory_files(self, tmp_path, monkeypatch):
        """When no memory files exist, flush still runs without the guard."""
        empty_dir = tmp_path / "no_memories"
        empty_dir.mkdir()

        runner, tmp_agent, _ = _make_flush_context(monkeypatch)

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: empty_dir)}),
        ):
            runner._flush_memories_for_session("session_456")

        tmp_agent.run_conversation.assert_called_once()
        flush_prompt = tmp_agent.run_conversation.call_args.kwargs.get("user_message", "")
        assert "Do NOT overwrite or remove entries" not in flush_prompt
        assert "Review the conversation above" in flush_prompt

    def test_empty_memory_files_no_injection(self, tmp_path, monkeypatch):
        """Empty memory files should not trigger the guard section."""
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("")
        (memory_dir / "USER.md").write_text("  \n  ")  # whitespace only

        runner, tmp_agent, _ = _make_flush_context(monkeypatch)

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: memory_dir)}),
        ):
            runner._flush_memories_for_session("session_789")

        tmp_agent.run_conversation.assert_called_once()
        flush_prompt = tmp_agent.run_conversation.call_args.kwargs.get("user_message", "")
        assert "current live state of memory" not in flush_prompt


class TestFlushAgentSilenced:
    """The flush agent must not produce any terminal output."""

    def test_print_fn_set_to_noop(self, tmp_path, monkeypatch):
        """_print_fn on the flush agent must be a no-op so tool output never leaks."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS

        captured_agent = {}

        def _fake_ai_agent(*args, **kwargs):
            agent = MagicMock()
            captured_agent["instance"] = agent
            return agent

        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _fake_ai_agent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: tmp_path)}),
        ):
            runner._flush_memories_for_session("session_silent")

        agent = captured_agent["instance"]
        assert agent._print_fn is not None, "_print_fn should be overridden to suppress output"
        # Confirm it is callable and produces no output (no exception)
        agent._print_fn("should be silenced")

    def test_kawaii_spinner_respects_print_fn(self):
        """KawaiiSpinner must route all output through print_fn when supplied."""
        from agent.display import KawaiiSpinner

        written = []
        spinner = KawaiiSpinner("test", print_fn=lambda *a, **kw: written.append(a))
        spinner._write("hello")
        assert written == [("hello",)], "spinner should route through print_fn"

        # A no-op print_fn must produce no output to stdout
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            silent_spinner = KawaiiSpinner("silent", print_fn=lambda *a, **kw: None)
            silent_spinner._write("should not appear")
            silent_spinner.stop("done")
        finally:
            sys.stdout = old_stdout
        assert buf.getvalue() == "", "no-op print_fn spinner must not write to stdout"

    def test_flush_agent_closes_resources_after_run(self, monkeypatch):
        """Memory flush should close temporary agent resources after the turn."""
        runner, tmp_agent, _ = _make_flush_context(monkeypatch)
        tmp_agent.shutdown_memory_provider = MagicMock()
        tmp_agent.close = MagicMock()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: Path("/nonexistent"))}),
        ):
            runner._flush_memories_for_session("session_cleanup")

        tmp_agent.shutdown_memory_provider.assert_called_once()
        tmp_agent.close.assert_called_once()


class TestFlushPromptStructure:
    """Verify the flush prompt retains its core instructions."""

    def test_core_instructions_present(self, monkeypatch):
        """The flush prompt should still contain the original guidance."""
        runner, tmp_agent, _ = _make_flush_context(monkeypatch)

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: Path("/nonexistent"))}),
        ):
            runner._flush_memories_for_session("session_struct")

        flush_prompt = tmp_agent.run_conversation.call_args.kwargs.get("user_message", "")
        assert "automatically reset" in flush_prompt
        assert "Save any important facts" in flush_prompt
        assert "consider saving it as a skill" in flush_prompt
        assert "Do NOT respond to the user" in flush_prompt
