"""Tests for memory flush stale-overwrite prevention (#2670).

Verifies that:
1. Cron sessions are skipped (no flush for headless cron runs)
2. Current memory state is injected into the flush prompt so the
   flush agent can see what's already saved and avoid overwrites
3. The flush still works normally when memory files don't exist
"""

import sys
import types
from types import SimpleNamespace
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


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


def _make_aux_response(*tool_calls):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=list(tool_calls) or None,
                )
            )
        ]
    )


def _make_tool_call(name: str, arguments: str):
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    )


def _make_flush_context():
    """Return a runner with a transcript loaded for flush tests."""
    runner = _make_runner()
    runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS
    return runner


class TestMemoryInjection:
    """The flush prompt should include current memory state from disk."""

    def test_memory_content_injected_into_flush_prompt(self, tmp_path, monkeypatch):
        """When memory files exist, their content appears in the flush prompt."""
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("Agent knows Python\n§\nUser prefers dark mode")
        (memory_dir / "USER.md").write_text("Name: Alice\n§\nTimezone: PST")

        runner = _make_flush_context()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("model_tools.get_tool_definitions", return_value=[
                {"type": "function", "function": {"name": "memory"}},
                {"type": "function", "function": {"name": "skill_manage"}},
            ]),
            patch("agent.auxiliary_client.call_llm", return_value=_make_aux_response()) as mock_call,
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: memory_dir)}),
        ):
            runner._flush_memories_for_session("session_123")

        mock_call.assert_called_once()
        flush_prompt = mock_call.call_args.kwargs["messages"][-1]["content"]

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

        runner = _make_flush_context()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("model_tools.get_tool_definitions", return_value=[
                {"type": "function", "function": {"name": "memory"}},
            ]),
            patch("agent.auxiliary_client.call_llm", return_value=_make_aux_response()) as mock_call,
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: empty_dir)}),
        ):
            runner._flush_memories_for_session("session_456")

        mock_call.assert_called_once()
        flush_prompt = mock_call.call_args.kwargs["messages"][-1]["content"]
        assert "Do NOT overwrite or remove entries" not in flush_prompt
        assert "Review the conversation above" in flush_prompt

    def test_empty_memory_files_no_injection(self, tmp_path, monkeypatch):
        """Empty memory files should not trigger the guard section."""
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("")
        (memory_dir / "USER.md").write_text("  \n  ")  # whitespace only

        runner = _make_flush_context()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("model_tools.get_tool_definitions", return_value=[
                {"type": "function", "function": {"name": "memory"}},
            ]),
            patch("agent.auxiliary_client.call_llm", return_value=_make_aux_response()) as mock_call,
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: memory_dir)}),
        ):
            runner._flush_memories_for_session("session_789")

        mock_call.assert_called_once()
        flush_prompt = mock_call.call_args.kwargs["messages"][-1]["content"]
        assert "current live state of memory" not in flush_prompt


class TestFlushAgentSilenced:
    """The flush path should use a silent one-shot tool call."""

    def test_flush_uses_auxiliary_call_not_run_conversation(self, tmp_path):
        """The expired-session flush should not spin a full agent conversation loop."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("model_tools.get_tool_definitions", return_value=[
                {"type": "function", "function": {"name": "memory"}},
            ]),
            patch("agent.auxiliary_client.call_llm", return_value=_make_aux_response()) as mock_call,
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: tmp_path)}),
            patch.dict("sys.modules", {"run_agent": MagicMock()}),
        ):
            runner._flush_memories_for_session("session_silent")

        mock_call.assert_called_once()

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


class TestFlushPromptStructure:
    """Verify the flush prompt retains its core instructions."""

    def test_core_instructions_present(self, monkeypatch):
        """The flush prompt should still contain the original guidance."""
        runner = _make_flush_context()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("model_tools.get_tool_definitions", return_value=[
                {"type": "function", "function": {"name": "memory"}},
            ]),
            patch("agent.auxiliary_client.call_llm", return_value=_make_aux_response()) as mock_call,
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock(get_memory_dir=lambda: Path("/nonexistent"))}),
        ):
            runner._flush_memories_for_session("session_struct")

        flush_prompt = mock_call.call_args.kwargs["messages"][-1]["content"]
        assert "automatically reset" in flush_prompt
        assert "Save any important facts" in flush_prompt
        assert "consider saving it as a skill" in flush_prompt
        assert "Do NOT respond to the user" in flush_prompt

    def test_memory_tool_calls_still_execute(self, tmp_path):
        """Tool calls from the quiet flush response should still be dispatched."""
        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS
        tool_call = _make_tool_call(
            "memory",
            '{"action":"add","target":"memory","content":"User prefers terse replies"}',
        )

        fake_memory_store = MagicMock()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("model_tools.get_tool_definitions", return_value=[
                {"type": "function", "function": {"name": "memory"}},
            ]),
            patch("agent.auxiliary_client.call_llm", return_value=_make_aux_response(tool_call)),
            patch("gateway.run._load_gateway_flush_memory_store", return_value=fake_memory_store),
            patch("tools.memory_tool.memory_tool") as mock_memory_tool,
        ):
            runner._flush_memories_for_session("session_tool_call")

        mock_memory_tool.assert_called_once_with(
            action="add",
            target="memory",
            content="User prefers terse replies",
            old_text=None,
            store=fake_memory_store,
        )
