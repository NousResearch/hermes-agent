"""End-to-end tests for ACP MCP server registration and tool-result reporting.

Exercises the full flow through the ACP server layer:
  new_session(mcpServers) → MCP tools registered → prompt() →
    tool_progress_callback (ToolCallStart) →
    step_callback with results (ToolCallUpdate with rawOutput) →
    session_update events arrive at the mock client
"""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import (
    EnvVariable,
    HttpHeader,
    McpServerHttp,
    McpServerStdio,
    NewSessionResponse,
    PromptResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_manager():
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def acp_agent(mock_manager):
    return HermesACPAgent(session_manager=mock_manager)


# ---------------------------------------------------------------------------
# E2E: MCP registration → prompt → tool events
# ---------------------------------------------------------------------------


class TestMcpRegistrationE2E:
    """Full flow: session with MCP servers → prompt with tool calls → ACP events."""

    @pytest.mark.asyncio
    async def test_session_with_mcp_servers_registers_tools(self, acp_agent, mock_manager):
        """new_session with mcpServers converts them to Hermes config and registers."""
        servers = [
            McpServerStdio(
                name="test-fs",
                command="/usr/bin/mcp-fs",
                args=["--root", "/tmp"],
                env=[EnvVariable(name="DEBUG", value="1")],
            ),
            McpServerHttp(
                name="test-api",
                url="https://api.example.com/mcp",
                headers=[HttpHeader(name="Authorization", value="Bearer tok123")],
            ),
        ]

        registered_configs = {}

        def mock_register(config_map):
            registered_configs.update(config_map)
            return ["mcp_test_fs_read", "mcp_test_fs_write", "mcp_test_api_search"]

        fake_tools = [
            {"function": {"name": "mcp_test_fs_read"}},
            {"function": {"name": "mcp_test_fs_write"}},
            {"function": {"name": "mcp_test_api_search"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=mock_register), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=servers)

        assert isinstance(resp, NewSessionResponse)
        state = mock_manager.get_session(resp.session_id)

        # Verify stdio server was converted correctly
        assert "test-fs" in registered_configs
        fs_cfg = registered_configs["test-fs"]
        assert fs_cfg["command"] == "/usr/bin/mcp-fs"
        assert fs_cfg["args"] == ["--root", "/tmp"]
        assert fs_cfg["env"] == {"DEBUG": "1"}

        # Verify HTTP server was converted correctly
        assert "test-api" in registered_configs
        api_cfg = registered_configs["test-api"]
        assert api_cfg["url"] == "https://api.example.com/mcp"
        assert api_cfg["headers"] == {"Authorization": "Bearer tok123"}

        # Verify agent tool surface was refreshed
        assert state.agent.tools == fake_tools
        assert state.agent.valid_tool_names == {
            "mcp_test_fs_read", "mcp_test_fs_write", "mcp_test_api_search", "terminal"
        }

    @pytest.mark.asyncio
    async def test_prompt_with_tool_calls_emits_acp_events(self, acp_agent, mock_manager):
        """Prompt → agent fires callbacks → ACP ToolCallStart + ToolCallUpdate events."""
        resp = await acp_agent.new_session(cwd="/tmp")
        session_id = resp.session_id
        state = mock_manager.get_session(session_id)

        # Wire up a mock ACP client connection
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        mock_conn.request_permission = AsyncMock()
        acp_agent._conn = mock_conn

        def mock_run_conversation(user_message, conversation_history=None, task_id=None):
            """Simulate an agent turn that calls terminal, gets a result, then responds."""
            agent = state.agent

            # 1) Agent fires tool_progress_callback (ToolCallStart)
            if agent.tool_progress_callback:
                agent.tool_progress_callback(
                    "tool.started", "terminal", "$ echo hello", {"command": "echo hello"}
                )

            # 2) Agent fires step_callback with tool results (ToolCallUpdate)
            if agent.step_callback:
                agent.step_callback(1, [
                    {"name": "terminal", "result": '{"output": "hello\\n", "exit_code": 0}'}
                ])

            return {
                "final_response": "The command output 'hello'.",
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "The command output 'hello'."},
                ],
            }

        state.agent.run_conversation = mock_run_conversation

        prompt = [TextContentBlock(type="text", text="run echo hello")]
        resp = await acp_agent.prompt(prompt=prompt, session_id=session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "end_turn"

        # Collect all session_update calls
        updates = []
        for call in mock_conn.session_update.call_args_list:
            # session_update(session_id, update) — grab the update
            update_arg = call[1].get("update") or call[0][1]
            updates.append(update_arg)

        # Find tool_call (start) and tool_call_update (completion) events
        starts = [u for u in updates if getattr(u, "session_update", None) == "tool_call"]
        completions = [u for u in updates if getattr(u, "session_update", None) == "tool_call_update"]

        # Should have at least one ToolCallStart for "terminal"
        assert len(starts) >= 1, f"Expected ToolCallStart, got updates: {[getattr(u, 'session_update', '?') for u in updates]}"
        start_event = starts[0]
        assert isinstance(start_event, ToolCallStart)
        assert start_event.title.startswith("terminal:")

        # Should have at least one ToolCallUpdate (completion) with rawOutput
        assert len(completions) >= 1, f"Expected ToolCallUpdate, got updates: {[getattr(u, 'session_update', '?') for u in updates]}"
        complete_event = completions[0]
        assert isinstance(complete_event, ToolCallProgress)
        assert complete_event.status == "completed"
        # rawOutput should contain the tool result string
        assert complete_event.raw_output is not None
        assert "hello" in str(complete_event.raw_output)

    @pytest.mark.asyncio
    async def test_prompt_tool_results_paired_by_call_id(self, acp_agent, mock_manager):
        """The ToolCallUpdate's toolCallId must match the ToolCallStart's."""
        resp = await acp_agent.new_session(cwd="/tmp")
        session_id = resp.session_id
        state = mock_manager.get_session(session_id)

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        mock_conn.request_permission = AsyncMock()
        acp_agent._conn = mock_conn

        def mock_run(user_message, conversation_history=None, task_id=None):
            agent = state.agent
            # Fire two tool calls
            if agent.tool_progress_callback:
                agent.tool_progress_callback("tool.started", "read_file", "read: /etc/hosts", {"path": "/etc/hosts"})
                agent.tool_progress_callback("tool.started", "web_search", "web search: test", {"query": "test"})

            if agent.step_callback:
                agent.step_callback(1, [
                    {"name": "read_file", "result": '{"content": "127.0.0.1 localhost"}'},
                    {"name": "web_search", "result": '{"data": {"web": []}}'},
                ])

            return {"final_response": "Done.", "messages": []}

        state.agent.run_conversation = mock_run

        prompt = [TextContentBlock(type="text", text="test")]
        await acp_agent.prompt(prompt=prompt, session_id=session_id)

        updates = []
        for call in mock_conn.session_update.call_args_list:
            update_arg = call[1].get("update") or call[0][1]
            updates.append(update_arg)

        starts = [u for u in updates if getattr(u, "session_update", None) == "tool_call"]
        completions = [u for u in updates if getattr(u, "session_update", None) == "tool_call_update"]

        assert len(starts) == 2, f"Expected 2 starts, got {len(starts)}"
        assert len(completions) == 2, f"Expected 2 completions, got {len(completions)}"

        # Each completion's toolCallId must match a start's toolCallId
        start_ids = {s.tool_call_id for s in starts}
        completion_ids = {c.tool_call_id for c in completions}
        assert start_ids == completion_ids, (
            f"IDs must match: starts={start_ids}, completions={completion_ids}"
        )


class TestMcpSanitizationE2E:
    """Verify server names with special chars work end-to-end."""

    @pytest.mark.asyncio
    async def test_slashed_server_name_registers_cleanly(self, acp_agent, mock_manager):
        """Server name 'ai.exa/exa' should not crash — tools get sanitized names."""
        servers = [
            McpServerHttp(
                name="ai.exa/exa",
                url="https://exa.ai/mcp",
                headers=[],
            ),
        ]

        registered_configs = {}
        def mock_register(config_map):
            registered_configs.update(config_map)
            return ["mcp_ai_exa_exa_search"]

        fake_tools = [{"function": {"name": "mcp_ai_exa_exa_search"}}]

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=mock_register), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=servers)

        state = mock_manager.get_session(resp.session_id)

        # Raw server name preserved as config key
        assert "ai.exa/exa" in registered_configs
        # Agent tools refreshed with sanitized name
        assert "mcp_ai_exa_exa_search" in state.agent.valid_tool_names


class TestSessionLifecycleMcpE2E:
    """Verify MCP servers are registered on all session lifecycle methods."""

    @pytest.mark.asyncio
    async def test_load_session_registers_mcp(self, acp_agent, mock_manager):
        """load_session re-registers MCP servers (spec says agents may not retain them)."""
        # Create a session first
        create_resp = await acp_agent.new_session(cwd="/tmp")
        sid = create_resp.session_id

        servers = [
            McpServerStdio(name="srv", command="/bin/test", args=[], env=[]),
        ]

        registered = {}
        def mock_register(config_map):
            registered.update(config_map)
            return []

        state = mock_manager.get_session(sid)
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=mock_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await acp_agent.load_session(cwd="/tmp", session_id=sid, mcp_servers=servers)

        assert "srv" in registered

    @pytest.mark.asyncio
    async def test_resume_session_registers_mcp(self, acp_agent, mock_manager):
        """resume_session re-registers MCP servers."""
        create_resp = await acp_agent.new_session(cwd="/tmp")
        sid = create_resp.session_id

        servers = [
            McpServerStdio(name="srv2", command="/bin/test2", args=[], env=[]),
        ]

        registered = {}
        def mock_register(config_map):
            registered.update(config_map)
            return []

        state = mock_manager.get_session(sid)
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=mock_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await acp_agent.resume_session(cwd="/tmp", session_id=sid, mcp_servers=servers)

        assert "srv2" in registered

    @pytest.mark.asyncio
    async def test_fork_session_registers_mcp(self, acp_agent, mock_manager):
        """fork_session registers MCP servers on the new forked session."""
        create_resp = await acp_agent.new_session(cwd="/tmp")
        sid = create_resp.session_id

        servers = [
            McpServerHttp(name="api", url="https://api.test/mcp", headers=[]),
        ]

        registered = {}
        def mock_register(config_map):
            registered.update(config_map)
            return []

        # Need to set up the forked session's agent too
        with patch("tools.mcp_tool.register_mcp_servers", side_effect=mock_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            fork_resp = await acp_agent.fork_session(
                cwd="/tmp", session_id=sid, mcp_servers=servers
            )

        assert fork_resp.session_id != ""
        assert "api" in registered
