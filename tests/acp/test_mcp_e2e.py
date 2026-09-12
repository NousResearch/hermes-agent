"""End-to-end tests for ACP MCP server registration and tool-result reporting.

Exercises the full flow through the ACP server layer:
  new_session(mcpServers) → MCP tools registered → prompt() →
    tool_progress_callback (ToolCallStart) →
    step_callback with results (ToolCallUpdate with rawOutput) →
    session_update events arrive at the mock client
"""

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
from acp_adapter.tools import build_tool_start


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

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=mock_register), \
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

        def mock_run_conversation(user_message, conversation_history=None, task_id=None, **kwargs):
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
        # Completion should contain human-readable output rather than forcing raw JSON panes.
        assert complete_event.content
        assert "hello" in complete_event.content[0].content.text
        assert complete_event.raw_output is None

    def test_patch_mode_tool_start_defers_diff_to_edit_approval_prompt(self):
        update = build_tool_start(
            "tc-1",
            "patch",
            {
                "mode": "patch",
                "patch": "*** Begin Patch\n*** Update File: src/app.py\n@@\n-old line\n+new line\n*** Add File: src/new.py\n+hello\n*** End Patch",
            },
        )

        assert len(update.content) == 1
        assert update.content[0].type == "content"
        assert "Approval prompt shows the diff" in update.content[0].content.text



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

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=mock_register), \
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

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=mock_register), \
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

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=mock_register), \
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
        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=mock_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            fork_resp = await acp_agent.fork_session(
                cwd="/tmp", session_id=sid, mcp_servers=servers
            )

        assert fork_resp.session_id != ""
        assert "api" in registered

    @pytest.mark.asyncio
    async def test_mcp_tools_skip_tool_search_assembly(self, acp_agent, mock_manager):
        """new_session passes skip_tool_search_assembly=True so MCP tools are not deferred."""
        servers = [McpServerStdio(name="direct-tool", command="/bin/echo", args=[], env=[])]
        captured_kwargs = {}

        def mock_get_tool_defs(**kwargs):
            captured_kwargs.update(kwargs)
            return [{"function": {"name": "mcp_direct_tool_run"}}]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_direct_tool_run"]), \
             patch("model_tools.get_tool_definitions", side_effect=mock_get_tool_defs):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=servers)

        assert captured_kwargs.get("skip_tool_search_assembly") is True
        state = mock_manager.get_session(resp.session_id)
        assert state.agent.tools == [{"function": {"name": "mcp_direct_tool_run"}}]

    @pytest.mark.asyncio
    async def test_set_session_model_preserves_mcp_tools(self, acp_agent, mock_manager):
        """set_session_model preserves mcp_servers on the session and refreshes agent tools."""
        servers = [McpServerStdio(name="persist-srv", command="/bin/srv", args=[], env=[])]
        fake_mcp_tool = {"function": {"name": "mcp_persist_srv_action"}}

        with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_persist_srv_action"]), \
             patch("model_tools.get_tool_definitions", return_value=[fake_mcp_tool]):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=servers)

        state = mock_manager.get_session(resp.session_id)
        assert getattr(state, "mcp_servers", None) == servers

        # Switch model
        with patch("model_tools.get_tool_definitions", return_value=[fake_mcp_tool]) as mock_defs:
            await acp_agent.set_session_model("gpt-4o", session_id=resp.session_id)

        assert state.agent.tools == [fake_mcp_tool]
        assert "mcp_persist_srv_action" in state.agent.valid_tool_names
        assert "mcp-persist-srv" in state.agent.enabled_toolsets

    @pytest.mark.asyncio
    async def test_fork_session_inherits_parent_mcp_servers_when_omitted(self, acp_agent, mock_manager):
        """fork_session preserves parent session's mcp_servers when mcp_servers is None."""
        servers = [McpServerStdio(name="parent-mcp", command="/bin/p", args=[], env=[])]
        fake_tool = {"function": {"name": "mcp_parent_mcp_tool"}}

        with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_parent_mcp_tool"]), \
             patch("model_tools.get_tool_definitions", return_value=[fake_tool]):
            create_resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=servers)

        sid = create_resp.session_id
        registered = {}

        def mock_register(config_map):
            registered.update(config_map)
            return ["mcp_parent_mcp_tool"]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=mock_register), \
             patch("model_tools.get_tool_definitions", return_value=[fake_tool]):
            fork_resp = await acp_agent.fork_session(cwd="/tmp", session_id=sid, mcp_servers=None)

        assert fork_resp.session_id != ""
        assert "parent-mcp" in registered
        forked_state = mock_manager.get_session(fork_resp.session_id)
        assert forked_state.mcp_servers == servers

    @pytest.mark.asyncio
    async def test_slash_cmd_tools_includes_mcp_servers(self, acp_agent, mock_manager):
        """The /tools slash command includes MCP tools directly without progressive disclosure deferral."""
        servers = [McpServerStdio(name="search-srv", command="/bin/s", args=[], env=[])]
        fake_tool = {"function": {"name": "mcp_search_srv_query", "description": "Execute a search"}}

        with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_search_srv_query"]), \
             patch("model_tools.get_tool_definitions", return_value=[fake_tool]):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=servers)

        state = mock_manager.get_session(resp.session_id)
        with patch("model_tools.get_tool_definitions", return_value=[fake_tool]) as mock_defs:
            out = acp_agent._cmd_tools("", state)

        assert "mcp_search_srv_query" in out
        assert mock_defs.call_args.kwargs.get("skip_tool_search_assembly") is True

    def test_refresh_agent_mcp_tools_defaults_skip_search_for_acp(self):
        """refresh_agent_mcp_tools defaults skip_tool_search_assembly=True when agent.platform == 'acp'."""
        from tools.mcp_tool_agent import refresh_agent_mcp_tools

        fake_agent = MagicMock()
        fake_agent.platform = "acp"
        fake_agent.enabled_toolsets = ["hermes-acp", "mcp-demo"]
        fake_agent.disabled_toolsets = []
        fake_agent._tool_snapshot_generation = 0

        captured = {}

        def fake_get_tool_definitions(**kwargs):
            captured.update(kwargs)
            return [{"function": {"name": "mcp_demo_tool"}}]

        with patch("model_tools.get_tool_definitions", side_effect=fake_get_tool_definitions), \
             patch("tools.registry.registry._generation", 1):
            refresh_agent_mcp_tools(fake_agent)

        assert captured.get("skip_tool_search_assembly") is True

