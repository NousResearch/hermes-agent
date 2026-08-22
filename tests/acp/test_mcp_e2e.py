"""End-to-end tests for ACP MCP server registration and tool-result reporting.

Exercises the full flow through the ACP server layer:
  new_session(mcpServers) → MCP tools registered → prompt() →
    tool_progress_callback (ToolCallStart) →
    step_callback with results (ToolCallUpdate with rawOutput) →
    session_update events arrive at the mock client
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.exceptions import RequestError
from acp.schema import (
    EnvVariable,
    HttpHeader,
    McpServerHttp,
    McpServerSse,
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
    async def test_sse_server_preserves_sse_transport(self, acp_agent):
        server = McpServerSse(
            name="events",
            url="https://example.test/sse",
            headers=[HttpHeader(name="X-Test", value="1")],
        )
        registered = {}

        with patch("tools.mcp_tool.get_registered_mcp_server_names", return_value=set()), \
             patch("tools.mcp_tool.register_mcp_servers", side_effect=lambda cfg: registered.update(cfg) or []), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await acp_agent.new_session(cwd="/tmp", mcp_servers=[server])

        assert registered["events"] == {
            "url": "https://example.test/sse",
            "headers": {"X-Test": "1"},
            "transport": "sse",
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

    @pytest.mark.asyncio
    async def test_close_releases_session_mcp_after_last_owner(self, acp_agent):
        server = McpServerStdio(name="owned", command="/bin/test", args=[], env=[])
        released = []

        with patch("tools.mcp_tool.get_registered_mcp_server_names", return_value=set()), \
             patch("tools.mcp_tool.register_mcp_servers", return_value=[]), \
             patch("tools.mcp_tool.release_mcp_servers", side_effect=lambda names: released.extend(names)), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            first = await acp_agent.new_session(cwd="/tmp", mcp_servers=[server])
            second = await acp_agent.new_session(cwd="/tmp", mcp_servers=[server])
            await acp_agent.close_session(first.session_id)
            assert released == []
            await acp_agent.close_session(second.session_id)

        assert released == ["owned"]

    @pytest.mark.asyncio
    async def test_close_mcp_release_failure_keeps_ownership_retryable(self, acp_agent):
        server = McpServerStdio(name="retry", command="/bin/test", args=[], env=[])
        attempts = []

        def release(names):
            attempts.append(list(names))
            if len(attempts) == 1:
                raise RuntimeError("release failed")

        with patch("tools.mcp_tool.get_registered_mcp_server_names", return_value=set()), \
             patch("tools.mcp_tool.register_mcp_servers", return_value=[]), \
             patch("tools.mcp_tool.release_mcp_servers", side_effect=release), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=[server])
            with pytest.raises(RequestError) as exc_info:
                await acp_agent.close_session(resp.session_id)
            assert exc_info.value.code == -32603
            assert resp.session_id in acp_agent.session_manager._sessions
            assert acp_agent._session_mcp_servers[resp.session_id] == {"retry"}
            await acp_agent.close_session(resp.session_id)

        assert attempts == [["retry"], ["retry"]]

    @pytest.mark.asyncio
    async def test_mcp_replacement_release_failure_rolls_back_ownership(self, acp_agent):
        old_server = McpServerStdio(name="old", command="/bin/old", args=[], env=[])
        new_server = McpServerStdio(name="new", command="/bin/new", args=[], env=[])
        release_attempts = []

        def release(names):
            release_attempts.append(list(names))
            if list(names) == ["old"]:
                raise RuntimeError("release failed")

        with patch("tools.mcp_tool.get_registered_mcp_server_names", return_value=set()), \
             patch("tools.mcp_tool.register_mcp_servers", return_value=[]), \
             patch("tools.mcp_tool.release_mcp_servers", side_effect=release), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            created = await acp_agent.new_session(cwd="/tmp", mcp_servers=[old_server])
            await acp_agent.load_session(
                "/tmp", created.session_id, [new_server]
            )

        assert acp_agent._session_mcp_servers[created.session_id] == {"old"}
        assert acp_agent._mcp_server_refcounts == {"old": 1}
        assert release_attempts == [["old"], ["new"]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["load", "resume"])
    async def test_closing_session_rejects_mcp_reregistration(self, acp_agent, method):
        created = await acp_agent.new_session(cwd="/tmp")
        state = acp_agent.session_manager.get_session(created.session_id)
        server = McpServerStdio(name="new", command="/bin/test", args=[], env=[])

        with patch("tools.mcp_tool.get_registered_mcp_server_names", return_value=set()), \
             patch("tools.mcp_tool.register_mcp_servers", return_value=[]), \
             patch("tools.mcp_tool.release_mcp_servers", return_value=[]), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await acp_agent._mcp_ownership_lock.acquire()
            try:
                if method == "load":
                    task = asyncio.create_task(
                        acp_agent.load_session("/tmp", created.session_id, [server])
                    )
                else:
                    task = asyncio.create_task(
                        acp_agent.resume_session("/tmp", created.session_id, [server])
                    )
                await asyncio.sleep(0)
                with state.runtime_lock:
                    state.closing = True
            finally:
                acp_agent._mcp_ownership_lock.release()

            with pytest.raises(RequestError) as exc_info:
                await task

        assert exc_info.value.code == -32002
        assert created.session_id not in acp_agent._session_mcp_servers
        assert acp_agent._mcp_server_refcounts == {}

    @pytest.mark.asyncio
    async def test_close_preserves_preexisting_mcp_server(self, acp_agent):
        server = McpServerStdio(name="shared", command="/bin/test", args=[], env=[])
        released = []

        with patch("tools.mcp_tool.get_registered_mcp_server_names", return_value={"shared"}), \
             patch("tools.mcp_tool.register_mcp_servers", return_value=[]), \
             patch("tools.mcp_tool.release_mcp_servers", side_effect=lambda names: released.extend(names)), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=[server])
            await acp_agent.close_session(resp.session_id)

        assert released == []

    @pytest.mark.asyncio
    async def test_close_preserves_globally_configured_mcp_server(self, acp_agent, monkeypatch):
        server = McpServerStdio(name="configured", command="/bin/test", args=[], env=[])
        released = []
        monkeypatch.delenv("HERMES_ACP_SKIP_CONFIGURED_MCP", raising=False)

        with patch("tools.mcp_tool.get_registered_mcp_server_names", return_value=set()), \
             patch("tools.mcp_tool.register_mcp_servers", return_value=[]), \
             patch("tools.mcp_tool.release_mcp_servers", side_effect=lambda names: released.extend(names)), \
             patch("hermes_cli.config.load_config", return_value={"mcp_servers": {"configured": {}}}), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            resp = await acp_agent.new_session(cwd="/tmp", mcp_servers=[server])
            await acp_agent.close_session(resp.session_id)

        assert released == []
