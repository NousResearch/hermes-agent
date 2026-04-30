"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SessionInfo,
    TextContentBlock,
    Usage,
    UserMessageChunk,
)
from acp_adapter.server import HermesACPAgent, HERMES_VERSION
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_initialize_returns_agent_info(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert resp.agent_info is not None
        assert isinstance(resp.agent_info, Implementation)
        assert resp.agent_info.name == "hermes-agent"
        assert resp.agent_info.version == HERMES_VERSION

    @pytest.mark.asyncio
    async def test_initialize_returns_capabilities(self, agent):
        resp = await agent.initialize(protocol_version=1)
        caps = resp.agent_capabilities
        assert isinstance(caps, AgentCapabilities)
        assert caps.load_session is True
        assert caps.session_capabilities is not None
        assert caps.session_capabilities.fork is not None
        assert caps.session_capabilities.list is not None
        assert caps.session_capabilities.resume is not None

    @pytest.mark.asyncio
    async def test_initialize_capabilities_wire_format(self, agent):
        """Verify the JSON wire format uses correct aliases so ACP clients see the right keys."""
        resp = await agent.initialize(protocol_version=1)
        payload = resp.agent_capabilities.model_dump(by_alias=True, exclude_none=True)
        assert payload["loadSession"] is True
        session_caps = payload["sessionCapabilities"]
        assert "fork" in session_caps
        assert "list" in session_caps
        assert "resume" in session_caps


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None


# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:
    @pytest.mark.asyncio
    async def test_new_session_creates_session(self, agent):
        resp = await agent.new_session(cwd="/home/user/project")
        assert isinstance(resp, NewSessionResponse)
        assert resp.session_id
        # Session should be retrievable from the manager
        state = agent.session_manager.get_session(resp.session_id)
        assert state is not None
        assert state.cwd == "/home/user/project"

    @pytest.mark.asyncio
    async def test_new_session_returns_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="gpt-5.4", provider="openai-codex")
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "hermes_cli.models.curated_models_for_provider",
            return_value=[("gpt-5.4", "recommended"), ("gpt-5.4-mini", "")],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert resp.models.available_models[0].model_id == "openai-codex:gpt-5.4"
        assert resp.models.available_models[0].description is not None
        assert "Provider:" in resp.models.available_models[0].description

    @pytest.mark.asyncio
    async def test_available_commands_include_help(self, agent):
        help_cmd = next(
            (cmd for cmd in agent._available_commands() if cmd.name == "help"),
            None,
        )

        assert help_cmd is not None
        assert help_cmd.description == "List available commands"
        assert help_cmd.input is None

    @pytest.mark.asyncio
    async def test_send_available_commands_update(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent._send_available_commands_update("session-123")

        mock_conn.session_update.assert_awaited_once()
        call = mock_conn.session_update.await_args
        assert call.kwargs["session_id"] == "session-123"
        update = call.kwargs["update"]
        assert isinstance(update, AvailableCommandsUpdate)
        assert update.session_update == "available_commands_update"
        assert [cmd.name for cmd in update.available_commands] == [
            "help",
            "model",
            "tools",
            "context",
            "reset",
            "compact",
            "version",
        ]
        model_cmd = next(
            cmd for cmd in update.available_commands if cmd.name == "model"
        )
        assert model_cmd.input is not None
        assert model_cmd.input.root.hint == "model name to switch to"

    @pytest.mark.asyncio
    async def test_cancel_sets_event(self, agent):
        resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(resp.session_id)
        assert not state.cancel_event.is_set()
        await agent.cancel(session_id=resp.session_id)
        assert state.cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session_is_noop(self, agent):
        # Should not raise
        await agent.cancel(session_id="does-not-exist")

    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None

    @pytest.mark.asyncio
    async def test_load_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "system", "content": "hidden system"},
            {"role": "user", "content": "what controls the / slash commands?"},
            {"role": "assistant", "content": "HermesACPAgent._ADVERTISED_COMMANDS controls them."},
            {"role": "tool", "content": "tool output should not replay"},
        ]

        mock_conn.session_update.reset_mock()
        resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)

        assert isinstance(resp, LoadSessionResponse)
        calls = mock_conn.session_update.await_args_list
        replay_calls = [
            call for call in calls
            if getattr(call.kwargs.get("update"), "session_update", None)
            in {"user_message_chunk", "agent_message_chunk"}
        ]
        assert len(replay_calls) == 2
        assert isinstance(replay_calls[0].kwargs["update"], UserMessageChunk)
        assert replay_calls[0].kwargs["update"].content.text == "what controls the / slash commands?"
        assert isinstance(replay_calls[1].kwargs["update"], AgentMessageChunk)
        assert replay_calls[1].kwargs["update"].content.text.startswith("HermesACPAgent")

    @pytest.mark.asyncio
    async def test_resume_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "So tell me the current state"}]

        mock_conn.session_update.reset_mock()
        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )

    @pytest.mark.asyncio
    async def test_resume_session_creates_new_if_missing(self, agent):
        resume_resp = await agent.resume_session(cwd="/tmp", session_id="nonexistent")
        assert isinstance(resume_resp, ResumeSessionResponse)


# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"

    @pytest.mark.asyncio
    async def test_list_sessions_passes_cwd_filter(self, agent):
        with patch.object(agent.session_manager, "list_sessions", return_value=[]) as mock_list:
            await agent.list_sessions(cwd="/mnt/e/Projects/AI/browser-link-3")

        mock_list.assert_called_once_with(cwd="/mnt/e/Projects/AI/browser-link-3")

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_first_page(self, agent):
        from acp_adapter import server as acp_server

        infos = [
            {"session_id": f"s{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(acp_server._LIST_SESSIONS_PAGE_SIZE + 5)
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        assert len(resp.sessions) == acp_server._LIST_SESSIONS_PAGE_SIZE
        assert resp.next_cursor == resp.sessions[-1].session_id

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_no_more(self, agent):
        infos = [
            {"session_id": f"s{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(3)
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        assert len(resp.sessions) == 3
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_cursor_resumes_after_match(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s2", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s3", "cwd": "/tmp", "title": None, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions(cursor="s1")

        assert [s.session_id for s in resp.sessions] == ["s2", "s3"]
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_unknown_cursor_returns_empty(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s2", "cwd": "/tmp", "title": None, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions(cursor="does-not-exist")

        assert resp.sessions == []
        assert resp.next_cursor is None

# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:
    @pytest.mark.asyncio
    async def test_set_session_mode_returns_response(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        resp = await agent.set_session_mode(mode_id="chat", session_id=new_resp.session_id)
        state = agent.session_manager.get_session(new_resp.session_id)

        assert isinstance(resp, SetSessionModeResponse)
        assert getattr(state, "mode", None) == "chat"

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "chat", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        assert config_result == {"configOptions": []}

    @pytest.mark.asyncio
    async def test_router_accepts_unstable_model_switch_when_enabled(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent, use_unstable_protocol=True)

        result = await router(
            "session/set_model",
            {"modelId": "gpt-5.4", "sessionId": new_resp.session_id},
            False,
        )
        state = agent.session_manager.get_session(new_resp.session_id)

        assert result == {}
        assert state.model == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_set_session_model_accepts_provider_prefixed_choice(self, tmp_path, monkeypatch):
        runtime_calls = []

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            runtime_calls.append(requested)
            provider = requested or "openrouter"
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "openrouter/gpt-5"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            acp_agent = HermesACPAgent(session_manager=manager)
            state = manager.create_session(cwd="/tmp")
            result = await acp_agent.set_session_model(
                model_id="anthropic:claude-sonnet-4-6",
                session_id=state.session_id,
            )

        assert isinstance(result, SetSessionModelResponse)
        assert state.model == "claude-sonnet-4-6"
        assert state.agent.provider == "anthropic"
        assert state.agent.base_url == "https://anthropic.example/v1"
        assert runtime_calls[-1] == "anthropic"


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_returns_end_turn_for_empty_message(self, agent):
        new_resp = await agent.new_session(cwd=".")
        prompt = [TextContentBlock(type="text", text="   ")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_prompt_runs_agent(self, agent):
        """The prompt method should call run_conversation on the agent."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        # Mock the agent's run_conversation
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Hello! How can I help?",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
        })

        # Set up a mock connection
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompt_updates_history(self, agent):
        """After a prompt, session history should be updated."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        expected_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "hey",
            "messages": expected_history,
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert state.history == expected_history

    @pytest.mark.asyncio
    async def test_prompt_sends_final_message_update(self, agent):
        """The final response should be sent as an AgentMessageChunk."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "I can help with that!",
            "messages": [],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="help me")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        # session_update should have been called with the final message
        mock_conn.session_update.assert_called()
        # Get the last call's update argument
        last_call = mock_conn.session_update.call_args_list[-1]
        update = last_call[1].get("update") or last_call[0][1]
        assert update.session_update == "agent_message_chunk"

    @pytest.mark.asyncio
    async def test_prompt_auto_titles_session(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Here is the fix.",
            "messages": [
                {"role": "user", "content": "fix the broken ACP history"},
                {"role": "assistant", "content": "Here is the fix."},
            ],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("agent.title_generator.maybe_auto_title") as mock_title:
            prompt = [TextContentBlock(type="text", text="fix the broken ACP history")]
            await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        mock_title.assert_called_once()
        assert mock_title.call_args.args[1] == new_resp.session_id
        assert mock_title.call_args.args[2] == "fix the broken ACP history"
        assert mock_title.call_args.args[3] == "Here is the fix."

    @pytest.mark.asyncio
    async def test_prompt_populates_usage_from_top_level_run_conversation_fields(self, agent):
        """ACP should map top-level token fields into PromptResponse.usage."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "usage attached",
            "messages": [],
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
            "reasoning_tokens": 7,
            "cache_read_tokens": 11,
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="show usage")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.usage is not None
        assert resp.usage.input_tokens == 123
        assert resp.usage.output_tokens == 45
        assert resp.usage.total_tokens == 168
        assert resp.usage.thought_tokens == 7
        assert resp.usage.cached_read_tokens == 11

    @pytest.mark.asyncio
    async def test_prompt_cancelled_returns_cancelled_stop_reason(self, agent):
        """If cancel is called during prompt, stop_reason should be 'cancelled'."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            # Simulate cancel being set during execution
            state.cancel_event.set()
            return {"final_response": "interrupted", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="do something")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "cancelled"


# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_stores_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        agent.on_connect(mock_conn)
        assert agent._conn is mock_conn


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state

    def test_help_lists_commands(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/help", state)
        assert result is not None
        assert "/help" in result
        assert "/model" in result
        assert "/tools" in result
        assert "/reset" in result

    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result

    def test_context_empty(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = []
        result = agent._handle_slash_command("/context", state)
        assert "empty" in result.lower()

    def test_context_with_messages(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = agent._handle_slash_command("/context", state)
        assert "2 messages" in result
        assert "user: 1" in result

    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        result = agent._handle_slash_command("/reset", state)
        assert "cleared" in result.lower()
        assert len(state.history) == 0

    def test_version(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/version", state)
        assert HERMES_VERSION in result

    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_messages_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compact", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert state.history == [{"role": "user", "content": "summary"}]
        assert state.agent._session_db is original_session_db
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            task_id=state.session_id,
        )
        mock_save.assert_called_once_with(state.session_id)

    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None

    @pytest.mark.asyncio
    async def test_slash_command_intercepted_in_prompt(self, agent, mock_manager):
        """Slash commands should be handled without calling the LLM."""
        new_resp = await agent.new_session(cwd="/tmp")
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="/help")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        mock_conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_slash_falls_through_to_llm(self, agent, mock_manager):
        """Unknown /commands should be sent to the LLM, not intercepted."""
        new_resp = await agent.new_session(cwd="/tmp")
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        mock_conn.request_permission = AsyncMock(return_value=None)
        agent._conn = mock_conn

        # Mock run_in_executor to avoid actually running the agent
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value={
                "final_response": "I processed /foo",
                "messages": [],
            })
            prompt = [TextContentBlock(type="text", text="/foo bar")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"

    def test_model_switch_uses_requested_provider(self, tmp_path, monkeypatch):
        """`/model provider:model` should rebuild the ACP agent on that provider."""
        runtime_calls = []

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            runtime_calls.append(requested)
            provider = requested or "openrouter"
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "openrouter/gpt-5"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            acp_agent = HermesACPAgent(session_manager=manager)
            state = manager.create_session(cwd="/tmp")
            result = acp_agent._cmd_model("anthropic:claude-sonnet-4-6", state)

        assert "Provider: anthropic" in result
        assert state.agent.provider == "anthropic"
        assert state.agent.base_url == "https://anthropic.example/v1"
        assert runtime_calls[-1] == "anthropic"


# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""

    @pytest.mark.asyncio
    async def test_noop_when_no_servers(self, agent, mock_manager):
        """No-op when mcp_servers is None or empty."""
        state = mock_manager.create_session(cwd="/tmp")
        # Should not raise
        await agent._register_session_mcp_servers(state, None)
        await agent._register_session_mcp_servers(state, [])

    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}

    @pytest.mark.asyncio
    async def test_registers_http_servers(self, agent, mock_manager):
        """McpServerHttp servers are converted correctly."""
        from acp.schema import McpServerHttp, HttpHeader

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerHttp(
            name="http-server",
            url="https://api.example.com/mcp",
            headers=[HttpHeader(name="Authorization", value="Bearer tok")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return []

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "http-server" in registered_config
        cfg = registered_config["http-server"]
        assert cfg["url"] == "https://api.example.com/mcp"
        assert cfg["headers"] == {"Authorization": "Bearer tok"}

    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools == fake_tools
        assert state.agent.valid_tool_names == {"mcp_srv_search", "terminal"}
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_logs_warning(self, agent, mock_manager):
        """If register_mcp_servers raises, warning is logged but no crash."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])
