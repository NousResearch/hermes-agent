"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import os
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionInfo,
    TextContentBlock,
    Usage,
)
from acp_adapter.server import HermesACPAgent, HERMES_VERSION
from acp_adapter.session import SessionManager


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
        assert caps.session_capabilities is not None
        assert caps.session_capabilities.fork is not None
        assert caps.session_capabilities.list is not None


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.has_provider",
            lambda: True,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.has_provider",
            lambda: False,
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
    async def test_load_session_returns_response(self, agent):
        resp = await agent.new_session(cwd="/tmp")
        load_resp = await agent.load_session(cwd="/tmp", session_id=resp.session_id)
        assert isinstance(load_resp, LoadSessionResponse)

    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None

    @pytest.mark.asyncio
    async def test_resume_session_returns_response(self, agent):
        resp = await agent.new_session(cwd="/tmp")
        resume_resp = await agent.resume_session(cwd="/tmp", session_id=resp.session_id)
        assert isinstance(resume_resp, ResumeSessionResponse)

    @pytest.mark.asyncio
    async def test_resume_session_creates_new_if_missing(self, agent):
        resume_resp = await agent.resume_session(cwd="/tmp", session_id="nonexistent")
        assert isinstance(resume_resp, ResumeSessionResponse)


# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_list_sessions(self, agent):
        await agent.new_session(cwd="/a")
        await agent.new_session(cwd="/b")
        resp = await agent.list_sessions()
        assert isinstance(resp, ListSessionsResponse)
        assert len(resp.sessions) == 2

    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id


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
