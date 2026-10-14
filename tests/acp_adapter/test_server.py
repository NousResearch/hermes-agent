"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AuthenticateResponse,
    InitializeResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SessionInfo,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
from acp_adapter.server import (
    HermesACPAgent,
)
from acp_adapter.session import SessionManager


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


@pytest.mark.asyncio
async def test_new_session_exposes_edit_approvals_as_modes_not_config_options(agent):
    resp = await agent.new_session(cwd="/tmp")

    assert resp.config_options is None
    assert isinstance(resp.modes, SessionModeState)
    assert resp.modes.current_mode_id == "default"
    assert [mode.id for mode in resp.modes.available_modes] == [
        "default",
        "accept_edits",
        "dont_ask",
    ]


@pytest.mark.asyncio
async def test_set_config_option_persists_edit_approval_policy_without_advertising_config(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "edit_approval_policy",
        resp.session_id,
        "workspace_session",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert isinstance(update, SetSessionConfigOptionResponse)
    assert update.config_options == []
    assert getattr(state, "mode", None) == "accept_edits"


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
    async def test_initialize_advertises_provider_and_terminal_auth_methods(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "openrouter")

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads[0]["id"] == "openrouter"
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]



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

    @pytest.mark.asyncio
    async def test_authenticate_accepts_terminal_setup_after_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert isinstance(resp, AuthenticateResponse)



# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:

    @pytest.mark.asyncio
    async def test_cancel_autonomous_turn_does_not_salvage_internal_notification(
        self, agent, monkeypatch
    ):
        state = agent.session_manager.create_session(cwd="/tmp")
        state.is_running = True
        state.active_turn_origin = "autonomous"
        state.current_prompt_text = "Internal background completion"
        state.interrupted_prompt_text = "Earlier user request"
        interrupt = MagicMock()
        monkeypatch.setattr("acp_adapter.server.request_hard_interrupt", interrupt)

        await agent.cancel(state.session_id)

        assert state.cancel_event.is_set()
        assert state.interrupted_prompt_text == "Earlier user request"
        interrupt.assert_called_once_with(state.agent)

    @pytest.mark.asyncio
    async def test_dispatches_only_background_notifications_owned_by_connected_sessions(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        conn = MagicMock()
        conn.session_update = AsyncMock()
        agent._conn = conn
        state = agent.session_manager.create_session(cwd="/tmp")
        state.agent.run_conversation.return_value = {
            "final_response": "Parent report.",
            "messages": [
                {"role": "user", "content": "Background process completed."},
                {"role": "assistant", "content": "Parent report."},
            ],
        }
        state.interrupted_prompt_text = "Cancelled user request"
        agent._connected_session_ids.add(state.session_id)
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)

        registry.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_owned",
                "session_key": state.session_id,
                "command": "printf done",
                "exit_code": 0,
                "output": "done",
            }
        )
        registry.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_foreign",
                "session_key": "acp-foreign",
                "command": "printf secret",
                "exit_code": 0,
                "output": "secret",
            }
        )
        registry._poll_observed.add("proc_owned")

        delivered = await agent._dispatch_background_notifications_once()

        assert delivered == 1
        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in conn.session_update.await_args_list
        ]
        notification = next(
            update for update in updates if update.session_update == "user_message_chunk"
        )
        assert notification.field_meta == {
            "hermes": {
                "backgroundNotification": True,
                "process": {
                    "id": "proc_owned",
                    "event": "completion",
                    "status": "completed",
                    "exitCode": 0,
                },
            }
        }
        assert "proc_owned completed normally" in notification.content.text
        assert any(
            update.session_update == "agent_message_chunk"
            and update.content.text == "Parent report."
            for update in updates
        )
        state.agent.run_conversation.assert_called_once()
        assert "proc_owned completed normally" in state.agent.run_conversation.call_args.kwargs[
            "user_message"
        ]
        assert "Cancelled user request" not in state.agent.run_conversation.call_args.kwargs[
            "user_message"
        ]
        assert state.interrupted_prompt_text == "Cancelled user request"
        assert registry.completion_queue.get_nowait()["session_id"] == "proc_foreign"

    @pytest.mark.asyncio
    async def test_session_not_attached_in_memory_keeps_notification_pending(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        conn = MagicMock()
        conn.session_update = AsyncMock()
        agent._conn = conn
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)

        # This connection still claims the session id (e.g. the claim outlived a
        # transport replacement) but no in-process state is attached for it, and
        # the dispatcher must not restore a database-only session here.
        detached = "acp-detached-not-in-memory"
        agent._connected_session_ids.add(detached)
        registry.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_detached",
                "session_key": detached,
                "command": "printf done",
                "exit_code": 0,
                "output": "done",
            }
        )

        delivered = await agent._dispatch_background_notifications_once()

        # Fail-closed: no autonomous turn and no client update for a session this
        # transport does not hold in memory ...
        assert delivered == 0
        state = agent.session_manager.get_in_memory(detached)
        assert state is None
        conn.session_update.assert_not_awaited()
        # ... and the stale claim goes away so it cannot keep the drain loop busy.
        assert detached not in agent._connected_session_ids
        # The completion is not dropped either: it stays pending for a later
        # reconnect that attaches the session again.
        assert registry.completion_queue.get_nowait()["session_id"] == "proc_detached"

    @pytest.mark.asyncio
    async def test_busy_session_keeps_background_notification_pending(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        conn = MagicMock()
        conn.session_update = AsyncMock()
        agent._conn = conn
        state = agent.session_manager.create_session(cwd="/tmp")
        state.is_running = True
        state.active_turn_origin = "user"
        agent._connected_session_ids.add(state.session_id)
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)
        registry.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_busy",
                "session_key": state.session_id,
                "command": "printf done",
                "exit_code": 0,
                "output": "done",
            }
        )

        delivered = await agent._dispatch_background_notifications_once()

        assert delivered == 0
        assert registry.completion_queue.get_nowait()["session_id"] == "proc_busy"
        state.agent.run_conversation.assert_not_called()
        conn.session_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_pass_serves_idle_peer_while_busy_session_retains(
        self, agent, monkeypatch
    ):
        """Two sessions attached on one connection, one dispatch pass.

        A completion owned by an idle session must still be delivered while the
        peer session is mid-turn, and the busy owner's own completion must stay
        queued instead of starting or interrupting anything.
        """
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        conn = MagicMock()
        conn.session_update = AsyncMock()
        agent._conn = conn
        busy = agent.session_manager.create_session(cwd="/tmp")
        busy.is_running = True
        busy.active_turn_origin = "user"
        idle = agent.session_manager.create_session(cwd="/tmp")
        idle.agent.run_conversation.return_value = {
            "final_response": "Idle peer report.",
            "messages": [
                {"role": "user", "content": "Background process completed."},
                {"role": "assistant", "content": "Idle peer report."},
            ],
        }
        agent._connected_session_ids.update({busy.session_id, idle.session_id})
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)
        registry.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_busy",
                "session_key": busy.session_id,
                "command": "printf busy",
                "exit_code": 0,
                "output": "busy",
            }
        )
        registry.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_idle",
                "session_key": idle.session_id,
                "command": "printf done",
                "exit_code": 0,
                "output": "done",
            }
        )

        delivered = await agent._dispatch_background_notifications_once()

        # The idle peer is served in this same pass, while the busy owner keeps
        # its event queued and its turn untouched.
        assert delivered == 1
        idle.agent.run_conversation.assert_called_once()
        assert registry.completion_queue.get_nowait()["session_id"] == "proc_busy"
        assert registry.completion_queue.empty()
        busy.agent.run_conversation.assert_not_called()
        assert busy.is_running is True

        calls = conn.session_update.await_args_list
        assert calls
        # Routing is per session: no frame of this delivery lands on the peer.
        assert {(call.kwargs.get("session_id") or call.args[0]) for call in calls} == {
            idle.session_id
        }
        updates = [call.kwargs.get("update") or call.args[1] for call in calls]
        notification = next(
            update for update in updates if update.session_update == "user_message_chunk"
        )
        assert notification.field_meta["hermes"]["process"]["id"] == "proc_idle"

    @pytest.mark.asyncio
    async def test_async_delegation_turn_completes_its_durable_claim(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        conn = MagicMock()
        conn.session_update = AsyncMock()
        agent._conn = conn
        state = agent.session_manager.create_session(cwd="/tmp")
        state.agent.run_conversation.return_value = {
            "final_response": "Consolidated parent report.",
            "messages": [],
        }
        agent._connected_session_ids.add(state.session_id)
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)

        claim = MagicMock(return_value="claim-1")
        complete = MagicMock()
        release = MagicMock()
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", claim)
        monkeypatch.setattr("tools.async_delegation.complete_event_delivery", complete)
        monkeypatch.setattr("tools.async_delegation.release_event_delivery", release)

        event = {
            "type": "async_delegation",
            "delegation_id": "deleg_test",
            "session_key": state.session_id,
            "status": "completed",
            "is_batch": True,
            "goals": ["inspect the regression"],
            "results": [
                {
                    "task_index": 0,
                    "status": "completed",
                    "summary": "Child found the missing ACP queue drain.",
                }
            ],
        }
        registry.completion_queue.put(event)

        delivered = await agent._dispatch_background_notifications_once()

        assert delivered == 1
        claimed_event, consumer = claim.call_args.args
        assert claimed_event == event
        assert consumer.startswith("acp:")
        complete.assert_called_once_with(event, "claim-1")
        release.assert_not_called()
        prompt_text = state.agent.run_conversation.call_args.kwargs["user_message"]
        assert "Child found the missing ACP queue drain." in prompt_text
        assert "deleg_test" in prompt_text

    @pytest.mark.asyncio
    async def test_background_notification_loop_stops_after_transport_failure(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        conn = MagicMock()
        conn.session_update = AsyncMock(side_effect=ConnectionError("disconnected"))
        agent._conn = conn
        state = agent.session_manager.create_session(cwd="/tmp")
        agent._connected_session_ids.add(state.session_id)
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)

        registry.completion_queue.put(
            {
                "type": "completion",
                "session_id": "proc_owned",
                "session_key": state.session_id,
                "command": "printf done",
                "exit_code": 0,
                "output": "done",
            }
        )

        await asyncio.wait_for(agent._background_notification_loop(conn), timeout=0.05)

        assert agent._conn is None
        conn.session_update.assert_awaited_once()
        assert registry.completion_queue.get_nowait()["session_id"] == "proc_owned"

    @pytest.mark.asyncio
    async def test_connection_replacement_requeues_cancelled_delivery_suffix(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        old_conn = MagicMock()
        new_conn = MagicMock()
        delivery_started = asyncio.Event()
        block_delivery = asyncio.Event()

        async def blocked_session_update(**_kwargs):
            delivery_started.set()
            await block_delivery.wait()

        old_conn.session_update = AsyncMock(side_effect=blocked_session_update)
        new_conn.session_update = AsyncMock()
        agent._conn = old_conn
        state = agent.session_manager.create_session(cwd="/tmp")
        agent._connected_session_ids.add(state.session_id)
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)

        for process_id in ("proc_first", "proc_second"):
            registry.completion_queue.put(
                {
                    "type": "completion",
                    "session_id": process_id,
                    "session_key": state.session_id,
                    "command": "printf done",
                    "exit_code": 0,
                    "output": "done",
                }
            )

        old_task = asyncio.create_task(agent._background_notification_loop(old_conn))
        agent._background_notification_task = old_task
        await asyncio.wait_for(delivery_started.wait(), timeout=0.1)

        agent.on_connect(new_conn)
        new_task = agent._background_notification_task
        try:
            with pytest.raises(asyncio.CancelledError):
                await old_task
            queued_ids = [
                registry.completion_queue.get_nowait()["session_id"],
                registry.completion_queue.get_nowait()["session_id"],
            ]
            assert queued_ids == ["proc_first", "proc_second"]
            assert registry.completion_queue.empty()
        finally:
            if new_task is not None:
                new_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await new_task

    @pytest.mark.asyncio
    async def test_connection_replacement_does_not_replay_started_autonomous_turn(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        old_conn = MagicMock()
        new_conn = MagicMock()
        old_conn.session_update = AsyncMock()
        new_conn.session_update = AsyncMock()
        agent._conn = old_conn
        state = agent.session_manager.create_session(cwd="/tmp")
        agent._connected_session_ids.add(state.session_id)
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)

        run_started = threading.Event()
        allow_finish = threading.Event()

        def run_parent(**_kwargs):
            run_started.set()
            assert allow_finish.wait(timeout=1)
            return {
                "final_response": "Parent finished once.",
                "messages": [
                    {"role": "user", "content": "Background process completed."},
                    {"role": "assistant", "content": "Parent finished once."},
                ],
            }

        state.agent.run_conversation.side_effect = run_parent
        for process_id in ("proc_started", "proc_waiting"):
            registry.completion_queue.put(
                {
                    "type": "completion",
                    "session_id": process_id,
                    "session_key": state.session_id,
                    "command": "printf done",
                    "exit_code": 0,
                    "output": "done",
                }
            )

        old_task = asyncio.create_task(agent._background_notification_loop(old_conn))
        agent._background_notification_task = old_task
        assert await asyncio.to_thread(run_started.wait, 1)

        agent.on_connect(new_conn)
        new_task = agent._background_notification_task
        allow_finish.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await old_task
            state.agent.run_conversation.assert_called_once()
            assert registry.completion_queue.get_nowait()["session_id"] == "proc_waiting"
            assert registry.completion_queue.empty()
        finally:
            if new_task is not None:
                new_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await new_task

    @pytest.mark.asyncio
    async def test_connection_replacement_after_model_completion_does_not_replay(
        self, agent, monkeypatch
    ):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        old_conn = MagicMock()
        new_conn = MagicMock()
        result_delivery_started = asyncio.Event()
        allow_result_delivery = asyncio.Event()

        async def block_final_result(*args, **kwargs):
            update = kwargs.get("update") or args[1]
            if update.session_update == "agent_message_chunk":
                result_delivery_started.set()
                await allow_result_delivery.wait()

        old_conn.session_update = AsyncMock(side_effect=block_final_result)
        new_conn.session_update = AsyncMock()
        agent._conn = old_conn
        state = agent.session_manager.create_session(cwd="/tmp")
        state.agent.run_conversation.return_value = {
            "final_response": "Parent result is durable.",
            "messages": [
                {"role": "user", "content": "Background process completed."},
                {"role": "assistant", "content": "Parent result is durable."},
            ],
        }
        agent._connected_session_ids.add(state.session_id)
        monkeypatch.setattr("acp_adapter.server.process_registry", registry)

        for process_id in ("proc_completed", "proc_waiting"):
            registry.completion_queue.put(
                {
                    "type": "completion",
                    "session_id": process_id,
                    "session_key": state.session_id,
                    "command": "printf done",
                    "exit_code": 0,
                    "output": "done",
                }
            )

        old_task = asyncio.create_task(agent._background_notification_loop(old_conn))
        agent._background_notification_task = old_task
        await asyncio.wait_for(result_delivery_started.wait(), timeout=1)

        agent.on_connect(new_conn)
        new_task = agent._background_notification_task
        allow_result_delivery.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await old_task
            state.agent.run_conversation.assert_called_once()
            assert registry.completion_queue.get_nowait()["session_id"] == "proc_waiting"
            assert registry.completion_queue.empty()
        finally:
            if new_task is not None:
                new_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await new_task

    @pytest.mark.asyncio
    async def test_new_session_returns_authenticated_cross_provider_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://api.openai.com/v1",
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        picker_context = MagicMock()
        picker_context.with_overrides.return_value = picker_context
        payload = {
            "providers": [
                {
                    "slug": "anthropic",
                    "name": "Anthropic",
                    "models": ["claude-sonnet-4-6", "claude-sonnet-4-6"],
                },
                {
                    "slug": "openai-codex",
                    "name": "OpenAI Codex",
                    "models": [
                        {"id": "gpt-5.4"},
                        "gpt-5.4-mini",
                    ],
                },
            ],
        }

        with (
            patch("hermes_cli.inventory.load_picker_context", return_value=picker_context),
            patch("hermes_cli.inventory.build_models_payload", return_value=payload),
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert [model.model_id for model in resp.models.available_models] == [
            "anthropic:claude-sonnet-4-6",
            "openai-codex:gpt-5.4",
            "openai-codex:gpt-5.4-mini",
        ]
        picker_context.with_overrides.assert_called_once_with(
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="https://api.openai.com/v1",
        )





    def test_build_usage_update_for_zed_context_indicator(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(context_length=100_000)
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            update = agent._build_usage_update(state)

        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 25_000




    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None






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
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )











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






# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "accept_edits", "sessionId": new_resp.session_id},
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
        assert config_result["configOptions"] == []





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
    async def test_prompt_binds_session_id_into_subprocess_env(self, agent, mock_manager):
        """The ACP prompt path must bridge the session id into child subprocesses.

        Regression: ``set_session_vars`` was called with ``session_key`` only,
        leaving the ``HERMES_SESSION_ID`` ContextVar bound to the explicit ""
        default. Once the session-context machinery is engaged, that empty value
        is authoritative — so ``_make_run_env`` handed child subprocesses an
        empty ``HERMES_SESSION_ID`` instead of the session's own id.
        """
        from tools.environments.local import _make_run_env

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        captured: dict[str, str | None] = {}

        def _run(*args, **kwargs):
            # Runs inside the session context copy set up by prompt().
            captured["child"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=resp.session_id,
        )

        assert captured.get("child") == resp.session_id

    @pytest.mark.asyncio
    async def test_user_prompt_queues_behind_autonomous_turn_without_redirecting(
        self, agent, mock_manager
    ):
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.is_running = True
        state.active_turn_origin = "autonomous"
        state.agent._supports_active_turn_redirect = True
        state.agent.redirect = MagicMock(return_value=True)
        conn = MagicMock(spec=acp.Client)
        conn.session_update = AsyncMock()
        agent._conn = conn

        result = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="new user request")],
            session_id=resp.session_id,
        )

        assert result.stop_reason == "end_turn"
        assert state.queued_prompts == ["new user request"]
        state.agent.redirect.assert_not_called()

    def test_title_update_does_not_create_coroutine_after_loop_closes(self, agent):
        class ClosedLoop:
            def call_soon_threadsafe(self, _callback):
                raise RuntimeError("event loop is closed")

        agent._send_session_info_update = MagicMock()

        agent._schedule_session_info_update_from_thread(
            ClosedLoop(),
            "closed-session",
        )

        agent._send_session_info_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_messages_list_replaces_stale_history(self, agent, mock_manager):
        """``run_conversation`` returning ``messages=[]`` clears the ACP transcript instead of
        leaving the previous turn's history in place (#10844)."""
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.history = [{"role": "user", "content": "old"}]
        state.agent.run_conversation = MagicMock(return_value={"final_response": "done", "messages": []})
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(prompt=[TextContentBlock(type="text", text="hi")], session_id=resp.session_id)

        assert state.history == []

    @pytest.mark.asyncio
    async def test_prompt_after_tail_exception_runs_instead_of_queueing(self, agent, mock_manager, monkeypatch):
        """A raise in the post-turn tail (here ``save_session``) costs at most that one turn:
        the next prompt runs instead of queueing forever behind a turn that already ended (#115588)."""
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={"final_response": "done", "messages": []})
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn
        monkeypatch.setattr(mock_manager, "save_session", MagicMock(side_effect=RuntimeError("disk full")))

        with pytest.raises(RuntimeError, match="disk full"):
            await agent.prompt(prompt=[TextContentBlock(type="text", text="first")], session_id=resp.session_id)

        monkeypatch.setattr(mock_manager, "save_session", MagicMock())
        second = await agent.prompt(prompt=[TextContentBlock(type="text", text="second")], session_id=resp.session_id)

        assert second.stop_reason == "end_turn"
        assert state.queued_prompts == []
        assert state.agent.run_conversation.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor_raises", [False, True])
    async def test_prompt_fails_tool_calls_left_open_before_responding(self, agent, mock_manager, executor_raises):
        """A ``tool.started`` that never sees ``tool.completed`` (blocked/denied/crashed turn) must
        reach the client as a terminal ``failed`` update BEFORE the PromptResponse — on the normal
        return path and when the executor body itself raises."""
        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)
        state.agent.model, state.agent.provider = "test-model", "openrouter"
        events: list = []
        mock_conn = MagicMock(spec=acp.Client)

        async def _record(_sid, update):
            events.append(update)

        mock_conn.session_update = _record
        agent._conn = mock_conn

        def _turn(*args, **kwargs):
            state.agent.tool_progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
            if executor_raises:
                raise RuntimeError("executor blew up")
            return {"final_response": "ok", "messages": []}

        with patch.object(HermesACPAgent, "_run_agent_turn", side_effect=_turn):
            started = asyncio.get_running_loop().time()
            response = await asyncio.wait_for(
                agent.prompt(prompt=[TextContentBlock(type="text", text="hi")], session_id=resp.session_id),
                timeout=3,
            )
            seen_before_response = list(events)
            # Flushing on the loop thread stalled the loop for ``_send_update``'s 5s wait per call.
            assert asyncio.get_running_loop().time() - started < 4

        assert isinstance(response, PromptResponse)
        start = next(e for e in seen_before_response if isinstance(e, ToolCallStart))
        closes = [e for e in seen_before_response if isinstance(e, ToolCallProgress) and e.tool_call_id == start.tool_call_id]
        assert [e.status for e in closes] == ["failed"]


# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_replaces_connection_session_ownership(self, agent):
        old_conn = MagicMock(spec=acp.Client)
        new_conn = MagicMock(spec=acp.Client)
        agent._conn = old_conn
        agent._connected_session_ids.add("old-session")

        agent.on_connect(new_conn)

        assert agent._conn is new_conn
        assert agent._connected_session_ids == set()


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


    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result





    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        agent._handle_slash_command("/reset", state)
        assert len(state.history) == 0




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
        state.agent.tools = None
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force, **kwargs):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            assert force is True
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

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
            focus_topic=None,
            force=True,
            defer_context_engine_notification=True,
            task_id=state.session_id,
        )
        mock_save.assert_called_once_with(state.session_id)


    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None


    def test_slash_handler_cwd_pin_does_not_leak(self, agent, mock_manager, tmp_path):
        """The pin is scoped to the handler's own context copy.

        Concurrent ACP sessions share the event loop, so a handler that pinned
        the ambient context would leave its workspace bound for whatever runs
        next. Asserting the ambient value is unchanged after dispatch keeps the
        fix from trading one cross-session leak for another.
        """
        from agent.runtime_cwd import resolve_agent_cwd

        workspace = tmp_path / "project"
        workspace.mkdir()
        state = mock_manager.create_session(cwd=str(workspace))
        state.cwd = str(workspace)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        before = str(resolve_agent_cwd())
        agent._handle_slash_command("/help", state)
        assert str(resolve_agent_cwd()) == before





# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""


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
        pinned_cwd = {}
        def capture_register(config_map):
            from agent.runtime_cwd import resolve_context_cwd
            registered_config.update(config_map)
            pinned_cwd["value"] = resolve_context_cwd()  # the stdio default cwd reads this pin
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}
        # Registration runs under the session's logical cwd so IDE-provided stdio servers spawn there.
        assert pinned_cwd["value"] == Path("/tmp")


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
        state.agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "hindsight_recall", "description": "Recall", "parameters": {}}
            ]
        )

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "memory"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools is fake_tools
        assert state.agent.tools[-1] == {
            "type": "function",
            "function": {
                "name": "hindsight_recall",
                "description": "Recall",
                "parameters": {},
            },
        }
        assert state.agent.valid_tool_names == {
            "hindsight_recall",
            "memory",
            "mcp_srv_search",
            "terminal",
        }
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

        with patch("tools.mcp_tool_discovery.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])


class TestDisabledToolsetsFilterToolSurface:
    def test_cmd_tools_strips_configured_disabled_toolsets(self, agent, mock_manager):
        """``/tools`` lists what the session can call: a config-disabled toolset is absent (real get_tool_definitions)."""
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent._memory_manager = None

        def listed() -> set:
            return {line.strip().split(":", 1)[0] for line in agent._cmd_tools("", state).splitlines()[1:]}

        state.agent.disabled_toolsets = None
        assert "execute_code" in listed()
        state.agent.disabled_toolsets = ["code_execution"]
        assert "execute_code" not in listed()
