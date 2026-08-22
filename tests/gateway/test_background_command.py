"""Tests for /background gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.session import (
    TELEGRAM_FOREGROUND_ROUTE_METADATA,
    SessionSource,
    build_session_key,
)


def _make_event(text="/background", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner with minimal mocks."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()

    mock_store = MagicMock()
    # A real SessionStore returns None when no persisted /model override exists.
    # MagicMock's default truthy return would otherwise rehydrate a fake model
    # and make the session-scoped reasoning resolver receive a MagicMock.
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()

    return runner


# ---------------------------------------------------------------------------
# _handle_background_command
# ---------------------------------------------------------------------------


class TestHandleBackgroundCommand:
    """Tests for GatewayRunner._handle_background_command."""

    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self):
        """Running /background with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result
        assert "/background" in result

    @pytest.mark.asyncio
    async def test_bg_alias_no_prompt_shows_usage(self):
        """Running /bg with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_shows_usage(self):
        """Running /background with only whitespace shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background   ")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_bare_busy_bg_detaches_exact_telegram_turn(self):
        runner = _make_runner()
        event = _make_event(text="/bg")
        physical_key = build_session_key(event.source)
        running_agent = MagicMock(session_id="20260822_020618_51467760")
        runner._running_agents[physical_key] = running_agent
        routed_entry = MagicMock(
            session_key=f"{physical_key}:route:fg_abc123",
            session_id="fresh-foreground-session",
        )
        runner.session_store._generate_session_key.return_value = physical_key
        runner._async_session_store = MagicMock()
        runner._async_session_store._store = runner.session_store
        runner._async_session_store.get_or_create_session = AsyncMock(
            return_value=routed_entry
        )
        runner._async_session_store.set_session_metadata = AsyncMock(
            return_value=True
        )
        runner._cache_session_source = MagicMock()

        result = await runner._busy_background_command(
            event, physical_key, event.source
        )

        assert "background" in result
        assert "20260822_020618_51467760" in result
        assert runner._running_agents[physical_key] is running_agent
        routed_source = (
            runner._async_session_store.get_or_create_session.await_args.args[0]
        )
        assert routed_source.session_route_id.startswith("fg_")
        runner._async_session_store.set_session_metadata.assert_awaited_once_with(
            physical_key,
            TELEGRAM_FOREGROUND_ROUTE_METADATA,
            routed_source.session_route_id,
        )

    @pytest.mark.asyncio
    async def test_busy_bg_with_prompt_still_spawns_isolated_task(self):
        runner = _make_runner()
        event = _make_event(text="/bg investigate this")
        runner._handle_background_command = AsyncMock(return_value="started")

        result = await runner._busy_background_command(
            event, build_session_key(event.source), event.source
        )

        assert result == "started"
        runner._handle_background_command.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_back_moves_named_current_agent_to_fresh_route(self):
        runner = _make_runner()
        event = _make_event(text="/back sess-live")
        physical_key = build_session_key(event.source)
        running_agent = MagicMock(session_id="sess-live")
        runner._running_agents[physical_key] = running_agent
        routed_entry = MagicMock(
            session_key=f"{physical_key}:route:fg_new",
            session_id="fresh-session",
        )
        runner.session_store._generate_session_key.side_effect = build_session_key
        runner._async_session_store = MagicMock(_store=runner.session_store)
        runner._async_session_store.get_or_create_session = AsyncMock(
            return_value=routed_entry
        )
        runner._async_session_store.set_session_metadata = AsyncMock(
            return_value=True
        )
        runner._cache_session_source = MagicMock()

        result = await runner._handle_back_command(event)

        assert "sess-live" in result
        assert "background" in result
        assert runner._running_agents[physical_key] is running_agent

    @pytest.mark.asyncio
    async def test_front_routes_chat_to_named_background_agent(self):
        runner = _make_runner()
        physical_source = _make_event(text="unused").source
        physical_key = build_session_key(physical_source)
        old_route = "fg_old"
        new_route = "fg_new"
        background_key = f"{physical_key}:route:{old_route}"
        event = _make_event(text="/front sess-background")
        event.source = dataclasses.replace(event.source, session_route_id=new_route)
        running_agent = MagicMock(session_id="sess-background")
        runner._running_agents[background_key] = running_agent
        runner.session_store._generate_session_key.side_effect = build_session_key
        runner._async_session_store = MagicMock(_store=runner.session_store)
        runner._async_session_store.set_session_metadata = AsyncMock(
            return_value=True
        )

        result = await runner._handle_front_command(event)

        assert "foreground" in result
        runner._async_session_store.set_session_metadata.assert_awaited_once_with(
            physical_key,
            TELEGRAM_FOREGROUND_ROUTE_METADATA,
            old_route,
        )

    @pytest.mark.asyncio
    async def test_front_rejects_agent_from_another_chat(self):
        runner = _make_runner()
        event = _make_event(text="/front foreign")
        foreign_source = _make_event(chat_id="other-chat").source
        foreign_key = build_session_key(foreign_source)
        runner._running_agents[foreign_key] = MagicMock(session_id="foreign")
        runner.session_store._generate_session_key.side_effect = build_session_key

        result = await runner._handle_front_command(event)

        assert "different chat" in result


class TestDetachedTelegramRouteIngress:
    def test_persisted_route_rekeys_external_messages_before_guard(self):
        adapter = MagicMock()
        adapter.config = MagicMock()
        adapter.config.extra = {
            "group_sessions_per_user": True,
            "thread_sessions_per_user": False,
        }
        adapter._owner_profile = None
        adapter._session_key_profile.return_value = None
        adapter._session_store = MagicMock()
        adapter._session_store._resolve_profile_for_key.return_value = None
        adapter._session_store.get_session_metadata.return_value = "fg_abc123"
        event = _make_event(text="new foreground work")
        physical_key = build_session_key(event.source)

        BasePlatformAdapter._apply_persisted_session_route(adapter, event)

        assert event.source.session_route_id == "fg_abc123"
        assert build_session_key(event.source).endswith(":route:fg_abc123")
        adapter._session_store.get_session_metadata.assert_called_once_with(
            physical_key,
            TELEGRAM_FOREGROUND_ROUTE_METADATA,
            "",
        )

    def test_strict_internal_event_keeps_own_route(self):
        adapter = MagicMock()
        adapter._session_store = MagicMock()
        event = _make_event(text="completion")
        event.internal = True
        event.metadata = {"gateway_session_key": build_session_key(event.source)}

        BasePlatformAdapter._apply_persisted_session_route(adapter, event)

        assert event.source.session_route_id is None
        adapter._session_store.get_session_metadata.assert_not_called()

    def test_detached_route_bypasses_physical_topic_binding(self):
        runner = _make_runner()
        runner._telegram_topic_mode_enabled = MagicMock(return_value=True)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="67890",
            chat_type="dm",
            user_id="12345",
            thread_id="42",
            session_route_id="fg_abc123",
        )

        assert runner._is_telegram_topic_lane(source) is False
        assert runner._is_telegram_topic_root_lobby(source) is False


# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    """Tests for GatewayRunner._run_background_task (the actual execution)."""


    @pytest.mark.asyncio
    async def test_no_credentials_sends_error(self):
        """When provider credentials are missing, an error is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_background_task("test prompt", source, "bg_test")

        # Should have sent an error message
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert "failed" in call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "").lower()

    @pytest.mark.asyncio
    async def test_successful_task_sends_result(self):
        """When the agent completes successfully, the result is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Hello from background!"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        mock_result = {"final_response": "Hello from background!", "messages": []}

        checkpoint_config = {
            "checkpoints": {
                "enabled": True,
                "max_snapshots": 8,
                "max_total_size_mb": 222,
                "max_file_size_mb": 3,
            }
        }
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value=checkpoint_config), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.shutdown_memory_provider = MagicMock()
            mock_agent_instance.close = MagicMock()
            mock_agent_instance.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task("say hello", source, "bg_test")

        # Should have sent the result
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "Background task complete" in content
        assert "Hello from background!" in content
        agent_kwargs = MockAgent.call_args.kwargs
        assert agent_kwargs["checkpoints_enabled"] is True
        assert agent_kwargs["checkpoint_max_snapshots"] == 8
        assert agent_kwargs["checkpoint_max_total_size_mb"] == 222
        assert agent_kwargs["checkpoint_max_file_size_mb"] == 3
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# /background in help and known_commands
# ---------------------------------------------------------------------------


class TestBackgroundInHelp:
    """Verify /background appears in help text and known commands."""

    @pytest.mark.asyncio
    async def test_background_in_help_output(self):
        """The /help output includes /background."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/background" in result


# ---------------------------------------------------------------------------
# CLI /background command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /background is registered in the CLI command system."""


    def test_background_autocompletes(self):
        """The /background command appears in autocomplete results."""
        pytest.importorskip("prompt_toolkit")
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("backgro")  # Partial match
        completions = list(completer.get_completions(doc, None))
        # Text doesn't start with / so no completions
        assert len(completions) == 0

        doc = Document("/backgro")  # With slash prefix
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/background" in d for d in cmd_displays)
