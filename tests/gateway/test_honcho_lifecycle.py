"""Tests for gateway-owned Honcho lifecycle helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


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
    runner.hooks.emit = AsyncMock()
    return runner


def _make_event(text="/reset"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="user-1",
            user_name="alice",
        ),
    )


class TestGatewayHonchoLifecycle:
    def test_gateway_reuses_honcho_manager_for_session_key(self):
        runner = _make_runner()
        hcfg = SimpleNamespace(
            enabled=True,
            api_key="honcho-key",
            ai_peer="hermes",
            peer_name="alice",
            context_tokens=123,
            peer_memory_mode=lambda peer: "hybrid",
        )
        manager = MagicMock()

        with (
            patch("honcho_integration.client.HonchoClientConfig.from_global_config", return_value=hcfg),
            patch("honcho_integration.client.get_honcho_client", return_value=MagicMock()),
            patch("honcho_integration.session.HonchoSessionManager", return_value=manager) as mock_mgr_cls,
        ):
            first_mgr, first_cfg = runner._get_or_create_gateway_honcho("session-key")
            second_mgr, second_cfg = runner._get_or_create_gateway_honcho("session-key")

        assert first_mgr is manager
        assert second_mgr is manager
        assert first_cfg is hcfg
        assert second_cfg is hcfg
        mock_mgr_cls.assert_called_once()

    def test_gateway_skips_honcho_manager_when_disabled(self):
        runner = _make_runner()
        hcfg = SimpleNamespace(
            enabled=False,
            api_key="honcho-key",
            ai_peer="hermes",
            peer_name="alice",
        )

        with (
            patch("honcho_integration.client.HonchoClientConfig.from_global_config", return_value=hcfg),
            patch("honcho_integration.client.get_honcho_client") as mock_client,
            patch("honcho_integration.session.HonchoSessionManager") as mock_mgr_cls,
        ):
            manager, cfg = runner._get_or_create_gateway_honcho("session-key")

        assert manager is None
        assert cfg is hcfg
        mock_client.assert_not_called()
        mock_mgr_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_shuts_down_gateway_honcho_manager(self):
        runner = _make_runner()
        event = _make_event()
        runner._shutdown_gateway_honcho = MagicMock()
        runner._async_flush_memories = AsyncMock()
        runner.session_store = MagicMock()
        runner.session_store._generate_session_key.return_value = "gateway-key"
        runner.session_store._entries = {
            "gateway-key": SimpleNamespace(session_id="old-session"),
        }
        runner.session_store.reset_session.return_value = SimpleNamespace(session_id="new-session")

        result = await runner._handle_reset_command(event)

        runner._shutdown_gateway_honcho.assert_called_once_with("gateway-key")
        runner._async_flush_memories.assert_called_once_with("old-session", "gateway-key")
        assert "Session reset" in result

    def test_flush_memories_reuses_gateway_session_key_and_skips_honcho_sync(self):
        runner = _make_runner()
        runner.session_store = MagicMock()
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        tmp_agent = MagicMock()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
            patch("gateway.run._resolve_gateway_model", return_value="model-name"),
            patch("run_agent.AIAgent", return_value=tmp_agent) as mock_agent_cls,
        ):
            runner._flush_memories_for_session("old-session", "gateway-key")

        mock_agent_cls.assert_called_once()
        _, kwargs = mock_agent_cls.call_args
        assert kwargs["session_id"] == "old-session"
        assert kwargs["honcho_session_key"] == "gateway-key"
        tmp_agent.run_conversation.assert_called_once()
        _, run_kwargs = tmp_agent.run_conversation.call_args
        assert run_kwargs["sync_honcho"] is False
