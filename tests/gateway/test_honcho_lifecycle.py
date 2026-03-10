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

    def test_gateway_skips_honcho_manager_in_local_mode(self):
        runner = _make_runner()
        hcfg = SimpleNamespace(
            enabled=True,
            api_key="honcho-key",
            ai_peer="hermes",
            peer_name="alice",
            peer_memory_mode=lambda peer: "local",
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
        runner.session_store = MagicMock()
        runner.session_store._generate_session_key.return_value = "gateway-key"
        runner.session_store._entries = {
            "gateway-key": SimpleNamespace(session_id="old-session"),
        }
        runner.session_store.reset_session.return_value = SimpleNamespace(session_id="new-session")

        result = await runner._handle_reset_command(event)

        runner._shutdown_gateway_honcho.assert_called_once_with("gateway-key")
        assert "Session reset" in result
