"""Tests for gateway-owned RetainDB lifecycle helpers."""

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
    runner._retaindb_managers = {}
    runner._retaindb_configs = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    return runner


def _make_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="alice",
    )


def _make_event(text="/reset"):
    return MessageEvent(text=text, source=_make_source())


class TestGatewayRetainDBLifecycle:
    def test_gateway_reuses_retaindb_manager_for_session_key(self):
        runner = _make_runner()
        identity_one = {"platform_user_id": "user-1", "session_id": "session-1"}
        identity_two = {"platform_user_id": "user-1", "session_id": "session-2"}
        rcfg = SimpleNamespace(should_activate=lambda: True)
        manager = MagicMock()

        with patch(
            "retaindb_integration.client.RetainDBClientConfig.from_global_config",
            return_value=rcfg,
        ), patch(
            "retaindb_integration.session.RetainDBSessionManager",
            return_value=manager,
        ) as mock_mgr_cls:
            first_mgr, first_cfg = runner._get_or_create_gateway_retaindb("session-key", identity_one)
            second_mgr, second_cfg = runner._get_or_create_gateway_retaindb("session-key", identity_two)

        assert first_mgr is manager
        assert second_mgr is manager
        assert first_cfg is rcfg
        assert second_cfg is rcfg
        mock_mgr_cls.assert_called_once_with(config=rcfg, runtime_identity=identity_one)
        manager.set_runtime_identity.assert_called_once_with(identity_two)

    def test_gateway_skips_retaindb_manager_when_inactive(self):
        runner = _make_runner()
        rcfg = SimpleNamespace(should_activate=lambda: False)

        with patch(
            "retaindb_integration.client.RetainDBClientConfig.from_global_config",
            return_value=rcfg,
        ), patch("retaindb_integration.session.RetainDBSessionManager") as mock_mgr_cls:
            manager, cfg = runner._get_or_create_gateway_retaindb("session-key")

        assert manager is None
        assert cfg is rcfg
        mock_mgr_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_shuts_down_gateway_retaindb_manager(self):
        runner = _make_runner()
        event = _make_event()
        origin = _make_source()
        runner._shutdown_gateway_honcho = MagicMock()
        runner._shutdown_gateway_retaindb = MagicMock()
        runner._async_flush_memories = AsyncMock()
        runner._evict_cached_agent = MagicMock()
        runner.session_store = MagicMock()
        runner.session_store._generate_session_key.return_value = "gateway-key"
        runner.session_store._entries = {
            "gateway-key": SimpleNamespace(session_id="old-session", origin=origin),
        }
        runner.session_store.reset_session.return_value = SimpleNamespace(session_id="new-session")

        result = await runner._handle_reset_command(event)

        runner._shutdown_gateway_retaindb.assert_called_once_with("gateway-key")
        runner._async_flush_memories.assert_called_once()
        assert runner._async_flush_memories.call_args.args == ("old-session", "gateway-key", origin)
        assert "Session reset" in result

    def test_flush_memories_passes_retaindb_identity_and_session_key(self):
        runner = _make_runner()
        runner.session_store = MagicMock()
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        runner._get_or_create_gateway_retaindb = MagicMock(
            return_value=(MagicMock(), SimpleNamespace())
        )
        tmp_agent = MagicMock()
        source = _make_source()

        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
            patch("gateway.run._resolve_gateway_model", return_value="model-name"),
            patch("run_agent.AIAgent", return_value=tmp_agent) as mock_agent_cls,
        ):
            runner._flush_memories_for_session("old-session", "gateway-key", source)

        _, kwargs = mock_agent_cls.call_args
        assert kwargs["session_id"] == "old-session"
        assert kwargs["retaindb_session_key"] == "old-session"
        assert kwargs["retaindb_identity"]["platform_user_id"] == "user-1"
        assert kwargs["retaindb_identity"]["session_id"] == "old-session"
        runner._get_or_create_gateway_retaindb.assert_called_once()
