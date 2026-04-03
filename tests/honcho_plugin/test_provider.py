"""Tests for HonchoMemoryProvider session initialization and tool behavior."""

from unittest.mock import MagicMock, patch

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


class TestHonchoMemoryProviderInitialize:
    def test_initialize_uses_resolved_session_name_for_cli_sessions(self):
        provider = HonchoMemoryProvider()
        cfg = HonchoClientConfig(api_key="key", enabled=True)
        fake_manager = MagicMock()
        fake_manager.get_or_create.return_value = object()

        with (
            patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg),
            patch("plugins.memory.honcho.client.get_honcho_client", return_value=object()),
            patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=fake_manager),
            patch.object(cfg, "resolve_session_name", return_value="resolved-session") as mock_resolve,
            patch("os.getcwd", return_value="/tmp/project"),
        ):
            provider.initialize(session_id="raw-session-id", platform="cli", hermes_home="/tmp/hermes")

        assert provider._session_key == "resolved-session"
        mock_resolve.assert_called_once_with(cwd="/tmp/project", session_title=None, session_id="raw-session-id")
        fake_manager.get_or_create.assert_called_once_with("resolved-session")

    def test_initialize_keeps_platform_user_id_for_gateway_sessions(self):
        provider = HonchoMemoryProvider()
        cfg = HonchoClientConfig(api_key="key", enabled=True)
        fake_manager = MagicMock()
        fake_manager.get_or_create.return_value = object()

        with (
            patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg),
            patch("plugins.memory.honcho.client.get_honcho_client", return_value=object()),
            patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=fake_manager),
            patch.object(cfg, "resolve_session_name") as mock_resolve,
        ):
            provider.initialize(
                session_id="raw-session-id",
                platform="telegram",
                user_id="12345",
                hermes_home="/tmp/hermes",
            )

        assert provider._session_key == "telegram:12345"
        mock_resolve.assert_not_called()
        fake_manager.get_or_create.assert_called_once_with("telegram:12345")


class TestHonchoMemoryProviderHandleToolCall:
    def test_handle_tool_call_primes_session_cache_before_dispatch(self):
        provider = HonchoMemoryProvider()
        provider._session_key = "resolved-session"
        provider._session_initialized = True
        provider._manager = MagicMock()
        provider._manager.get_peer_card.return_value = ["fact one"]

        result = provider.handle_tool_call("honcho_profile", {})

        provider._manager.get_or_create.assert_called_once_with("resolved-session")
        provider._manager.get_peer_card.assert_called_once_with("resolved-session")
        assert "fact one" in result
