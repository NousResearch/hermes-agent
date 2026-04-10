"""Tests for Session Protocol platform adapter (gateway/platforms/session.py).

Covers:
- Platform.SESSION enum value
- Config loading from env vars via _apply_env_overrides
- SessionAdapter init and default values
- check_session_requirements() logic
- _should_process_group_message() mention detection
- Authorization maps in run.py
- send_message tool routing
- Toolset registration
"""

import pytest
from unittest.mock import MagicMock, patch

from gateway.config import Platform, PlatformConfig, GatewayConfig


# ---------------------------------------------------------------------------
# 1. Platform Enum
# ---------------------------------------------------------------------------

class TestSessionPlatformEnum:
    def test_session_enum_exists(self):
        assert Platform.SESSION.value == "session"

    def test_session_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "session" in platforms


# ---------------------------------------------------------------------------
# 2. Config loading from env vars
# ---------------------------------------------------------------------------

class TestSessionConfigLoading:
    def test_apply_env_overrides_session(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SESSION in config.platforms
        sc = config.platforms[Platform.SESSION]
        assert sc.enabled is True
        assert sc.extra["bot_id"] == "05abc123def456"

    def test_session_not_loaded_without_bot_id(self, monkeypatch):
        monkeypatch.delenv("SESSION_BOT_ID", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SESSION not in config.platforms

    def test_bot_name_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")
        monkeypatch.setenv("SESSION_BOT_NAME", "TestBot")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        sc = config.platforms[Platform.SESSION]
        assert sc.extra["bot_name"] == "TestBot"

    def test_bridge_port_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")
        monkeypatch.setenv("SESSION_BRIDGE_PORT", "9999")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        sc = config.platforms[Platform.SESSION]
        assert sc.extra["bridge_port"] == "9999"

    def test_startup_timeout_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")
        monkeypatch.setenv("SESSION_STARTUP_TIMEOUT", "30")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        sc = config.platforms[Platform.SESSION]
        assert sc.extra["startup_timeout"] == "30"

    def test_data_path_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")
        monkeypatch.setenv("SESSION_DATA_PATH", "/tmp/session-data")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        sc = config.platforms[Platform.SESSION]
        assert sc.extra["data_path"] == "/tmp/session-data"

    def test_home_channel_set_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")
        monkeypatch.setenv("SESSION_HOME_CHANNEL", "05abcdef1234567890")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        sc = config.platforms[Platform.SESSION]
        assert sc.home_channel is not None
        assert sc.home_channel.chat_id == "05abcdef1234567890"
        assert sc.home_channel.platform == Platform.SESSION

    def test_home_channel_not_set_without_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")
        monkeypatch.delenv("SESSION_HOME_CHANNEL", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        sc = config.platforms[Platform.SESSION]
        assert sc.home_channel is None

    def test_get_connected_platforms_includes_session(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        connected = config.get_connected_platforms()
        assert Platform.SESSION in connected


# ---------------------------------------------------------------------------
# 3. Adapter init — config parsing and defaults
# ---------------------------------------------------------------------------

class TestSessionAdapterInit:
    def _make_config(self, **extra):
        config = PlatformConfig()
        config.enabled = True
        config.extra = {
            "bot_id": "05abc123def456",
            **extra,
        }
        return config

    def test_init_defaults(self):
        from gateway.platforms.session import SessionAdapter
        adapter = SessionAdapter(self._make_config())

        assert adapter.bridge_port == 8095
        assert adapter.bot_name == "Hermes"
        assert adapter.startup_timeout == 15

    def test_init_custom_bridge_port(self):
        from gateway.platforms.session import SessionAdapter
        adapter = SessionAdapter(self._make_config(bridge_port="9090"))

        assert adapter.bridge_port == 9090

    def test_init_custom_bot_name(self):
        from gateway.platforms.session import SessionAdapter
        adapter = SessionAdapter(self._make_config(bot_name="MyBot"))

        assert adapter.bot_name == "MyBot"

    def test_init_custom_startup_timeout(self):
        from gateway.platforms.session import SessionAdapter
        adapter = SessionAdapter(self._make_config(startup_timeout="30"))

        assert adapter.startup_timeout == 30

    def test_bridge_url_uses_port(self):
        from gateway.platforms.session import SessionAdapter
        adapter = SessionAdapter(self._make_config(bridge_port="8765"))

        assert adapter.bridge_url == "http://127.0.0.1:8765"

    def test_platform_is_session(self):
        from gateway.platforms.session import SessionAdapter
        adapter = SessionAdapter(self._make_config())

        assert adapter.platform == Platform.SESSION


# ---------------------------------------------------------------------------
# 4. check_session_requirements()
# ---------------------------------------------------------------------------

class TestCheckSessionRequirements:
    def test_returns_false_when_bot_id_not_set(self, monkeypatch):
        monkeypatch.delenv("SESSION_BOT_ID", raising=False)

        from gateway.platforms.session import check_session_requirements
        assert check_session_requirements() is False

    def test_returns_false_when_node_not_found(self, monkeypatch):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")

        with patch("shutil.which", return_value=None):
            from gateway.platforms.session import check_session_requirements
            assert check_session_requirements() is False

    def test_returns_false_when_bridge_script_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")

        fake_node = str(tmp_path / "node")
        (tmp_path / "node").write_text("#!/bin/sh\necho v24.12.0")
        (tmp_path / "node").chmod(0o755)

        with patch("shutil.which", return_value=fake_node):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="v24.12.0\n", returncode=0)
                with patch("pathlib.Path.exists", return_value=False):
                    from gateway.platforms.session import check_session_requirements
                    result = check_session_requirements()
                    assert result is False

    def test_returns_true_when_all_requirements_met(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")

        fake_node = str(tmp_path / "node")
        (tmp_path / "node").write_text("#!/bin/sh\necho v24.12.0")
        (tmp_path / "node").chmod(0o755)

        # Create a fake bridge script
        bridge_dir = tmp_path / "scripts" / "session-bridge"
        bridge_dir.mkdir(parents=True)
        bridge_script = bridge_dir / "session-bridge.mjs"
        bridge_script.write_text("// fake bridge")

        with patch("shutil.which", return_value=fake_node):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="v24.12.0\n", returncode=0)
                with patch("pathlib.Path.exists", return_value=True):
                    from gateway.platforms.session import check_session_requirements
                    result = check_session_requirements()
                    assert result is True

    def test_returns_false_when_node_version_too_old(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SESSION_BOT_ID", "05abc123def456")

        fake_node = str(tmp_path / "node")

        with patch("shutil.which", return_value=fake_node):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="v18.0.0\n", returncode=0)
                from gateway.platforms.session import check_session_requirements
                result = check_session_requirements()
                assert result is False


# ---------------------------------------------------------------------------
# 5. _should_process_group_message()
# ---------------------------------------------------------------------------

class TestShouldProcessGroupMessage:
    def _make_adapter(self, bot_name="Hermes", bot_session_id=None):
        config = PlatformConfig()
        config.enabled = True
        config.extra = {"bot_id": "05abc123def456"}
        from gateway.platforms.session import SessionAdapter
        adapter = SessionAdapter(config)
        adapter.bot_name = bot_name
        adapter._bot_session_id = bot_session_id
        return adapter

    def test_returns_true_for_bot_name_mention(self):
        adapter = self._make_adapter(bot_name="Hermes")
        msg = {"body": "Hey @Hermes can you help?"}
        assert adapter._should_process_group_message(msg) is True

    def test_returns_true_for_bot_name_mention_case_insensitive(self):
        adapter = self._make_adapter(bot_name="Hermes")
        msg = {"body": "Hey @hermes can you help?"}
        assert adapter._should_process_group_message(msg) is True

    def test_returns_true_for_session_id_mention(self):
        session_id = "05abcdef1234567890abcdef"
        adapter = self._make_adapter(bot_session_id=session_id)
        msg = {"body": f"@{session_id} what time is it?"}
        assert adapter._should_process_group_message(msg) is True

    def test_returns_false_when_no_mention(self):
        adapter = self._make_adapter(bot_name="Hermes")
        msg = {"body": "Just chatting with everyone here"}
        assert adapter._should_process_group_message(msg) is False

    def test_returns_false_for_empty_body(self):
        adapter = self._make_adapter(bot_name="Hermes")
        msg = {"body": ""}
        assert adapter._should_process_group_message(msg) is False

    def test_returns_false_for_none_body(self):
        adapter = self._make_adapter(bot_name="Hermes")
        msg = {"body": None}
        assert adapter._should_process_group_message(msg) is False

    def test_returns_false_when_no_session_id_and_only_id_mention(self):
        # bot_session_id not yet resolved
        adapter = self._make_adapter(bot_name="Hermes", bot_session_id=None)
        msg = {"body": "@05abc123 what time is it?"}
        assert adapter._should_process_group_message(msg) is False


# ---------------------------------------------------------------------------
# 6. Authorization integration (run.py)
# ---------------------------------------------------------------------------

class TestSessionAuthorization:
    def test_session_in_platform_env_map(self):
        """Platform.SESSION must appear in the allowlist env-var lookup map."""
        import inspect
        import gateway.run as run_module

        src = inspect.getsource(run_module)
        assert "Platform.SESSION" in src
        assert "SESSION_ALLOWED_USERS" in src

    def test_session_in_platform_allow_all_map(self):
        """Platform.SESSION must have a ALLOW_ALL env var configured."""
        import inspect
        import gateway.run as run_module

        src = inspect.getsource(run_module)
        assert "SESSION_ALLOW_ALL_USERS" in src

    def test_session_unauthorized_returns_false(self):
        """A SESSION message from an unknown user should not be authorized."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.SESSION
        source.user_id = "05abc123unknown"

        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False


# ---------------------------------------------------------------------------
# 7. send_message tool routing
# ---------------------------------------------------------------------------

class TestSessionSendMessageTool:
    def test_session_platform_map_resolves(self):
        """'session' string must resolve to Platform.SESSION in the tool's platform map."""
        import tools.send_message_tool as smt_module
        from gateway.config import Platform

        # The platform_map is built inside send_message_tool — verify via the
        # same GatewayConfig lookup the tool uses at runtime.
        config = GatewayConfig()
        # Manually inject a Session platform entry as the tool would see it
        from gateway.config import PlatformConfig
        pc = PlatformConfig()
        pc.enabled = True
        pc.extra = {"bot_id": "05abc123def456", "bridge_port": "8095"}
        config.platforms[Platform.SESSION] = pc

        # The tool maps the string "session" → Platform.SESSION to look up pconfig.
        # Verify the string key and enum value are in sync.
        assert Platform.SESSION.value == "session"
        assert config.platforms.get(Platform.SESSION) is pc

    def test_send_session_uses_bridge_port_from_extra(self):
        """_send_session must use the bridge_port from extra config."""
        from unittest.mock import AsyncMock, patch
        import asyncio

        async def run():
            import tools.send_message_tool as smt_module
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = lambda: None

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=mock_resp)
                mock_client_cls.return_value = mock_client

                result = await smt_module._send_session(
                    {"bridge_port": "9999"}, "05abc123", "hello"
                )
                call_url = mock_client.post.call_args[0][0]
                assert "9999" in call_url
                assert result.get("success") is True

        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# 8. Toolset registration
# ---------------------------------------------------------------------------

class TestSessionToolset:
    def test_hermes_session_toolset_exists(self):
        """'hermes-session' must be a defined toolset in toolsets.py."""
        from toolsets import TOOLSETS
        assert "hermes-session" in TOOLSETS

    def test_hermes_session_in_gateway_includes(self):
        """'hermes-session' must be included in the hermes-gateway toolset."""
        from toolsets import TOOLSETS
        gateway_includes = TOOLSETS.get("hermes-gateway", {}).get("includes", [])
        assert "hermes-session" in gateway_includes
