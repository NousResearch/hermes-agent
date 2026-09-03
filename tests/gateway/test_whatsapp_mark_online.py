"""Tests for the whatsapp mark_online config (#99829).

``send_read_receipts`` was dead config: the Baileys bridge hardwired
``markOnlineOnConnect: false``, so inbound messages arrived via the
offline-notification path, WhatsApp emitted no delivery receipt for them,
and the bridge's ``readMessages()`` call was discarded before a read
receipt could ever render (messages stuck at one grey tick, #27198).

These tests pin the fix: ``platforms.whatsapp.mark_online`` (default
``false`` — online presence is visible to contacts, so it stays opt-in)
flows config.yaml → ``PlatformConfig.extra`` → ``WhatsAppAdapter`` →
``WHATSAPP_MARK_ONLINE`` env → ``bridge.js markOnlineOnConnect``, with a
startup warning when ``send_read_receipts`` is enabled while presence is
off, and a stale-bridge restart when the running bridge's presence mode
no longer matches the config.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _mock_health(json_data):
    """Mock aiohttp.ClientSession whose GET returns 200 + *json_data*."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=_AsyncCM(mock_resp))
    mock_session.close = AsyncMock()
    return MagicMock(return_value=_AsyncCM(mock_session))


def _setup_bridge_dir(tmp_path: Path) -> Path:
    bridge_dir = tmp_path / "whatsapp-bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge.js").write_text("// current bridge code\n")
    (bridge_dir / "package.json").write_text('{"name": "bridge"}\n')
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}")
    return bridge_dir


def _fresh_node_modules(bridge_dir: Path) -> None:
    from plugins.platforms.whatsapp.adapter import _file_content_hash

    nm = bridge_dir / "node_modules"
    nm.mkdir()
    (nm / ".hermes-pkg-hash").write_text(
        _file_content_hash(bridge_dir / "package.json")
    )


def _make_adapter(bridge_script: str = "/tmp/test-bridge.js",
                  session_path: Path = Path("/tmp/test-wa-session")):
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = bridge_script
    adapter._session_path = session_path
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._send_read_receipts = False
    adapter._mark_online = False
    adapter._running = False
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = None
    return adapter


# ---------------------------------------------------------------------------
# Config bridging from config.yaml
# ---------------------------------------------------------------------------


class TestConfigYamlBridging:
    """whatsapp.mark_online in config.yaml flows into PlatformConfig.extra."""

    def test_mark_online_bridged_from_yaml(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("whatsapp:\n  mark_online: true\n")

        with patch("gateway.config.get_hermes_home", return_value=tmp_path):
            from gateway.config import load_gateway_config
            with patch.dict("os.environ", {"WHATSAPP_ENABLED": "true"}, clear=False):
                config = load_gateway_config()

        wa_config = config.platforms.get(Platform.WHATSAPP)
        assert wa_config is not None
        assert wa_config.extra.get("mark_online") is True

    def test_mark_online_absent_by_default(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("whatsapp:\n  reply_prefix: \"Bot\"\n")

        with patch("gateway.config.get_hermes_home", return_value=tmp_path):
            from gateway.config import load_gateway_config
            with patch.dict("os.environ", {"WHATSAPP_ENABLED": "true"}, clear=False):
                config = load_gateway_config()

        wa_config = config.platforms.get(Platform.WHATSAPP)
        assert wa_config is not None
        assert "mark_online" not in wa_config.extra


# ---------------------------------------------------------------------------
# WhatsAppAdapter __init__ parsing + startup warning
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_mark_online_true(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"mark_online": True}))
        assert adapter._mark_online is True

    def test_mark_online_default_false(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter._mark_online is False

    def test_mark_online_truthy_string(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"mark_online": "yes"}))
        assert adapter._mark_online is True

    def test_warns_when_receipts_enabled_without_presence(self, capsys):
        from gateway.config import PlatformConfig
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        WhatsAppAdapter(PlatformConfig(enabled=True, extra={"send_read_receipts": True}))
        out = capsys.readouterr().out
        assert "send_read_receipts=true has no effect" in out
        assert "mark_online" in out

    def test_no_warning_when_presence_enabled(self, capsys):
        from gateway.config import PlatformConfig
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        WhatsAppAdapter(PlatformConfig(enabled=True, extra={
            "send_read_receipts": True, "mark_online": True,
        }))
        out = capsys.readouterr().out
        assert "has no effect" not in out

    def test_no_warning_when_receipts_disabled(self, capsys):
        from gateway.config import PlatformConfig
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
        out = capsys.readouterr().out
        assert "has no effect" not in out


# ---------------------------------------------------------------------------
# Bridge subprocess env bridging
# ---------------------------------------------------------------------------


class TestBridgeEnvBridging:
    @pytest.mark.asyncio
    async def test_bridge_env_carries_mark_online(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        adapter._mark_online = True
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", _mock_health({"status": "disconnected"})), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        env = mock_popen.call_args.kwargs["env"]
        assert env["WHATSAPP_MARK_ONLINE"] == "true"

    @pytest.mark.asyncio
    async def test_bridge_env_defaults_mark_online_false(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", _mock_health({"status": "disconnected"})), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        env = mock_popen.call_args.kwargs["env"]
        assert env["WHATSAPP_MARK_ONLINE"] == "false"


# ---------------------------------------------------------------------------
# Stale-bridge handshake: presence mismatch restarts, match reuses
# ---------------------------------------------------------------------------


class TestStaleBridgeHandshake:

    @pytest.mark.asyncio
    async def test_restarts_bridge_when_mark_online_config_changed(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        adapter._mark_online = True
        disk_hash = _file_content_hash(bridge_dir / "bridge.js")
        # Running bridge: same code hash, receipts match, but offline mode.
        mock_client = _mock_health({
            "status": "connected",
            "scriptHash": disk_hash,
            "sendReadReceipts": False,
            "markOnline": False,
        })
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        mock_popen.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_bridge_when_mark_online_matches(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        adapter._send_read_receipts = True
        adapter._mark_online = True
        disk_hash = _file_content_hash(bridge_dir / "bridge.js")
        # Running bridge: same code hash and both flags match the config.
        mock_client = _mock_health({
            "status": "connected",
            "scriptHash": disk_hash,
            "sendReadReceipts": True,
            "markOnline": True,
        })
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True), \
             patch.object(adapter, "_mark_connected", create=True), \
             patch.object(adapter, "_wire_plugin_handlers", create=True), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.create_task"):
            result = await adapter.connect()

        assert result is True
        mock_popen.assert_not_called()
        assert adapter._bridge_process is None  # reused, not managed by us
