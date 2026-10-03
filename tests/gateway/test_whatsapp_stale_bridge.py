"""Tests for the WhatsApp stale-bridge staleness handshake.

Regression tests for the stale-bridge trap: ``connect()`` reused any
already-running bridge with ``status: connected`` unconditionally, and
``disconnect()`` only kills bridges the adapter spawned itself.  A
long-lived bridge process therefore survived gateway restarts AND
``hermes update``, serving pre-update bridge.js behavior forever (e.g.
no inbound media download → images/voice notes arrive as placeholders).

The fix: bridge.js reports a hash of its own source in ``/health``
(``scriptHash``); the adapter compares it against the bridge.js on disk
and restarts the bridge on mismatch.  Bridges that predate the handshake
report no hash and are treated as stale by definition.

Also covers the npm dependency-refresh stamp: deps are reinstalled when
package.json changes, not only when node_modules is missing.
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



@pytest.fixture(autouse=True)
def _pm_node(monkeypatch):
    """Stand-in for PM's Node/npm; the user's PATH copy is never picked up."""
    from plugins.platforms.whatsapp import adapter as whatsapp_adapter
    monkeypatch.setattr(whatsapp_adapter, "find_node_executable", lambda name: f"/pm/{name}")


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
    adapter._dm_policy = adapter._group_policy = "pairing"
    adapter._allow_from = adapter._group_allow_from = set()
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
    """Create a real bridge dir with bridge.js + package.json + creds."""
    bridge_dir = tmp_path / "whatsapp-bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge.js").write_text("// current bridge code\n", encoding="utf-8")
    (bridge_dir / "package.json").write_text('{"name": "bridge"}\n', encoding="utf-8")
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}", encoding="utf-8")
    return bridge_dir


def _fresh_node_modules(bridge_dir: Path) -> None:
    """Create node_modules with a stamp matching the current package.json."""
    from plugins.platforms.whatsapp.adapter import _file_content_hash

    nm = bridge_dir / "node_modules"
    nm.mkdir()
    (nm / ".hermes-pkg-hash").write_text(
        _file_content_hash(bridge_dir / "package.json")
    )




class TestStaleBridgeHandshake:


    @pytest.mark.asyncio
    async def test_restarts_bridge_when_read_receipt_config_changed(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        adapter._send_read_receipts = True
        disk_hash = _file_content_hash(bridge_dir / "bridge.js")
        mock_client = _mock_health(
            {
                "status": "connected",
                "scriptHash": disk_hash,
                "sendReadReceipts": False,
            }
        )
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


class TestDepRefreshStamp:
    @pytest.mark.asyncio
    async def test_skips_install_when_stamp_fresh(self, tmp_path):
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
             patch("subprocess.run") as mock_run, \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        mock_run.assert_not_called()


class TestCacheDirEnvPassthrough:
    @pytest.mark.asyncio
    async def test_bridge_spawn_env_has_cache_dirs(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        adapter._send_read_receipts = True
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
        from gateway.platforms.base import (
            get_audio_cache_dir,
            get_document_cache_dir,
            get_image_cache_dir,
        )
        assert env["HERMES_IMAGE_CACHE_DIR"] == str(get_image_cache_dir())
        assert env["HERMES_AUDIO_CACHE_DIR"] == str(get_audio_cache_dir())
        assert env["HERMES_DOCUMENT_CACHE_DIR"] == str(get_document_cache_dir())
        assert env["WHATSAPP_SEND_READ_RECEIPTS"] == "true"


class TestConfigFingerprintAdoption:
    """Adoption must also verify the spawn-time config: bridge.js reads its policy/allowlist env
    once at startup, so adopting a bridge spawned with older env keeps gating DMs with the stale
    allowlist until the bridge is killed by hand (#126824)."""

    def _record_fingerprint(self, adapter, bridge_env: dict) -> None:
        from plugins.platforms.whatsapp.adapter import (
            _BRIDGE_FINGERPRINT_FILE, _bridge_config_fingerprint,
        )
        (adapter._session_path / _BRIDGE_FINGERPRINT_FILE).write_text(
            _bridge_config_fingerprint(bridge_env), encoding="utf-8"
        )

    async def _reuse(self, adapter, bridge_path):
        with patch.object(adapter, "_mark_connected"), \
             patch.object(adapter, "_attach_to_bridge"), \
             patch.object(adapter, "_wire_plugin_handlers"):
            return await adapter._reuse_running_bridge(bridge_path)

    @pytest.mark.asyncio
    async def test_adopts_bridge_when_fingerprint_matches(self, tmp_path, monkeypatch, capsys):
        for key in ("WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS", "WHATSAPP_MODE", "WHATSAPP_REPLY_PREFIX"):
            monkeypatch.delenv(key, raising=False)
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        self._record_fingerprint(adapter, adapter._bridge_env())

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _file_content_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False, "uptime": 230000.5})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is True

        assert "uptime: 2d 15h" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_restarts_bridge_when_allowlist_changed(self, tmp_path, monkeypatch, capsys):
        for key in ("WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS", "WHATSAPP_MODE", "WHATSAPP_REPLY_PREFIX"):
            monkeypatch.delenv(key, raising=False)
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        stale_env = dict(adapter._bridge_env())
        stale_env["WHATSAPP_DM_POLICY"] = "allowlist"
        stale_env["WHATSAPP_ALLOWED_USERS"] = "1111111111@s.whatsapp.net"
        self._record_fingerprint(adapter, stale_env)

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _file_content_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is False

        assert "Running bridge is stale (config changed), restarting" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_missing_fingerprint_file_is_stale(self, tmp_path):
        """A bridge from before the fingerprint existed (or a lost file) is never adopted."""
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )

        with patch("aiohttp.ClientSession", _mock_health(
            {"status": "connected", "scriptHash": _file_content_hash(bridge_dir / "bridge.js"),
             "sendReadReceipts": False})):
            assert await self._reuse(adapter, bridge_dir / "bridge.js") is False

    @pytest.mark.asyncio
    async def test_spawn_writes_fingerprint_of_spawn_env(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import (
            _BRIDGE_FINGERPRINT_FILE, _bridge_config_fingerprint,
        )

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

        recorded = (tmp_path / "session" / _BRIDGE_FINGERPRINT_FILE).read_text(encoding="utf-8").strip()
        assert recorded == _bridge_config_fingerprint(mock_popen.call_args.kwargs["env"])

    @pytest.mark.asyncio
    async def test_disconnect_keeps_fingerprint_file_for_adopted_bridge(self, tmp_path):
        """The adopted bridge keeps running with its recorded config; the next gateway start
        still needs the record to decide adoption, so disconnect() must not remove it."""
        from plugins.platforms.whatsapp.adapter import _BRIDGE_FINGERPRINT_FILE

        bridge_dir = _setup_bridge_dir(tmp_path)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        self._record_fingerprint(adapter, adapter._bridge_env())
        (tmp_path / "session" / "bridge.pid").write_text("4242\n12345", encoding="utf-8")
        adapter._bridge_process = None  # adopted: not managed by us
        adapter._poll_task = None

        with patch.object(adapter, "_release_platform_lock", create=True), \
             patch.object(adapter, "_mark_disconnected", create=True):
            await adapter.disconnect()

        assert not (tmp_path / "session" / "bridge.pid").exists()
        assert (tmp_path / "session" / _BRIDGE_FINGERPRINT_FILE).exists()


class TestFmtUptime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (None, ""),
            ("not-a-number", ""),
            (0, "0s"),
            (59.9, "59s"),
            (60, "1m"),
            (3661, "1h 1m"),
            (230000.5, "2d 15h"),
        ],
    )
    def test_compact_rendering(self, seconds, expected):
        from plugins.platforms.whatsapp.adapter import _fmt_uptime

        assert _fmt_uptime(seconds) == expected
