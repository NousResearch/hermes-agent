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
    (bridge_dir / "bridge.js").write_text("// current bridge code\n")
    (bridge_dir / "package.json").write_text('{"name": "bridge"}\n')
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}")
    return bridge_dir


def _fresh_node_modules(bridge_dir: Path) -> None:
    """Create node_modules with a stamp matching the current package.json."""
    from plugins.platforms.whatsapp.adapter import _file_content_hash

    nm = bridge_dir / "node_modules"
    nm.mkdir()
    (nm / ".hermes-pkg-hash").write_text(
        _file_content_hash(bridge_dir / "package.json")
    )


class TestFileContentHash:
    def test_hashes_file(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        f = tmp_path / "x.js"
        f.write_text("abc")
        h = _file_content_hash(f)
        assert len(h) == 16
        assert h == _file_content_hash(f)  # deterministic

    def test_changes_with_content(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        f = tmp_path / "x.js"
        f.write_text("abc")
        h1 = _file_content_hash(f)
        f.write_text("def")
        assert _file_content_hash(f) != h1

    def test_missing_file_returns_empty(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        assert _file_content_hash(tmp_path / "nope.js") == ""

    def test_matches_bridge_js_self_hash_algorithm(self, tmp_path):
        """Python and Node must compute the same hash for the same bytes."""
        import hashlib

        from plugins.platforms.whatsapp.adapter import _file_content_hash

        f = tmp_path / "bridge.js"
        f.write_bytes(b"const x = 1;\n")
        # Node side: createHash('sha256').update(bytes).digest('hex').slice(0, 16)
        expected = hashlib.sha256(b"const x = 1;\n").hexdigest()[:16]
        assert _file_content_hash(f) == expected


class TestStaleBridgeHandshake:
    @pytest.mark.asyncio
    async def test_reuses_bridge_when_hash_matches(self, tmp_path):
        from plugins.platforms.whatsapp.adapter import _file_content_hash

        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        disk_hash = _file_content_hash(bridge_dir / "bridge.js")
        mock_client = _mock_health({"status": "connected", "scriptHash": disk_hash})

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.create_task") as mock_task, \
             patch("subprocess.Popen") as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True), \
             patch.object(adapter, "_mark_connected", create=True):
            result = await adapter.connect()

        assert result is True
        mock_popen.assert_not_called()  # reused, never spawned
        mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_restarts_bridge_on_hash_mismatch(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        mock_client = _mock_health(
            {"status": "connected", "scriptHash": "deadbeefdeadbeef"}
        )
        # Spawned bridge dies immediately → connect() returns False, but the
        # assertion that matters is that the stale bridge was NOT reused and
        # a new process spawn was attempted.
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", mock_client), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process") as mock_kill_port, \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            result = await adapter.connect()

        assert result is False  # mock proc died; not the point of the test
        mock_popen.assert_called_once()  # stale bridge replaced, not reused
        mock_kill_port.assert_called_once_with(adapter._bridge_port)

    @pytest.mark.asyncio
    async def test_restarts_unversioned_bridge(self, tmp_path):
        """Bridges predating the handshake report no scriptHash → stale."""
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        # Old bridge /health payload: no scriptHash key at all
        mock_client = _mock_health({"status": "connected"})
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

    @pytest.mark.asyncio
    async def test_reinstalls_when_package_json_changed(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)
        _fresh_node_modules(bridge_dir)
        # Simulate `hermes update` bumping the Baileys pin
        (bridge_dir / "package.json").write_text('{"name": "bridge", "v": 2}\n')
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
             patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run, \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        mock_run.assert_called_once()
        assert "install" in mock_run.call_args[0][0]
        # Stamp updated to the new package.json hash
        from plugins.platforms.whatsapp.adapter import _file_content_hash
        stamp = (bridge_dir / "node_modules" / ".hermes-pkg-hash").read_text().strip()
        assert stamp == _file_content_hash(bridge_dir / "package.json")

    @pytest.mark.asyncio
    async def test_installs_when_node_modules_missing(self, tmp_path):
        bridge_dir = _setup_bridge_dir(tmp_path)  # no node_modules
        adapter = _make_adapter(
            bridge_script=str(bridge_dir / "bridge.js"),
            session_path=tmp_path / "session",
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        def _npm_install(*args, **kwargs):
            # npm creates node_modules as a side effect
            (bridge_dir / "node_modules").mkdir(exist_ok=True)
            return MagicMock(returncode=0)

        with patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=True), \
             patch("aiohttp.ClientSession", _mock_health({"status": "disconnected"})), \
             patch("plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock), \
             patch("plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"), \
             patch("plugins.platforms.whatsapp.adapter._kill_port_process"), \
             patch("subprocess.run", side_effect=_npm_install) as mock_run, \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch.object(adapter, "_acquire_platform_lock", return_value=True, create=True):
            await adapter.connect()

        mock_run.assert_called_once()


class TestCacheDirEnvPassthrough:
    @pytest.mark.asyncio
    async def test_bridge_spawn_env_has_cache_dirs(self, tmp_path):
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
        from gateway.platforms.base import (
            get_audio_cache_dir,
            get_document_cache_dir,
            get_image_cache_dir,
        )
        assert env["HERMES_IMAGE_CACHE_DIR"] == str(get_image_cache_dir())
        assert env["HERMES_AUDIO_CACHE_DIR"] == str(get_audio_cache_dir())
        assert env["HERMES_DOCUMENT_CACHE_DIR"] == str(get_document_cache_dir())
