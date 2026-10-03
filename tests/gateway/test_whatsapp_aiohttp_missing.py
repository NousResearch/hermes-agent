"""WhatsApp must name a missing aiohttp instead of looping forever (#126358).

A sealed env built by ``hermes update`` can ship a partial messaging extra
without aiohttp. The adapter's requirement check only verified Node.js, so
``connect()`` spawned the bridge and every health poll raised
``ModuleNotFoundError`` — swallowed by ``except Exception: continue`` — and
the gateway looped forever on the misleading
"Bridge HTTP server did not start in 15s".
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19877
    adapter._bridge_script = "/tmp/test-bridge-126358.js"
    adapter._session_path = Path("/tmp/test-wa-session-126358")
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


class TestMissingAiohttpNamed:
    @pytest.mark.asyncio
    async def test_connect_fails_fast_with_named_aiohttp_error(self):
        """Node present + aiohttp missing -> non-retryable fatal naming aiohttp, no bridge spawn."""
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = _make_adapter()

        with patch(
            "plugins.platforms.whatsapp.adapter.find_node_executable",
            return_value="/fake/node",
        ), patch.object(Path, "exists", return_value=True), patch(
            "subprocess.run", return_value=MagicMock(returncode=0)
        ), patch(
            "subprocess.Popen", return_value=MagicMock()
        ) as mock_popen, patch(
            "plugins.platforms.whatsapp.adapter.asyncio.sleep",
            new_callable=AsyncMock,
        ), patch.dict(
            sys.modules, {"aiohttp": None}
        ):
            result = await adapter.connect()

        assert result is False
        # Fail fast: the bridge process must never be spawned.
        mock_popen.assert_not_called()
        # Named error, not the misleading "did not start in 15s" loop.
        assert adapter.fatal_error_code == "whatsapp_aiohttp_missing"
        assert "aiohttp" in (adapter.fatal_error_message or "").lower()
        assert adapter.fatal_error_retryable is False

    def test_preflight_names_missing_aiohttp(self):
        """_preflight alone reports whatsapp_aiohttp_missing when the import fails."""
        adapter = _make_adapter()

        with patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ), patch.object(Path, "exists", return_value=True), patch.dict(
            sys.modules, {"aiohttp": None}
        ):
            ok = adapter._preflight()

        assert ok is False
        assert adapter.fatal_error_code == "whatsapp_aiohttp_missing"

    def test_preflight_passes_with_aiohttp(self):
        """With aiohttp importable the preflight keeps its Node/bridge/creds semantics."""
        adapter = _make_adapter()

        with patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ), patch.object(Path, "exists", return_value=True):
            ok = adapter._preflight()

        assert ok is True
        assert adapter.fatal_error_code is None
