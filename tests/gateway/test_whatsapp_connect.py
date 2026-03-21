"""Tests for WhatsApp connect() error handling.

Regression tests for two bugs in WhatsAppAdapter.connect():

1. Uninitialized ``data`` variable: when ``resp.json()`` raised after the
   health endpoint returned HTTP 200, ``http_ready`` was set to True but
   ``data`` was never assigned.  The subsequent ``data.get("status")``
   check raised ``NameError``.

2. Bridge log file handle leaked on error paths: the file was opened before
   the health-check loop but never closed when ``connect()`` returned False.
   Repeated connection failures accumulated open file descriptors.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = Path("/tmp/test-wa-session")
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
    return adapter


def _mock_aiohttp(status=200, json_data=None, json_side_effect=None):
    """Build a mock ``aiohttp.ClientSession`` returning a fixed response."""
    mock_resp = MagicMock()
    mock_resp.status = status
    if json_side_effect:
        mock_resp.json = AsyncMock(side_effect=json_side_effect)
    else:
        mock_resp.json = AsyncMock(return_value=json_data or {})

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=_AsyncCM(mock_resp))

    return MagicMock(return_value=_AsyncCM(mock_session))


def _connect_patches(mock_proc, mock_fh, mock_client_cls=None):
    """Return a dict of common patches needed to reach the health-check loop."""
    patches = {
        "gateway.platforms.whatsapp.check_whatsapp_requirements": True,
        "gateway.platforms.whatsapp.asyncio.create_task": MagicMock(),
    }
    base = [
        patch("gateway.platforms.whatsapp.check_whatsapp_requirements", return_value=True),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "mkdir", return_value=None),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("builtins.open", return_value=mock_fh),
        patch("gateway.platforms.whatsapp.asyncio.sleep", new_callable=AsyncMock),
        patch("gateway.platforms.whatsapp.asyncio.create_task"),
    ]
    if mock_client_cls is not None:
        base.append(patch("aiohttp.ClientSession", mock_client_cls))
    return base


# ---------------------------------------------------------------------------
# _close_bridge_log() unit tests
# ---------------------------------------------------------------------------

class TestCloseBridgeLog:
    """Direct tests for the _close_bridge_log() helper method."""

    @staticmethod
    def _bare_adapter():
        from gateway.platforms.whatsapp import WhatsAppAdapter
        a = WhatsAppAdapter.__new__(WhatsAppAdapter)
        a._bridge_log_fh = None
        return a

    def test_closes_open_handle(self):
        adapter = self._bare_adapter()
        mock_fh = MagicMock()
        adapter._bridge_log_fh = mock_fh

        adapter._close_bridge_log()

        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None

    def test_noop_when_no_handle(self):
        adapter = self._bare_adapter()

        adapter._close_bridge_log()  # must not raise

        assert adapter._bridge_log_fh is None

    def test_suppresses_close_exception(self):
        adapter = self._bare_adapter()
        mock_fh = MagicMock()
        mock_fh.close.side_effect = OSError("already closed")
        adapter._bridge_log_fh = mock_fh

        adapter._close_bridge_log()  # must not raise

        assert adapter._bridge_log_fh is None


# ---------------------------------------------------------------------------
# data variable initialization
# ---------------------------------------------------------------------------

class TestDataInitialized:
    """Verify ``data = {}`` prevents NameError when resp.json() fails."""

    @pytest.mark.asyncio
    async def test_no_name_error_when_json_always_fails(self):
        """HTTP 200 sets http_ready but json() always raises.

        Without the fix, ``data`` was never assigned and the Phase 2 check
        ``data.get("status")`` raised NameError.  With ``data = {}``, the
        check evaluates to ``None != "connected"`` and Phase 2 runs normally.
        """
        adapter = _make_adapter()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # bridge stays alive

        mock_client_cls = _mock_aiohttp(
            status=200, json_side_effect=ValueError("bad json"),
        )
        mock_fh = MagicMock()

        patches = _connect_patches(mock_proc, mock_fh, mock_client_cls)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], \
             patch.object(type(adapter), "_poll_messages", return_value=MagicMock()):
            # Must NOT raise NameError
            result = await adapter.connect()

        # connect() returns True (warn-and-proceed path)
        assert result is True
        assert adapter._running is True


# ---------------------------------------------------------------------------
# File handle cleanup on error paths
# ---------------------------------------------------------------------------

class TestFileHandleClosedOnError:
    """Verify the bridge log file handle is closed on every failure path."""

    @pytest.mark.asyncio
    async def test_closed_when_bridge_dies_phase1(self):
        """Bridge process exits during Phase 1 health-check loop."""
        adapter = _make_adapter()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # dead immediately
        mock_proc.returncode = 1

        mock_fh = MagicMock()
        patches = _connect_patches(mock_proc, mock_fh)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7]:
            result = await adapter.connect()

        assert result is False
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None


class TestBridgeRuntimeFailure:
    """Verify runtime bridge death is surfaced as a fatal adapter error."""

    @pytest.mark.asyncio
    async def test_send_marks_retryable_fatal_when_managed_bridge_exits(self):
        adapter = _make_adapter()
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        mock_fh = MagicMock()
        adapter._bridge_log_fh = mock_fh

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 7
        adapter._bridge_process = mock_proc

        result = await adapter.send("chat-123", "hello")

        assert result.success is False
        assert "exited unexpectedly" in result.error
        assert adapter.fatal_error_code == "whatsapp_bridge_exited"
        assert adapter.fatal_error_retryable is True
        fatal_handler.assert_awaited_once()
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None

    @pytest.mark.asyncio
    async def test_poll_messages_marks_retryable_fatal_when_managed_bridge_exits(self):
        adapter = _make_adapter()
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        mock_fh = MagicMock()
        adapter._bridge_log_fh = mock_fh

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 23
        adapter._bridge_process = mock_proc

        await adapter._poll_messages()

        assert adapter.fatal_error_code == "whatsapp_bridge_exited"
        assert adapter.fatal_error_retryable is True
        fatal_handler.assert_awaited_once()
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None

    @pytest.mark.asyncio
    async def test_closed_when_http_not_ready(self):
        """Health endpoint never returns 200 within 15 attempts."""
        adapter = _make_adapter()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # bridge alive

        mock_client_cls = _mock_aiohttp(status=503)
        mock_fh = MagicMock()
        patches = _connect_patches(mock_proc, mock_fh, mock_client_cls)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8]:
            result = await adapter.connect()

        assert result is False
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None

    @pytest.mark.asyncio
    async def test_closed_when_bridge_dies_phase2(self):
        """Bridge alive during Phase 1 but dies during Phase 2."""
        adapter = _make_adapter()

        # Phase 1 (15 iterations): alive.  Phase 2 (iteration 16): dead.
        call_count = [0]

        def poll_side_effect():
            call_count[0] += 1
            return None if call_count[0] <= 15 else 1

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = poll_side_effect
        mock_proc.returncode = 1

        # Health returns 200 with status != "connected" -> triggers Phase 2
        mock_client_cls = _mock_aiohttp(
            status=200, json_data={"status": "disconnected"},
        )
        mock_fh = MagicMock()
        patches = _connect_patches(mock_proc, mock_fh, mock_client_cls)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8]:
            result = await adapter.connect()

        assert result is False
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None

    @pytest.mark.asyncio
    async def test_closed_on_unexpected_exception(self):
        """Popen raises, outer except block must still close the handle."""
        adapter = _make_adapter()

        mock_fh = MagicMock()

        with patch("gateway.platforms.whatsapp.check_whatsapp_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "mkdir", return_value=None), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("subprocess.Popen", side_effect=OSError("spawn failed")), \
             patch("builtins.open", return_value=mock_fh):
            result = await adapter.connect()

        assert result is False
        mock_fh.close.assert_called_once()
        assert adapter._bridge_log_fh is None


# ---------------------------------------------------------------------------
# _kill_port_process() cross-platform tests
# ---------------------------------------------------------------------------

class TestKillPortProcess:
    """Verify _kill_port_process uses platform-appropriate commands."""

    def test_uses_netstat_and_taskkill_on_windows(self):
        from gateway.platforms.whatsapp import _kill_port_process

        netstat_output = (
            "  Proto  Local Address          Foreign Address        State           PID\n"
            "  TCP    0.0.0.0:3000           0.0.0.0:0              LISTENING       12345\n"
            "  TCP    0.0.0.0:3001           0.0.0.0:0              LISTENING       99999\n"
        )
        mock_netstat = MagicMock(stdout=netstat_output)
        mock_taskkill = MagicMock()

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "netstat":
                return mock_netstat
            if cmd[0] == "taskkill":
                return mock_taskkill
            return MagicMock()

        with patch("gateway.platforms.whatsapp._IS_WINDOWS", True), \
             patch("gateway.platforms.whatsapp.subprocess.run", side_effect=run_side_effect) as mock_run:
            _kill_port_process(3000)

        # netstat called
        assert any(
            call.args[0][0] == "netstat" for call in mock_run.call_args_list
        )
        # taskkill called with correct PID
        assert any(
            call.args[0] == ["taskkill", "/PID", "12345", "/F"]
            for call in mock_run.call_args_list
        )

    def test_does_not_kill_wrong_port_on_windows(self):
        from gateway.platforms.whatsapp import _kill_port_process

        netstat_output = (
            "  TCP    0.0.0.0:30000          0.0.0.0:0              LISTENING       55555\n"
        )
        mock_netstat = MagicMock(stdout=netstat_output)

        with patch("gateway.platforms.whatsapp._IS_WINDOWS", True), \
             patch("gateway.platforms.whatsapp.subprocess.run", return_value=mock_netstat) as mock_run:
            _kill_port_process(3000)

        # Should NOT call taskkill because port 30000 != 3000
        assert not any(
            call.args[0][0] == "taskkill"
            for call in mock_run.call_args_list
        )

    def test_uses_fuser_on_linux(self):
        from gateway.platforms.whatsapp import _kill_port_process

        mock_check = MagicMock(returncode=0)

        with patch("gateway.platforms.whatsapp._IS_WINDOWS", False), \
             patch("gateway.platforms.whatsapp.subprocess.run", return_value=mock_check) as mock_run:
            _kill_port_process(3000)

        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["fuser", "3000/tcp"] in calls
        assert ["fuser", "-k", "3000/tcp"] in calls

    def test_skips_fuser_kill_when_port_free(self):
        from gateway.platforms.whatsapp import _kill_port_process

        mock_check = MagicMock(returncode=1)  # port not in use

        with patch("gateway.platforms.whatsapp._IS_WINDOWS", False), \
             patch("gateway.platforms.whatsapp.subprocess.run", return_value=mock_check) as mock_run:
            _kill_port_process(3000)

        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["fuser", "3000/tcp"] in calls
        assert ["fuser", "-k", "3000/tcp"] not in calls

    def test_suppresses_exceptions(self):
        from gateway.platforms.whatsapp import _kill_port_process

        with patch("gateway.platforms.whatsapp._IS_WINDOWS", True), \
             patch("gateway.platforms.whatsapp.subprocess.run", side_effect=OSError("no netstat")):
            _kill_port_process(3000)  # must not raise
