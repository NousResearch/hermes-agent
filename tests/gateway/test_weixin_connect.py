import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


class _AsyncCM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _session_response(*, status=200, json_data=None):
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=json_data or {})

    session = MagicMock()
    session.get = MagicMock(return_value=_AsyncCM(response))
    return _AsyncCM(session)


def _make_adapter():
    from gateway.platforms.weixin import WeixinAdapter

    adapter = WeixinAdapter.__new__(WeixinAdapter)
    adapter.platform = Platform.WEIXIN
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = "/tmp/test-weixin-bridge.js"
    adapter._session_path = Path("/tmp/test-weixin-session")
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._poll_task = None
    adapter._http_session = None
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
    adapter._typing_paused = set()
    adapter._message_queue = asyncio.Queue()
    return adapter


def _fake_create_task(coro):
    coro.close()
    return MagicMock()


class TestConnectRequiresTruePairing:
    @pytest.mark.asyncio
    async def test_existing_connected_bridge_is_reused(self):
        adapter = _make_adapter()
        persistent_session = MagicMock()
        persistent_session.closed = False

        with patch("gateway.platforms.weixin.check_weixin_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "mkdir", return_value=None), \
             patch(
                 "aiohttp.ClientSession",
                 side_effect=[
                     _session_response(json_data={"status": "connected"}),
                     persistent_session,
                 ],
             ), \
             patch("gateway.platforms.weixin.asyncio.create_task", side_effect=_fake_create_task) as create_task, \
             patch("subprocess.Popen") as popen:
            result = await adapter.connect()

        assert result is True
        assert adapter._running is True
        assert adapter._http_session is persistent_session
        create_task.assert_called_once()
        popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_existing_unpaired_bridge_does_not_count_as_connected(self):
        adapter = _make_adapter()
        mock_proc = MagicMock()
        mock_proc.pid = 4321
        mock_proc.poll.return_value = None
        mock_fh = MagicMock()

        client_sessions = [
            _session_response(json_data={"status": "needs_login"}),
            _session_response(json_data={"status": "qr_ready"}),
        ]
        client_sessions.extend(_session_response(json_data={"status": "qr_ready"}) for _ in range(15))

        with patch("gateway.platforms.weixin.check_weixin_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "mkdir", return_value=None), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("builtins.open", return_value=mock_fh), \
             patch("gateway.platforms.weixin._kill_port_process") as kill_port, \
             patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock), \
             patch("gateway.platforms.weixin.asyncio.create_task") as create_task, \
             patch("gateway.platforms.weixin.os.getpgid", return_value=mock_proc.pid), \
             patch("gateway.platforms.weixin.os.killpg"), \
             patch("aiohttp.ClientSession", side_effect=client_sessions):
            result = await adapter.connect()

        assert result is False
        assert adapter._running is False
        assert adapter._http_session is None
        assert adapter._bridge_process is None
        kill_port.assert_called_once_with(adapter._bridge_port)
        create_task.assert_not_called()
        mock_fh.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_bridge_http_ready_without_connected_state_returns_false(self):
        adapter = _make_adapter()
        mock_proc = MagicMock()
        mock_proc.pid = 9876
        mock_proc.poll.return_value = None
        mock_fh = MagicMock()

        client_sessions = [_session_response(status=500)]
        client_sessions.append(_session_response(json_data={"status": "needs_login"}))
        client_sessions.extend(_session_response(json_data={"status": "needs_login"}) for _ in range(15))

        with patch("gateway.platforms.weixin.check_weixin_requirements", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "mkdir", return_value=None), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("builtins.open", return_value=mock_fh), \
             patch("gateway.platforms.weixin._kill_port_process"), \
             patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock), \
             patch("gateway.platforms.weixin.asyncio.create_task") as create_task, \
             patch("gateway.platforms.weixin.os.getpgid", return_value=mock_proc.pid), \
             patch("gateway.platforms.weixin.os.killpg"), \
             patch("aiohttp.ClientSession", side_effect=client_sessions):
            result = await adapter.connect()

        assert result is False
        assert adapter._running is False
        assert adapter._http_session is None
        assert adapter._bridge_process is None
        create_task.assert_not_called()
        mock_fh.close.assert_called_once()


class TestBridgeRuntimeFailure:
    @pytest.mark.asyncio
    async def test_send_marks_retryable_fatal_when_managed_bridge_exits(self):
        adapter = _make_adapter()
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        adapter._http_session = MagicMock()
        log_handle = MagicMock()
        adapter._bridge_log_fh = log_handle

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 9
        adapter._bridge_process = mock_proc

        result = await adapter.send("wxid-123", "hello")

        assert result.success is False
        assert "exited unexpectedly" in result.error
        assert adapter.fatal_error_code == "weixin_bridge_exited"
        assert adapter.fatal_error_retryable is True
        fatal_handler.assert_awaited_once()
        log_handle.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_messages_marks_retryable_fatal_when_managed_bridge_exits(self):
        adapter = _make_adapter()
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        adapter._http_session = MagicMock()
        log_handle = MagicMock()
        adapter._bridge_log_fh = log_handle

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 17
        adapter._bridge_process = mock_proc

        await adapter._poll_messages()

        assert adapter.fatal_error_code == "weixin_bridge_exited"
        assert adapter.fatal_error_retryable is True
        fatal_handler.assert_awaited_once()
        log_handle.close.assert_called_once()
