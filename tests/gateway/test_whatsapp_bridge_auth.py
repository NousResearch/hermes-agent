import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig
from gateway.platforms.whatsapp import WhatsAppAdapter


class _AsyncCM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _connected_adapter():
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    adapter._running = True
    adapter._bridge_port = 3000
    adapter._bridge_process = None
    adapter._http_session = MagicMock()
    return adapter


def _ok_response(payload=None):
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value=payload or {"messageId": "msg-1"})
    response.text = AsyncMock(return_value="")
    return response


def _consume_task(coro):
    coro.close()
    return MagicMock()


def test_bridge_headers_use_per_adapter_token():
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))

    headers = adapter._bridge_headers()

    assert adapter._bridge_token
    assert headers == {"Authorization": f"Bearer {adapter._bridge_token}"}


def test_send_adds_bridge_authorization_header():
    adapter = _connected_adapter()
    response = _ok_response({"messageId": "msg-1"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(response))

    result = asyncio.run(adapter.send("chat-1", "hello"))

    assert result.success is True
    assert adapter._http_session.post.call_args.kwargs["headers"] == adapter._bridge_headers()


def test_chat_info_adds_bridge_authorization_header():
    adapter = _connected_adapter()
    response = _ok_response({"name": "chat", "isGroup": False, "participants": []})
    adapter._http_session.get = MagicMock(return_value=_AsyncCM(response))

    result = asyncio.run(adapter.get_chat_info("chat-1"))

    assert result == {"name": "chat", "type": "dm", "participants": []}
    assert adapter._http_session.get.call_args.kwargs["headers"] == adapter._bridge_headers()


def test_existing_bridge_reuse_requires_auth_check():
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    adapter._bridge_port = 3000
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = Path("/tmp/test-wa-session")

    health_response = _ok_response({"status": "connected"})
    auth_response = _ok_response({"ok": False})
    auth_response.status = 401

    mock_session = MagicMock()

    def _get(url, **_kwargs):
        if url.endswith("/auth-check"):
            return _AsyncCM(auth_response)
        return _AsyncCM(health_response)

    mock_session.get = MagicMock(side_effect=_get)
    client_session = MagicMock(return_value=_AsyncCM(mock_session))
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_fh = MagicMock()

    with patch("gateway.platforms.whatsapp.check_whatsapp_requirements", return_value=True), \
         patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "mkdir", return_value=None), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)), \
         patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
         patch("builtins.open", return_value=mock_fh), \
         patch("gateway.platforms.whatsapp._kill_stale_bridge_by_pidfile"), \
         patch("gateway.platforms.whatsapp._kill_port_process"), \
         patch("gateway.platforms.whatsapp.asyncio.sleep", new_callable=AsyncMock), \
         patch("gateway.platforms.whatsapp.asyncio.create_task", side_effect=_consume_task), \
         patch.object(WhatsAppAdapter, "_poll_messages", return_value=MagicMock()), \
         patch("aiohttp.ClientSession", client_session):
        result = asyncio.run(adapter.connect())

    assert result is True
    assert any(call.args[0].endswith("/auth-check") for call in mock_session.get.call_args_list)
    assert not any(call.args[0].endswith("/messages") for call in mock_session.get.call_args_list)
    mock_popen.assert_called_once()
