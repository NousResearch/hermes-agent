"""A receive loop that stops for good reports to the gateway runner.

The runner learns of a post-startup failure only through the fatal-error handler it installs
with ``set_fatal_error_handler`` (``GatewayRunner._handle_adapter_fatal_error``). An adapter whose
receive loop ends without calling ``_notify_fatal_error`` leaves a deaf platform registered as
live: no background reconnect, no parking, no exit. Each test drives a real adapter's receive loop
into a terminal branch through a fake transport and asserts that the real runner took the adapter
out of service (and queued it for reconnection when the failure is retryable).
"""

import asyncio
import json

import aiohttp
import httpx
import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.run import GatewayRunner
from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tests.gateway.test_google_chat import GoogleChatAdapter, _base_config
from tests.gateway.test_matrix import _make_adapter as _make_matrix_adapter, _sync_error
from tests.waiting import async_wait_until


class _ScriptedSocket:
    """The aiohttp client WebSocket surface the receive loops read: ``receive``, ``closed``, ``close``."""

    def __init__(self, *messages: aiohttp.WSMessage):
        self._messages = list(messages)
        self.closed = False

    async def receive(self) -> aiohttp.WSMessage:
        if self._messages:
            return self._messages.pop(0)
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


def _supervise(adapter, tmp_path) -> GatewayRunner:
    """Register ``adapter`` with a real runner the way ``_create_adapter`` wires it."""
    runner = GatewayRunner(GatewayConfig(
        platforms={adapter.platform: adapter.config}, sessions_dir=tmp_path / "sessions"))
    runner.adapters = {adapter.platform: adapter}
    runner.delivery_router.adapters = runner.adapters
    adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)
    return runner


async def _assert_runner_took_adapter_out(runner: GatewayRunner, adapter, *, queued: bool) -> None:
    await async_wait_until(lambda: adapter.platform not in runner.adapters, timeout=5.0,
                           what=f"runner notified that the {adapter.platform.value} receive loop stopped")
    assert (adapter.platform in runner._failed_platforms) is queued


@pytest.mark.asyncio
async def test_qqbot_fatal_close_code_reaches_runner(tmp_path):
    from plugins.platforms.qqbot.adapter import QQAdapter
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "s"}))
    runner = _supervise(adapter, tmp_path)
    adapter._running = True
    adapter._ws = _ScriptedSocket(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 4915, "banned"))
    adapter._listen_task = asyncio.create_task(adapter._listen_loop())

    await _assert_runner_took_adapter_out(runner, adapter, queued=False)
    assert adapter.fatal_error_code == "qq_banned"


@pytest.mark.asyncio
async def test_qqbot_reconnect_exhaustion_queues_runner_reconnect(tmp_path, monkeypatch):
    import plugins.platforms.qqbot.adapter as qq_mod
    monkeypatch.setattr(qq_mod, "RECONNECT_BACKOFF", [0])
    monkeypatch.setattr(qq_mod, "MAX_RECONNECT_ATTEMPTS", 2)
    adapter = qq_mod.QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "s"}))
    runner = _supervise(adapter, tmp_path)
    adapter._running = True  # no socket and no HTTP client: every read and every re-dial fails
    adapter._listen_task = asyncio.create_task(adapter._listen_loop())

    await _assert_runner_took_adapter_out(runner, adapter, queued=True)


@pytest.mark.asyncio
async def test_google_chat_pubsub_exhaustion_queues_runner_reconnect(tmp_path):
    adapter = GoogleChatAdapter(_base_config())
    runner = _supervise(adapter, tmp_path)
    adapter._subscription_path = "projects/test-project/subscriptions/test-sub"
    adapter._RECONNECT_BASE_DELAY = 0

    class _UnreachablePubSub:
        def subscribe(self, *args, **kwargs):
            raise RuntimeError("stream died")

        def close(self):
            pass

    adapter._subscriber = _UnreachablePubSub()
    adapter._supervisor_task = asyncio.create_task(adapter._run_supervisor())

    await _assert_runner_took_adapter_out(runner, adapter, queued=True)
    assert adapter.fatal_error_code == "pubsub_reconnect_exhausted"


@pytest.mark.asyncio
async def test_ntfy_unauthorized_stream_reaches_runner(tmp_path):
    ntfy = load_plugin_adapter("ntfy")
    adapter = ntfy.NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t", "server": "http://ntfy.test"}))
    runner = _supervise(adapter, tmp_path)
    adapter._running = True
    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
    adapter._stream_task = asyncio.create_task(adapter._run_stream())

    await _assert_runner_took_adapter_out(runner, adapter, queued=False)
    assert adapter.fatal_error_code == "ntfy_unauthorized"


@pytest.mark.asyncio
async def test_wecom_server_kick_reaches_runner(tmp_path):
    from plugins.platforms.wecom.adapter import APP_CMD_EVENT_CALLBACK, WeComAdapter
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"bot_id": "bot-1", "secret": "s"}))
    runner = _supervise(adapter, tmp_path)
    adapter._running = True
    kick = json.dumps({"cmd": APP_CMD_EVENT_CALLBACK, "headers": {"req_id": "evt-1"},
                       "body": {"event_type": "disconnected_event"}})
    adapter._ws = _ScriptedSocket(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, kick, None))
    adapter._listen_task = asyncio.create_task(adapter._listen_loop())

    await _assert_runner_took_adapter_out(runner, adapter, queued=False)
    assert adapter.has_fatal_error


@pytest.mark.asyncio
async def test_matrix_permanent_auth_error_reaches_runner(tmp_path):
    adapter = _make_matrix_adapter()
    runner = _supervise(adapter, tmp_path)
    adapter._closing = False

    class _RevokedTokenClient:
        class sync_store:
            @staticmethod
            async def get_next_batch():
                return None

        async def sync(self, **kwargs):
            raise _sync_error("Invalid access token", errcode="M_UNKNOWN_TOKEN", http_status=401)

    adapter._client = _RevokedTokenClient()
    adapter._sync_task = asyncio.create_task(adapter._sync_loop())

    await _assert_runner_took_adapter_out(runner, adapter, queued=False)
    assert adapter.has_fatal_error
