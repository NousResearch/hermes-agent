"""The Feishu adapter only reports ``connected`` once the WebSocket link is provably up.

``lark_oapi``'s ``Client.start()`` returns only on fatal errors and performs its handshake
asynchronously on the WS thread, so submitting that thread is not evidence of a live link. A
handshake that never completed (hung socket, rejected app) used to let ``connect()`` log
"Connected in websocket mode" while the profile received nothing: group traffic and DMs were
dropped silently until an operator noticed and restarted the gateway. The adapter now waits for
the SDK's receive-loop entry — the only in-thread proof a link is up — and fails the attempt when
it never arrives, so ``_connect_with_retry`` rebuilds instead of stamping ``connected`` on a
connection that does not exist.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from plugins.platforms.feishu import adapter as adapter_module
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FakeWSClient:
    """Stands in for ``lark_oapi.ws.Client`` — ``_connect_websocket`` only constructs it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _adapter_for_connect(monkeypatch, runner, *, timeout_s: float, tmp_path) -> FeishuAdapter:
    """A websocket-mode adapter with the SDK/identity edges stubbed and ``runner`` as the WS thread."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.config import PlatformConfig

    adapter = FeishuAdapter(PlatformConfig(extra={"connection_mode": "websocket"}))
    adapter._loop = asyncio.get_running_loop()
    adapter._running = False  # ``_mark_connected`` is what flips this, and it has not run yet
    adapter._ws_client = adapter._ws_future = adapter._ws_supervisor = adapter._ws_thread_loop = None
    adapter._ws_link_up_event = None
    monkeypatch.setattr(adapter, "_prepare_client", lambda: None)

    async def _skip_hydrate() -> None:
        return None

    monkeypatch.setattr(adapter, "_hydrate_bot_identity", _skip_hydrate)
    monkeypatch.setattr(adapter_module, "FEISHU_WEBSOCKET_AVAILABLE", True)
    monkeypatch.setattr(adapter_module, "lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO="INFO")))
    monkeypatch.setattr(adapter_module, "FeishuWSClient", _FakeWSClient)
    monkeypatch.setattr(adapter_module, "_run_official_feishu_ws_client", runner)
    monkeypatch.setattr(adapter_module, "_FEISHU_WS_HANDSHAKE_CONFIRM_TIMEOUT", timeout_s)
    return adapter


@pytest.mark.asyncio
async def test_connect_websocket_fails_when_the_link_never_comes_up(monkeypatch, tmp_path):
    """A WS thread that never proves its link must fail the attempt, not report success."""

    def _sdk_thread_without_link_up(ws_client, adapter):
        # A hung handshake: the SDK thread is alive, the link is not up, nothing ever signals.
        threading.Event().wait(1.0)

    adapter = _adapter_for_connect(monkeypatch, _sdk_thread_without_link_up, timeout_s=0.2, tmp_path=tmp_path)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(adapter._connect_websocket(), timeout=10)
        # The failure leaves the adapter un-``connected``, which is what ``connect()`` reports on.
        assert adapter._running is False
    finally:
        adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
async def test_link_up_proof_lets_connect_websocket_finish(monkeypatch, tmp_path):
    """The SDK thread's link-up report releases the wait, so a real handshake still connects."""
    loop = asyncio.get_running_loop()

    def _sdk_thread_that_links_up(ws_client, adapter):
        # Fired from the WS thread, exactly as ``_receive_message_loop_exit_notify`` does.
        loop.call_soon_threadsafe(adapter._ws_link_up, ws_client)

    adapter = _adapter_for_connect(monkeypatch, _sdk_thread_that_links_up, timeout_s=10, tmp_path=tmp_path)
    try:
        await asyncio.wait_for(adapter._connect_websocket(), timeout=10)
        assert adapter._ws_link_up_event is not None
        assert adapter._ws_link_up_event.is_set() is True
    finally:
        adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
async def test_a_detached_connect_releases_the_app_lock(monkeypatch, tmp_path):
    """The gateway detaches a slow connect by cancelling it: that must not strand the app lock.

    ``CancelledError`` derives from ``BaseException``, so the failure path that releases the lock does
    not run. The lock then stays owned by a live PID, and the next start reads it as the non-retryable
    "another gateway owns this app_id" case and drops the platform from the reconnect queue.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.config import PlatformConfig

    adapter = FeishuAdapter(PlatformConfig(extra={"connection_mode": "websocket"}))
    setattr(adapter, "_app_id", "cli_test_app")  # test fixture: connect() bails early without these
    setattr(adapter, "_app_secret", "secret")
    monkeypatch.setattr(adapter_module, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(
        adapter_module, "acquire_scoped_lock", lambda scope, identity, metadata=None: (True, {})
    )

    entered = asyncio.Event()

    async def _connect_that_never_returns() -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "_connect_with_retry", _connect_that_never_returns)
    releases = []

    async def _record_release() -> None:
        releases.append(True)

    monkeypatch.setattr(adapter, "_release_app_lock", _record_release)

    task = asyncio.ensure_future(adapter.connect())
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert releases, "a detached connect left the app lock owned by a live PID"


@pytest.mark.asyncio
async def test_the_retry_ladder_stays_inside_the_connect_budget(monkeypatch, tmp_path):
    """A handshake that never completes must exhaust the ladder on its own terms, not the outer wait."""

    def _sdk_thread_without_link_up(ws_client, adapter):
        threading.Event().wait(1.0)

    adapter = _adapter_for_connect(monkeypatch, _sdk_thread_without_link_up, timeout_s=0.3, tmp_path=tmp_path)
    monkeypatch.setattr(adapter_module, "_FEISHU_CONNECT_BUDGET_SECONDS", 0.5)
    monkeypatch.setattr(adapter, "_disable_websocket_auto_reconnect", lambda: None)

    async def _no_webhook_server() -> None:
        return None

    monkeypatch.setattr(adapter, "_stop_webhook_server", _no_webhook_server)

    waits = []
    inner = adapter._connect_websocket

    async def _counting_connect_websocket(*, confirm_timeout=None):
        waits.append(confirm_timeout)
        return await inner(confirm_timeout=confirm_timeout)

    monkeypatch.setattr(adapter, "_connect_websocket", _counting_connect_websocket)

    started = asyncio.get_running_loop().time()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(adapter._connect_with_retry(), timeout=10)
    finally:
        adapter._shutdown_sdk_executor()
    elapsed = asyncio.get_running_loop().time() - started

    assert len(waits) == 1, "the ladder started a retry the remaining budget could not cover"
    assert waits[0] <= 0.5, "an attempt waited longer than the whole connect budget"
    assert elapsed < 1.0, "the ladder slept through its backoff instead of failing at the budget"


@pytest.mark.asyncio
async def test_stale_link_up_does_not_release_the_next_attempt(monkeypatch, tmp_path):
    """A link-up from an abandoned client must not vouch for the attempt that replaced it."""
    loop = asyncio.get_running_loop()

    def _sdk_thread_linking_up_as_an_abandoned_client(ws_client, adapter):
        # A supervisor rebuild's late link-up arrives after ``connect()`` replaced ``_ws_client``.
        loop.call_soon_threadsafe(adapter._ws_link_up, _FakeWSClient())

    adapter = _adapter_for_connect(
        monkeypatch, _sdk_thread_linking_up_as_an_abandoned_client, timeout_s=0.2, tmp_path=tmp_path,
    )
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(adapter._connect_websocket(), timeout=10)
    finally:
        adapter._shutdown_sdk_executor()
