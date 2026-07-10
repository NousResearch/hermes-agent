from __future__ import annotations

import asyncio
import contextlib
from importlib.metadata import version
import logging
import time
from typing import Any

import pytest
from slack_bolt.app.async_app import AsyncApp
from slack_sdk.socket_mode.aiohttp import SocketModeClient

from gateway.config import PlatformConfig
from plugins.platforms.slack import adapter as slack_adapter_module
from plugins.platforms.slack.adapter import SlackAdapter


class ControlledClientSession:
    """No-network aiohttp boundary that records transport-close ordering."""

    def __init__(self, order: list[str]) -> None:
        self.closed = False
        self.order = order

    async def close(self) -> None:
        self.closed = True
        self.order.append("session:closed")


class ControlledWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self._receive_forever = asyncio.Event()

    async def close(self) -> None:
        self.closed = True

    async def ping(self, _payload: bytes) -> None:
        return None

    async def receive(self) -> None:
        await self._receive_forever.wait()


class RetryingClientSession(ControlledClientSession):
    """Fail once, then connect without network and record retry timing."""

    def __init__(self, order: list[str]) -> None:
        super().__init__(order)
        self.attempt_times: list[float] = []
        self.websocket = ControlledWebSocket()

    async def ws_connect(self, _wss_uri: str, **_kwargs: Any) -> ControlledWebSocket:
        self.attempt_times.append(time.monotonic())
        if len(self.attempt_times) == 1:
            raise ConnectionError("controlled transient failure")
        return self.websocket

    async def close(self) -> None:
        await self.websocket.close()
        await super().close()


async def _cancellation_delayed_task(
    name: str,
    started: asyncio.Event,
    cancelling: asyncio.Event,
    release_cleanup: asyncio.Event,
    order: list[str],
) -> None:
    started.set()
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        cancelling.set()
        await release_cleanup.wait()
        order.append(f"{name}:settled")
        raise


async def _cancellation_resistant_task(
    started: asyncio.Event,
    cancelling: asyncio.Event,
    release_cleanup: asyncio.Event,
) -> None:
    started.set()
    while not release_cleanup.is_set():
        try:
            await release_cleanup.wait()
        except asyncio.CancelledError:
            cancelling.set()
            continue


async def _real_handler_with_controlled_session(
    order: list[str],
) -> tuple[Any, SocketModeClient, ControlledClientSession]:
    handler = slack_adapter_module.AsyncSocketModeHandler(
        AsyncApp(token="xoxb-test-token"),
        "xapp-test-token",
    )
    client = handler.client
    assert isinstance(client, SocketModeClient)
    assert version("slack-sdk") == "3.40.1"
    assert version("slack-bolt") == "1.27.0"

    client.message_processor.cancel()
    await asyncio.gather(client.message_processor, return_exceptions=True)
    await client.aiohttp_client_session.close()
    session = ControlledClientSession(order)
    setattr(client, "aiohttp_client_session", session)
    return handler, client, session


async def _wait_for_attempt_count(
    session: RetryingClientSession,
    *,
    expected: int,
) -> None:
    while len(session.attempt_times) < expected:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_adapter_settles_real_sdk_tasks_before_closing_client_session() -> None:
    """The adapter must quiesce old Socket Mode owners before transport close."""

    order: list[str] = []
    handler, client, session = await _real_handler_with_controlled_session(order)
    release_cleanup = asyncio.Event()
    started = [asyncio.Event() for _ in range(4)]
    cancelling = [asyncio.Event() for _ in range(4)]
    tasks = [
        asyncio.create_task(
            _cancellation_delayed_task(
                name,
                started[index],
                cancelling[index],
                release_cleanup,
                order,
            )
        )
        for index, name in enumerate(
            ("outer", "message_processor", "current_session_monitor", "message_receiver")
        )
    ]
    outer, client.message_processor, client.current_session_monitor, client.message_receiver = tasks
    for event in started:
        await event.wait()

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
    adapter._handler = handler
    adapter._socket_mode_task = outer
    stop_task = asyncio.create_task(adapter._stop_socket_mode_handler())

    try:
        for event in cancelling:
            await asyncio.wait_for(event.wait(), timeout=0.2)

        assert not stop_task.done()
        assert session.closed is False

        release_cleanup.set()
        await asyncio.wait_for(stop_task, timeout=0.2)

        assert session.closed is True
        assert all(task.done() for task in tasks)
        assert order[-1] == "session:closed"
        assert set(order[:-1]) == {
            "outer:settled",
            "message_processor:settled",
            "current_session_monitor:settled",
            "message_receiver:settled",
        }
        assert adapter._handler is None
        assert adapter._socket_mode_task is None
    finally:
        release_cleanup.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if not session.closed:
            with contextlib.suppress(Exception):
                await handler.close_async()


@pytest.mark.asyncio
async def test_adapter_does_not_cancel_or_await_current_shutdown_owner() -> None:
    order: list[str] = []
    handler, client, session = await _real_handler_with_controlled_session(order)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
    adapter._handler = handler
    adapter._socket_mode_task = asyncio.current_task()
    client.current_session_monitor = asyncio.current_task()

    await adapter._stop_socket_mode_handler()

    current = asyncio.current_task()
    assert current is not None
    assert not current.cancelled()
    assert client.current_session_monitor is None
    assert session.closed is True
    assert adapter._handler is None
    assert adapter._socket_mode_task is None


@pytest.mark.asyncio
async def test_adapter_bounds_stubborn_task_cleanup_and_redacts_state_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    order: list[str] = []
    handler, client, session = await _real_handler_with_controlled_session(order)
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release_cleanup = asyncio.Event()
    stubborn_task = asyncio.create_task(
        _cancellation_resistant_task(started, cancelling, release_cleanup)
    )
    await started.wait()
    client.message_receiver = stubborn_task

    adapter = SlackAdapter(
        PlatformConfig(enabled=True, token="xoxb-secret-must-not-appear")
    )
    adapter._handler = handler
    adapter._socket_mode_task = stubborn_task
    adapter._proxy_url = "http://proxy-user:proxy-secret@proxy.invalid:8080"
    handler.client.wss_uri = "wss://socket-mode.invalid/secret-path"
    monkeypatch.setattr(
        slack_adapter_module,
        "_SOCKET_MODE_TASK_SETTLE_TIMEOUT_S",
        0.01,
    )
    caplog.set_level(logging.WARNING, logger=slack_adapter_module.logger.name)

    try:
        await asyncio.wait_for(adapter._stop_socket_mode_handler(), timeout=0.2)
        await asyncio.wait_for(cancelling.wait(), timeout=0.2)

        assert session.closed is True
        warning_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "did not settle before transport close" in warning_text
        for secret in (
            "xoxb-secret-must-not-appear",
            "secret-path",
            "proxy-user",
            "proxy-secret",
        ):
            assert secret not in warning_text
    finally:
        release_cleanup.set()
        await asyncio.gather(stubborn_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_live_real_client_retry_keeps_ping_interval_and_stop_is_terminal() -> None:
    baseline = {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    }
    order: list[str] = []
    handler, client, replaced_session = await _real_handler_with_controlled_session(order)
    await replaced_session.close()
    retrying_session = RetryingClientSession(order)
    setattr(client, "aiohttp_client_session", retrying_session)
    client.wss_uri = "wss://socket-mode.invalid/no-network"
    client.ping_interval = 0.02
    connect_task = asyncio.create_task(client.connect())

    try:
        await asyncio.wait_for(
            _wait_for_attempt_count(retrying_session, expected=2),
            timeout=0.2,
        )
        assert (
            retrying_session.attempt_times[1] - retrying_session.attempt_times[0]
            >= client.ping_interval
        )

        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
        adapter._handler = handler
        adapter._socket_mode_task = connect_task
        await adapter._stop_socket_mode_handler()
        await asyncio.sleep(client.ping_interval * 2)

        assert connect_task.done()
        assert len(retrying_session.attempt_times) == 2
        assert retrying_session.closed is True
        assert retrying_session.websocket.closed is True
        assert replaced_session.closed is True
    finally:
        if not connect_task.done():
            connect_task.cancel()
        await asyncio.gather(connect_task, return_exceptions=True)
        if not retrying_session.closed:
            await retrying_session.close()

    leaked = {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task not in baseline
    }
    assert leaked == set()
