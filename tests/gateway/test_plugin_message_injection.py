"""Tests for plugin-triggered turns in existing gateway sessions."""

import asyncio
import concurrent.futures
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from hermes_cli.plugins import PluginManager


def _entry(*, origin=True) -> SessionEntry:
    source = None
    if origin:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="42",
            user_name="tester",
        )
    now = datetime.now()
    return SessionEntry(
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
    )


def _runner(entry: SessionEntry | None, adapter=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        lookup_by_session_key=AsyncMock(return_value=entry)
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner._running = True
    runner._background_tasks = set()
    return runner


@pytest.mark.asyncio
async def test_dispatch_uses_stored_origin_and_adapter_message_path():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="check the deployment",
        plugin_id="notify-plugin",
    )

    assert accepted is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "check the deployment"
    assert event.internal is True
    assert event.source == entry.origin
    assert event.source is not entry.origin
    assert event.metadata == {
        "hermes_plugin_id": "notify-plugin",
        "hermes_plugin_injection": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "with_adapter"),
    [
        (None, True),
        (_entry(origin=False), True),
        (_entry(), False),
    ],
)
async def test_dispatch_rejects_unroutable_session(entry, with_adapter):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter if with_adapter else None)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_submits_dispatch_on_live_gateway_loop():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)

    assert runner._schedule_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    ) is True

    await asyncio.sleep(0)
    runner._dispatch_plugin_message_injection.assert_awaited_once_with(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )


@pytest.mark.asyncio
async def test_scheduler_ignores_same_loop_task_cancellation():
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))

    blocker = asyncio.Event()

    async def _wait_for_cancellation(**_kwargs):
        await blocker.wait()

    runner._dispatch_plugin_message_injection = _wait_for_cancellation

    try:
        assert runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        ) is True

        task = next(iter(runner._background_tasks))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []


def test_scheduler_uses_threadsafe_bridge_outside_gateway_loop():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, target_loop, **_kwargs):
        assert target_loop is loop
        coro.close()
        future = concurrent.futures.Future()
        future.set_result(True)
        return future

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit) as submit:
        assert runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        ) is True

    submit.assert_called_once()


def test_scheduler_ignores_threadsafe_future_cancellation():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, _target_loop, **_kwargs):
        coro.close()
        future = concurrent.futures.Future()
        future.cancel()
        return future

    with (
        patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit),
        patch("gateway.run.logger.warning") as warning,
    ):
        assert runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        ) is True

    warning.assert_not_called()


def test_scheduler_rejects_stopped_or_closed_gateway():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    runner._running = False

    assert runner._schedule_plugin_message_injection(
        session_key="key",
        content="wake up",
        plugin_id="notify-plugin",
    ) is False
    loop.call_soon_threadsafe.assert_not_called()

    runner._running = True
    loop.is_closed.return_value = True
    assert runner._schedule_plugin_message_injection(
        session_key="key",
        content="wake up",
        plugin_id="notify-plugin",
    ) is False
    loop.call_soon_threadsafe.assert_not_called()


def test_scheduler_rejects_submission_failure():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    with patch("gateway.run.safe_schedule_threadsafe", return_value=None):
        assert runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        ) is False


def test_install_and_clear_gateway_injector_preserves_newer_owner():
    runner = _runner(_entry())
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert manager.has_gateway_message_injector is True

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False

        runner._install_plugin_message_injector()

        newer_owner = MagicMock()
        newer_injector = MagicMock(return_value=True)
        manager.set_gateway_message_injector(newer_owner, newer_injector)
        runner._clear_plugin_message_injector()

    assert manager.has_gateway_message_injector is True
    assert manager.inject_gateway_message(value="kept") is True
    newer_injector.assert_called_once_with(value="kept")
