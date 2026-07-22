"""A2A registers through the generic bundled platform plugin interface."""

import asyncio
import socket
from unittest.mock import MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.a2a.adapter import A2AAdapter, register


def test_register_uses_platform_plugin_surface():
    ctx = MagicMock()

    register(ctx)

    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "a2a"
    assert kwargs["adapter_factory"](PlatformConfig())
    assert callable(kwargs["apply_yaml_config_fn"])


def test_adapter_implements_base_platform_contract():
    adapter = A2AAdapter(PlatformConfig(enabled=True))

    assert isinstance(adapter, BasePlatformAdapter)


def test_adapter_returns_false_when_port_is_already_bound():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    adapter = A2AAdapter(
        PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": port})
    )

    async def exercise():
        assert await adapter.connect() is False
        assert adapter.has_fatal_error
        await adapter.disconnect()

    try:
        asyncio.run(exercise())
    finally:
        listener.close()


def test_unexpected_server_exit_notifies_without_self_await():
    adapter = A2AAdapter(PlatformConfig(enabled=True))

    class StoppedServer:
        should_exit = False

        async def serve(self):
            return None

    async def exercise():
        notified = asyncio.Event()

        async def fatal_handler(failed_adapter):
            await failed_adapter.disconnect()
            notified.set()

        adapter._server = StoppedServer()
        adapter._running = True
        adapter.set_fatal_error_handler(fatal_handler)
        serve_task = asyncio.create_task(adapter._serve_embedded())
        adapter._serve_task = serve_task

        await serve_task
        await asyncio.wait_for(notified.wait(), timeout=2)

        assert adapter.has_fatal_error
        assert adapter._serve_task is None

    asyncio.run(exercise())
