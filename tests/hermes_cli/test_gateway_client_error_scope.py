import asyncio

import pytest

from hermes_cli.gateway_client import (
    GatewayClientError,
    _open_gateway_websocket,
)


class FakeWebSocketConnection:
    def __init__(self, open_error=None):
        self.open_error = open_error
        self.exit_exception_type = None

    async def __aenter__(self):
        if self.open_error is not None:
            raise self.open_error
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_exception_type = exc_type
        return False


@pytest.mark.parametrize(
    "open_error",
    [
        OSError("connection refused"),
        TimeoutError("WebSocket opening timed out"),
    ],
)
def test_websocket_open_errors_are_gateway_connection_errors(open_error):
    async def run():
        connection = FakeWebSocketConnection(open_error)

        def fake_connect(*args, **kwargs):
            return connection

        with pytest.raises(
            GatewayClientError,
            match="Gateway connection failed",
        ) as caught:
            async with _open_gateway_websocket(
                fake_connect,
                "ws://127.0.0.1:12345/api/ws",
                subprotocols=None,
                max_size=1024,
            ):
                pytest.fail("WebSocket opening should have failed")

        assert caught.value.__cause__ is open_error

    asyncio.run(run())


def test_timeout_after_websocket_opens_is_not_misclassified():
    async def run():
        connection = FakeWebSocketConnection()

        def fake_connect(*args, **kwargs):
            return connection

        with pytest.raises(
            TimeoutError,
            match="session.resume timed out",
        ):
            async with _open_gateway_websocket(
                fake_connect,
                "ws://127.0.0.1:12345/api/ws",
                subprotocols=None,
                max_size=1024,
            ):
                raise TimeoutError("session.resume timed out")

        assert connection.exit_exception_type is TimeoutError

    asyncio.run(run())
