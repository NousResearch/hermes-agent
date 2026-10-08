"""Expected keepalive close preserves an honest unknown outcome and no task traceback."""
import asyncio

import pytest
from websockets.exceptions import ConnectionClosedError

from hermes_cli.gateway_client import GatewayClient, GatewayClientError


@pytest.mark.asyncio
async def test_keepalive_close_releases_waiters_without_unhandled_reader_exception():
    class Disconnected:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionClosedError(None, None)

    client = GatewayClient(Disconnected())
    pending = asyncio.get_running_loop().create_future()
    client.pending[1] = pending
    await client._read()
    with pytest.raises(GatewayClientError, match='outcome is unknown'):
        await pending
    assert 'do not resend' in str(await client.events.get())
