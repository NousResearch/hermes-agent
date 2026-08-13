import pytest

from gateway.becky_loops import (
    BeckyLoopsConfig,
    BeckyLoopsBridgeServer,
    SessionDBBeckyLoopsStore,
    start_becky_loops_bridge,
    stop_becky_loops_bridge,
)


class EmptyDB:
    def list_sessions_rich(self, **kwargs):
        return []

    def get_messages(self, session_id, include_inactive=False):
        return []


@pytest.mark.asyncio
async def test_lifecycle_starts_loopback_bridge_and_stops_it() -> None:
    config = BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=0,
    )
    server = await start_becky_loops_bridge(config=config, db=EmptyDB())
    assert isinstance(server, BeckyLoopsBridgeServer)
    assert server.bound_port > 0
    await stop_becky_loops_bridge(server)


@pytest.mark.asyncio
async def test_lifecycle_is_noop_when_disabled() -> None:
    config = BeckyLoopsConfig(
        enabled=False,
        chat_id="123456789",
        token="t" * 64,
        port=0,
    )
    assert await start_becky_loops_bridge(config=config, db=EmptyDB()) is None
    await stop_becky_loops_bridge(None)
