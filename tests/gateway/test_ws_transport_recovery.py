import asyncio
import json


def test_ws_transport_failed_send_marks_closed_and_invokes_reap_hook():
    from tui_gateway import ws as ws_mod

    ws_mod.reset_ws_transport_health_for_tests()
    broken = []

    class ClosedWS:
        async def send_text(self, _line):
            raise RuntimeError("Cannot call send once a close message has been sent")

    async def run():
        transport = ws_mod.WSTransport(
            ClosedWS(),
            asyncio.get_running_loop(),
            peer="test-peer",
            on_broken=lambda _transport, reason: broken.append(reason),
        )
        assert await transport.write_async({"jsonrpc": "2.0", "method": "event"}) is False
        assert transport.closed is True

    asyncio.run(run())

    health = ws_mod.get_ws_transport_health(close_wait_count=6)
    assert broken == ["send_failed"]
    assert health["active_clients"] == 0
    assert health["closed_clients"] == 1
    assert health["stale_closed_clients"] == 1
    assert health["send_failures"] == 1
    assert health["last_send_failure_type"] == "RuntimeError"
    assert health["close_wait_count"] == 6


def test_ws_transport_health_is_value_free():
    from tui_gateway import ws as ws_mod

    ws_mod.reset_ws_transport_health_for_tests()

    class ClosedWS:
        async def send_text(self, _line):
            raise RuntimeError("private socket details must not leak")

    async def run():
        transport = ws_mod.WSTransport(
            ClosedWS(),
            asyncio.get_running_loop(),
            peer="test-peer",
        )
        await transport.write_async({"payload": "not in health"})

    asyncio.run(run())

    dumped = json.dumps(ws_mod.get_ws_transport_health())
    assert "private socket details" not in dumped
    assert "not in health" not in dumped
