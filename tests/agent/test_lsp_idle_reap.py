import time

from agent.lsp.manager import LSPService


class DummyClient:
    def __init__(self):
        self.shutdown_calls = 0

    async def shutdown(self):
        self.shutdown_calls += 1


def test_reap_idle_clients_shuts_down_only_clients_past_idle_timeout():
    service = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.1,
        install_strategy="never",
        idle_timeout=10,
    )
    service._loop.stop()

    old_client = DummyClient()
    fresh_client = DummyClient()
    old_key = ("typescript", "/tmp/old")
    fresh_key = ("typescript", "/tmp/fresh")
    now = time.time()

    service._clients = {old_key: old_client, fresh_key: fresh_client}
    service._last_used = {old_key: now - 30, fresh_key: now}

    reaped = service.reap_idle_clients(now=now)

    assert reaped == 1
    assert old_client.shutdown_calls == 1
    assert fresh_client.shutdown_calls == 0
    assert old_key not in service._clients
    assert fresh_key in service._clients
