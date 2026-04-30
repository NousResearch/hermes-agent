from types import SimpleNamespace

import cron.scheduler as scheduler
from cron.scheduler import _deliver_result


class DummyLoop:
    def is_running(self):
        return True


class DummyFuture:
    def __init__(self, result):
        self._result = result

    def result(self, timeout=None):
        return self._result

    def cancel(self):
        return None


def test_live_adapter_success_without_message_id_falls_back_to_error(monkeypatch):
    job = {
        "id": "job1",
        "name": "test",
        "deliver": "telegram:226252250:7072",
        "origin": None,
    }

    class DummyAdapter:
        async def send(self, chat_id, text, metadata=None):
            return SimpleNamespace(success=True, message_id=None)

    from gateway.config import Platform
    adapters = {Platform.TELEGRAM: DummyAdapter()}

    class DisabledPlatformConfig:
        enabled = False
        token = None
        extra = {}

    class DummyGatewayConfig:
        def __init__(self):
            self.platforms = {Platform.TELEGRAM: DisabledPlatformConfig()}

    monkeypatch.setattr(scheduler, "load_gateway_config", lambda: DummyGatewayConfig(), raising=False)
    def fake_run_coroutine_threadsafe(coro, loop):
        coro.close()
        return DummyFuture(SimpleNamespace(success=True, message_id=None))

    monkeypatch.setattr(
        scheduler.asyncio,
        "run_coroutine_threadsafe",
        fake_run_coroutine_threadsafe,
    )

    err = _deliver_result(job, "hello world", adapters=adapters, loop=DummyLoop())
    assert err is not None
    assert "platform 'telegram' not configured/enabled" in err
