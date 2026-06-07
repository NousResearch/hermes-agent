from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace


class FakeReplyMessageResponse:
    """SDK-style response object: truthy, but deliberately no .get()."""

    pass


def _start_background_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    ready.wait(timeout=5)
    return loop, thread


def _stop_background_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_live_adapter_sdk_raw_response_does_not_fallback(monkeypatch):
    """A successful Feishu live-adapter send must not fallback just because
    raw_response is an SDK object rather than a dict.

    Regression coverage for duplicate cron deliveries: the message.reply call
    can succeed, then scheduler diagnostics used raw_response.get(...), raised
    AttributeError, and sent the same content again via standalone fallback.
    """
    from cron import scheduler
    from gateway.config import Platform
    import gateway.config as gateway_config
    import tools.send_message_tool as send_message_tool

    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "feishu", "chat_id": "oc_test", "thread_id": "om_test"}],
    )
    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: SimpleNamespace(platforms={Platform.FEISHU: SimpleNamespace(enabled=True)}),
    )

    fallback_calls = []

    async def fake_send_to_platform(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return {"error": "fallback should not be called"}

    monkeypatch.setattr(send_message_tool, "_send_to_platform", fake_send_to_platform)

    live_calls = []

    class FakeLiveAdapter:
        async def send(self, chat_id, content, metadata=None):
            live_calls.append((chat_id, content, metadata))
            return SimpleNamespace(success=True, raw_response=FakeReplyMessageResponse())

    loop, thread = _start_background_loop()
    try:
        error = scheduler._deliver_result(
            {"id": "job1", "deliver": "origin", "name": "test"},
            "hello",
            adapters={Platform.FEISHU: FakeLiveAdapter()},
            loop=loop,
        )
    finally:
        _stop_background_loop(loop, thread)

    assert error is None
    assert len(live_calls) == 1
    assert live_calls[0][2] == {"thread_id": "om_test"}
    assert fallback_calls == []


def test_raw_response_get_handles_dict_and_sdk_object():
    from cron.scheduler import _raw_response_get

    assert _raw_response_get({"thread_fallback": True}, "thread_fallback") is True
    assert _raw_response_get(FakeReplyMessageResponse(), "thread_fallback") is None
