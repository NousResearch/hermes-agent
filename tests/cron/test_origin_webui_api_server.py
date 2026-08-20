"""Regression tests for WebUI/API-server cron origin delivery."""

from concurrent.futures import Future
from types import SimpleNamespace

from cron.scheduler import _deliver_result, _resolve_delivery_target, _resolve_origin
from gateway.config import Platform
from gateway.session_context import clear_session_vars, set_session_vars
from tools.cronjob_tools import _origin_from_env


def test_webui_session_origin_captures_api_server_raw_session_id():
    tokens = set_session_vars(
        platform="webui",
        chat_id="transcript-session-id",
        ui_session_id="raw-ui-session-id",
        async_delivery=False,
    )
    try:
        origin = _origin_from_env()
    finally:
        clear_session_vars(tokens)

    assert origin["platform"] == "api_server"
    assert origin["chat_id"] == "raw-ui-session-id"
    assert origin["thread_id"] is None


def test_legacy_webui_origin_normalizes_to_api_server_without_mutating_job():
    job = {
        "id": "legacy-webui-job",
        "deliver": "origin",
        "origin": {"platform": "webui", "chat_id": "raw-ui-session-id"},
    }

    assert _resolve_origin(job) == {
        "platform": "api_server",
        "chat_id": "raw-ui-session-id",
    }
    assert job["origin"]["platform"] == "webui"
    assert _resolve_delivery_target(job) == {
        "platform": "api_server",
        "chat_id": "raw-ui-session-id",
        "thread_id": None,
    }


def test_api_server_origin_delivery_uses_wake_self_post(monkeypatch):
    calls = []

    async def fake_deliver_wake(adapter, *, text, session_id="", source=None):
        calls.append({"adapter": adapter, "text": text, "session_id": session_id, "source": source})

    def fake_schedule(coro, loop):
        future = Future()
        try:
            import asyncio

            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - failure assertion path
            future.set_exception(exc)
        return future

    pconfig = SimpleNamespace(enabled=True, extra={})
    adapter = SimpleNamespace(supports_async_delivery=False)
    loop = SimpleNamespace(is_running=lambda: True)
    config = SimpleNamespace(platforms={Platform.API_SERVER: pconfig})
    transport = SimpleNamespace(config=pconfig, adapter=adapter, is_relay=False)

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.delivery.resolve_delivery_transport", lambda *args, **kwargs: transport)
    monkeypatch.setattr("gateway.wake.deliver_wake", fake_deliver_wake)
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", fake_schedule)

    err = _deliver_result(
        {
            "id": "webui-cron",
            "name": "WebUI cron",
            "deliver": "origin",
            "origin": {"platform": "webui", "chat_id": "raw-ui-session-id"},
        },
        "done",
        adapters={Platform.API_SERVER: adapter},
        loop=loop,
    )

    assert err is None
    assert len(calls) == 1
    assert calls[0]["adapter"] is adapter
    assert calls[0]["session_id"] == "raw-ui-session-id"
    assert "Cronjob Response: WebUI cron" in calls[0]["text"]
    assert calls[0]["source"] is None
