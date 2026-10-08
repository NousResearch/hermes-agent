"""A terminal topic policy remains terminal across cron's router and fallback lanes."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cron.scheduler import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget
from plugins.platforms.feishu.adapter import FeishuAdapter


def _adapter(policy, *, text_succeeds=False):
    pytest.importorskip("lark_oapi")
    from plugins.platforms.feishu.adapter import _load_lark_oapi
    assert _load_lark_oapi()
    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={"topic_delivery_fallback": policy}))
    missing = SimpleNamespace(success=lambda: False, code=230011, msg="message withdrawn")
    success = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_sent"))
    wire = SimpleNamespace(
        reply=Mock(side_effect=[success, missing] if text_succeeds else None, return_value=missing),
        create=Mock(return_value=success),
        list=Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=[]))),
    )
    upload = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(file_key="file_key")))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message=wire, file=SimpleNamespace(create=upload))))

    async def blocking(func, *args):
        return func(*args)

    adapter._run_blocking = blocking
    return adapter, wire, upload


def _run_coro(coro, _loop):
    future = Future()
    try:
        future.set_result(asyncio.run(coro))
    except BaseException as exc:
        future.set_exception(exc)
    return future


@pytest.mark.parametrize("policy", ["error_notice", "silent"])
@pytest.mark.parametrize("stage", ["text", "media_only", "after_text"])
def test_cron_terminal_topic_policy_does_not_replay_or_mirror(monkeypatch, tmp_path, policy, stage):
    pytest.importorskip("lark_oapi")
    adapter, wire, upload = _adapter(policy, text_succeeds=stage == "after_text")
    config = GatewayConfig(platforms={Platform.FEISHU: adapter.config})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", _run_coro)
    standalone = AsyncMock(return_value={"success": True, "message_id": "unexpected"})
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone)
    mirror = Mock()
    monkeypatch.setattr("cron.scheduler_delivery._maybe_mirror_cron_delivery", mirror)
    loop = Mock()
    loop.is_running.return_value = True
    job = {
        "id": "feishu-policy", "name": "Topic result", "deliver": "origin",
        "origin": {"platform": "feishu", "chat_id": "oc_chat", "chat_type": "group",
                   "thread_id": "omt_topic", "message_id": "om_origin", "user_id": "ou_user"},
    }
    files = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path in files:
        path.write_text("private attachment")
    content = ("RESULT BODY\n" if stage != "media_only" else "") + "\n".join(f"MEDIA:{path}" for path in files)

    error = _deliver_result(job, content, adapters={Platform.FEISHU: adapter}, loop=loop)

    assert error and "stopped by policy" in error
    standalone.assert_not_awaited()
    mirror.assert_not_called()
    assert wire.list.call_count == 1
    assert wire.create.call_count == (1 if policy == "error_notice" else 0)
    assert upload.call_count == (0 if stage == "text" else 1)
    assert wire.reply.call_count == (2 if stage == "after_text" else 1)
    if policy == "error_notice":
        assert "RESULT BODY" not in wire.create.call_args.args[0].request_body.content
        assert "private attachment" not in wire.create.call_args.args[0].request_body.content


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["error_notice", "silent"])
@pytest.mark.parametrize("boundary", ["router", "standalone", "standalone_cron"])
async def test_terminal_flag_survives_public_delivery_boundaries(monkeypatch, policy, boundary):
    pytest.importorskip("lark_oapi")
    adapter, wire, _upload = _adapter(policy)
    if boundary == "router":
        router = DeliveryRouter(GatewayConfig(), {Platform.FEISHU: adapter})
        results = await router.deliver("RESULT BODY", [DeliveryTarget.parse("feishu:oc_chat:omt_topic")])
        result = results["feishu:oc_chat:omt_topic"]
        assert not router.dead_targets.is_dead("feishu", "oc_chat")
    else:
        from plugins.platforms.feishu import adapter as module
        monkeypatch.setattr(module, "_load_lark_oapi", lambda: True)
        monkeypatch.setattr(FeishuAdapter, "_build_lark_client", lambda self, domain: adapter._client)
        async def blocking(self, func, *args):
            return func(*args)
        monkeypatch.setattr(FeishuAdapter, "_run_blocking", blocking)
        if boundary == "standalone_cron":
            async def send(platform, config, chat_id, content, **kwargs):
                return await module._standalone_send(config, chat_id, content, **kwargs)
            monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
            monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig(
                platforms={Platform.FEISHU: adapter.config}))
            monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
            reconnect = Mock()
            monkeypatch.setattr("cron.scheduler_delivery._queue_for_live_reconnect", reconnect)
            result = await asyncio.to_thread(_deliver_result, {
                "id": "standalone-policy", "deliver": "feishu:oc_chat:omt_topic",
            }, "RESULT BODY")
            assert result and "stopped by policy" in result
            reconnect.assert_not_called()
        else:
            result = await module._standalone_send(adapter.config, "oc_chat", "RESULT BODY", thread_id="omt_topic")
    if boundary != "standalone_cron":
        assert result.get("success") is not True
        assert result["retry_suppressed"] is True
        assert result["error"]
    assert wire.create.call_count == (1 if policy == "error_notice" else 0)
