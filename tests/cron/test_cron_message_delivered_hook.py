"""``gateway_message_delivered`` kind="cron" fire sites (#64176 delivery half): the live-adapter
lane and the standalone lane of ``cron/scheduler_delivery.py`` fire once per confirmed delivery."""

import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler_delivery as sched_delivery
from cron.scheduler import _deliver_result
from gateway import delivery_hooks as dh
from gateway.config import Platform, PlatformConfig

CHAT_ID = "-1001234567890"


class _SendResult:
    def __init__(self, success=True, message_id=None, raw_response=None):
        self.success, self.message_id, self.raw_response = success, message_id, raw_response
        self.continuation_message_ids = ()


def _job(thread_id=None):
    origin = {"platform": "telegram", "chat_id": CHAT_ID}
    if thread_id is not None:
        origin["thread_id"] = thread_id
    return {"id": "job-42", "name": "Report", "deliver": "origin", "origin": origin}


@pytest.fixture
def fired():
    calls = []
    async def _async(name, **kw):
        calls.append((name, kw))
        return []

    # The live lane dispatches onto the gateway loop (asyncio.run_coroutine_threadsafe, which the
    # harness runs inline); a loop-less call uses the sync path — capture both.
    with patch.object(dh, "_has_subscribers", return_value=True), \
         patch("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: calls.append((name, kw)) or []), \
         patch("hermes_cli.lifecycle.ainvoke_hook", _async), \
         patch.object(dh, "_active_profile", return_value=None):
        yield calls


def _run(job, content, send_result, standalone_result=None):
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            future.set_exception(e)
        return future

    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata, transport=None):
        return send_result

    router._deliver_to_platform = _deliver_to_platform
    standalone_calls = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone_calls.append(text)
        return standalone_result if standalone_result is not None else {"success": True, "message_id": "s-1"}

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._record_delivery_verification"), \
         patch("gateway.delivery.DeliveryRouter", return_value=router), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, content, adapters={Platform.TELEGRAM: MagicMock()}, loop=loop)
    return error, standalone_calls


def test_live_lane_fires_once_with_job_and_all_ids(fired):
    result = _SendResult(message_id="100", raw_response={"message_ids": ["100", "101"]})
    error, standalone = _run(_job(thread_id="5"), "ACTION_REQUIRED: apply", result)
    assert error is None and standalone == []
    (name, payload), = fired
    assert name == "gateway_message_delivered"
    assert payload["kind"] == "cron" and payload["job_id"] == "job-42"
    assert payload["platform"] == "telegram" and payload["chat_id"] == CHAT_ID and payload["thread_id"] == "5"
    assert payload["message_ids"] == ["100", "101"] and payload["last_message_id"] == "101"
    assert payload["text"] == "ACTION_REQUIRED: apply" and payload["streamed"] is False


def test_unconfirmed_live_send_fires_only_from_the_standalone_lane(fired):
    filtered = {"success": True, "filtered": "silence_narration", "delivered": False}
    error, standalone = _run(_job(), "report", filtered)
    assert error is None and standalone == ["report"]
    (_, payload), = fired
    assert payload["message_ids"] == ["s-1"] and payload["job_id"] == "job-42"


def test_in_flight_timeout_is_assumed_not_confirmed_so_no_fire(fired):
    with patch.object(sched_delivery, "_live_send_text", return_value=(True, True, None)):
        error, standalone = _run(_job(), "report", _SendResult(message_id="1"))
    assert error is None and standalone == []
    assert fired == []


def test_standalone_error_does_not_fire(fired):
    filtered = {"success": True, "filtered": "silence_narration", "delivered": False}
    _run(_job(), "report", filtered, standalone_result={"error": "chat not found"})
    assert fired == []


def test_hook_failure_never_changes_the_delivery_outcome():
    def boom(*a, **k):
        raise RuntimeError("plugin exploded")

    with patch.object(dh, "_has_subscribers", return_value=True), patch("hermes_cli.lifecycle.invoke_hook", boom), \
         patch("hermes_cli.lifecycle.ainvoke_hook", boom):
        error, _ = _run(_job(), "report", _SendResult(message_id="1"))
    assert error is None
