"""A live send that already succeeded must not be re-read through a dict-only accessor.

``SendResult.raw_response`` is adapter-owned: Telegram publishes a dict (and is the only producer of
the ``thread_fallback`` diagnostic key), while the Feishu adapter stores the SDK's response object
(``lark_oapi`` ``ReplyMessageResponse`` — attributes, no ``.get()``).  ``_live_send_text`` read that
payload with ``.get()`` AFTER the send was confirmed, so a Feishu topic delivery raised
``AttributeError`` inside ``_deliver_via_live_adapter``'s blanket handler, reported the target as
failed, and let the caller re-send through the standalone lane: every cron delivery to a Feishu topic
arrived twice (#153500).  A successful send must stay successful.
"""

import asyncio
import threading

import pytest

import cron.scheduler_delivery as sd
from gateway.config import GatewayConfig, Platform
from gateway.delivery import PartialDeliveryError
from gateway.platforms.base import SendResult

THREAD_ID = "omt_19969188f04f9b90"


class _SdkResponse:
    """``lark_oapi``'s reply object: readable fields, no mapping protocol."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _TopicAdapter:
    """A live adapter for one ``thread_id`` target; ``result`` is what the send returns."""

    def __init__(self, result):
        self.result = result
        self.sent: list[dict] = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(dict(metadata or {}))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _deliver(monkeypatch, result):
    """Send one text through the real live lane for a Feishu topic target."""
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    # Seeding a continuation session needs a live gateway store; the delivery verdict is the subject.
    monkeypatch.setattr(sd, "_seed_live_delivery_sessions", lambda *a, **k: None)
    adapter = _TopicAdapter(result)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        target = sd._prepare_target_delivery(
            {"id": "job-1", "origin": _origin()},
            {"platform": "feishu", "chat_id": "oc_abdae", "thread_id": THREAD_ID},
            adapters={Platform.FEISHU: adapter}, loop=loop, config=GatewayConfig(),
            notify_delivery=True, mirror_enabled=False, mirror_text="", delivery_errors=[],
        )
        assert target is not None
        target_errors: list = []
        delivery_errors: list = []
        delivered = sd._deliver_via_live_adapter(
            target, "brief", [], target_errors=target_errors,
            delivery_errors=delivery_errors, unverified_targets=[])
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
    return delivered, target_errors, delivery_errors, adapter


def _origin():
    return {"platform": "feishu", "chat_id": "oc_abdae", "thread_id": THREAD_ID}


def test_successful_topic_send_with_a_non_dict_raw_response_is_delivered(monkeypatch):
    """The reported defect: a confirmed send was reported failed, so the caller re-sent it."""
    delivered, target_errors, _delivery_errors, adapter = _deliver(
        monkeypatch, SendResult(success=True, message_id="om_1", raw_response=_SdkResponse(code=0)))
    assert delivered is True
    assert target_errors == []
    assert len(adapter.sent) == 1


def test_dict_raw_response_still_reports_a_thread_fallback(monkeypatch):
    """The Telegram diagnostic the read exists for keeps working (dict shape)."""
    delivered, target_errors, delivery_errors, _adapter = _deliver(
        monkeypatch,
        SendResult(success=True, message_id="om_1",
                   raw_response={"thread_fallback": True, "requested_thread_id": "999"}),
    )
    assert delivered is True
    assert target_errors == []
    assert any("999" in error and "delivered without thread_id" in error
               for error in delivery_errors)


def test_object_raw_response_does_not_turn_a_partial_send_into_a_resend(monkeypatch):
    """Same accessor in the PartialDeliveryError arm: a non-dict payload must not escape as failure."""
    partial = SendResult(success=False, message_id="om_head",
                         raw_response=_SdkResponse(delivered_chunks=1, total_chunks=3))
    delivered, target_errors, delivery_errors, _adapter = _deliver(
        monkeypatch, PartialDeliveryError("split send failed", partial))
    assert delivered is True
    assert target_errors == []
    assert any("1 of 3 chunks" in error for error in delivery_errors)


@pytest.mark.parametrize("raw", [None, {}, _SdkResponse()])
def test_a_send_without_thread_routing_never_reads_the_diagnostic(monkeypatch, raw):
    """A DM target (no thread_id) short-circuits before the read, as before."""
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    monkeypatch.setattr(sd, "_seed_live_delivery_sessions", lambda *a, **k: None)
    adapter = _TopicAdapter(SendResult(success=True, message_id="om_1", raw_response=raw))
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        target = sd._prepare_target_delivery(
            {"id": "job-1", "origin": {"platform": "feishu", "chat_id": "oc_abdae"}},
            {"platform": "feishu", "chat_id": "oc_abdae"},
            adapters={Platform.FEISHU: adapter}, loop=loop, config=GatewayConfig(),
            notify_delivery=True, mirror_enabled=False, mirror_text="", delivery_errors=[],
        )
        assert target is not None
        target_errors: list = []
        delivered = sd._deliver_via_live_adapter(
            target, "brief", [], target_errors=target_errors,
            delivery_errors=[], unverified_targets=[])
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
    assert delivered is True
    assert target_errors == []
