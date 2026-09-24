"""Live-adapter delivery confirmation for cron (#77763).

A ``no_agent`` job fired, the scheduler logged
``delivered to telegram:<chat> via live adapter``, and the user received
nothing — no message row, no delivery obligation. The log line was not
evidence of a send:

* the silence-narration filter returns ``{"success": True, "delivered": False}``
  (a successful *drop*), and the normalization block read only ``success``;
* an empty payload skipped the send entirely and still fell into the
  "delivered" branch;
* the log line named the chat but not the lane, so a wrong-thread delivery and
  a phantom one look identical after the fact.

These tests pin the confirmation contract: positive evidence, honest logging,
and fail-closed on nothing-to-send.
"""

import asyncio
import logging
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler_delivery as sched_delivery
from cron.scheduler import _deliver_result
from cron.scheduler_delivery import _confirm_adapter_delivery
from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# _confirm_adapter_delivery: the contract in isolation
# ---------------------------------------------------------------------------

class _SendResult:
    """Minimal stand-in for an adapter SendResult."""

    def __init__(self, success=True, message_id=None, raw_response=None, **extra):
        self.success = success
        self.message_id = message_id
        self.raw_response = raw_response
        for key, value in extra.items():
            setattr(self, key, value)


class TestConfirmAdapterDelivery:
    def test_none_is_not_delivered(self):
        assert _confirm_adapter_delivery(None, "j1") is False

    def test_missing_success_is_not_delivered(self):
        assert _confirm_adapter_delivery(object(), "j1") is False
        assert _confirm_adapter_delivery({"message_id": 7}, "j1") is False

    def test_explicit_failure_is_not_delivered(self):
        assert _confirm_adapter_delivery(_SendResult(success=False), "j1") is False
        assert _confirm_adapter_delivery({"success": False}, "j1") is False

    def test_filtered_dict_is_not_delivered(self):
        """The exact silence-filter shape: a successful DROP is not a delivery."""
        filtered = {"success": True, "filtered": "silence_narration", "delivered": False}
        assert _confirm_adapter_delivery(filtered, "j1") is False


    def test_positive_evidence_is_delivered_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            assert _confirm_adapter_delivery(_SendResult(message_id=1234), "j1") is True
        assert "UNVERIFIED" not in caplog.text

    def test_raw_response_alone_counts_as_evidence(self, caplog):
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            result = _SendResult(raw_response={"ok": True})
            assert _confirm_adapter_delivery(result, "j1") is True
        assert "UNVERIFIED" not in caplog.text

    def test_evidence_free_success_is_accepted_but_warned(self, caplog):
        """Not proof of failure either — accept it, but say so in the log."""
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            assert _confirm_adapter_delivery(_SendResult(), "92e639af907f") is True
        assert "UNVERIFIED" in caplog.text
        assert "92e639af907f" in caplog.text



# ---------------------------------------------------------------------------
# _deliver_result: the live lane end to end
# ---------------------------------------------------------------------------

CHAT_ID = "-1001234567890"


def _job(thread_id=None):
    origin = {"platform": "telegram", "chat_id": CHAT_ID}
    if thread_id is not None:
        origin["thread_id"] = thread_id
    return {
        "id": "92e639af907f",
        "name": "Ghost Delivery",
        "deliver": "origin",
        "origin": origin,
    }


def _gateway_config(relay=False):
    config = MagicMock()
    platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    if relay:
        platforms[Platform.RELAY] = PlatformConfig(enabled=True)
    config.platforms = platforms
    config.get_home_channel = lambda p: None
    return config


def _adapters(relay=False):
    adapter = MagicMock()
    if relay:
        adapter.fronts_platform = lambda p: p == Platform.TELEGRAM
        return {Platform.RELAY: adapter}
    return {Platform.TELEGRAM: adapter}


RECORDED_VERIFICATION = []


def _record_verification(job, unverified_targets):
    RECORDED_VERIFICATION.append((job["id"], list(unverified_targets)))


def _run(job, content, send_result, relay=False, standalone_result=None, cron_cfg=None):
    """Drive ``_deliver_result`` over the live lane with a stubbed router.

    Returns ``(error, router_calls, standalone_calls)``. ``cron_cfg`` extends
    the ``cron:`` section handed to the scheduler (default: unwrapped output).
    """
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            future.set_exception(e)
        return future

    router_calls = []
    standalone_calls = []
    RECORDED_VERIFICATION.clear()

    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata, transport=None):
        router_calls.append({"target": target, "text": text, "metadata": metadata})
        return send_result

    router._deliver_to_platform = _deliver_to_platform

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone_calls.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})
        return standalone_result if standalone_result is not None else {}

    with patch("gateway.config.load_gateway_config", return_value=_gateway_config(relay)), \
         patch("cron.scheduler.load_config",
               return_value={"cron": {"wrap_response": False, **(cron_cfg or {})}}), \
         patch("cron.scheduler_delivery._record_delivery_verification", side_effect=_record_verification), \
         patch("gateway.delivery.DeliveryRouter", return_value=router), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, content, adapters=_adapters(relay), loop=loop)
    return error, router_calls, standalone_calls


class TestFilteredResultIsNotDelivered:
    FILTERED = {"success": True, "filtered": "silence_narration", "delivered": False}

    def test_filtered_dict_does_not_log_a_live_delivery(self, caplog):
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            _, router_calls, standalone_calls = _run(_job(), "...", self.FILTERED)

        assert len(router_calls) == 1                     # the live send was attempted
        assert "via live adapter" not in caplog.text      # but never claimed as delivered
        assert len(standalone_calls) == 1                 # fell back instead of lying

    def test_filtered_dict_fails_closed_on_the_relay_lane(self):
        """Relay owns the destination, so there is no fallback — report it."""
        error, _, standalone_calls = _run(_job(), "...", self.FILTERED, relay=True)

        assert error is not None
        assert "unconfirmed result" in error
        assert "silence_narration" in error  # names the filter, not "unknown"
        assert standalone_calls == []

    def test_confirmed_send_result_still_delivers(self, caplog):
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, router_calls, standalone_calls = _run(
                _job(), "Nightly report.", _SendResult(message_id=1234),
            )

        assert error is None
        assert len(router_calls) == 1
        assert standalone_calls == []
        assert "via live adapter" in caplog.text


class TestEmptyPayloadFailsClosed:

    def test_empty_payload_never_reaches_the_standalone_sender(self, caplog):
        """The native fallback must not re-open the hole the live lane closed.

        Telegram's adapter returns ``SendResult(success=True)`` for empty
        content without an API call, so an unguarded fallback would log a
        standalone "delivered" for the same phantom payload (#77763).
        """
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, router_calls, standalone_calls = _run(
                _job(), "   ", _SendResult(message_id=1),
            )

        assert router_calls == []
        assert standalone_calls == []  # _send_to_platform never called
        assert error is not None
        assert "standalone send skipped (empty text and no media)" in error
        assert "delivered to" not in caplog.text

    def test_empty_payload_is_reported_on_the_relay_lane(self):
        error, router_calls, _ = _run(_job(), "", _SendResult(message_id=1), relay=True)

        assert router_calls == []
        assert error is not None
        assert "live adapter send skipped (empty text and no media)" in error




class TestLiveDeliveryIsAFinalNotification:
    """Cron output is a final user-visible delivery, not a progress send.

    Telegram's adapter defaults to ``_notifications_mode = "important"`` and
    sends with ``disable_notification=True`` unless ``metadata["notify"]`` is
    set — so a cron brief without the marker lands silently, which users
    report as "never delivered" (#77763 thread, #58258 typing bubble). The
    marker must ride both the text route and the media route, in every
    Telegram routing mode.
    """

    def test_text_route_metadata_carries_notify(self):
        _, router_calls, _ = _run(_job(), "Nightly report.", _SendResult(message_id=1))
        assert len(router_calls) == 1
        metadata = router_calls[0]["metadata"]
        assert metadata["job_id"] == "92e639af907f"
        assert metadata["notify"] is True

    def test_forum_topic_route_keeps_thread_and_notify(self):
        _, router_calls, _ = _run(
            _job(thread_id="99"), "Nightly report.", _SendResult(message_id=1),
        )
        metadata = router_calls[0]["metadata"]
        assert metadata["thread_id"] == "99"
        assert metadata["notify"] is True

    def test_media_route_metadata_carries_notify(self, tmp_path):
        media = tmp_path / "report.png"
        media.write_bytes(b"\x89PNG\r\n\x1a\n")
        sent = []

        def fake_send_media(adapter, chat_id, media_files, metadata, loop, job, platform=None):
            sent.append({"media": list(media_files), "metadata": metadata})
            return []

        with patch("cron.scheduler_delivery._send_media_via_adapter", side_effect=fake_send_media), \
             patch("gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
                   side_effect=lambda files: files):
            error, router_calls, _ = _run(
                _job(), f"Nightly report.\nMEDIA:{media}", _SendResult(message_id=1),
            )

        assert error is None
        assert len(router_calls) == 1
        assert len(sent) == 1
        assert sent[0]["metadata"]["notify"] is True


class TestNotifyIsConfigurable:
    """``cron.delivery.notify`` (config.yaml) gates the notify marker.

    The current behaviour (push notification) stays the default; only an
    explicit ``false`` restores silent deliveries. The knob rides both the
    text route and the media route so the two never disagree.
    """


    def test_explicit_false_disables_notify_on_text_route(self):
        _, router_calls, _ = _run(
            _job(thread_id="99"), "Nightly report.", _SendResult(message_id=1),
            cron_cfg={"delivery": {"notify": False}},
        )
        metadata = router_calls[0]["metadata"]
        assert metadata["notify"] is False
        assert metadata["thread_id"] == "99"  # routing untouched

    def test_explicit_false_disables_notify_on_media_route(self, tmp_path):
        media = tmp_path / "report.png"
        media.write_bytes(b"\x89PNG\r\n\x1a\n")
        sent = []

        def fake_send_media(adapter, chat_id, media_files, metadata, loop, job, platform=None):
            sent.append(metadata)
            return []

        with patch("cron.scheduler_delivery._send_media_via_adapter", side_effect=fake_send_media), \
             patch("gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
                   side_effect=lambda files: files):
            _run(
                _job(), f"Nightly report.\nMEDIA:{media}", _SendResult(message_id=1),
                cron_cfg={"delivery": {"notify": False}},
            )
        assert sent[0]["notify"] is False

    @pytest.mark.parametrize("cron_cfg", [
        {"delivery": None},            # `delivery:` with no body parses to null
        {"delivery": "yes"},           # malformed scalar
        {"delivery": {"notify": None}},  # `notify:` with no value
    ])
    def test_malformed_section_keeps_the_default(self, cron_cfg):
        _, router_calls, _ = _run(_job(), "Nightly report.", _SendResult(message_id=1), cron_cfg=cron_cfg)
        assert router_calls[0]["metadata"]["notify"] is True



class TestUnverifiedDeliveryIsRecordedOnTheJob:
    """An evidence-free ack is accepted, but the state must reach the job
    record (and from there ``hermes cron list`` / ``cron doctor``), not only a
    WARNING log line."""

    def test_evidence_free_ack_records_the_target(self):
        error, _, _ = _run(_job(), "Nightly report.", _SendResult())
        assert error is None
        assert RECORDED_VERIFICATION == [("92e639af907f", [f"telegram:{CHAT_ID}"])]

    def test_positive_evidence_clears_the_marker(self):
        error, _, _ = _run(_job(), "Nightly report.", _SendResult(message_id=1234))
        assert error is None
        assert RECORDED_VERIFICATION == [("92e639af907f", [])]


    def test_recorder_clears_a_stale_marker(self):
        with patch("cron.jobs.update_job") as update_job:
            sched_delivery._record_delivery_verification({"id": "j1", "last_delivery_unverified": ["slack:C1"]}, [])
            update_job.assert_called_once_with("j1", {"last_delivery_unverified": None})





class TestStandaloneSendIsBounded:
    """The standalone fallback lane must not wait on its send unbounded (#115469).

    ``_send_to_platform``'s gateway-loop dispatch awaits with a deliberate no-timeout shield
    whose comment assumes an outer ``_run_async`` bound — but this lane's outer runner is a bare
    ``asyncio.run``, so a mid-reconnect transport pinned the run (and the restart drain behind
    it) for hours while the job's script had finished in seconds.
    """

    @staticmethod
    def _deliver_standalone(sender, cron_cfg):
        """Drive the production entry point with no live adapters (the standalone lane)."""
        with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False, **cron_cfg}}), \
             patch("cron.scheduler_delivery._record_delivery_verification"), \
             patch("tools.send_message_tool._send_to_platform", sender):
            return _deliver_result(_job(), "Nightly report.")

    def test_hung_send_is_released_at_the_configured_bound(self, caplog):
        async def _hang(*_args, **_kwargs):
            await asyncio.Event().wait()  # transport mid-reconnect: the send never resolves

        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            started = time.monotonic()
            error = self._deliver_standalone(_hang, {"standalone_send_timeout_seconds": 1})

        assert time.monotonic() - started < 30  # released at the bound, not never
        assert error is not None
        assert "timed out after 1s" in error
        assert "in flight" in error  # an un-cancelled shielded send may still land
        assert "via live adapter" not in caplog.text and "delivered to" not in caplog.text

    def test_a_timely_send_is_unaffected(self, caplog):
        async def _ok(*_args, **_kwargs):
            return {"success": True, "message_id": 7}

        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error = self._deliver_standalone(_ok, {})

        assert error is None
        assert f"delivered to telegram:{CHAT_ID}" in caplog.text


class TestStandaloneLaneGetsTheEvidenceGate:
    """#121725: the #100908 evidence gate covered only the live-adapter lane;
    the standalone fallback still logged ``delivered`` for evidence-free acks
    and left no trace on the job."""

    def test_evidence_free_standalone_ack_is_flagged_unverified(self, caplog):
        """A bare-ack success is accepted, but it must leave the same UNVERIFIED
        trace (WARNING + ``last_delivery_unverified``) the live lane has left
        since #100908 — not a silent ``delivered`` (#121725)."""
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, _, standalone_calls = _run(
                _job(), "Nightly report.", None, standalone_result={"success": True})

        assert error is None
        assert len(standalone_calls) == 1
        assert "UNVERIFIED" in caplog.text
        assert RECORDED_VERIFICATION == [("92e639af907f", [f"telegram:{CHAT_ID}"])]

    def test_standalone_positive_evidence_clears_the_marker(self):
        error, _, _ = _run(
            _job(), "Nightly report.", None,
            standalone_result={"success": True, "message_id": 7})

        assert error is None
        assert RECORDED_VERIFICATION == [("92e639af907f", [])]

    def test_standalone_unconfirmed_result_is_not_delivered(self, caplog):
        """A success-less sender response is a failed delivery, not a delivery."""
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, _, _ = _run(_job(), "Nightly report.", None, standalone_result={})

        assert error is not None
        assert "unconfirmed result" in error
        assert "delivered to" not in caplog.text
