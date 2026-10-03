"""Standalone delivery lane needs the same positive-evidence gate as the live lane (#121725).

Follow-up to #77763 / #100908: that fix's ``_confirm_adapter_delivery`` gate has exactly one
production caller (the live lane in ``_live_send_text``). ``_deliver_standalone`` never calls it,
so a standalone send with no receipt logs a bare ``delivered to <platform>:<chat>`` that is
indistinguishable from an evidenced delivery, and a live -> standalone downgrade leaves no trace
(``target_errors`` is discarded whenever standalone succeeds).
"""

import logging
from unittest.mock import MagicMock, patch

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig

CHAT_ID = "-1001234567890"
WHERE = f"telegram:{CHAT_ID}"


def _job():
    return {
        "id": "92e639af907f",
        "name": "Standalone Evidence",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": CHAT_ID},
    }


def _gateway_config():
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    return config


def _run_standalone(content, standalone_result, record_sink):
    """Drive ``_deliver_result`` down the standalone lane (no live adapters/loop)."""
    async def _sender(*_args, **_kwargs):
        return standalone_result

    with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("cron.scheduler.load_config",
               return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._record_delivery_verification",
               side_effect=record_sink), \
         patch("tools.send_message_tool._send_to_platform", _sender):
        return _deliver_result(_job(), content)


class TestStandaloneEvidenceGate:
    def test_bare_success_is_accepted_but_recorded_unverified(self, caplog):
        """``{"success": True}`` with no message_id/raw_response still delivers (live-lane parity)
        but the target must reach ``last_delivery_unverified``, not just a WARNING line."""
        recorded = []
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            error = _run_standalone(
                "Nightly report.", {"success": True},
                lambda job, targets: recorded.append(list(targets)))

        assert error is None
        assert recorded == [[WHERE]]
        assert "UNVERIFIED" in caplog.text

    def test_evidenced_success_stays_clean(self, caplog):
        recorded = []
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error = _run_standalone(
                "Nightly report.", {"success": True, "message_id": 7},
                lambda job, targets: recorded.append(list(targets)))

        assert error is None
        assert recorded == [[]]
        assert f"delivered to {WHERE}" in caplog.text
        assert "UNVERIFIED" not in caplog.text

    def test_empty_dict_result_fails_closed(self, caplog):
        """``{}`` has no ``success`` key: logging ``delivered`` for it is the #77763 shape."""
        recorded = []
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error = _run_standalone(
                "Nightly report.", {},
                lambda job, targets: recorded.append(list(targets)))

        assert error is not None
        assert "delivered to" not in caplog.text

    def test_none_result_fails_closed(self, caplog):
        recorded = []
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error = _run_standalone(
                "Nightly report.", None,
                lambda job, targets: recorded.append(list(targets)))

        assert error is not None
        assert "delivered to" not in caplog.text


class TestDowngradeIsVisible:
    def test_standalone_lane_logs_the_downgrade(self, caplog):
        """Live lane never attempted (no live transport/loop): the fallback reason must be logged,
        otherwise the bare ``delivered to`` line hides which lane served the target."""
        recorded = []
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            error = _run_standalone(
                "Nightly report.", {"success": True, "message_id": 7},
                lambda job, targets: recorded.append(list(targets)))

        assert error is None
        assert "falling back to standalone" in caplog.text

    def test_event_loop_scheduling_failure_warns_with_fallback(self, caplog):
        """``safe_schedule_threadsafe`` returning None is an ``adapter_ok=False`` path that skips
        ``_warn_live_lane_failure`` entirely — the soft downgrade emits nothing."""
        adapter = MagicMock()
        adapters = {Platform.TELEGRAM: adapter}
        loop = MagicMock()
        loop.is_running.return_value = True

        async def _sender(*_args, **_kwargs):
            return {"success": True, "message_id": 7}

        with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch("cron.scheduler_delivery._record_delivery_verification"), \
             patch("agent.async_utils.safe_schedule_threadsafe", return_value=None), \
             patch("tools.send_message_tool._send_to_platform", _sender):
            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                error = _deliver_result(
                    _job(), "Nightly report.", adapters=adapters, loop=loop)

        assert error is None  # standalone still delivered
        assert "falling back to standalone" in caplog.text
        assert "event loop scheduling failed" in caplog.text
