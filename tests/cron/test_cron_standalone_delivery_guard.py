"""Regression tests for the cron standalone fallback lane's delivery guards.

The live lane sends through ``gateway.delivery.DeliveryRouter._deliver_to_platform``, where the
degeneracy detector and the ``cron.max_delivery_chars`` cap run before any adapter sees a payload.
The standalone fallback lane called ``tools.send_message_tool._send_to_platform`` directly, so
whenever the live adapter failed — the exact condition the fallback exists for — a degenerate
72,000-character payload reached the wire verbatim, re-opening the 2026-11-02 incident on the one
lane that had no guard (PR 148960 review finding 1).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import cron.scheduler_delivery as sd
from cron.scheduler import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.degenerate_output import DEFAULT_MAX_DELIVERY_CHARS

CHAT_ID = "-1001234567890"

# The incident shape: one phrase repeated to exhaustion, non-degenerate nowhere.
_PURE_LOOP = "NOW FIN. " * 8000  # 72,000 chars — the size measured on the review

# A legitimately long report: every line differs, so the detector must leave it alone.
_LONG_REPORT = "\n".join(
    f"- board-{i % 4}: item {i} verified by the reviewer, closing now ({i * 7 % 101})"
    for i in range(1500)
)


def _job():
    return {
        "id": "92e639af907f",
        "name": "Standalone Lane Report",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": CHAT_ID},
    }


def _gateway_config():
    return GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})


def _deliver_standalone_lane(payload: str, monkeypatch) -> str:
    """Run one fire with NO live adapters — the standalone fallback lane — and return the text
    handed to ``tools.send_message_tool._send_to_platform`` (i.e. what the wire would carry)."""
    sent: list = []

    async def _send_to_platform(platform, pconfig, chat_id, message, **_kwargs):
        sent.append(message)
        return {"success": True, "message_id": 7}

    with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._record_delivery_verification"), \
         patch("tools.send_message_tool._send_to_platform", _send_to_platform):
        error = _deliver_result(_job(), payload)

    assert error is None
    assert len(sent) == 1
    return sent[0]


def _target():
    """A ``_TargetDelivery`` for the fallback lane, with no live transport."""
    return sd._TargetDelivery(
        job={"id": "92e639af907f"}, platform=Platform.TELEGRAM, platform_name="telegram",
        chat_id=CHAT_ID, thread_id=None, transport=None, pconfig=None, runtime_adapter=None,
        target_adapters={}, config=GatewayConfig(), loop=None, notify_delivery=False,
        origin={}, origin_target=True, origin_user_id=None, is_dm_target=True, mirror_text="",
        mirror_this_target=False, in_channel_surface=False, inchannel_continuable=False,
        opened_thread_id=None,
    )


class TestStandaloneLaneIsGuarded:
    def test_degenerate_payload_is_not_delivered_verbatim(self, monkeypatch, tmp_path):
        delivered = _deliver_standalone_lane(_PURE_LOOP, monkeypatch)

        assert delivered != _PURE_LOOP
        assert "NOW FIN" not in delivered, "the looping tail reached the wire on the fallback lane"
        assert len(delivered) < len(_PURE_LOOP) / 100

    def test_degenerate_payload_keeps_an_audit_pointer_to_the_original(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        delivered = _deliver_standalone_lane(_PURE_LOOP, monkeypatch)

        saved = list((tmp_path / "cron" / "output").glob("92e639af907f_*.txt"))
        assert len(saved) == 1, "the collapse must leave the full original on disk"
        assert saved[0].read_text(encoding="utf-8") == _PURE_LOOP
        assert str(saved[0]) in delivered, "the delivered message must point at the audit copy"

    def test_oversized_normal_report_is_capped(self, monkeypatch):
        assert len(_LONG_REPORT) > DEFAULT_MAX_DELIVERY_CHARS
        delivered = _deliver_standalone_lane(_LONG_REPORT, monkeypatch)

        assert len(delivered) <= DEFAULT_MAX_DELIVERY_CHARS
        assert delivered.startswith(_LONG_REPORT[:200])
        assert "truncated at" in delivered

    def test_short_report_is_delivered_unchanged(self, monkeypatch):
        report = "Nightly patrol done — 2 open cards, 0 failures."
        assert _deliver_standalone_lane(report, monkeypatch) == report

    def test_payload_queued_for_reconnect_redelivery_is_also_bounded(self, monkeypatch):
        """The ledger replay has no guards of its own, so the fallback lane must queue bounded text."""
        queued: list = []
        monkeypatch.setattr(sd, "_standalone_send", lambda t, content, media: (None, "no token"))
        monkeypatch.setattr(
            sd, "_queue_for_live_reconnect",
            lambda t, content, media, errors: queued.append(content))

        sd._deliver_standalone(_target(), _PURE_LOOP, [], [], [])

        assert len(queued) == 1
        assert "NOW FIN" not in queued[0]
        assert len(queued[0]) < len(_PURE_LOOP) / 100


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
