"""Watchdog outage notice (CORP-836): the Windows hang watchdog records an estimated outage window in
the planned-restart marker, and the recovered gateway turns it into its home-channel notice telling
people which window to resend requests from.

unittest.TestCase so it runs under ``python -m unittest`` too; ``gateway.watchdog_outage`` is stdlib-only.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gateway import watchdog_outage as outage

_T = 1_793_400_000.0  # a realistic epoch: Windows refuses local-time conversion near 1970
_KEY_KNOWN = "gateway.startup.watchdog_recovered"
_KEY_UNKNOWN_START = "gateway.startup.watchdog_recovered_start_unknown"


class _Home(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.marker = self.home / outage.PLANNED_RESTART_MARKER_FILENAME

    def read(self):
        return json.loads(self.marker.read_text(encoding="utf-8"))


class RecordOutageTest(_Home):
    def test_records_only_the_window_and_owes_every_home_the_new_notice(self):
        self.marker.write_text(json.dumps({"requested_at": 1.0, "delivered_targets": [["", "slack", "C1", None]]}),
                               encoding="utf-8")
        with mock.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-SECRET-123"}):
            outage.record_outage(self.home, last_healthy_at=1000.0, detected_at=1060.0)
        data = self.read()
        self.assertEqual(data[outage.OUTAGE_KEY], {"last_healthy_at": 1000.0, "detected_at": 1060.0})
        self.assertEqual(data["requested_at"], 1.0)  # an earlier planned restart's fields survive
        self.assertNotIn("delivered_targets", data)  # homes told "online" before still owe this notice
        self.assertNotIn("xoxb-SECRET-123", self.marker.read_text(encoding="utf-8"))

    def test_a_second_record_before_recovery_keeps_the_earlier_start(self):
        outage.record_outage(self.home, last_healthy_at=1000.0, detected_at=1060.0)
        outage.record_outage(self.home, last_healthy_at=2000.0, detected_at=2060.0)
        self.assertEqual(self.read()[outage.OUTAGE_KEY]["last_healthy_at"], 1000.0)

    def test_a_new_outage_after_a_recovered_one_replaces_it(self):
        outage.record_outage(self.home, last_healthy_at=1000.0, detected_at=1060.0)
        data = self.read()
        outage.stamp_recovered_at(data, 1200.0)
        self.marker.write_text(json.dumps(data), encoding="utf-8")
        outage.record_outage(self.home, last_healthy_at=5000.0, detected_at=5060.0)
        self.assertEqual(self.read()[outage.OUTAGE_KEY], {"last_healthy_at": 5000.0, "detected_at": 5060.0})

    def test_unknown_start_is_recorded_as_unknown_not_guessed(self):
        outage.record_outage(self.home, last_healthy_at=None, detected_at=1060.0)
        self.assertEqual(self.read()[outage.OUTAGE_KEY], {"last_healthy_at": None, "detected_at": 1060.0})


class RetractOutageTest(_Home):
    def test_retracting_restores_the_marker_exactly_or_removes_a_new_one(self):
        earlier = json.dumps({"requested_at": 1.0, "delivered_targets": [["", "slack", "C1", None]]})
        self.marker.write_text(earlier, encoding="utf-8")
        outage.retract_outage(self.home, outage.record_outage(self.home, last_healthy_at=1.0, detected_at=2.0))
        self.assertEqual(self.marker.read_text(encoding="utf-8"), earlier)
        self.marker.unlink()
        outage.retract_outage(self.home, outage.record_outage(self.home, last_healthy_at=1.0, detected_at=2.0))
        self.assertFalse(self.marker.exists())

    def test_a_record_that_wrote_nothing_retracts_nothing(self):
        outage.record_outage(self.home, last_healthy_at=1000.0, detected_at=1060.0)
        receipt = outage.record_outage(self.home, last_healthy_at=2000.0, detected_at=2060.0)
        self.assertIsNone(receipt)  # the earlier, still-pending outage stays the one owed
        outage.retract_outage(self.home, receipt)
        self.assertEqual(self.read()[outage.OUTAGE_KEY]["detected_at"], 1060.0)


class CarryPendingOutageTest(_Home):
    def test_a_rewritten_planned_restart_marker_keeps_the_pending_outage(self):
        outage.record_outage(self.home, last_healthy_at=1000.0, detected_at=1060.0)
        marker = outage.carry_pending_outage(self.marker, {"requested_at": 2.0})
        self.assertEqual(marker, {"requested_at": 2.0,
                                  outage.OUTAGE_KEY: {"last_healthy_at": 1000.0, "detected_at": 1060.0}})

    def test_a_missing_or_torn_marker_carries_nothing(self):
        self.assertEqual(outage.carry_pending_outage(self.marker, {"requested_at": 2.0}), {"requested_at": 2.0})
        self.marker.write_text("{torn", encoding="utf-8")
        self.assertEqual(outage.carry_pending_outage(self.marker, {"requested_at": 2.0}), {"requested_at": 2.0})


class OutageNoticeTest(unittest.TestCase):
    def test_no_outage_means_the_plain_online_notice(self):
        self.assertIsNone(outage.outage_notice({"requested_at": 1.0}))

    def test_an_unrecovered_record_is_never_announced(self):
        self.assertIsNone(outage.outage_notice({outage.OUTAGE_KEY: {"last_healthy_at": 1000.0, "detected_at": 1060.0}}))

    def test_known_window_names_start_and_end(self):
        data = {outage.OUTAGE_KEY: {"last_healthy_at": _T, "detected_at": _T + 60}}
        self.assertTrue(outage.stamp_recovered_at(data, _T + 300))
        self.assertFalse(outage.stamp_recovered_at(data, _T + 9999))  # replays keep the first recovery time
        key, kwargs = outage.outage_notice(data)
        self.assertEqual(key, _KEY_KNOWN)
        self.assertEqual(kwargs, {"start": outage.format_time(_T), "end": outage.format_time(_T + 300)})
        self.assertNotEqual(kwargs["start"], kwargs["end"])

    def test_unknown_or_inconsistent_start_is_not_presented_as_a_window(self):
        for start in (None, "x", _T + 1000):  # missing, malformed, after detection
            data = {outage.OUTAGE_KEY: {"last_healthy_at": start, "detected_at": _T + 60, "recovered_at": _T + 300}}
            key, kwargs = outage.outage_notice(data)
            self.assertEqual(key, _KEY_UNKNOWN_START, start)
            self.assertEqual(kwargs, {"detected": outage.format_time(_T + 60), "end": outage.format_time(_T + 300)})

    def test_times_are_shown_as_wall_clock_with_an_offset(self):
        self.assertRegex(outage.format_time(_T), r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} [+-]\d{4}$")
        self.assertRegex(outage.format_time(1000.0), r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} [+-]\d{4}$")

    def test_malformed_detection_or_recovery_falls_back_to_the_plain_notice(self):
        for record in ({"detected_at": "x", "recovered_at": 1300.0}, {"detected_at": 1060.0, "recovered_at": 1000.0},
                       "garbage"):
            self.assertIsNone(outage.outage_notice({outage.OUTAGE_KEY: record}), record)


if __name__ == "__main__":
    unittest.main()
