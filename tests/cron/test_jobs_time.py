"""Unit contract for the per-job schedule knobs: ``cron/jobs_time.py``.

Covers the two values a job may pin — an IANA ``timezone`` for its cron wall clock and a
``grace_minutes`` override for the catch-up window — including the two traps that cost a
silent misbehaviour rather than an exception: ``0 == False`` (a grace of 0 means "never catch
up", not "unset") and a corrupt persisted zone (must degrade to the profile zone, warned once,
never break the 60s tick).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from cron import jobs_time

NY = ZoneInfo("America/New_York")


class TestNormalizeTimezone:
    def test_iana_key_is_stored_canonical(self):
        assert jobs_time.normalize_timezone("America/New_York") == "America/New_York"

    def test_surrounding_whitespace_is_trimmed(self):
        assert jobs_time.normalize_timezone("  Asia/Hong_Kong  ") == "Asia/Hong_Kong"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_empty_means_inherit_the_profile(self, value):
        assert jobs_time.normalize_timezone(value) is None

    @pytest.mark.parametrize("value", ["Mars/Olympus_Mons", "not a zone", "/etc/passwd"])
    def test_unknown_zone_is_rejected_before_storage(self, value):
        with pytest.raises(ValueError, match="timezone"):
            jobs_time.normalize_timezone(value)

    def test_non_string_is_rejected(self):
        with pytest.raises(ValueError, match="IANA name"):
            jobs_time.normalize_timezone(9)


class TestGraceMinutesNormalization:
    """The ``0 == False`` trap: 0 must survive as a real override."""

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_empty_means_use_the_computed_window(self, value):
        assert jobs_time.normalize_grace_minutes(value) is None

    def test_zero_is_a_real_value_not_unset(self):
        assert jobs_time.normalize_grace_minutes(0) == 0
        assert jobs_time.normalize_grace_minutes("0") == 0
        # The whole point: 0 is falsy but must NOT read as "absent".
        assert jobs_time.normalize_grace_minutes(0) is not None

    def test_upper_bound_is_one_day(self):
        assert jobs_time.normalize_grace_minutes(24 * 60) == 24 * 60
        with pytest.raises(ValueError, match="between"):
            jobs_time.normalize_grace_minutes(24 * 60 + 1)

    def test_negative_is_rejected(self):
        with pytest.raises(ValueError, match="between"):
            jobs_time.normalize_grace_minutes(-1)

    @pytest.mark.parametrize("value", [True, False])
    def test_booleans_are_rejected(self, value):
        with pytest.raises(ValueError, match="boolean"):
            jobs_time.normalize_grace_minutes(value)

    @pytest.mark.parametrize("value", ["soon", "1.5", [30]])
    def test_non_numeric_is_rejected(self, value):
        with pytest.raises(ValueError, match="grace_minutes"):
            jobs_time.normalize_grace_minutes(value)

    def test_fractional_float_is_rejected(self):
        with pytest.raises(ValueError, match="whole number"):
            jobs_time.normalize_grace_minutes(1.5)


class TestZoneForJob:
    def test_absent_or_null_inherits_the_profile(self):
        assert jobs_time.zone_for_job({}) is None
        assert jobs_time.zone_for_job({"timezone": None}) is None
        assert jobs_time.zone_for_job({"timezone": ""}) is None

    def test_pinned_zone_is_returned(self):
        assert jobs_time.zone_for_job({"timezone": "America/New_York"}) == NY

    def test_accepts_a_bare_zone_name(self):
        assert jobs_time.zone_for_job("America/New_York") == NY

    def test_corrupt_value_degrades_instead_of_raising(self, monkeypatch):
        """One bad record must never abort the tick it is scanned in."""
        jobs_time.reset_warning_state()
        warnings: list[str] = []
        monkeypatch.setattr(
            jobs_time.logger, "warning",
            lambda msg, *a, **kw: warnings.append(msg % a if a else msg))
        job = {"timezone": "Mars/Olympus_Mons"}

        assert jobs_time.zone_for_job(job) is None
        assert jobs_time.zone_for_job(job) is None  # scanned again a minute later
        assert jobs_time.zone_for_job({"timezone": "Mars/Olympus_Mons"}) is None

        assert len(warnings) == 1, f"a bad zone must warn ONCE, got {warnings}"
        assert "Mars/Olympus_Mons" in warnings[0]


class TestGraceSecondsOverride:
    def test_absent_override_defers_to_the_cadence_window(self):
        assert jobs_time.grace_seconds_override(None) is None

    def test_minutes_become_seconds(self):
        assert jobs_time.grace_seconds_override(30) == 1800

    def test_zero_override_means_never_catch_up(self):
        assert jobs_time.grace_seconds_override(0) == 0

    def test_corrupt_override_is_ignored_with_one_warning(self, monkeypatch):
        jobs_time.reset_warning_state()
        warnings: list[str] = []
        monkeypatch.setattr(
            jobs_time.logger, "warning",
            lambda msg, *a, **kw: warnings.append(msg % a if a else msg))

        assert jobs_time.grace_seconds_override("tomorrow") is None
        assert jobs_time.grace_seconds_override("tomorrow") is None
        assert len(warnings) == 1


class TestPruneInheritedKnobs:
    def test_cleared_zone_drops_the_key(self):
        record = {"id": "a", "timezone": "", "grace_minutes": 0}
        jobs_time.prune_inherited_knobs(record)
        assert "timezone" not in record
        assert record["grace_minutes"] == 0, "grace 0 is a value, not an unset key"

    def test_pinned_zone_is_kept(self):
        record = {"timezone": "Asia/Hong_Kong"}
        jobs_time.prune_inherited_knobs(record)
        assert record["timezone"] == "Asia/Hong_Kong"


class TestOneshotInstant:
    def test_offsetless_run_at_is_the_jobs_wall_clock(self):
        """``09:00`` for a New-York job is 09:00 in New York, not on the host."""
        instant = jobs_time.oneshot_instant("2026-06-01T09:00:00", NY)
        assert instant == datetime(2026, 6, 1, 9, 0, tzinfo=NY)

    def test_offset_bearing_run_at_is_already_an_instant(self):
        instant = jobs_time.oneshot_instant("2026-06-01T09:00:00+00:00", NY)
        assert instant.astimezone(NY) == datetime(2026, 6, 1, 5, 0, tzinfo=NY)

    def test_garbage_is_none(self):
        assert jobs_time.oneshot_instant("not-a-time", NY) is None
