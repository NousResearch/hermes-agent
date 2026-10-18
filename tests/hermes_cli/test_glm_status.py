"""Peak-pricing reminder fragment in the GLM status bar (local patch).

Peak = Mon-Fri 14:00-18:00 UTC+8 (Singapore = Beijing time). GLM-5.3 bills
3x in peak (1x off), Flash-family 1.2x (0.4x off)."""
from datetime import datetime, timedelta, timezone

import hermes_cli.glm_status as gs
from hermes_cli.glm_status import _peak_fragment, build_segment

SG = timezone(timedelta(hours=8))
# 2026-10-14 is a Wednesday, 10-16 Friday, 10-17 Saturday.


def _wed(h, m=0):
    return datetime(2026, 10, 14, h, m, tzinfo=SG)


def test_peak_active_shows_multiplier():
    assert _peak_fragment("glm-5.3", _wed(15))[0] == " ⚡峰3×"


def test_peak_flash_uses_flash_multiplier():
    assert _peak_fragment("glm-5.2-flash", _wed(15))[0] == " ⚡峰1.2×"


def test_warning_within_one_hour_before_peak():
    assert _peak_fragment("glm-5.3", _wed(13, 30))[0] == " ⚡30m后入峰3×"


def test_offpeak_weekend_and_far_away_are_quiet():
    assert _peak_fragment("glm-5.3", _wed(20)) == ("", "")  # evening
    assert _peak_fragment("glm-5.3", _wed(10)) == ("", "")  # morning, >60m out
    sat = datetime(2026, 10, 17, 15, 0, tzinfo=SG)
    assert _peak_fragment("glm-5.3", sat) == ("", "")  # weekend
    fri = datetime(2026, 10, 16, 18, 30, tzinfo=SG)  # next peak is Monday
    assert _peak_fragment("glm-5.3", fri) == ("", "")


def test_peak_window_boundaries():
    assert _peak_fragment("glm-5.3", _wed(14))[0] == " ⚡峰3×"  # 14:00 sharp in
    assert _peak_fragment("glm-5.3", _wed(18)) == ("", "")  # 18:00 sharp out


def test_build_segment_appends_peak_fragment(monkeypatch):
    monkeypatch.setattr(gs, "_peak_fragment", lambda model="": (" ⚡峰3×", "warn"))
    data = {"level": "pro", "five_pct": 2, "five_reset_ms": None,
            "week_pct": 6, "week_reset_ms": None}
    text = "".join(t for _k, t in build_segment(data, 120, "glm-5.3"))
    assert "⚡峰3×" in text
    assert text.startswith(" │ GLM pro")  # existing layout intact


def test_build_segment_without_model_still_renders():
    data = {"level": "pro", "five_pct": 2, "five_reset_ms": None,
            "week_pct": 6, "week_reset_ms": None}
    text = "".join(t for _k, t in build_segment(data, 120))
    assert "GLM" in text and "5h 2%" in text
