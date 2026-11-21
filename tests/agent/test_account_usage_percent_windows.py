"""Regression tests for provider account-usage window parsing.

The specific bug these pin: Anthropic's OAuth usage endpoint reports ``utilization`` as a
PERCENT (0-100). An older parser scaled any value <= 1 by 100 to accommodate a fraction shape,
so a weekly window legitimately reporting ``1.0`` ("1% of your weekly limit used", per the
Claude dashboard) was rendered as 100% used / 0% remaining.

Live payload captured 2026-11-03 from https://api.anthropic.com/api/oauth/usage:

    "five_hour": {"utilization": 5.0, "resets_at": "2026-11-04T00:39:59.597027+00:00"}
    "seven_day": {"utilization": 1.0, "resets_at": "2026-11-08T07:59:59.597051+00:00"}
    "limits": [
        {"kind": "session",    "group": "session", "percent": 5, "resets_at": "...", "is_active": True},
        {"kind": "weekly_all", "group": "weekly",  "percent": 1, "resets_at": "...", "is_active": False},
    ]
"""

from agent import account_usage


def _anthropic_payload(five_hour=5.0, seven_day=1.0):
    """A trimmed but structurally faithful Anthropic OAuth usage payload."""
    return {
        "five_hour": {"utilization": five_hour, "resets_at": "2026-11-04T00:39:59.597027+00:00",
                      "limit_dollars": None, "used_dollars": None, "remaining_dollars": None,
                      "locked_reason": None},
        "seven_day": {"utilization": seven_day, "resets_at": "2026-11-08T07:59:59.597051+00:00",
                      "limit_dollars": None, "used_dollars": None, "remaining_dollars": None,
                      "locked_reason": None},
        "seven_day_opus": None,
        "seven_day_sonnet": None,
        "extra_usage": {"is_enabled": False, "monthly_limit": None, "used_credits": None,
                        "utilization": None, "currency": None},
        "limits": [
            {"kind": "session", "group": "session", "percent": int(five_hour),
             "severity": "normal", "resets_at": "2026-11-04T00:39:59.597027+00:00",
             "scope": None, "is_active": True},
            {"kind": "weekly_all", "group": "weekly", "percent": int(seven_day),
             "severity": "normal", "resets_at": "2026-11-08T07:59:59.597051+00:00",
             "scope": None, "is_active": False},
        ],
    }


def _windows(payload):
    return account_usage._usage_windows(
        payload,
        (("five_hour", "Current session"), ("seven_day", "Current week"),
         ("seven_day_opus", "Opus week"), ("seven_day_sonnet", "Sonnet week")),
        "utilization", "resets_at",
        model_scoped={"seven_day_opus", "seven_day_sonnet"},
    )


def test_utilization_of_one_is_one_percent_not_one_hundred():
    """Regression: utilization 1.0 is '1% used', never '100% used'."""
    windows = _windows(_anthropic_payload())
    by_label = {w.label: w for w in windows}
    assert by_label["Current session"].used_percent == 5.0
    assert by_label["Current week"].used_percent == 1.0, (
        "utilization is a percent; 1.0 means 1% used, not 100%"
    )


def test_zero_utilization_is_zero_percent():
    windows = _windows(_anthropic_payload(five_hour=0.0, seven_day=0.0))
    by_label = {w.label: w for w in windows}
    assert by_label["Current session"].used_percent == 0.0
    assert by_label["Current week"].used_percent == 0.0


def test_utilization_above_one_is_not_rescaled():
    """A genuinely high percentage must survive untouched."""
    windows = _windows(_anthropic_payload(five_hour=87.5, seven_day=99.0))
    by_label = {w.label: w for w in windows}
    assert by_label["Current session"].used_percent == 87.5
    assert by_label["Current week"].used_percent == 99.0


def test_missing_and_null_windows_are_skipped():
    windows = _windows(_anthropic_payload())
    labels = {w.label for w in windows}
    assert labels == {"Current session", "Current week"}, "null model-scoped windows are skipped"


def test_usage_windows_helper_takes_no_fraction_scaling_flag():
    """The fraction heuristic was the bug; it must not come back."""
    import inspect
    params = inspect.signature(account_usage._usage_windows).parameters
    assert "fraction" not in params
