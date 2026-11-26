"""Which window of a Codex usage snapshot the admission policy judges."""

from datetime import UTC, datetime, timedelta

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.usage_admission.policy import _select_codex_policy_window


def _snapshot(*windows: AccountUsageWindow) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(UTC),
        windows=tuple(windows),
    )


def test_policy_window_uses_longest_duration_not_primary_secondary_label():
    seven_day_primary = AccountUsageWindow(
        label="Session",
        used_percent=79,
        limit_window_seconds=604_800,
    )
    five_hour_secondary = AccountUsageWindow(
        label="Weekly",
        used_percent=95,
        limit_window_seconds=18_000,
    )

    selected = _select_codex_policy_window(
        _snapshot(seven_day_primary, five_hour_secondary)
    )

    assert selected is seven_day_primary


def test_policy_window_falls_back_to_farthest_reset():
    now = datetime.now(UTC)
    earlier = AccountUsageWindow(
        label="Primary",
        used_percent=95,
        reset_at=now + timedelta(hours=2),
    )
    later = AccountUsageWindow(
        label="Secondary",
        used_percent=79,
        reset_at=now + timedelta(days=5),
    )

    selected = _select_codex_policy_window(_snapshot(earlier, later))

    assert selected is later


def test_policy_window_without_duration_or_reset_fails_open():
    unknown = AccountUsageWindow(label="Unknown", used_percent=95)

    assert _select_codex_policy_window(_snapshot(unknown)) is None


def test_policy_window_mixed_durations_fail_open():
    now = datetime.now(UTC)
    shorter_with_duration = AccountUsageWindow(
        label="Short",
        used_percent=95,
        limit_window_seconds=18_000,
        reset_at=now + timedelta(hours=2),
    )
    longer_without_duration = AccountUsageWindow(
        label="Long",
        used_percent=79,
        reset_at=now + timedelta(days=5),
    )

    assert (
        _select_codex_policy_window(
            _snapshot(shorter_with_duration, longer_without_duration)
        )
        is None
    )


@pytest.mark.parametrize("malformed_duration", ["604800", 0, -1, float("nan")])
def test_policy_window_malformed_duration_with_complete_resets_fails_open(
    malformed_duration,
):
    now = datetime.now(UTC)
    valid = AccountUsageWindow(
        label="Valid",
        used_percent=79,
        limit_window_seconds=18_000,
        reset_at=now + timedelta(hours=2),
    )
    malformed = AccountUsageWindow(
        label="Malformed",
        used_percent=95,
        limit_window_seconds=malformed_duration,
        reset_at=now + timedelta(days=5),
    )

    assert _select_codex_policy_window(_snapshot(valid, malformed)) is None


def test_policy_window_mixed_durations_without_complete_resets_fails_open():
    shorter_with_duration = AccountUsageWindow(
        label="Short",
        used_percent=95,
        limit_window_seconds=18_000,
        reset_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    unknown_longer = AccountUsageWindow(label="Unknown", used_percent=79)

    assert (
        _select_codex_policy_window(_snapshot(shorter_with_duration, unknown_longer))
        is None
    )


def test_policy_window_equal_longest_duration_tie_fails_open():
    first = AccountUsageWindow(
        label="First",
        used_percent=90,
        limit_window_seconds=604_800,
    )
    second = AccountUsageWindow(
        label="Second",
        used_percent=10,
        limit_window_seconds=604_800,
    )

    assert _select_codex_policy_window(_snapshot(first, second)) is None


def test_policy_window_past_reset_fallback_fails_open():
    stale = AccountUsageWindow(
        label="Stale",
        used_percent=90,
        reset_at=datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
    )

    assert _select_codex_policy_window(_snapshot(stale)) is None


def test_policy_window_duration_with_past_reset_fails_open():
    stale = AccountUsageWindow(
        label="Stale weekly window",
        used_percent=90,
        limit_window_seconds=604_800,
        reset_at=datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
    )

    assert _select_codex_policy_window(_snapshot(stale)) is None


def test_policy_window_huge_duration_fails_open_without_raising():
    malformed = AccountUsageWindow(
        label="Malformed",
        used_percent=90,
        limit_window_seconds=10**400,
    )

    assert _select_codex_policy_window(_snapshot(malformed)) is None


@pytest.mark.parametrize("used_percent", ["80", " 80 ", "8e1"])
def test_policy_window_rejects_string_percentages(used_percent):
    malformed = AccountUsageWindow(
        label="Weekly",
        used_percent=used_percent,
        limit_window_seconds=604_800,
    )

    assert _select_codex_policy_window(_snapshot(malformed)) is None
