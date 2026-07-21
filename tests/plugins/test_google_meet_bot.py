"""Tests for plugins.google_meet.meet_bot URL handling."""

from __future__ import annotations


def test_force_english_ui_appends_hl():
    from plugins.google_meet.meet_bot import _force_english_ui

    assert (
        _force_english_ui("https://meet.google.com/abc-defg-hij")
        == "https://meet.google.com/abc-defg-hij?hl=en"
    )


def test_force_english_ui_overrides_existing_hl_and_keeps_params():
    from plugins.google_meet.meet_bot import _force_english_ui

    assert (
        _force_english_ui("https://meet.google.com/abc-defg-hij?hl=zh-TW&authuser=1")
        == "https://meet.google.com/abc-defg-hij?hl=en&authuser=1"
    )


def test_force_english_ui_result_stays_safe():
    from plugins.google_meet.meet_bot import _force_english_ui, _is_safe_meet_url

    for url in (
        "https://meet.google.com/abc-defg-hij",
        "https://meet.google.com/lookup/some-id",
        "https://meet.google.com/new",
    ):
        assert _is_safe_meet_url(_force_english_ui(url))
