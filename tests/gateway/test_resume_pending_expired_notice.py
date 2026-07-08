from __future__ import annotations

from types import SimpleNamespace

from gateway.run import (
    _auto_reset_adjust_hint,
    _auto_reset_context_note,
    _auto_reset_reason_text,
)


def _policy(*, idle_minutes: int = 1440, at_hour: int = 9) -> SimpleNamespace:
    return SimpleNamespace(idle_minutes=idle_minutes, at_hour=at_hour)


def test_resume_pending_expired_has_dedicated_context_note() -> None:
    note = _auto_reset_context_note("resume_pending_expired")
    assert "gateway resume timeout" in note
    assert "inactivity" not in note


def test_resume_pending_expired_has_dedicated_reason_text() -> None:
    text = _auto_reset_reason_text("resume_pending_expired", _policy())
    assert text == "previous session expired after gateway resume timeout"


def test_resume_pending_expired_has_dedicated_config_hint() -> None:
    hint = _auto_reset_adjust_hint("resume_pending_expired")
    assert hint == (
        "Adjust gateway resume freshness in config.yaml under "
        "agent.gateway_auto_continue_freshness."
    )


def test_idle_reason_text_still_uses_session_reset_idle_minutes() -> None:
    text = _auto_reset_reason_text("idle", _policy(idle_minutes=90))
    assert text == "inactive for 1h 30m"


def test_daily_reason_text_is_unchanged() -> None:
    text = _auto_reset_reason_text("daily", _policy(at_hour=6))
    assert text == "daily schedule at 6:00"


def test_non_resume_pending_adjust_hint_stays_on_session_reset() -> None:
    hint = _auto_reset_adjust_hint("idle")
    assert hint == "Adjust reset timing in config.yaml under session_reset."
