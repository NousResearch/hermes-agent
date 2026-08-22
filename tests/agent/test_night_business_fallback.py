from datetime import datetime
from zoneinfo import ZoneInfo

from agent.night_business_fallback import decide_night_business_route


KST = ZoneInfo("Asia/Seoul")


def _config(**overrides):
    config = {
        "enabled": True,
        "timezone": "Asia/Seoul",
        "start_time": "21:00",
        "end_time": "06:00",
        "profiles": {
            "anthropic": "night-business-claude",
            "openai-codex": "night-business-codex",
        },
    }
    config.update(overrides)
    return config


def _at(hour, minute=0):
    return datetime(2026, 8, 3, hour, minute, tzinfo=KST)


def test_before_night_window_never_routes_even_when_personal_is_exhausted():
    decision = decide_night_business_route(
        _config(),
        provider="anthropic",
        personal_has_credentials=True,
        personal_has_available=False,
        business_has_available=True,
        now=_at(20, 59),
    )

    assert decision.profile is None
    assert decision.reason == "outside_window"


def test_at_window_start_routes_only_the_exhausted_provider():
    claude = decide_night_business_route(
        _config(),
        provider="anthropic",
        personal_has_credentials=True,
        personal_has_available=False,
        business_has_available=True,
        now=_at(21),
    )
    codex = decide_night_business_route(
        _config(),
        provider="openai-codex",
        personal_has_credentials=True,
        personal_has_available=True,
        business_has_available=True,
        now=_at(21),
    )

    assert claude.profile == "night-business-claude"
    assert claude.reason == "personal_exhausted"
    assert codex.profile is None
    assert codex.reason == "personal_available"


def test_end_boundary_is_exclusive_and_disabled_never_routes():
    at_end = decide_night_business_route(
        _config(),
        provider="openai-codex",
        personal_has_credentials=True,
        personal_has_available=False,
        business_has_available=True,
        now=_at(6),
    )
    disabled = decide_night_business_route(
        _config(enabled=False),
        provider="anthropic",
        personal_has_credentials=True,
        personal_has_available=False,
        business_has_available=True,
        now=_at(22),
    )

    assert at_end.profile is None
    assert at_end.reason == "outside_window"
    assert disabled.profile is None
    assert disabled.reason == "disabled"


def test_missing_personal_or_business_capacity_fails_closed():
    no_personal = decide_night_business_route(
        _config(),
        provider="anthropic",
        personal_has_credentials=False,
        personal_has_available=False,
        business_has_available=True,
        now=_at(5, 59),
    )
    no_business = decide_night_business_route(
        _config(),
        provider="anthropic",
        personal_has_credentials=True,
        personal_has_available=False,
        business_has_available=False,
        now=_at(5, 59),
    )

    assert no_personal.reason == "personal_unknown"
    assert no_business.reason == "business_unavailable"
