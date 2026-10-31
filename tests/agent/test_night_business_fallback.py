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


def test_codex_only_exhausted_routes_codex_to_business():
    """21:00 KST, Codex만 개인 소진 → Codex만 business로 라우팅.

    Item 4: 21:00 Codex만 부족 → Codex business 1회.
    Claude는 이 호출에서 판단하지 않으므로 테스트하지 않는다.
    """
    codex = decide_night_business_route(
        _config(),
        provider="openai-codex",
        personal_has_credentials=True,
        personal_has_available=False,
        business_has_available=True,
        now=_at(21),
    )

    assert codex.profile == "night-business-codex"
    assert codex.reason == "personal_exhausted"
    assert codex.provider == "openai-codex"


def test_seoul_tz_is_fixed_regardless_of_system_tz():
    """Item 7: Asia/Seoul 고정 — 시스템 TZ(UTC) 무관.

    now를 UTC datetime으로 전달해도 Asia/Seoul로 변환 후 창 판정한다.
    21:00 KST = 12:00 UTC (여름, UTC+9).
    """
    import os

    UTC = ZoneInfo("UTC")
    # 2026-08-03 21:00 KST = 2026-08-03 12:00 UTC
    now_utc = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

    # 환경 TZ를 UTC로 고정해도 Seoul 창 내에서 business 라우팅
    old_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "UTC"
        decision = decide_night_business_route(
            _config(),
            provider="anthropic",
            personal_has_credentials=True,
            personal_has_available=False,
            business_has_available=True,
            now=now_utc,
        )
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz

    assert decision.profile == "night-business-claude", (
        f"UTC now={now_utc.isoformat()} should resolve to 21:00 KST inside window; "
        f"got reason={decision.reason}"
    )
    assert decision.timezone == "Asia/Seoul"


def test_personal_unknown_blocks_business_routing_fail_closed():
    """fail-closed: 미터 판정불가(personal_has_credentials=False) → business 라우팅 금지.

    unknown/429/401 상황에서 integration layer가 personal_has_credentials=False를 전달하면
    pure function은 'personal_unknown'을 반환해야 한다 — '부족'이 아님.
    """
    for now_hour in (21, 2, 5):  # 야간 창 내 여러 시각
        decision = decide_night_business_route(
            _config(),
            provider="anthropic",
            personal_has_credentials=False,
            personal_has_available=False,
            business_has_available=True,
            now=_at(now_hour),
        )
        assert decision.profile is None, (
            f"미터 판정불가 시 {now_hour}:00 KST에서 business 라우팅되면 안 됨"
        )
        assert decision.reason == "personal_unknown"
