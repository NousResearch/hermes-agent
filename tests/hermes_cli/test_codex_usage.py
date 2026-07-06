from hermes_cli.codex_usage import (
    apply_alert_policy,
    compute_recommendation,
    render_compact,
    summarize_window,
    usage_bar,
)


def test_risk_policy_supports_per_window_thresholds():
    payload = {
        "checked_at": "2026-07-06T15:00:00+09:00",
        "accounts": [
            {
                "label": "personal-backup",
                "ok": True,
                "primary_window": {"used_percent": 94, "reset_at": "2026-07-06T16:00:00+09:00"},
                "secondary_window": {"used_percent": 86, "reset_at": "2026-07-07T10:00:00+09:00"},
            },
            {
                "label": "company-plus-100",
                "ok": True,
                "primary_window": {"used_percent": 10, "reset_at": "2026-07-06T19:00:00+09:00"},
                "secondary_window": {"used_percent": 54, "reset_at": "2026-07-09T18:00:00+09:00"},
            },
        ],
    }

    events = apply_alert_policy(payload, primary_threshold=95, secondary_threshold=85)

    assert [event["label"] for event in events] == ["personal-backup"]
    assert events[0]["window"] == "7d"
    assert events[0]["used"] == 86


def test_recommendation_prefers_lowest_weekly_then_five_hour_usage():
    accounts = [
        {"label": "a", "ok": True, "primary_window": {"used_percent": 1}, "secondary_window": {"used_percent": 70}},
        {"label": "b", "ok": True, "primary_window": {"used_percent": 60}, "secondary_window": {"used_percent": 40}},
    ]

    rec = compute_recommendation(accounts)

    assert rec == {"label": "b", "reason": "7d 40%, 5h 60%"}


def test_render_compact_includes_risk_and_recommendation():
    payload = {
        "checked_at": "2026-07-06T15:00:00+09:00",
        "accounts": [
            {
                "label": "company-plus-100",
                "ok": True,
                "primary_window": summarize_window({"used_percent": 1, "reset_at": 1783333044}),
                "secondary_window": summarize_window({"used_percent": 54, "reset_at": 1783609614}),
            }
        ],
        "recommendation": {"label": "company-plus-100", "reason": "7d 54%, 5h 1%"},
    }

    text = render_compact(payload)

    assert "🧭 Codex 사용량" in text
    assert "✅ 추천 company-plus-100" in text
    assert "company-plus-100" in text
    assert "5h  1% 🟢" in text
    assert "7d 54% 🟢" in text
    assert "[" in text and "]" in text


def test_usage_bar_visualizes_percent_buckets():
    assert usage_bar(0, width=10) == "░░░░░░░░░░"
    assert usage_bar(54, width=10) == "█████░░░░░"
    assert usage_bar(98, width=10) == "██████████"
    assert usage_bar(None, width=10) == "??????????"
