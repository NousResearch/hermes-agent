from hermes_cli.codex_usage import (
    apply_alert_policy,
    compute_recommendation,
    render_alert,
    render_compact,
    render_credential_insights,
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
                "label": "personal-backup",
                "ok": True,
                "plan_type": "pro",
                "primary_window": summarize_window({"used_percent": 92, "reset_at": "2099-07-06T15:20:00+09:00"}),
                "secondary_window": summarize_window({"used_percent": 98, "reset_at": "2099-07-07T10:42:00+09:00"}),
            },
            {
                "label": "company-plus-100",
                "ok": True,
                "plan_type": "prolite",
                "primary_window": summarize_window({"used_percent": 1, "reset_at": "2099-07-06T19:17:00+09:00"}),
                "secondary_window": summarize_window({"used_percent": 54, "reset_at": "2099-07-09T18:46:00+09:00"}),
            },
        ],
        "recommendation": {"label": "company-plus-100", "reason": "7d 54%, 5h 1%"},
    }

    text = render_compact(payload)

    assert "🧭 Codex 사용량" in text
    assert "✅ 추천 company-plus-100" in text
    assert "사유:" in text
    assert "personal-backup 7d 98%" in text
    assert "⏱ 다음 회복: personal-backup 5h" in text
    assert "🚦 위험 회복: personal-backup 7d" in text
    assert "company-plus-100" in text
    assert "5h  1% 🟢" in text
    assert "7d 54% 🟢" in text
    assert "[" in text and "]" in text


def test_render_alert_uses_card_layout_with_bar_and_recommendation():
    payload = {
        "checked_at": "2026-07-06T15:00:00+09:00",
        "accounts": [
            {
                "label": "personal-backup",
                "ok": True,
                "primary_window": summarize_window({"used_percent": 17, "reset_at": "2099-07-06T16:05:00+09:00"}),
                "secondary_window": summarize_window({"used_percent": 98, "reset_at": "2099-07-07T10:42:00+09:00"}),
            },
            {
                "label": "company-plus-100",
                "ok": True,
                "primary_window": summarize_window({"used_percent": 1, "reset_at": "2099-07-06T19:17:00+09:00"}),
                "secondary_window": summarize_window({"used_percent": 54, "reset_at": "2099-07-09T18:46:00+09:00"}),
            },
        ],
        "recommendation": {"label": "company-plus-100", "reason": "7d 54%, 5h 1%"},
    }

    text = render_alert(payload, primary_threshold=95, secondary_threshold=85)

    assert "🚨 Codex 한도 주의" in text
    assert "personal-backup · 7d 98%" in text
    assert "[██████████]" in text
    assert "✅ 추천 company-plus-100" in text


def test_usage_bar_visualizes_percent_buckets():
    assert usage_bar(0, width=10) == "░░░░░░░░░░"
    assert usage_bar(54, width=10) == "█████░░░░░"
    assert usage_bar(98, width=10) == "██████████"
    assert usage_bar(None, width=10) == "??????????"


def test_render_credential_insights_groups_rows_readably():
    rows = [
        {
            "credential_label": "personal-backup",
            "model": "gpt-5.5",
            "total_tokens": 1477482,
            "input_tokens": 77847,
            "output_tokens": 2996,
            "api_call_count": 10,
        },
        {
            "credential_label": "company-plus-100",
            "model": "gpt-5.5",
            "total_tokens": 200000,
            "input_tokens": 1000,
            "output_tokens": 2000,
            "api_call_count": 2,
        },
    ]

    text = render_credential_insights(rows, provider="openai-codex", days=30)

    assert "📊 Codex credential 사용량 · 30d" in text
    assert "personal-backup" in text
    assert "1.48M tokens · 10 calls" in text
    assert "avg 147.7k/call" in text
    assert "company-plus-100" in text
    assert "200.0k tokens · 2 calls" in text


def test_render_credential_insights_shows_share_and_cache_breakdown():
    rows = [
        {
            "credential_label": "personal-backup",
            "model": "gpt-5.5",
            "total_tokens": 900,
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_tokens": 700,
            "reasoning_tokens": 80,
            "api_calls": 3,
        },
        {
            "credential_label": "company-plus-100",
            "model": "gpt-5.5",
            "total_tokens": 100,
            "input_tokens": 50,
            "output_tokens": 50,
            "api_calls": 1,
        },
    ]

    text = render_credential_insights(rows, provider="openai-codex", days=7)

    assert "900 tokens · 3 calls · avg 300/call · 90%" in text
    assert "cache 700" in text
    assert "reason 80" in text


def test_render_credential_insights_empty_explains_future_rows():
    text = render_credential_insights([], provider="openai-codex", days=30)

    assert "아직 credential별 사용 기록 없음" in text
    assert "새 모델 턴부터 쌓임" in text
