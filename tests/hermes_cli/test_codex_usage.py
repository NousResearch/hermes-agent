from hermes_cli.codex_usage import (
    annotate_usage_trends,
    apply_alert_policy,
    compute_recommendation,
    load_history,
    render_alert,
    render_compact,
    render_credential_insights,
    save_history_snapshot,
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


def test_recommendation_prefers_soonest_weekly_reset_then_usage_fallback():
    accounts = [
        {
            "label": "a",
            "ok": True,
            "primary_window": {"used_percent": 1},
            "secondary_window": {"used_percent": 70, "reset_at": "2099-07-09T18:00:00+09:00", "remaining": "2d"},
        },
        {
            "label": "b",
            "ok": True,
            "primary_window": {"used_percent": 60},
            "secondary_window": {"used_percent": 40, "reset_at": "2099-07-07T10:00:00+09:00", "remaining": "1h"},
        },
    ]

    rec = compute_recommendation(accounts)

    assert rec is not None
    assert rec["label"] == "b"
    assert rec["policy"] == "7d-reset-aware"
    assert "7d reset" in rec["reason"]


def test_recommendation_prefers_unstarted_weekly_window_before_known_resets():
    accounts = [
        {
            "label": "started",
            "ok": True,
            "available": True,
            "primary_window": {"used_percent": 1},
            "secondary_window": {"used_percent": 10, "reset_at": "2099-07-07T10:00:00+09:00", "remaining": "1h"},
        },
        {
            "label": "unstarted",
            "ok": True,
            "available": True,
            "primary_window": {"used_percent": 1},
            "secondary_window": {"used_percent": 0},
        },
    ]

    rec = compute_recommendation(accounts)

    assert rec is not None
    assert rec["label"] == "unstarted"


def test_annotate_usage_trends_uses_recent_history_for_burn_and_eta():
    payload = {
        "checked_at": "2026-07-06T20:00:00+09:00",
        "accounts": [
            {
                "label": "company-plus-100",
                "ok": True,
                "plan_type": "prolite",
                "primary_window": summarize_window(
                    {
                        "used_percent": 22,
                        "limit_window_seconds": 18000,
                        "reset_at": "2026-07-07T00:00:00+09:00",
                    }
                ),
                "secondary_window": summarize_window(
                    {
                        "used_percent": 57,
                        "limit_window_seconds": 604800,
                        "reset_at": "2026-07-09T18:00:00+09:00",
                    }
                ),
            }
        ],
    }
    history = [
        {
            "checked_at": "2026-07-06T19:30:00+09:00",
            "accounts": [
                {
                    "label": "company-plus-100",
                    "ok": True,
                    "primary_window": {
                        "used_percent": 20,
                        "reset_at": "2026-07-07T00:00:00+09:00",
                    },
                    "secondary_window": {
                        "used_percent": 56,
                        "reset_at": "2026-07-09T18:00:00+09:00",
                    },
                }
            ],
        }
    ]

    annotate_usage_trends(payload, history=history)

    primary_trend = payload["accounts"][0]["primary_window"]["trend"]
    assert primary_trend["source"] == "recent"
    assert primary_trend["burn_percent_per_hour"] == 4.0
    assert primary_trend["eta"]["95"]["at"] == "2026-07-07T14:15:00+09:00"


def test_render_compact_includes_burn_and_eta_when_trend_is_available():
    payload = {
        "checked_at": "2026-07-06T20:00:00+09:00",
        "accounts": [
            {
                "label": "company-plus-100",
                "ok": True,
                "plan_type": "prolite",
                "primary_window": summarize_window(
                    {
                        "used_percent": 22,
                        "limit_window_seconds": 18000,
                        "reset_at": "2026-07-07T00:00:00+09:00",
                    }
                ),
                "secondary_window": summarize_window(
                    {
                        "used_percent": 57,
                        "limit_window_seconds": 604800,
                        "reset_at": "2026-07-09T18:00:00+09:00",
                    }
                ),
            }
        ],
        "recommendation": {"label": "company-plus-100", "reason": "7d 57%, 5h 22%"},
    }
    annotate_usage_trends(
        payload,
        history=[
            {
                "checked_at": "2026-07-06T19:30:00+09:00",
                "accounts": [
                    {
                        "label": "company-plus-100",
                        "ok": True,
                        "primary_window": {"used_percent": 20, "reset_at": "2026-07-07T00:00:00+09:00"},
                        "secondary_window": {"used_percent": 56, "reset_at": "2026-07-09T18:00:00+09:00"},
                    }
                ],
            }
        ],
    )

    text = render_compact(payload)

    assert "🔥 Burn:" in text
    assert "5h +4.0%/h" in text
    assert "burn +4.0%/h · ETA95 07/07 14:15" in text


def test_annotate_usage_trends_falls_back_to_window_average_on_first_run():
    payload = {
        "checked_at": "2026-07-06T20:00:00+09:00",
        "accounts": [
            {
                "label": "company-plus-100",
                "ok": True,
                "primary_window": summarize_window(
                    {
                        "used_percent": 25,
                        "limit_window_seconds": 18000,
                        "reset_at": "2026-07-06T21:00:00+09:00",
                    }
                ),
                "secondary_window": {},
            }
        ],
    }

    annotate_usage_trends(payload, history=[])

    trend = payload["accounts"][0]["primary_window"]["trend"]
    assert trend["source"] == "window_avg"
    assert trend["burn_percent_per_hour"] == 6.25


def test_annotate_usage_trends_ignores_usage_rollbacks_and_uses_average():
    payload = {
        "checked_at": "2026-07-06T20:00:00+09:00",
        "accounts": [
            {
                "label": "company-plus-100",
                "ok": True,
                "primary_window": summarize_window(
                    {
                        "used_percent": 20,
                        "limit_window_seconds": 18000,
                        "reset_at": "2026-07-06T21:00:00+09:00",
                    }
                ),
                "secondary_window": {},
            }
        ],
    }

    annotate_usage_trends(
        payload,
        history=[
            {
                "checked_at": "2026-07-06T19:30:00+09:00",
                "accounts": [
                    {
                        "label": "company-plus-100",
                        "ok": True,
                        "primary_window": {"used_percent": 25, "reset_at": "2026-07-06T21:00:00+09:00"},
                    }
                ],
            }
        ],
    )

    trend = payload["accounts"][0]["primary_window"]["trend"]
    assert trend["source"] == "window_avg"
    assert trend["burn_percent_per_hour"] == 5.0


def test_history_load_skips_corrupt_lines_and_save_is_nonfatal(tmp_path):
    history_path = tmp_path / "history.jsonl"
    history_path.write_text('{"checked_at":"old"}\nnot-json\n{"checked_at":"new"}\n', encoding="utf-8")

    rows = load_history(history_path)

    assert [row["checked_at"] for row in rows] == ["old", "new"]
    save_history_snapshot(tmp_path / "missing" / "nested" / "history.jsonl", {"checked_at": "now", "accounts": []})
    save_history_snapshot(tmp_path, {"checked_at": "now", "accounts": []})


def test_render_compact_marks_eta_after_reset():
    payload = {
        "checked_at": "2026-07-06T20:00:00+09:00",
        "accounts": [
            {
                "label": "company-plus-100",
                "ok": True,
                "plan_type": "prolite",
                "primary_window": summarize_window(
                    {
                        "used_percent": 22,
                        "limit_window_seconds": 18000,
                        "reset_at": "2026-07-06T21:00:00+09:00",
                    }
                ),
                "secondary_window": {},
            }
        ],
        "recommendation": {"label": "company-plus-100", "reason": "7d ?, 5h 22%"},
    }
    annotate_usage_trends(
        payload,
        history=[
            {
                "checked_at": "2026-07-06T19:30:00+09:00",
                "accounts": [
                    {
                        "label": "company-plus-100",
                        "ok": True,
                        "primary_window": {"used_percent": 20, "reset_at": "2026-07-06T21:00:00+09:00"},
                    }
                ],
            }
        ],
    )

    text = render_compact(payload)

    assert "ETA95 07/07 14:15*" in text


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
