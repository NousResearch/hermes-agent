from suggest_router_changes import suggest_from_summary


def test_suggests_high_fallback_low_success_and_task_issue():
    summary = {
        "primary_models": {
            "deepseek": 10,
            "gpt-5.4": 12,
        },
        "joined": {
            "by_primary_model": {
                "deepseek": {
                    "total": 10,
                    "with_feedback": 8,
                    "success": 2,
                    "bad_fit": 2,
                    "failed": 1,
                    "fallback_used": 4,
                    "abandoned": 0,
                    "feedback_coverage_rate": 0.8,
                    "success_rate": 0.25,
                    "fallback_rate": 0.50,
                    "average_rating": 2.8,
                    "mismatch_actual_model": 3,
                },
                "gpt-5.4": {
                    "total": 12,
                    "with_feedback": 10,
                    "success": 8,
                    "bad_fit": 1,
                    "failed": 0,
                    "fallback_used": 1,
                    "abandoned": 0,
                    "feedback_coverage_rate": 0.83,
                    "success_rate": 0.80,
                    "fallback_rate": 0.10,
                    "average_rating": 4.5,
                    "mismatch_actual_model": 0,
                },
            },
            "by_task_type": {
                "coding": {
                    "total": 14,
                    "with_feedback": 10,
                    "success": 9,
                    "bad_fit": 0,
                    "failed": 0,
                    "fallback_used": 1,
                    "abandoned": 0,
                    "feedback_coverage_rate": 0.71,
                    "success_rate": 0.90,
                    "fallback_rate": 0.10,
                    "average_rating": 4.7,
                    "mismatch_actual_model": 0,
                },
                "chat": {
                    "total": 8,
                    "with_feedback": 7,
                    "success": 2,
                    "bad_fit": 2,
                    "failed": 1,
                    "fallback_used": 3,
                    "abandoned": 0,
                    "feedback_coverage_rate": 0.87,
                    "success_rate": 0.29,
                    "fallback_rate": 0.43,
                    "average_rating": 2.9,
                    "mismatch_actual_model": 2,
                },
            },
        },
    }

    suggestions = suggest_from_summary(summary, min_feedback_samples=5)
    messages = [s["message"] for s in suggestions]

    assert any("deepseek" in msg and "fallback_rate גבוה" in msg for msg in messages)
    assert any("deepseek" in msg and "success_rate נמוך" in msg for msg in messages)
    assert any("deepseek" in msg and "דירוג ממוצע נמוך" in msg for msg in messages)
    assert any("קטגוריית chat" in msg and "success_rate נמוך" in msg for msg in messages)


def test_returns_no_action_when_no_patterns_found():
    summary = {
        "primary_models": {
            "claude-sonnet-4.6": 5,
            "gpt-5.4": 6,
        },
        "joined": {
            "by_primary_model": {
                "claude-sonnet-4.6": {
                    "total": 5,
                    "with_feedback": 5,
                    "success": 5,
                    "bad_fit": 0,
                    "failed": 0,
                    "fallback_used": 0,
                    "abandoned": 0,
                    "feedback_coverage_rate": 1.0,
                    "success_rate": 1.0,
                    "fallback_rate": 0.0,
                    "average_rating": 4.8,
                    "mismatch_actual_model": 0,
                },
            },
            "by_task_type": {},
        },
    }

    suggestions = suggest_from_summary(summary, min_feedback_samples=3, low_usage_threshold=0)
    assert len(suggestions) == 1
    assert suggestions[0]["type"] == "no_action"
