from suggest_router_changes_v2 import segmented_suggestions


def test_segmented_suggestions_flags_bad_primary_model_for_task_type():
    joined = [
        {
            "request_id": "1",
            "task_type": "chat",
            "priority": "medium",
            "quota": "critical",
            "privacy": "normal",
            "mode": "draft",
            "primary_model": "deepseek",
            "feedback_present": True,
            "outcome": "fallback_used",
            "actual_model_used": "claude-sonnet-4.6",
            "fallback_used": True,
            "user_rating": 3,
        },
        {
            "request_id": "2",
            "task_type": "chat",
            "priority": "medium",
            "quota": "critical",
            "privacy": "normal",
            "mode": "draft",
            "primary_model": "deepseek",
            "feedback_present": True,
            "outcome": "bad_fit",
            "actual_model_used": "claude-sonnet-4.6",
            "fallback_used": True,
            "user_rating": 2,
        },
        {
            "request_id": "3",
            "task_type": "chat",
            "priority": "medium",
            "quota": "critical",
            "privacy": "normal",
            "mode": "draft",
            "primary_model": "deepseek",
            "feedback_present": True,
            "outcome": "failed",
            "actual_model_used": "claude-sonnet-4.6",
            "fallback_used": True,
            "user_rating": 2,
        },
        {
            "request_id": "4",
            "task_type": "chat",
            "priority": "medium",
            "quota": "critical",
            "privacy": "normal",
            "mode": "draft",
            "primary_model": "deepseek",
            "feedback_present": True,
            "outcome": "success",
            "actual_model_used": "deepseek",
            "fallback_used": False,
            "user_rating": 4,
        },
    ]

    suggestions = segmented_suggestions(joined, min_feedback_samples=4)
    messages = [s["message"] for s in suggestions]

    assert any("primary_model=deepseek | task_type=chat" in msg for msg in messages)
    assert any("task_type=chat | priority=medium" in msg for msg in messages)


def test_segmented_suggestions_returns_no_action_when_clean():
    joined = [
        {
            "request_id": "1",
            "task_type": "coding",
            "priority": "high",
            "quota": "normal",
            "privacy": "normal",
            "mode": "execute",
            "primary_model": "gpt-5.4",
            "feedback_present": True,
            "outcome": "success",
            "actual_model_used": "gpt-5.4",
            "fallback_used": False,
            "user_rating": 5,
        },
        {
            "request_id": "2",
            "task_type": "coding",
            "priority": "high",
            "quota": "normal",
            "privacy": "normal",
            "mode": "execute",
            "primary_model": "gpt-5.4",
            "feedback_present": True,
            "outcome": "success",
            "actual_model_used": "gpt-5.4",
            "fallback_used": False,
            "user_rating": 5,
        },
    ]

    suggestions = segmented_suggestions(joined, min_feedback_samples=2)
    assert len(suggestions) == 1
    assert suggestions[0]["type"] == "no_segment_action"
