import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from tools.delegate_tool import delegate_task


def _parent():
    return SimpleNamespace(
        base_url="https://example.test/v1",
        api_key="test-key",
        provider="openrouter",
        api_mode="chat_completions",
        model="test/model",
        platform="cli",
        enabled_toolsets=["terminal", "file"],
        valid_tool_names=set(),
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
        session_id="parent-session",
        max_iterations=20,
        _api_call_count=19,
    )


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 0})
def test_invalid_review_child_is_surfaced_without_counting_as_independent_review(
    _mock_config,
):
    result = json.loads(
        delegate_task(
            goal="Review the paid-live safety verdict before merge.",
            parent_agent=_parent(),
        )
    )

    assert result["error"] == "Delegation preflight failed"
    summary = result["review_evidence_summary"]
    assert summary["valid_count"] == 0
    assert summary["invalid_count"] == 1
    assert summary["invalid_children"][0]["review_evidence_status"] == "blocked_zero_budget"
    assert "rerun_recommendation" in summary["invalid_children"][0]


@patch("tools.delegate_tool._load_config", return_value={"max_iterations": 0})
def test_review_evidence_metadata_redacts_secret_like_goal_and_caps_prompt_preview(
    _mock_config,
):
    secret_body = "abcdefghijklmnopqrstuvwxyz0123456789"
    raw_goal = (
        "Review safety for token sk-proj-"
        + secret_body
        + " and then inspect every file. "
        + ("extra context " * 40)
    )

    result = json.loads(delegate_task(goal=raw_goal, parent_agent=_parent()))
    entry = result["results"][0]

    assert entry["review_evidence_status"] == "blocked_zero_budget"
    assert "raw_goal" not in entry
    assert len(entry["goal_preview"]) <= 180
    assert secret_body not in entry["goal_preview"]
    assert secret_body not in json.dumps(entry["rerun_recommendation"])
    assert entry["rerun_recommendation"]["goal_preview"] == entry["goal_preview"]
