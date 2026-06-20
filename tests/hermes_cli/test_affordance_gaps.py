"""Tests for affordance gap logging."""

import json

import pytest

from hermes_cli.affordance_gaps import (
    format_affordance_gaps,
    load_affordance_gaps,
    log_affordance_gap,
)


def test_log_and_load_affordance_gap(tmp_path):
    log_path = tmp_path / "affordance_gaps.jsonl"

    event = log_affordance_gap(
        goal="Summarize a video",
        missing_capability="video transcript fetcher",
        failure_description="No tool could retrieve the transcript.",
        available_tools=["web", "web", "memory"],
        session_id="session-1",
        source="test",
        log_path=log_path,
    )

    assert event["failure_mode"] == "missing_affordance"
    assert event["available_tools"] == ["memory", "web"]

    loaded = load_affordance_gaps(log_path=log_path)
    assert len(loaded) == 1
    assert loaded[0]["goal"] == "Summarize a video"
    assert loaded[0]["missing_capability"] == "video transcript fetcher"


def test_load_skips_invalid_jsonl_rows(tmp_path):
    log_path = tmp_path / "affordance_gaps.jsonl"
    log_path.write_text(
        "\n".join([
            "{bad json",
            json.dumps({"goal": "ok", "missing_capability": "tool"}),
        ]),
        encoding="utf-8",
    )

    loaded = load_affordance_gaps(log_path=log_path)

    assert loaded == [{"goal": "ok", "missing_capability": "tool"}]


def test_format_affordance_gaps_empty():
    assert format_affordance_gaps([]) == "No affordance gaps logged."


def test_format_affordance_gaps_includes_key_fields():
    output = format_affordance_gaps([
        {
            "timestamp": "2026-04-25T00:00:00+00:00",
            "goal": "Do the task",
            "missing_capability": "browser auth",
            "failure_description": "Could not access the site.",
            "available_tools": ["web"],
            "session_id": "abc",
        }
    ])

    assert "browser auth" in output
    assert "Goal: Do the task" in output
    assert "Session: abc" in output


def test_log_requires_goal_and_missing_capability(tmp_path):
    with pytest.raises(ValueError, match="goal is required"):
        log_affordance_gap(goal="", missing_capability="tool", log_path=tmp_path / "gaps.jsonl")

    with pytest.raises(ValueError, match="missing_capability is required"):
        log_affordance_gap(goal="goal", missing_capability="", log_path=tmp_path / "gaps.jsonl")
