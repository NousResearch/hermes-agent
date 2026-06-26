from __future__ import annotations

import json

from hermes_cli import looper


def test_build_looper_run_writes_artifacts_for_oddsedge(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    spec, artifacts, preview = looper.build_looper_run(
        "Sample OddsEdge task: tighten the Picks review flow, keep nearby date tabs, and preserve archive naming.",
        session_id="sid-oddsedge-loop",
        session_key="telegram:oddsedge:session",
        source_platform="telegram",
        source_chat_id="chat-oddsedge",
        source_thread_id="14054",
    )

    assert spec.project_hint == "OddsEdge"
    assert spec.risk_level == "medium"
    assert spec.verification_commands[0] == "python -m pytest tests -q"
    assert "Gates: plan → implementation → delivery" in preview
    assert spec.goal.startswith("Sample OddsEdge task")

    assert artifacts.run_in_session.exists()
    assert artifacts.loop_yaml.exists()
    assert artifacts.loop_resolved_json.exists()
    assert artifacts.review_rubric.exists()
    assert artifacts.state_json.exists()

    state = json.loads(artifacts.state_json.read_text(encoding="utf-8"))
    assert state["status"] == "awaiting_approval"
    assert state["approval_required"] is True

    resolved = json.loads(artifacts.loop_resolved_json.read_text(encoding="utf-8"))
    assert resolved["status"] == "awaiting_approval"
    assert resolved["spec"]["goal"] == spec.goal
    assert "final_goal_prompt" in resolved


def test_finalize_looper_run_marks_ready(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    spec, artifacts, _ = looper.build_looper_run(
        "Implement a review-gated goal designer for Hermes/OpenClaw.",
        session_id="sid-hermes-loop",
        session_key="telegram:hermes:session",
        source_platform="telegram",
        source_chat_id="chat-hermes",
        source_thread_id="6",
    )

    report, goal = looper.finalize_looper_run(spec, artifacts, approval_choice="once")

    assert goal == spec.goal
    assert report.startswith("✅ ready")
    assert "Final /goal prompt:" in report

    state = json.loads(artifacts.state_json.read_text(encoding="utf-8"))
    assert state["status"] == "ready"
    assert state["approved"] is True

    resolved = json.loads(artifacts.loop_resolved_json.read_text(encoding="utf-8"))
    assert resolved["status"] == "ready"
    assert resolved["approved"] is True
