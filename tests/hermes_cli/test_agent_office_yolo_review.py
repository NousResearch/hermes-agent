"""Regression tests for Office yolo hands-free review routing policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import agent_office


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_kanban_worker_prompt_does_not_force_human_review_required_in_yolo_mode():
    from agent import prompt_builder

    guidance = prompt_builder.KANBAN_GUIDANCE

    assert "review-required" not in guidance
    assert "human_review_required" in guidance
    assert "handoff_for_supervisor_review" in guidance


def test_office_supervisor_routes_review_required_blocks_to_reviewer_in_yolo_mode(kanban_home, monkeypatch):
    monkeypatch.setattr(agent_office, "office_config", lambda: {
        "default_mode": "yolo",
        "quality_gates": {"default_reviewer": "reviewer"},
        "role_seats": {"reviewer": ["reviewer"]},
    })
    monkeypatch.setattr(agent_office, "resolve_profile_for_role", lambda conn, role: "reviewer")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Implement feature", assignee="coder")
        assert kb.block_task(conn, tid, reason="review-required: implemented feature")

        touched = agent_office.supervise(conn)
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()

    assert tid in touched
    assert task.status == "ready"
    assert task.assignee == "reviewer"
    assert any("Auto-routed review-required block" in c.body for c in comments)
    assert any(row["kind"] == "office.review_routed" for row in events)


@pytest.mark.parametrize(
    "reason",
    [
        "review-required: explicit approval required before merge",
        "review-required: ask Akhil before production deploy",
        "review-required: human_review_required: true",
        "review-required: security approval required",
        "review-required: human review required before continuing",
        "review-required: requires human approval from Akhil",
        "review-required: needs approval from Akhil before continuing",
        "review-required: must be approved by user",
        "review-required: get approval from legal",
        "review-required: await user sign-off",
    ],
)
def test_office_supervisor_does_not_auto_route_explicit_approval_blocks(
    kanban_home, monkeypatch, reason
):
    monkeypatch.setattr(agent_office, "office_config", lambda: {
        "default_mode": "yolo",
        "quality_gates": {"default_reviewer": "reviewer"},
        "role_seats": {"reviewer": ["reviewer"]},
    })
    monkeypatch.setattr(agent_office, "resolve_profile_for_role", lambda conn, role: "reviewer")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Approval-gated work", assignee="coder")
        assert kb.block_task(conn, tid, reason=reason)

        touched = agent_office.supervise(conn)
        task = kb.get_task(conn, tid)
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()

    assert task.status == "blocked"
    assert task.assignee == "coder"
    assert not any(row["kind"] == "office.review_routed" for row in events)
