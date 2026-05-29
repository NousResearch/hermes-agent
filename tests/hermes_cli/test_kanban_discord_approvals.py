import json

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_discord_approvals as kap


def test_autonomous_review_required_block_is_not_human_ping():
    task = {"id": "t_review", "title": "Implement feature", "body": "", "assignee": "forge"}
    payload = {"reason": "review-required: implementation complete", "metadata": {"review_required": True}}

    assert kap.is_autonomous_review_gate(task, payload) is True
    assert kap.is_human_approval_gate(task, "blocked", payload) is False


def test_human_gate_approval_prompt_contains_required_context_and_buttons():
    task = {"id": "t_gate", "title": "Ship production config", "assignee": "forge"}
    payload = {
        "reason": "human-gate: approve production-facing config change",
        "metadata": {
            "human_approval_required": True,
            "what_is_approved": "shipping the config to the shared repo",
            "if_approved": "Forge will unblock the Kanban task and continue the lane",
            "risk_rollback": "Rollback is reverting the config commit before merge.",
        },
    }

    assert kap.is_human_approval_gate(task, "blocked", payload) is True
    req = kap.build_approval_request(task, payload, {"project_title": "Build Lane"})
    body = kap.build_message_payload(req, "<@123>")

    content = body["content"]
    assert "<@123>" in content
    assert "`t_gate`" in content
    assert "Ship production config" in content
    assert "Build Lane" in content
    assert "shipping the config" in content
    assert "Forge will unblock" in content
    assert "Rollback is reverting" in content
    assert "human-gate" in content
    labels = [c["label"] for c in body["components"][0]["components"]]
    assert labels == ["Approve", "Deny", "Needs changes"]
    assert kap.parse_custom_id(body["components"][0]["components"][0]["custom_id"]) == ("t_gate", "approve")


def test_human_gate_prompt_redacts_secret_looking_reason_and_metadata():
    task = {"id": "t_gate", "title": "Ship production config", "assignee": "forge"}
    payload = {
        "reason": "human-gate: deploy with token=sk-live-secret and email matthew@example.com",
        "metadata": {
            "human_approval_required": True,
            "what_is_approved": "use api_key='abc123secret' for a long approval explanation " + ("with details " * 80),
            "if_approved": "password: correct-horse-battery-staple; continue deployment",
            "risk_rollback": "rollback_token=ghp_abcdefghijklmnopqrstuvwxyz123456",
        },
    }

    req = kap.build_approval_request(task, payload, {"project_title": "Build Lane"})
    body = kap.build_message_payload(req)
    content = body["content"]

    assert "sk-live-secret" not in content
    assert "matthew@example.com" not in content
    assert "abc123secret" not in content
    assert "correct-horse-battery-staple" not in content
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in content
    assert "[redacted]" in content
    assert "…[truncated]" in content
    for line in content.splitlines():
        if line.startswith(("**Approve:**", "**If approved:**", "**Risk / rollback:**", "**Source:")):
            assert len(line) <= 260


def test_approval_decision_comments_and_unblocks(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect_closing(db_path=db_path) as conn:
        task_id = kb.create_task(conn, title="Human gate", assignee="forge", body="", initial_status="blocked")

    result = kap.apply_approval_decision(task_id, "approve", "Matthew (123)", db_path=db_path)

    assert result == "approved_unblocked"
    with kb.connect_closing(db_path=db_path) as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["status"] == "ready"
        comments = kb.list_comments(conn, task_id)
        assert len(comments) == 1
        assert comments[0].author == "discord:Matthew (123)"
        assert "APPROVED via Discord approval button" in comments[0].body


def test_needs_changes_records_comment_without_unblocking(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect_closing(db_path=db_path) as conn:
        task_id = kb.create_task(conn, title="Human gate", assignee="forge", body="", initial_status="blocked")

    result = kap.apply_approval_decision(task_id, "needs_changes", "Matthew (123)", db_path=db_path)

    assert result == "needs_changes_recorded"
    with kb.connect_closing(db_path=db_path) as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["status"] == "blocked"
        assert "NEEDS CHANGES" in kb.list_comments(conn, task_id)[0].body
