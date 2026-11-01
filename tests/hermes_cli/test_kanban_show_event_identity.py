"""CLI event identity must support annotations bound to the current handoff."""

import json

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_attention import set_review_attention
from hermes_cli.kanban_db_connect import connect_closing


def test_show_preserves_event_identity_for_current_review(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    kb.init_db()
    with connect_closing() as conn:
        task_id = kb.create_task(conn, title="Current review", assignee="forge")
        run = kb.claim_task(conn, task_id)
        assert kb.request_review(
            conn, task_id, expected_run_id=run.current_run_id,
            metadata={"wait_reason": "awaiting-test"}, summary="Ready for testing",
        )
        events = kb.list_events(conn, task_id)
        before = kb.get_task(conn, task_id), kb.list_runs(conn, task_id), events

    shown = json.loads(cli.run_slash(f"show {task_id} --json"))
    assert shown["events"] == [
        {key: getattr(event, key) for key in ("id", "kind", "payload", "created_at", "run_id")}
        for event in events
    ]
    handoff = next(event for event in reversed(shown["events"]) if event["kind"] == "review_requested")
    with connect_closing() as conn:
        assert (kb.get_task(conn, task_id), kb.list_runs(conn, task_id), kb.list_events(conn, task_id)) == before
        assert set_review_attention(
            conn, task_id, expected_handoff_event_id=handoff["id"], badge={"label": "Acceptance ready", "tone": "success"},
        )
