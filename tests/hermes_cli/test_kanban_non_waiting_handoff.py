"""Non-waiting Kanban handoff contracts."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db(board="tech-coe")
    return home


def test_concurrent_idempotent_create_has_one_winner(kanban_home):
    def create(_: int) -> str:
        with kbc.connect_closing(board="tech-coe") as conn:
            return kb.create_task(
                conn,
                title="same intent",
                assignee="software-eng",
                idempotency_key="daily:2026-10-05:v1",
                board="tech-coe",
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        ids = list(pool.map(create, range(20)))

    assert len(set(ids)) == 1
    with kbc.connect_closing(board="tech-coe") as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
            ("daily:2026-10-05:v1",),
        ).fetchone()[0]
    assert count == 1


def test_idempotent_replay_with_different_intent_fails_closed(kanban_home):
    with kbc.connect_closing(board="tech-coe") as conn:
        kb.create_task(
            conn, title="original", assignee="software-eng",
            idempotency_key="phase:key", board="tech-coe",
        )
        with pytest.raises(ValueError, match="immutable intent"):
            kb.create_task(
                conn, title="changed", assignee="software-eng",
                idempotency_key="phase:key", board="tech-coe",
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = 'phase:key'"
        ).fetchone()[0] == 1


def test_completion_commits_evidence_and_promotes_child_immediately(kanban_home):
    with kbc.connect_closing(board="tech-coe") as conn:
        parent = kb.create_task(conn, title="parent", assignee="software-eng", board="tech-coe")
        child = kb.create_task(
            conn, title="child", assignee="qa-security", parents=[parent], board="tech-coe",
        )
        assert kb.get_task(conn, child).status == "todo"
        assert kb.complete_task(
            conn,
            parent,
            summary="implementation complete",
            metadata={"evidence": ["tests:pass"], "commit": "abc123"},
        )
        assert kb.get_task(conn, child).status == "ready"
        run = kb.latest_run(conn, parent)
        assert run.metadata["evidence"] == ["tests:pass"]
        kinds = [
            row["kind"] for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (child,)
            )
        ]
        assert "promoted" in kinds


def test_escalation_is_deduplicated_privacy_safe_and_kill_switchable(kanban_home):
    secret = "LINE-ID-raw-secret"
    with kbc.connect_closing(board="tech-coe") as conn:
        source = kb.create_task(
            conn, title="customer phone 0812345678", body=secret,
            assignee="software-eng", board="tech-coe",
        )
        assert kb.block_task(conn, source, reason=secret, kind="needs_input")

        assert kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=False,
        ) == []
        first = kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        )
        assert len(first) == 1
        assert kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        ) == []

        escalation = kb.get_task(conn, first[0])
        assert escalation.assignee == "tech-cto"
        serialized = json.dumps(
            {
                "title": escalation.title,
                "body": escalation.body,
                "events": [
                    json.loads(row["payload"] or "{}")
                    for row in conn.execute(
                        "SELECT payload FROM task_events WHERE task_id = ? ORDER BY id", (source,)
                    )
                ],
            },
            sort_keys=True,
        )
        assert secret not in escalation.body
        assert "0812345678" not in escalation.body
        handoff_rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'handoff_escalated' ORDER BY id",
            (source,),
        ).fetchall()
        assert len(handoff_rows) == 1
        handoff_payload = json.loads(handoff_rows[0]["payload"])
        assert set(handoff_payload) == {
            "source_event_id", "board", "task_ref", "state", "failure_class",
            "attempt_count", "owner_profile", "timestamp", "run_id",
        }
        assert serialized.count("Kanban escalation") == 1
