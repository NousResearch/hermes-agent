"""Authoritative Kanban context is injected before a worker agent starts."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.prompt_builder import KANBAN_GUIDANCE
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_large_task_context_is_injected_complete_without_tool_reference(kanban_home):
    body = "BEGIN-LARGE-CONTEXT\n" + ("worker requirement\n" * 350) + "END-CONTEXT"
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="Large worker context",
            body=body,
            assignee="worker-a",
        )
        query = kb.build_worker_query(
            conn,
            task_id,
            f"work kanban task {task_id}",
        )
    finally:
        conn.close()

    assert query.startswith(f"work kanban task {task_id}\n\n# Kanban task {task_id}")
    assert body in query
    assert query.count(body) == 1
    assert "<<ccr:" not in query


def test_unknown_task_prevents_contextless_worker_bootstrap(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError, match="unknown task t_missing"):
            kb.build_worker_query(conn, "t_missing", "work kanban task t_missing")
    finally:
        conn.close()


def test_worker_query_caps_many_parent_handoffs_without_losing_task_body(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="Fan-in worker",
            body="BEGIN-TASK-BODY\nworker requirement\nEND-TASK-BODY",
            assignee="worker-a",
        )
        for index in range(20):
            parent_id = kb.create_task(
                conn,
                title=f"Parent {index}",
                assignee="worker-a",
            )
            assert kb.complete_task(conn, parent_id, result="P" * 4096)
            kb.link_tasks(conn, parent_id, task_id)

        query = kb.build_worker_query(
            conn,
            task_id,
            f"work kanban task {task_id}",
        )
    finally:
        conn.close()

    assert len(query) == kb._CTX_MAX_WORKER_QUERY_CHARS
    assert query.startswith(f"work kanban task {task_id}\n\n# Kanban task {task_id}")
    assert "BEGIN-TASK-BODY\nworker requirement\nEND-TASK-BODY" in query
    assert "[worker context truncated," in query
    assert query.endswith(" chars omitted]\n")


def test_worker_query_preserves_parent_handoff_before_attachment_overflow(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="Attachment-heavy fan-in worker",
            body="Worker requirement",
            assignee="worker-a",
        )
        parent_id = kb.create_task(conn, title="Parent", assignee="worker-a")
        assert kb.complete_task(conn, parent_id, result="PARENT-HANDOFF-SENTINEL")
        kb.link_tasks(conn, parent_id, task_id)
        for index in range(100):
            kb.add_attachment(
                conn,
                task_id,
                filename=f"attachment-{index}.txt",
                stored_path="C:/attachments/" + ("a" * 1024) + f"/{index}",
            )

        query = kb.build_worker_query(
            conn,
            task_id,
            f"work kanban task {task_id}",
        )
    finally:
        conn.close()

    assert len(query) == kb._CTX_MAX_WORKER_QUERY_CHARS
    assert "PARENT-HANDOFF-SENTINEL" in query
    assert query.index("## Parent task results") < query.index("## Attachments")
    assert "[worker context truncated," in query


def test_worker_query_caps_title_before_authoritative_task_body(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="T" * (kb._CTX_MAX_WORKER_QUERY_CHARS + 100),
            body="BEGIN-TASK-BODY\nworker requirement\nEND-TASK-BODY",
            assignee="worker-a",
        )
        query = kb.build_worker_query(
            conn,
            task_id,
            f"work kanban task {task_id}",
        )
    finally:
        conn.close()

    assert len(query) < kb._CTX_MAX_WORKER_QUERY_CHARS
    assert "[truncated," in query.splitlines()[2]
    assert "BEGIN-TASK-BODY\nworker requirement\nEND-TASK-BODY" in query


def test_worker_guidance_consumes_injected_context_before_refresh_tool():
    assert "includes an authoritative `# Kanban task …` context block" in KANBAN_GUIDANCE
    assert "Call `kanban_show()` only if the block is absent" in KANBAN_GUIDANCE
