"""Regression tests for single-query CLI failure exit semantics."""

from pathlib import Path

import pytest


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_single_query_exit_code_treats_failed_result_as_error():
    from cli import _single_query_exit_code

    assert _single_query_exit_code({"failed": True, "error": "API call failed after 3 retries"}) == 1


def test_single_query_exit_code_treats_success_result_as_success():
    from cli import _single_query_exit_code

    assert _single_query_exit_code({"failed": False, "final_response": "ok"}) == 0


def test_single_query_exit_code_treats_missing_chat_result_as_error():
    from cli import _single_query_exit_code

    assert _single_query_exit_code(None) == 1


def test_kanban_fail_safe_blocks_current_task_on_agent_failure(kanban_home, monkeypatch):
    from cli import _maybe_block_current_kanban_task_on_agent_failure
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="provider timeout", assignee="coder")
        assert kb.claim_task(conn, tid, claimer="host:worker") is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    assert _maybe_block_current_kanban_task_on_agent_failure(
        "API call failed after 3 retries",
        {
            "failure_type": "rate_limit",
            "failure_provider": "openai-codex",
            "failure_model": "gpt-5.5",
            "failure_status_code": 429,
        },
    ) is True

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
        run = conn.execute(
            "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()

    assert task.status == "blocked"
    assert "infrastructure-failure" in (run["summary"] or "")
    assert "rate_limit" in (run["summary"] or "")
    assert "status=429" in (run["summary"] or "")
    assert any("Runtime fail-safe blocked this task" in c.body for c in comments)


def test_kanban_fail_safe_does_not_persist_raw_provider_error(kanban_home, monkeypatch):
    from cli import _maybe_block_current_kanban_task_on_agent_failure
    from hermes_cli import kanban_db as kb

    secret = "sk-live-secret-token"
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="provider secret", assignee="coder")
        assert kb.claim_task(conn, tid, claimer="host:worker") is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    assert _maybe_block_current_kanban_task_on_agent_failure(
        f"provider dumped raw request with token {secret}",
        {
            "failure_type": "provider_error",
            "failure_provider": "openai-codex",
            "failure_model": "gpt-5.5",
        },
    ) is True

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        run = conn.execute(
            "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()

    joined = "\n".join(
        [run["summary"] or ""]
        + [c.body for c in comments]
        + [e["payload"] or "" for e in events]
    )
    assert secret not in joined
    assert "raw request" not in joined
