"""Regression coverage for natural-language Discord progress questions."""

import pytest

from hermes_cli import kanban_db as kb


def test_plain_english_burndown_pr_question_is_a_progress_query():
    try:
        from gateway.progress_queries import is_progress_query
    except ModuleNotFoundError:
        pytest.fail("Discord progress-query routing is missing from the runtime")

    assert is_progress_query(
        "How’s the burndown patches going how much more do we have to do till we can send a PR"
    )


def test_plain_english_plural_burndown_question_summarizes_all_matching_roots(
    tmp_path, monkeypatch
):
    from gateway.progress_queries import ProgressSource, resolve_progress_query

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    board = "project-burndown"
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        first = kb.create_task(
            conn,
            title="July exception burndown patches",
            initial_status="running",
            created_by="test",
        )
        second = kb.create_task(
            conn,
            title="August exception burndown patches",
            initial_status="running",
            created_by="test",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (first,))
        for task_id in (first, second):
            kb.add_notify_sub(
                conn,
                task_id=task_id,
                platform="discord",
                chat_id="project-updates",
                chat_type="group",
                delivery_mode="notify",
            )

    result = resolve_progress_query(
        "How’s the burndown patches going how much more do we have to do till we can send a PR",
        source=ProgressSource(platform="discord", chat_id="project-updates"),
        board=board,
    )

    assert result.handled is True
    assert result.reason == "resolved_multiple"
    assert "July exception burndown patches" in result.response
    assert "August exception burndown patches" in result.response
    assert "Please name one" not in result.response
