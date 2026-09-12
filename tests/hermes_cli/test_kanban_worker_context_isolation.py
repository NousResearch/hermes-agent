"""Task-scoped authority boundaries for Kanban worker construction."""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _completed_task(conn, *, title: str, summary: str) -> str:
    task_id = kb.create_task(conn, title=title, assignee="kai")
    assert kb.complete_task(conn, task_id, summary=summary)
    return task_id


def _worker_cli_args(monkeypatch, task, home):
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    command = kbd._worker_argv(task, "kai", str(home))
    assert command[1:3] == ["-p", "kai"]
    parser, _subparsers, _chat_parser = build_top_level_parser()
    return command, parser.parse_args(command[3:])


def test_task_isolation_preserves_explicit_context_and_gates_worker_argv(
    kanban_home, monkeypatch
):
    with kbc.connect() as conn:
        unrelated_id = _completed_task(
            conn, title="unrelated prior work", summary="UNRELATED_ASSIGNEE_HISTORY"
        )
        parent_id = _completed_task(
            conn, title="explicit parent", summary="EXPLICIT_PARENT_HANDOFF"
        )

    created = json.loads(
        kc.run_slash(
            "create 'isolated worker' --body 'CURRENT_TASK_BODY' --assignee kai "
            f"--parent {parent_id} --context-isolation task --json"
        )
    )
    task_id = created["id"]
    assert created["context_isolation"] == "task"

    with kbc.connect() as conn:
        kb.add_comment(conn, task_id, author="operator", body="CURRENT_TASK_COMMENT")
        kb.store_attachment_bytes(
            conn, task_id, "current-task.txt", b"attachment", uploaded_by="operator"
        )
        kb._synthesize_ended_run(
            conn, task_id, outcome="crashed", summary="SAME_TASK_PRIOR_ATTEMPT"
        )
        conn.commit()
        task = kb.get_task(conn, task_id)
        context = kb.build_worker_context(conn, task_id)

    assert task is not None
    assert task.context_isolation == "task"
    for explicit_context in (
        "CURRENT_TASK_BODY",
        "CURRENT_TASK_COMMENT",
        "current-task.txt",
        "SAME_TASK_PRIOR_ATTEMPT",
        parent_id,
        "EXPLICIT_PARENT_HANDOFF",
    ):
        assert explicit_context in context
    assert "## Recent work by @kai" not in context
    assert unrelated_id not in context
    assert "UNRELATED_ASSIGNEE_HISTORY" not in context

    command, args = _worker_cli_args(monkeypatch, task, kanban_home)
    assert "--ignore-rules" in command
    assert args.command == "chat"
    assert args.ignore_rules is True
    assert args.query == f"work kanban task {task_id}"


def test_unknown_stored_policy_fails_closed_without_assignee_history(
    kanban_home, monkeypatch
):
    with kbc.connect() as conn:
        prior_id = _completed_task(
            conn, title="prior work", summary="PRIVATE_ASSIGNEE_HISTORY"
        )
        task_id = kb.create_task(conn, title="future policy", assignee="kai")
        conn.execute(
            "UPDATE tasks SET context_isolation = ? WHERE id = ?",
            ("future-policy", task_id),
        )
        conn.commit()
        task = kb.get_task(conn, task_id)
        context = kb.build_worker_context(conn, task_id)

    assert task is not None
    assert "## Recent work by @kai" not in context
    assert prior_id not in context
    assert "PRIVATE_ASSIGNEE_HISTORY" not in context
    assert task.context_isolation == "task"
    command, args = _worker_cli_args(monkeypatch, task, kanban_home)
    assert "--ignore-rules" in command
    assert args.ignore_rules is True


def test_task_isolation_descriptions_include_preloaded_skills():
    root = argparse.ArgumentParser()
    kanban_parser = kc.build_parser(root.add_subparsers())
    actions = next(
        action
        for action in kanban_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    create_help = actions.choices["create"].format_help().lower()

    from tools import kanban_tools as kt

    policy_schema = kt.KANBAN_CREATE_SCHEMA["parameters"]["properties"][
        "context_isolation"
    ]
    assert "preloaded skills" in policy_schema["description"].lower()
    assert "preloaded skills" in create_help


def test_create_tool_roundtrips_policy_and_defaults_to_legacy_context(
    kanban_home, monkeypatch
):
    from tools import kanban_tools as kt

    policy_schema = kt.KANBAN_CREATE_SCHEMA["parameters"]["properties"][
        "context_isolation"
    ]
    assert set(policy_schema["enum"]) == {"none", "task"}

    isolated_created = json.loads(
        kt._handle_create(
            {
                "title": "tool-created isolated task",
                "assignee": "kai",
                "context_isolation": "task",
            }
        )
    )
    assert isolated_created["ok"] is True
    assert isolated_created["context_isolation"] == "task"
    isolated_shown = json.loads(
        kt._handle_show({"task_id": isolated_created["task_id"]})
    )
    assert isolated_shown["task"]["context_isolation"] == "task"

    with kbc.connect() as conn:
        isolated = kb.get_task(conn, isolated_created["task_id"])
        assert isolated is not None
        assert isolated.context_isolation == "task"
        prior_id = _completed_task(
            conn, title="visible ordinary history", summary="VISIBLE_ASSIGNEE_HISTORY"
        )

    ordinary_created = json.loads(
        kt._handle_create({"title": "ordinary worker", "assignee": "kai"})
    )
    assert ordinary_created["ok"] is True
    assert ordinary_created["context_isolation"] == "none"
    ordinary_shown = json.loads(
        kt._handle_show({"task_id": ordinary_created["task_id"]})
    )
    assert ordinary_shown["task"]["context_isolation"] == "none"

    listed_raw = kt.registry.dispatch("kanban_list", {})
    assert isinstance(listed_raw, str)
    listed = json.loads(listed_raw)
    listed_by_id = {task["id"]: task for task in listed["tasks"]}
    assert listed_by_id[isolated_created["task_id"]]["context_isolation"] == "task"
    assert listed_by_id[ordinary_created["task_id"]]["context_isolation"] == "none"

    with kbc.connect() as conn:
        ordinary = kb.get_task(conn, ordinary_created["task_id"])
        assert ordinary is not None
        assert ordinary.context_isolation == "none"
        ordinary_context = kb.build_worker_context(conn, ordinary.id)
        with pytest.raises(ValueError, match="context_isolation"):
            kb.create_task(
                conn,
                title="invalid policy",
                assignee="kai",
                context_isolation="future-policy",
            )

    assert prior_id in ordinary_context
    assert "VISIBLE_ASSIGNEE_HISTORY" in ordinary_context
    ordinary_command, ordinary_args = _worker_cli_args(
        monkeypatch, ordinary, kanban_home
    )
    assert "--ignore-rules" not in ordinary_command
    assert ordinary_args.ignore_rules is False
