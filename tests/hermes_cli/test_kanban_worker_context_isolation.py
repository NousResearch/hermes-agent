"""Task-scoped authority boundaries for Kanban worker construction."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


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


def _owner_policy(monkeypatch, task_id, home, config):
    """Build the owner's frozen launch policy for the task's live claim."""
    from contextlib import closing

    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy

    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env", lambda _name: str(home)
    )
    with closing(kbc.connect()) as conn:
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    assert task is not None
    actor = Principal("owner", str(home), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(
        authority=SimpleNamespace(profile_id=str(home)), actor=actor, native_owner=True
    )
    params = dict(
        board="default",
        task_id=task.id,
        run_id=task.current_run_id,
        claim_lock=task.claim_lock,
    )
    policy, _secrets = build_kanban_policy(connection, params, config)
    return policy, json.loads(policy.kanban_json)


def _isolated_task(conn, workspace, *, context_isolation="task", skills=None):
    return kb.create_task(
        conn,
        title="isolated worker",
        assignee="kai",
        workspace_kind="dir",
        workspace_path=str(workspace),
        skills=skills,
        context_isolation=context_isolation,
    )


def _bind_workspace(task_id, workspace):
    workspace.mkdir(exist_ok=True)
    with kbc.connect() as conn:
        conn.execute(
            "UPDATE tasks SET workspace_kind = 'dir', workspace_path = ? WHERE id = ?",
            (str(workspace), task_id),
        )
        conn.commit()


_CONFIG = {
    "model": {"default": "offline-dummy", "provider": "custom"},
    "platform_toolsets": {"cli": ["file", "memory"]},
}


@pytest.mark.parametrize(
    ("context_isolation", "profile_toolsets", "memory_loaded"),
    [
        ("none", ["file", "memory"], True),
        ("task", ["file", "memory"], False),
        ("task", ["memory"], False),
    ],
)
def test_owner_policy_excludes_private_memory_but_keeps_explicit_skill(
    kanban_home,
    tmp_path,
    monkeypatch,
    context_isolation,
    profile_toolsets,
    memory_loaded,
):
    memory_sentinel = "PRIVATE_MEMORY_SENTINEL"
    user_sentinel = "PRIVATE_USER_SENTINEL"
    memories = kanban_home / "memories"
    memories.mkdir(exist_ok=True)
    (memories / "MEMORY.md").write_text(memory_sentinel, encoding="utf-8")
    (memories / "USER.md").write_text(user_sentinel, encoding="utf-8")
    monkeypatch.setenv("HOME", str(kanban_home.parent))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kbc.connect() as conn:
        task_id = _isolated_task(
            conn,
            workspace,
            context_isolation=context_isolation,
            skills=["explicit-task-skill"],
        )
    config = {
        "model": {"default": "offline-dummy", "provider": "custom"},
        "platform_toolsets": {"cli": list(profile_toolsets)},
    }
    policy, context = _owner_policy(monkeypatch, task_id, kanban_home, config)

    assert context["skills"] == ["explicit-task-skill"]
    assert policy.ignore_rules is (context_isolation == "task")
    assert "kanban" in policy.toolsets
    assert ("memory" in policy.toolsets) is memory_loaded
    for toolset in set(profile_toolsets) - {"memory"}:
        assert toolset in policy.toolsets

    # Construct the agent with the same policy-derived kwargs the owner's worker uses.
    from run_agent import AIAgent

    agent = AIAgent(
        model="offline-dummy",
        provider="custom",
        base_url="http://127.0.0.1:9/v1",
        api_key="offline-dummy",
        enabled_toolsets=list(policy.toolsets),
        skip_context_files=policy.ignore_rules,
        load_soul_identity=not policy.ignore_rules,
        skip_memory=policy.ignore_rules,
        skip_background_review=True,
        quiet_mode=True,
    )
    store = agent._memory_store
    snapshot = store._system_prompt_snapshot if store else {"memory": "", "user": ""}
    loaded = json.dumps(snapshot)
    assert (memory_sentinel in loaded) is memory_loaded
    assert (user_sentinel in loaded) is memory_loaded
    assert (store is not None) is memory_loaded


def test_task_isolation_preserves_explicit_context_and_gates_owner_policy(
    kanban_home, tmp_path, monkeypatch
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

    _bind_workspace(task_id, tmp_path / "workspace")
    policy, _context = _owner_policy(monkeypatch, task_id, kanban_home, _CONFIG)
    assert policy.ignore_rules is True
    assert "memory" not in policy.toolsets


def test_unknown_stored_policy_fails_closed_without_assignee_history(
    kanban_home, tmp_path, monkeypatch
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
    _bind_workspace(task_id, tmp_path / "workspace")
    policy, _context = _owner_policy(monkeypatch, task_id, kanban_home, _CONFIG)
    assert policy.ignore_rules is True
    assert "memory" not in policy.toolsets


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
    kanban_home, tmp_path, monkeypatch
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
    _bind_workspace(ordinary.id, tmp_path / "workspace")
    ordinary_policy, _context = _owner_policy(
        monkeypatch, ordinary.id, kanban_home, _CONFIG
    )
    assert ordinary_policy.ignore_rules is False
    assert "memory" in ordinary_policy.toolsets
