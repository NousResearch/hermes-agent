"""Result-aware Kanban dependency edge behavior.

These tests protect the distinction between a parent's lifecycle state and
its machine-readable outcome.  Legacy links stay terminal-status gated;
opt-in links additionally require an exact successful result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban as kb_cli


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with kb.connect_closing() as connection:
        yield connection


def _completed_parent(conn, *, result: str | None) -> str:
    parent_id = kb.create_task(conn, title="qa", assignee="reviewer")
    assert kb.complete_task(conn, parent_id, result=result)
    return parent_id


@pytest.mark.parametrize(
    ("status", "result"),
    [
        ("done", "qa_failed"),
        ("done", None),
        ("done", "READY_TO_DEPLOY"),
        ("archived", "ready_to_deploy"),
    ],
)
def test_result_gated_creation_waits_for_exact_done_result(
    conn, status: str, result: str | None,
) -> None:
    parent_id = _completed_parent(conn, result=result)
    if status == "archived":
        assert kb.archive_task(conn, parent_id)

    child_id = kb.create_task(
        conn,
        title="deploy",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )

    assert kb.get_task(conn, child_id).status == "todo"
    assert not kb._parents_satisfied(conn, child_id)


def test_result_gated_creation_accepts_exact_done_result(conn) -> None:
    parent_id = _completed_parent(conn, result="ready_to_deploy")

    child_id = kb.create_task(
        conn,
        title="deploy",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )

    assert kb.get_task(conn, child_id).status == "ready"
    assert kb._parents_satisfied(conn, child_id)


def test_idempotent_create_reuses_exact_dependency_contract(conn) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")
    expected_id = kb.create_task(
        conn,
        title="deploy",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
        idempotency_key="deploy-once",
    )

    actual_id = kb.create_task(
        conn,
        title="ignored on exact retry",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
        idempotency_key="deploy-once",
    )

    assert actual_id == expected_id


@pytest.mark.parametrize(
    ("original_required_result", "retry_required_result"),
    [
        pytest.param(None, "ready_to_deploy", id="legacy-to-strict"),
        pytest.param(
            "ready_to_deploy",
            "qa_failed",
            id="different-required-result",
        ),
    ],
)
def test_idempotent_create_rejects_different_dependency_contract(
    conn,
    original_required_result: str | None,
    retry_required_result: str,
) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")
    original_requirements = (
        {parent_id: original_required_result}
        if original_required_result is not None
        else None
    )
    existing_id = kb.create_task(
        conn,
        title="original deploy",
        parents=[parent_id],
        required_parent_results=original_requirements,
        idempotency_key="deploy-once",
    )

    with pytest.raises(ValueError, match="different dependency contract"):
        kb.create_task(
            conn,
            title="conflicting deploy retry",
            parents=[parent_id],
            required_parent_results={parent_id: retry_required_result},
            idempotency_key="deploy-once",
        )

    tasks = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ?",
        ("deploy-once",),
    ).fetchall()
    assert [row["id"] for row in tasks] == [existing_id]


def test_idempotent_create_rejects_different_parent_set(conn) -> None:
    first_parent_id = _completed_parent(conn, result="ready_to_deploy")
    second_parent_id = _completed_parent(conn, result="ready_to_deploy")
    existing_id = kb.create_task(
        conn,
        title="original deploy",
        parents=[first_parent_id],
        required_parent_results={first_parent_id: "ready_to_deploy"},
        idempotency_key="deploy-once",
    )

    with pytest.raises(ValueError, match="different dependency contract"):
        kb.create_task(
            conn,
            title="conflicting deploy retry",
            parents=[first_parent_id, second_parent_id],
            required_parent_results={
                first_parent_id: "ready_to_deploy",
                second_parent_id: "ready_to_deploy",
            },
            idempotency_key="deploy-once",
        )

    tasks = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ?",
        ("deploy-once",),
    ).fetchall()
    assert [row["id"] for row in tasks] == [existing_id]


def test_summary_or_metadata_claim_does_not_satisfy_result_gate(conn) -> None:
    parent_id = kb.create_task(conn, title="qa", assignee="reviewer")
    assert kb.complete_task(
        conn,
        parent_id,
        result="qa_failed",
        summary="ready_to_deploy",
        metadata={"result": "ready_to_deploy"},
    )

    child_id = kb.create_task(
        conn,
        title="deploy",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )

    assert kb.get_task(conn, child_id).status == "todo"


@pytest.mark.parametrize("terminal_status", ["done", "archived"])
def test_legacy_null_predicate_keeps_terminal_status_behavior(
    conn, terminal_status: str,
) -> None:
    parent_id = _completed_parent(conn, result="anything")
    if terminal_status == "archived":
        assert kb.archive_task(conn, parent_id)

    child_id = kb.create_task(conn, title="ordinary", parents=[parent_id])

    assert kb.get_task(conn, child_id).status == "ready"
    edge = conn.execute(
        "SELECT required_parent_result FROM task_links "
        "WHERE parent_id = ? AND child_id = ?",
        (parent_id, child_id),
    ).fetchone()
    assert edge["required_parent_result"] is None


def test_linking_unsatisfied_result_gate_demotes_ready_child(conn) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")
    child_id = kb.create_task(conn, title="deploy")

    kb.link_tasks(
        conn,
        parent_id,
        child_id,
        required_parent_result="ready_to_deploy",
    )

    assert kb.get_task(conn, child_id).status == "todo"
    conn.execute(
        "UPDATE tasks SET result = 'ready_to_deploy' WHERE id = ?",
        (parent_id,),
    )
    assert kb.recompute_ready(conn) == 1
    assert kb.get_task(conn, child_id).status == "ready"


def test_claim_and_repeated_recompute_fail_closed(conn) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")
    child_id = kb.create_task(
        conn,
        title="deploy",
        assignee="deployer",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )

    for _ in range(3):
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child_id).status == "todo"

    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child_id,))
    assert kb.claim_task(conn, child_id, claimer="dispatcher") is None
    assert kb.get_task(conn, child_id).status == "todo"


def test_claim_holds_child_after_completed_parent_result_stops_matching(
    conn,
) -> None:
    parent_id = _completed_parent(conn, result="ready_to_deploy")
    child_id = kb.create_task(
        conn,
        title="deploy",
        assignee="deployer",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )
    assert kb.get_task(conn, child_id).status == "ready"

    assert kb.edit_completed_task_result(
        conn,
        parent_id,
        result="qa_failed",
    )

    assert kb.claim_task(conn, child_id, claimer="dispatcher") is None
    assert kb.get_task(conn, child_id).status == "todo"


def test_recompute_releases_child_after_completed_parent_result_starts_matching(
    conn,
) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")
    child_id = kb.create_task(
        conn,
        title="deploy",
        assignee="deployer",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )
    assert kb.get_task(conn, child_id).status == "todo"

    assert kb.edit_completed_task_result(
        conn,
        parent_id,
        result="ready_to_deploy",
    )

    assert kb.recompute_ready(conn) == 1
    claimed = kb.claim_task(conn, child_id, claimer="dispatcher")
    assert claimed is not None
    assert claimed.id == child_id


def test_manual_force_cannot_bypass_result_predicate(conn) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")
    child_id = kb.create_task(
        conn,
        title="deploy",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )

    ok, reason = kb.promote_task(
        conn,
        child_id,
        actor="operator",
        force=True,
    )

    assert not ok
    assert "required parent results" in (reason or "")
    assert kb.get_task(conn, child_id).status == "todo"


def test_manual_force_still_bypasses_legacy_status_dependency(conn) -> None:
    parent_id = kb.create_task(conn, title="ordinary parent")
    child_id = kb.create_task(conn, title="ordinary child", parents=[parent_id])

    ok, reason = kb.promote_task(
        conn,
        child_id,
        actor="operator",
        force=True,
    )

    assert ok
    assert reason is None
    assert kb.get_task(conn, child_id).status == "ready"


def test_dispatch_ticks_never_spawn_unsatisfied_result_gate(
    conn, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    parent_id = _completed_parent(conn, result="qa_failed")
    child_id = kb.create_task(
        conn,
        title="deploy",
        assignee="deployer",
        parents=[parent_id],
        required_parent_results={parent_id: "ready_to_deploy"},
    )
    spawned: list[str] = []

    def fake_spawn(task, _workspace, board=None):
        spawned.append(task.id)
        return 4242

    for _ in range(3):
        result = kb.dispatch_once(
            conn,
            spawn_fn=fake_spawn,
            reconcile_orphans=False,
        )
        assert result.promoted == 0
        assert spawned == []
        assert kb.get_task(conn, child_id).status == "todo"

    conn.execute(
        "UPDATE tasks SET result = 'ready_to_deploy' WHERE id = ?",
        (parent_id,),
    )
    result = kb.dispatch_once(
        conn,
        spawn_fn=fake_spawn,
        reconcile_orphans=False,
    )
    assert result.promoted == 1
    assert spawned == [child_id]


def test_creation_rejects_requirement_for_unlinked_parent(conn) -> None:
    parent_id = kb.create_task(conn, title="qa")

    with pytest.raises(ValueError, match="not present in parents"):
        kb.create_task(
            conn,
            title="deploy",
            required_parent_results={parent_id: "ready_to_deploy"},
        )


def test_legacy_task_links_schema_migrates_to_nullable_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = home / "legacy.db"

    with kb.connect_closing(db_path) as conn:
        parent_id = _completed_parent(conn, result="ordinary")
        child_id = kb.create_task(conn, title="ordinary", parents=[parent_id])
        conn.execute("ALTER TABLE task_links RENAME TO task_links_current")
        conn.execute(
            "CREATE TABLE task_links ("
            "parent_id TEXT NOT NULL, child_id TEXT NOT NULL, "
            "PRIMARY KEY (parent_id, child_id))"
        )
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) "
            "SELECT parent_id, child_id FROM task_links_current"
        )
        conn.execute("DROP TABLE task_links_current")

    kb.init_db(db_path)

    with kb.connect_closing(db_path) as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_links)")
        }
        assert "required_parent_result" in columns
        edge = conn.execute(
            "SELECT required_parent_result FROM task_links "
            "WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        ).fetchone()
        assert edge["required_parent_result"] is None
        assert kb._parents_satisfied(conn, child_id)


def _create_namespace(**overrides) -> argparse.Namespace:
    values = {
        "title": "deploy",
        "body": None,
        "assignee": "deployer",
        "created_by": "operator",
        "workspace": "scratch",
        "branch": None,
        "project": None,
        "tenant": None,
        "priority": 0,
        "parent": [],
        "required_parent_results": [],
        "triage": False,
        "idempotency_key": None,
        "max_runtime": None,
        "skills": None,
        "max_retries": None,
        "model_override": None,
        "provider_override": None,
        "goal_mode": False,
        "goal_max_turns": None,
        "initial_status": "running",
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_cli_create_parses_and_persists_parent_result_requirement(
    conn, capsys,
) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")

    rc = kb_cli._cmd_create(
        _create_namespace(
            parent=[parent_id],
            required_parent_results=[f"{parent_id}=ready_to_deploy"],
        )
    )

    assert rc == 0, capsys.readouterr().err
    child = conn.execute(
        "SELECT id, status FROM tasks WHERE title = 'deploy' ORDER BY created_at DESC"
    ).fetchone()
    assert child["status"] == "todo"
    edge = conn.execute(
        "SELECT required_parent_result FROM task_links "
        "WHERE parent_id = ? AND child_id = ?",
        (parent_id, child["id"]),
    ).fetchone()
    assert edge["required_parent_result"] == "ready_to_deploy"


def test_cli_link_persists_parent_result_requirement(conn, capsys) -> None:
    parent_id = _completed_parent(conn, result="qa_failed")
    child_id = kb.create_task(conn, title="child")

    rc = kb_cli._cmd_link(
        argparse.Namespace(
            parent_id=parent_id,
            child_id=child_id,
            required_parent_result="ready_to_deploy",
        )
    )

    assert rc == 0, capsys.readouterr().err
    assert kb.get_task(conn, child_id).status == "todo"
    edge = conn.execute(
        "SELECT required_parent_result FROM task_links "
        "WHERE parent_id = ? AND child_id = ?",
        (parent_id, child_id),
    ).fetchone()
    assert edge["required_parent_result"] == "ready_to_deploy"


def test_cli_parser_exposes_typed_result_gate_options() -> None:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    kb_cli.build_parser(subparsers)

    create_args = root.parse_args(
        [
            "kanban",
            "create",
            "deploy",
            "--parent",
            "t_qa",
            "--require-parent-result",
            "t_qa=ready_to_deploy",
        ]
    )
    link_args = root.parse_args(
        [
            "kanban",
            "link",
            "t_qa",
            "t_deploy",
            "--required-parent-result",
            "ready_to_deploy",
        ]
    )

    assert create_args.required_parent_results == ["t_qa=ready_to_deploy"]
    assert link_args.required_parent_result == "ready_to_deploy"


def test_worker_create_tool_persists_typed_result_gate(conn) -> None:
    from tools import kanban_tools as kt

    parent_id = _completed_parent(conn, result="qa_failed")
    payload = kt._handle_create(
        {
            "title": "tool child",
            "assignee": "deployer",
            "parents": [parent_id],
            "required_parent_results": {parent_id: "ready_to_deploy"},
        }
    )
    result = __import__("json").loads(payload)

    assert result["ok"] is True
    assert kb.get_task(conn, result["task_id"]).status == "todo"
    edge = conn.execute(
        "SELECT required_parent_result FROM task_links "
        "WHERE parent_id = ? AND child_id = ?",
        (parent_id, result["task_id"]),
    ).fetchone()
    assert edge["required_parent_result"] == "ready_to_deploy"


def test_worker_link_tool_persists_typed_result_gate(conn) -> None:
    from tools import kanban_tools as kt

    parent_id = _completed_parent(conn, result="qa_failed")
    child_id = kb.create_task(conn, title="tool linked child")
    payload = kt._handle_link(
        {
            "parent_id": parent_id,
            "child_id": child_id,
            "required_parent_result": "ready_to_deploy",
        }
    )
    result = __import__("json").loads(payload)

    assert result["ok"] is True
    assert kb.get_task(conn, child_id).status == "todo"


def test_worker_tool_schemas_type_result_gate_options() -> None:
    from tools import kanban_tools as kt

    create_prop = kt.KANBAN_CREATE_SCHEMA["parameters"]["properties"][
        "required_parent_results"
    ]
    link_prop = kt.KANBAN_LINK_SCHEMA["parameters"]["properties"][
        "required_parent_result"
    ]

    assert create_prop["type"] == "object"
    assert create_prop["additionalProperties"] == {"type": "string"}
    assert link_prop["type"] == "string"
