"""Fleet Linear contracts are validated before a Kanban worker is claimed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import projects_db as pdb
from hermes_cli.kanban_contract_guard import CONTRACT_BEGIN, CONTRACT_END, CONTRACT_HASH_FIELDS


@pytest.fixture
def fleet_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with pdb.connect_closing() as conn:
        project_id = pdb.create_project(
            conn, name="Hermes Agent", slug="hermes-agent", folders=[str(tmp_path / "repo")],
        )
    return home, project_id


def _hash(contract):
    canonical = json.dumps(
        {key: contract[key] for key in CONTRACT_HASH_FIELDS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _contract(**changes):
    contract = {
        "schema_version": 1,
        "issue_id": "ENS-8",
        "linear_id": "linear-uuid-8",
        "title": "Fleet acceptance test",
        "description": "Bounded test-only work.",
        "status": "Todo",
        "project_slug": "hermes-agent",
        "repository": "hermes-agent",
        "url": "https://linear.example/ENS-8",
        "labels": ["agent-ready"],
        "relations": {"blockedBy": [], "blocks": [], "duplicateOf": None, "relatedTo": []},
        "agent_ready": True,
        "blocked": False,
        "redactions": [],
        "execution_mode": "plan_only",
        "source_updated_at": "2026-10-07T08:02:03Z",
        "fetched_at": "2026-10-07T08:03:00Z",
    }
    contract.update(changes)
    contract["contract_sha256"] = _hash(contract)
    return contract


def _body(contract):
    return (
        "Frozen Linear contract.\n\n"
        f"{CONTRACT_BEGIN}\n```json\n"
        f"{json.dumps(contract, indent=2, sort_keys=True)}\n"
        f"```\n{CONTRACT_END}\n"
    )


def _create_fleet_task(conn, linked_project_id, contract=None, **changes):
    contract = contract or _contract()
    values = {
        "title": f"{contract['issue_id']}: {contract['title']}",
        "body": _body(contract),
        "assignee": "builder",
        "created_by": "fleet-linear-reconciler",
        "idempotency_key": f"linear:{contract['linear_id']}",
        "project_id": linked_project_id,
    }
    values.update(changes)
    return kb.create_task(conn, **values)


def test_valid_contract_is_spawnable(fleet_home, all_assignees_spawnable):
    _home, project_id = fleet_home
    with kbc.connect() as conn:
        task_id = _create_fleet_task(conn, project_id)
        assert kbd.has_spawnable_ready(conn) is True
        result = kbd.dispatch_once(conn, dry_run=True)
    assert [row[0] for row in result.spawned] == [task_id]
    assert result.contract_guarded == []


def test_hash_tamper_is_refused_without_mutation_on_dry_run(fleet_home, all_assignees_spawnable):
    _home, project_id = fleet_home
    contract = _contract()
    contract["description"] = "tampered after hashing"
    with kbc.connect() as conn:
        task_id = _create_fleet_task(conn, project_id, contract)
        assert kbd.has_spawnable_ready(conn) is False
        result = kbd.dispatch_once(conn, dry_run=True)
        task = kb.get_task(conn, task_id)
    assert result.spawned == []
    assert result.contract_guarded == [(task_id, "snapshot_hash_mismatch")]
    assert task.status == "ready"


@pytest.mark.parametrize(
    ("task_changes", "reason"),
    [
        ({"created_by": "user"}, "provenance_mismatch"),
        ({"idempotency_key": "linear:different"}, "linear_identity_mismatch"),
        ({"title": "ENS-8: altered title"}, "task_title_mismatch"),
        ({"project_id": None, "workspace_kind": "scratch"}, "task_project_missing"),
    ],
)
def test_identity_or_routing_mismatch_is_blocked(
    fleet_home, all_assignees_spawnable, task_changes, reason,
):
    _home, project_id = fleet_home
    spawned = []
    with kbc.connect() as conn:
        task_id = _create_fleet_task(conn, project_id, **task_changes)
        result = kbd.dispatch_once(conn, spawn_fn=lambda *_args: spawned.append(True))
        task = kb.get_task(conn, task_id)
    assert spawned == []
    assert result.contract_guarded == [(task_id, reason)]
    assert task.status == "blocked"
    assert task.block_kind == "capability"


def test_contract_repository_must_match_first_class_project(fleet_home, all_assignees_spawnable):
    _home, project_id = fleet_home
    contract = _contract(repository="other-project")
    with kbc.connect() as conn:
        task_id = _create_fleet_task(conn, project_id, contract)
        result = kbd.dispatch_once(conn, spawn_fn=lambda *_args: pytest.fail("must not spawn"))
        task = kb.get_task(conn, task_id)
    assert result.contract_guarded == [(task_id, "repository_project_mismatch")]
    assert task.status == "blocked"


def test_invalid_review_contract_is_reopened_then_blocked(fleet_home, all_assignees_spawnable):
    _home, project_id = fleet_home
    contract = _contract()
    contract["description"] = "tampered after hashing"
    with kbc.connect() as conn:
        task_id = _create_fleet_task(conn, project_id, contract)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        result = kbd.dispatch_once(conn, spawn_fn=lambda *_args: pytest.fail("must not spawn"))
        task = kb.get_task(conn, task_id)
    assert result.contract_guarded == [(task_id, "snapshot_hash_mismatch")]
    assert task.status == "blocked"
    assert task.block_kind == "capability"


def test_ordinary_kanban_task_is_unchanged(fleet_home, all_assignees_spawnable):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="ordinary", assignee="builder")
        result = kbd.dispatch_once(conn, dry_run=True)
    assert [row[0] for row in result.spawned] == [task_id]
    assert result.contract_guarded == []


def test_dispatch_json_exposes_contract_guard_reason(fleet_home, monkeypatch, capsys):
    from hermes_cli import kanban_ops

    monkeypatch.setattr(
        kanban_ops.kbd,
        "dispatch_once",
        lambda *args, **kwargs: kbd.DispatchResult(
            contract_guarded=[("t_guarded", "snapshot_hash_mismatch")],
        ),
    )
    args = SimpleNamespace(
        dry_run=True, max=None, failure_limit=kbd.DEFAULT_FAILURE_LIMIT, json=True,
    )
    assert kanban_ops._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract_guarded"] == [
        {"task_id": "t_guarded", "reason": "snapshot_hash_mismatch"},
    ]
