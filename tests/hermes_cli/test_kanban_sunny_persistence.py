from __future__ import annotations

import sqlite3
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban as kc


def _legacy_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(kb.SCHEMA_SQL)
    conn.execute("DROP TABLE sunny_route_policies")
    for column in ("route_policy_version", "route_policy_ref", "bucket_key"):
        conn.execute(f"ALTER TABLE tasks DROP COLUMN {column}")
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, ?, ?)",
        ("legacy-1", "Legacy", "done", 100),
    )
    conn.execute(
        "INSERT INTO task_events (task_id, kind, created_at) VALUES (?, ?, ?)",
        ("legacy-1", "completed", 101),
    )
    conn.commit()
    conn.close()


def test_sunny_migration_preserves_legacy_rows_with_null_references(tmp_path):
    path = tmp_path / "kanban.db"
    _legacy_db(path)
    with kbc.connect_closing(db_path=path) as conn:
        task = kb.get_task(conn, "legacy-1")
        assert task is not None
        assert (task.bucket_key, task.route_policy_ref, task.route_policy_version) == (
            None, None, None,
        )
        assert [event.kind for event in kb.list_events(conn, task.id)] == ["completed"]
        assert conn.execute("SELECT COUNT(*) FROM sunny_route_policies").fetchone()[0] == 0


def test_route_policies_are_immutable_and_task_references_are_queryable(tmp_path):
    with kbc.connect_closing(db_path=tmp_path / "kanban.db") as conn:
        policy = kb.put_route_policy(
            conn, policy_ref="sunny.worker", policy_version=1,
            provider_ref="openai-codex", model_ref="gpt-5.6-sol", effort_ref="low",
        )
        assert kb.put_route_policy(
            conn, policy_ref="sunny.worker", policy_version=1,
            provider_ref="openai-codex", model_ref="gpt-5.6-sol", effort_ref="low",
        ) == policy
        with pytest.raises(ValueError, match="different contract"):
            kb.put_route_policy(
                conn, policy_ref="sunny.worker", policy_version=1,
                provider_ref="openrouter", model_ref="openai/gpt-5.6-sol",
            )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sunny_route_policies)")}
        assert not columns & {"api_key", "credential", "credentials", "entitlement"}

        task_id = kb.create_task(
            conn, title="Routed", bucket_key="coding",
            route_policy_ref=policy.policy_ref, route_policy_version=policy.policy_version,
        )
        assert [task.id for task in kb.list_tasks(
            conn, bucket_key="coding", route_policy_ref="sunny.worker",
            route_policy_version=1,
        )] == [task_id]
        before = kb.get_task(conn, task_id)
        assert kb.update_task_route_references(conn, task_id, bucket_key="review")
        after = kb.get_task(conn, task_id)
        assert after.bucket_key == "review"
        assert (after.status, after.claim_lock, after.current_run_id) == (
            before.status, before.claim_lock, before.current_run_id,
        )


def test_cli_round_trips_sunny_references(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    created = json.loads(kc.run_slash(
        "create Routed --bucket-key coding --route-policy-ref sunny.worker "
        "--route-policy-version 1 --json"
    ))
    assert (created["bucket_key"], created["route_policy_ref"],
            created["route_policy_version"]) == ("coding", "sunny.worker", 1)
    listed = json.loads(kc.run_slash(
        "list --bucket-key coding --route-policy-ref sunny.worker "
        "--route-policy-version 1 --json"
    ))
    assert [task["id"] for task in listed] == [created["id"]]
    assert "Edited" in kc.run_slash(f"edit {created['id']} --bucket-key review")
    shown = json.loads(kc.run_slash(f"show {created['id']} --json"))
    assert shown["task"]["bucket_key"] == "review"
