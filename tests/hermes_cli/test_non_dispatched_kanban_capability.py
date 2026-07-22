"""Native non-dispatched Kanban task capability tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def request_kwargs(status="todo", correlation_key="batch9-contract"):
    return {
        "title": "non-dispatched audit pickup",
        "body": "Audit pickup task carrying audit_task_contract.v1.",
        "assignee": "bafuxunan",
        "initial_status": status,
        "no_dispatch": True,
        "correlation_key": correlation_key,
        "metadata": {
            "audit_task_contract": {
                "schema_version": "audit_task_contract.v1",
                "verdict_source": "structured_completion_metadata",
                "github_comment_verdict_allowed": False,
            },
            "source_issue": "#595",
        },
        "created_by": "test",
    }


@pytest.mark.parametrize("status", ["todo", "ready"])
def test_create_non_dispatched_todo_ready_read_after_create(kanban_home, status):
    with kb.connect() as conn:
        proof = kb.create_non_dispatched_task(
            conn, **request_kwargs(status=status, correlation_key=f"key-{status}")
        )
        assert proof["status"] == status
        assert proof["assignee"] == "bafuxunan"
        assert proof["metadata"]["audit_task_contract"]["schema_version"] == "audit_task_contract.v1"
        assert proof["no_dispatch"] is True
        assert proof["read_after_create_verified"] is True
        assert proof["worker_execution_started"] is False
        assert proof["dispatcher_pickup_count"] == 0

        task = kb.get_task(conn, proof["task_id"])
        assert task is not None
        assert task.no_dispatch is True
        assert task.correlation_key == f"key-{status}"
        assert task.task_metadata == proof["metadata"]


def test_no_dispatch_ready_cannot_be_claimed_by_dispatcher(kanban_home):
    with kb.connect() as conn:
        proof = kb.create_non_dispatched_task(conn, **request_kwargs(status="ready"))
        assert kb.claim_task(conn, proof["task_id"]) is None
        after = kb.read_after_create_non_dispatched_task(conn, proof["task_id"])
        assert after["status"] == "ready"
        assert after["dispatcher_pickup_count"] == 0


def test_correlation_key_duplicate_rejected_before_second_create(kanban_home):
    with kb.connect() as conn:
        first = kb.create_non_dispatched_task(conn, **request_kwargs(correlation_key="dup"))
        with pytest.raises(ValueError, match="duplicate correlation_key"):
            kb.create_non_dispatched_task(conn, **request_kwargs(status="ready", correlation_key="dup"))
        rows = conn.execute("SELECT id FROM tasks WHERE correlation_key = 'dup'").fetchall()
        assert [r["id"] for r in rows] == [first["task_id"]]


@pytest.mark.parametrize(
    ("status", "message"),
    [("running", "running cannot be used or interpreted as ready"), ("blocked", "blocked cannot be used or interpreted as todo")],
)
def test_status_substitution_rejected_fail_closed(kanban_home, status, message):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match=message):
            kb.create_non_dispatched_task(conn, **request_kwargs(status=status))


@pytest.mark.parametrize("value", [False, None])
def test_missing_or_false_no_dispatch_rejected(kanban_home, value):
    kwargs = request_kwargs()
    kwargs["no_dispatch"] = value
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="no_dispatch must be true"):
            kb.create_non_dispatched_task(conn, **kwargs)


def test_missing_correlation_key_rejected(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="correlation_key is required"):
            kb.create_non_dispatched_task(conn, **request_kwargs(correlation_key=""))


@pytest.mark.parametrize("source", ["github_comment_regex", "github_comment", "regex", "timeout"])
def test_github_comment_regex_timeout_cannot_produce_verdict(kanban_home, source):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="GitHub comment / regex / timeout cannot produce audit verdict"):
            kb.create_non_dispatched_task(conn, **request_kwargs(), audit_verdict_source=source)


@pytest.mark.parametrize("kwargs", [{"created_via": "direct_db_write"}, {"direct_db_write": True}])
def test_direct_db_write_cannot_satisfy_native_contract(kanban_home, kwargs):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="direct Kanban DB write"):
            kb.create_non_dispatched_task(conn, **request_kwargs(), **kwargs)


def test_dispatch_leakage_fails_closed(kanban_home):
    with kb.connect() as conn:
        proof = kb.create_non_dispatched_task(conn, **request_kwargs())
        conn.execute("UPDATE tasks SET dispatcher_pickup_count = 1 WHERE id = ?", (proof["task_id"],))
        with pytest.raises(ValueError, match="dispatcher pickup observed"):
            kb.read_after_create_non_dispatched_task(conn, proof["task_id"])
