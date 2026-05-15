"""Focused tests for the Kanban lifecycle plugin hook seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import plugins


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_events(monkeypatch):
    events: list[dict] = []

    def fake_invoke_hook(name: str, **kwargs):
        assert name == "on_kanban_event"
        events.append(kwargs["event"])
        return []

    monkeypatch.setattr(plugins, "invoke_hook", fake_invoke_hook)
    return events


def test_valid_hooks_contains_kanban_lifecycle_hook():
    assert "on_kanban_event" in plugins.VALID_HOOKS


def test_create_task_emits_after_commit_and_uses_task_event_id(kanban_home, monkeypatch):
    observed: list[tuple[dict, str]] = []

    def fake_invoke_hook(name: str, **kwargs):
        assert name == "on_kanban_event"
        event = kwargs["event"]
        with kb.connect() as read_conn:
            committed = kb.get_task(read_conn, event["task_id"])
        assert committed is not None
        observed.append((event, committed.status))
        return []

    monkeypatch.setattr(plugins, "invoke_hook", fake_invoke_hook)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="hook me",
            body="raw task body must not enter hook payload",
            assignee="worker",
        )
        row_event = kb.list_events(conn, tid)[0]

    assert len(observed) == 1
    event, committed_status = observed[0]
    assert committed_status == "ready"
    assert event["schema_version"] == 1
    assert event["event_type"] == "kanban.task_created"
    assert event["event_id"] == row_event.id
    assert event["task_id"] == tid
    assert event["status_after"] == "ready"
    assert event["assignee_after"] == "worker"
    assert event["payload"]["title_preview"] == "hook me"
    assert "raw task body" not in json.dumps(event)


def test_hook_failure_does_not_break_lifecycle_operations(kanban_home, monkeypatch, caplog):
    calls: list[str] = []

    def failing_hook(name: str, **kwargs):
        calls.append(kwargs["event"]["event_type"])
        raise RuntimeError("observer exploded")

    monkeypatch.setattr(plugins, "invoke_hook", failing_hook)

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker")
        kb.link_tasks(conn, parent, child)
        kb.add_comment(conn, parent, "reviewer", "looks okay")
        assert kb.block_task(conn, parent, reason="needs review")
        assert kb.complete_task(conn, parent, summary="done", result="finished")
        assert kb.get_task(conn, parent).status == "done"

    assert "kanban.task_created" in calls
    assert "kanban.dependency_linked" in calls
    assert "kanban.comment_added" in calls
    assert "kanban.task_blocked" in calls
    assert "kanban.task_completed" in calls
    assert "Kanban lifecycle hook delivery failed" in caplog.text


def test_failed_mutation_does_not_emit_lifecycle_success(kanban_home, captured_events):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        kb.link_tasks(conn, a, b)
        count_after_success = len(captured_events)
        with pytest.raises(ValueError):
            kb.link_tasks(conn, b, a)

    assert len(captured_events) == count_after_success
    assert [e["event_type"] for e in captured_events].count("kanban.dependency_linked") == 1


def test_append_event_requires_write_txn_for_after_commit_guarantee(kanban_home, captured_events):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="manual txn guard")
        event_count = len(kb.list_events(conn, tid))
        hook_count = len(captured_events)

        conn.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(RuntimeError, match="write_txn"):
                kb._append_event(conn, tid, "commented", {"author": "tester", "len": 1})
        finally:
            conn.execute("ROLLBACK")

        assert len(kb.list_events(conn, tid)) == event_count
        assert len(captured_events) == hook_count


def test_bounded_jsonish_does_not_process_items_beyond_cap():
    class ExplodingRepr:
        def __str__(self):
            raise AssertionError("item past the cap should not be stringified")

    big: dict[str, object] = {f"k{i}": i for i in range(20)}
    big["k20"] = ExplodingRepr()
    bounded = kb._bounded_jsonish(big)
    assert len(bounded) == 20
    assert list(bounded) == [f"k{i}" for i in range(20)]

    values = list(range(20)) + [ExplodingRepr()]
    assert kb._bounded_jsonish(values) == list(range(20))

    large_set = set(range(1000))
    assert len(kb._bounded_jsonish(large_set)) == 20


def test_lifecycle_payloads_are_sanitized_and_bounded(kanban_home, captured_events):
    big_body = "body-secret-" + "x" * 5000
    big_comment = "comment-secret-" + "y" * 5000
    big_result = "result-secret-" + "z" * 5000
    metadata = {"token": "metadata-secret", "count": 1}

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="redaction", body=big_body, assignee="worker")
        kb.add_comment(conn, tid, "human", big_comment)
        assert kb.block_task(conn, tid, reason="waiting on reviewer " + "r" * 1000)
        assert kb.complete_task(conn, tid, summary="summary-secret " + "s" * 1000, result=big_result, metadata=metadata)
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                tid,
                "commented",
                {"author": "human", "len": 7, "new_future_field": "future-secret"},
            )

    serialized = "\n".join(json.dumps(event, sort_keys=True) for event in captured_events)
    assert "body-secret" not in serialized
    assert "comment-secret" not in serialized
    assert "result-secret" not in serialized
    assert "metadata-secret" not in serialized
    assert "summary-secret" not in serialized
    assert "future-secret" not in serialized
    assert "token" in serialized  # metadata keys are useful; values are not.
    for event in captured_events:
        assert len(json.dumps(event)) < 4096

    blocked = next(e for e in captured_events if e["event_type"] == "kanban.task_blocked")
    assert blocked["payload"]["reason_len"] > len("waiting on reviewer")
    assert blocked["payload"]["reason_present"] is True
    assert "reason_preview" not in blocked["payload"]
    completed = next(e for e in captured_events if e["event_type"] == "kanban.task_completed")
    assert completed["payload"]["summary_len"] > len("summary")
    assert completed["payload"]["summary_present"] is True
    assert "summary_preview" not in completed["payload"]
    assert completed["payload"]["result_len"] == len(big_result)
    assert completed["payload"]["metadata_keys"] == ["count", "token"]
    future = [e for e in captured_events if e["event_type"] == "kanban.comment_added"][-1]
    assert future["payload"]["new_future_field_present"] is True
    assert future["payload"]["new_future_field_len"] == len("future-secret")
    assert "new_future_field" not in future["payload"]


def test_assignment_dependency_and_run_events_include_core_ids(
    kanban_home, captured_events, all_assignees_spawnable
):
    def spawn(_task, _workspace):
        return 12345

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alpha")
        child = kb.create_task(conn, title="child", assignee="alpha")
        kb.link_tasks(conn, parent, child)
        assert kb.assign_task(conn, child, "beta")
        assert kb.complete_task(conn, parent, summary="parent done")
        kb.dispatch_once(conn, spawn_fn=spawn)

    assigned = next(e for e in captured_events if e["event_type"] == "kanban.task_assigned")
    assert assigned["assignee_before"] == "alpha"
    assert assigned["assignee_after"] == "beta"
    assert assigned["payload"]["previous_assignee"] == "alpha"
    assert assigned["payload"]["assignee"] == "beta"

    linked = next(e for e in captured_events if e["event_type"] == "kanban.dependency_linked")
    assert linked["payload"]["parent_id"] == parent
    assert linked["payload"]["child_id"] == child

    promoted = next(e for e in captured_events if e["event_type"] == "kanban.task_promoted")
    assert promoted["task_id"] == child

    claimed = next(e for e in captured_events if e["event_type"] == "kanban.task_claimed")
    assert claimed["task_id"] == child
    assert isinstance(claimed["run_id"], int)
    assert claimed["payload"]["run_id"] == claimed["run_id"]

    spawned = next(e for e in captured_events if e["event_type"] == "kanban.worker_spawned")
    assert spawned["task_id"] == child
    assert spawned["run_id"] == claimed["run_id"]
    assert spawned["payload"]["pid_present"] is True
    assert "pid" not in spawned["payload"]
