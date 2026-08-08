"""C-1 generic per-row Kanban event observer contract."""

from __future__ import annotations

import ast
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def generic_hooks():
    manager = get_plugin_manager()
    seen: list[dict] = []
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    manager._hooks.setdefault("kanban_task_event", []).append(
        lambda **kwargs: seen.append(kwargs)
    )
    try:
        yield seen
    finally:
        manager._hooks = saved


def test_generic_hook_is_registered_without_replacing_legacy_hooks():
    assert "kanban_task_event" in VALID_HOOKS
    assert {
        "kanban_task_claimed",
        "kanban_task_completed",
        "kanban_task_blocked",
    } <= VALID_HOOKS


def test_one_committed_row_matches_exact_database_scalars(
    kanban_home, generic_hooks, monkeypatch
):
    monkeypatch.setattr(kb.time, "time", lambda: 1_700_000_123)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="event", assignee="worker", initial_status="running"
        )
        row = conn.execute(
            "SELECT id, task_id, run_id, kind, created_at FROM task_events "
            "WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchone()

    assert len(generic_hooks) == 1
    event = generic_hooks[0]
    assert event == {
        "task_id": row["task_id"],
        "kind": row["kind"],
        "core_event_seq": row["id"],
        "created_at_epoch_s": row["created_at"],
        "run_id": row["run_id"],
        "board": "default",
        "profile_name": "default",
        # create_task normalizes unsupported direct-running creation to ready.
        "status_to": "ready",
        "kanban_event_schema_version": "hermes.kanban.event.v1",
        "telemetry_schema_version": event["telemetry_schema_version"],
    }


def test_multirow_flush_is_post_commit_ordered_and_off_lock(
    kanban_home, generic_hooks
):
    manager = get_plugin_manager()
    callback_writes: list[str] = []

    def observe_and_write(*, kind, **_kwargs):
        assert origin.in_transaction is False
        if not kind.startswith("synthetic-"):
            return
        with kb.connect_closing() as observer:
            callback_writes.append(
                kb.create_task(observer, title=f"callback-{kind}", assignee="observer")
            )

    manager._hooks["kanban_task_event"].insert(0, observe_and_write)
    with kb.connect_closing() as origin:
        task_id = kb.create_task(origin, title="first", assignee="worker")
        generic_hooks.clear()
        callback_writes.clear()
        with kb.write_txn(origin):
            kb._append_event(origin, task_id, "synthetic-first")
            kb._append_event(origin, task_id, "synthetic-second")

    assert [
        event["kind"]
        for event in generic_hooks
        if event["kind"].startswith("synthetic-")
    ] == [
        "synthetic-first",
        "synthetic-second",
    ]
    assert len(callback_writes) == 2


def test_body_rollback_commit_failure_and_invariant_failure_suppress_and_reset(
    kanban_home, generic_hooks, monkeypatch
):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="failure paths", assignee="worker")
        generic_hooks.clear()
        with pytest.raises(RuntimeError, match="body"):
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "rollback-event")
                raise RuntimeError("body")
        assert generic_hooks == []

        real_boundary = kb._execute_boundary_with_retry

        def fail_commit(candidate, sql):
            if sql == "COMMIT":
                raise sqlite3.OperationalError("forced commit")
            return real_boundary(candidate, sql)

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", fail_commit)
        with pytest.raises(sqlite3.OperationalError, match="forced commit"):
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "commit-event")
        assert generic_hooks == []
        monkeypatch.setattr(kb, "_execute_boundary_with_retry", real_boundary)

        monkeypatch.setattr(
            kb,
            "_check_file_length_invariant",
            lambda _conn: (_ for _ in ()).throw(RuntimeError("invariant")),
        )
        with pytest.raises(RuntimeError, match="invariant"):
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "invariant-event")
        assert generic_hooks == []

    # Frame state was cleared despite each failure path.
    monkeypatch.setattr(kb, "_check_file_length_invariant", lambda _conn: None)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="after", assignee="worker")
    assert generic_hooks[-1]["task_id"] == task_id


def test_append_without_frame_fails_before_insert(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="owner", assignee="worker")
        before = len(kb.list_events(conn, task_id))
        with pytest.raises(RuntimeError, match="active write_txn"):
            kb._append_event(conn, task_id, "escaped")
        assert len(kb.list_events(conn, task_id)) == before


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (7, 7),
        (kb.FAILURE_COUNT_MAX, kb.FAILURE_COUNT_MAX),
        (True, None),
        ("7", None),
        (7.0, None),
        (-1, None),
        (1001, None),
        (None, None),
    ],
)
def test_failure_count_is_strict_bounded_scalar(
    kanban_home, generic_hooks, value, expected
):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="failure", assignee="worker")
        generic_hooks.clear()
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "synthetic-failure",
                {"failures": value, "error": "must-not-leak"},
                failure_count=value,
            )

    event = generic_hooks[0]
    if expected is None:
        assert "failure_count" not in event
    else:
        assert event["failure_count"] == expected
    assert not ({"payload", "reason", "error", "summary", "title", "body"} & event.keys())


def test_failure_writer_queues_its_local_count(kanban_home, generic_hooks):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="real failure", assignee="worker")
        assert kb.claim_task(conn, task_id, claimer="worker") is not None
        generic_hooks.clear()

        blocked = kb._record_spawn_failure(
            conn, task_id, "must-not-leak", failure_limit=2
        )

    assert blocked is False
    assert len(generic_hooks) == 1
    assert generic_hooks[0]["kind"] == "spawn_failed"
    assert generic_hooks[0]["failure_count"] == 1
    assert "error" not in generic_hooks[0]


def test_unknown_kind_dispatches_and_observer_failures_are_isolated(
    kanban_home, generic_hooks
):
    manager = get_plugin_manager()
    manager._hooks["kanban_task_event"].insert(
        0, lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("observer"))
    )
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="unknown", assignee="worker")
        generic_hooks.clear()
        with kb.write_txn(conn):
            kb._append_event(
                conn, task_id, "future-unknown", {"failures": 7}
            )
        assert kb.list_events(conn, task_id)[-1].kind == "future-unknown"

    assert generic_hooks[0]["kind"] == "future-unknown"
    assert "failure_count" not in generic_hooks[0]
    assert "future-unknown" not in kb.KANBAN_EVENT_KINDS


def test_synchronous_observer_latency_is_on_originating_return_path(
    kanban_home, generic_hooks
):
    manager = get_plugin_manager()
    delay_s = 0.03
    manager._hooks["kanban_task_event"].insert(
        0, lambda **_kwargs: time.sleep(delay_s)
    )

    started = time.monotonic()
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="latency", assignee="worker")
    elapsed = time.monotonic() - started

    assert elapsed >= delay_s
    assert len(generic_hooks) == 1


def test_dispatcher_and_worker_manual_origins_emit_in_their_own_processes(
    kanban_home, generic_hooks
):
    child_result = kanban_home.parent / "dispatcher-events.json"
    child_code = """
import json
import os
import sys
os.environ["HERMES_HOME"] = sys.argv[1]
from hermes_cli import kanban_db as kb
from hermes_cli.plugins import get_plugin_manager
seen = []
get_plugin_manager()._hooks.setdefault("kanban_task_event", []).append(
    lambda **kwargs: seen.append(kwargs)
)
with kb.connect_closing() as conn:
    task_id = kb.create_task(conn, title="cross-process", assignee="worker")
    assert kb.claim_task(conn, task_id, claimer="dispatcher") is not None
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({"pid": os.getpid(), "task_id": task_id, "events": seen}, handle)
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_code, str(kanban_home), str(child_result)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    dispatcher = json.loads(child_result.read_text())
    assert [event["kind"] for event in dispatcher["events"]] == ["created", "claimed"]
    assert generic_hooks == []

    with kb.connect_closing() as conn:
        assert kb.block_task(
            conn, dispatcher["task_id"], reason="manual worker boundary"
        ) is True

    assert [event["kind"] for event in generic_hooks] == ["blocked"]


def test_status_to_is_not_changed_by_reentrant_post_commit_mutation(
    kanban_home, generic_hooks
):
    manager = get_plugin_manager()
    mutated = False

    def mutate_after_created(*, task_id, kind, **_kwargs):
        nonlocal mutated
        if kind != "created" or mutated:
            return
        mutated = True
        with kb.connect_closing() as observer:
            with kb.write_txn(observer):
                observer.execute(
                    "UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,)
                )
                kb._append_event(
                    observer, task_id, "status", {"status": "todo"},
                    status_to="todo",
                )

    manager._hooks["kanban_task_event"].insert(0, mutate_after_created)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="capture", assignee="worker")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "todo"

    created = next(event for event in generic_hooks if event["kind"] == "created")
    assert created["status_to"] == "ready"


def test_source_inventory_declares_kinds_and_exact_status_matrix():
    source_path = Path(kb.__file__)
    tree = ast.parse(source_path.read_text())
    calls: list[tuple[str, str | None]] = []
    dynamic_kinds: set[str] = set()
    failure_count_calls: set[tuple[str, str]] = set()

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_append_event"
            and len(node.args) >= 3
        ):
            continue
        kind_node = node.args[2]
        status_to = next(
            (ast.unparse(kw.value) for kw in node.keywords if kw.arg == "status_to"),
            None,
        )
        failure_count = next(
            (
                ast.unparse(kw.value)
                for kw in node.keywords
                if kw.arg == "failure_count"
            ),
            None,
        )
        if failure_count is not None:
            failure_count_calls.add((ast.unparse(kind_node), failure_count))
        if isinstance(kind_node, ast.Constant) and isinstance(kind_node.value, str):
            calls.append((kind_node.value, status_to))
        else:
            dynamic_kinds.add(ast.unparse(kind_node))

    literal_kinds = {kind for kind, _status in calls}
    assert literal_kinds <= kb.KANBAN_EVENT_KINDS
    assert dynamic_kinds == {"event_kind", "outcome"}
    assert failure_count_calls == {
        ("'gave_up'", "failures"),
        ("outcome", "failures"),
    }
    assert kb.KANBAN_EVENT_KINDS - literal_kinds == {
        "crashed", "protocol_violation", "rate_limited", "spawn_failed",
        "reprioritized", "status",
    }

    actual: dict[str, set[str | None]] = {}
    for kind, status_to in calls:
        actual.setdefault(kind, set()).add(status_to)
    assert actual["created"] == {"task_status", "'todo'"}
    assert actual["linked"] == {"linked_status_to", None}
    expected_statuses = {
        "promoted": "'ready'",
        "promoted_manual": "'ready'",
        "claimed": "'running'",
        "claim_rejected": "'todo'",
        "reclaimed": "'ready'",
        "completed": "'done'",
        "dependency_wait": "'todo'",
        "blocked": "'blocked'",
        "block_loop_detected": "'triage'",
        "unblocked": "new_status",
        "specified": "'todo'",
        "decomposed": "'todo'",
        "scheduled": "'scheduled'",
        "gave_up": "'blocked'",
        "archived": "'archived'",
        "stale": "'ready'",
        "timed_out": "'ready'",
    }
    for kind, status_to in expected_statuses.items():
        assert actual[kind] == {status_to}

    for non_transition in (
        "assigned", "commented", "heartbeat", "spawned", "attached",
        "attachment_removed", "claim_extended", "reclaim_deferred",
    ):
        assert actual[non_transition] == {None}


def test_all_production_task_event_inserts_use_shared_append_seam():
    repo = Path(__file__).resolve().parents[2]
    direct_insert_files = []
    for path in repo.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(errors="replace")
        if "INSERT INTO task_events" in text:
            direct_insert_files.append(path.relative_to(repo).as_posix())
    assert direct_insert_files == ["hermes_cli/kanban_db.py"]

    dashboard = (
        repo / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    ).read_text()
    assert "INSERT INTO task_events" not in dashboard
    assert dashboard.count("kanban_db._append_event(") == 5


def test_public_plugin_docs_describe_generic_event_contract():
    repo = Path(__file__).resolve().parents[2]
    developer_guide = (
        repo / "website" / "docs" / "developer-guide" / "plugins" / "index.md"
    ).read_text()
    kanban_guide = (
        repo / "website" / "docs" / "user-guide" / "features" / "kanban.md"
    ).read_text()
    for document in (developer_guide, kanban_guide):
        assert "kanban_task_event" in document
        assert "core_event_seq" in document
        assert "at most once" in document
        assert "synchronous" in document
