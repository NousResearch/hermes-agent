"""Contract tests for the Kanban sanitized signal read surface."""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

CAPABILITIES = {
    "kanban.read_model.signal.v1",
    "kanban.events.signal.v1",
}
DENIED_KEYS = {
    "title",
    "body",
    "result",
    "prompt",
    "workspace_path",
    "branch_name",
    "claim_lock",
    "error",
    "summary",
    "metadata",
    "payload",
    "last_failure_error",
}


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_signal_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def signal_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def signal_client(signal_home, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "_ws_upgrade_authorized", lambda _ws: True)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    with TestClient(app) as client:
        yield client


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _seed_sensitive_graph():
    secret = "ghp_fake-canary-credential"
    conn = kbc.connect()
    try:
        parent = kb.create_task(
            conn,
            title=f"Deploy {secret}",
            body=f"password={secret}",
            assignee="dev",
            priority=3,
        )
        child = kb.create_task(
            conn,
            title="Verify release",
            body=f"/root/private/{secret}",
            assignee="qa",
            priority=1,
        )
        conn.execute(
            "UPDATE tasks SET result=?, workspace_path=?, branch_name=?, "
            "last_failure_error=? WHERE id=?",
            (secret, f"/tmp/{secret}", secret, secret, parent),
        )
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent, child),
        )
        conn.execute(
            "INSERT INTO task_runs "
            "(task_id, profile, status, started_at, summary, metadata, error) "
            "VALUES (?, ?, 'done', 10, ?, ?, ?)",
            (parent, "dev", secret, json.dumps({"token": secret}), secret),
        )
        return secret, parent, child
    finally:
        conn.close()


def test_signal_connector_is_query_only_and_never_creates_board(signal_home):
    conn = kbc._connect_existing_query_only(board="default")
    try:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO kanban_metadata (key, value) VALUES ('x', 'y')")
    finally:
        conn.close()

    missing_path = kb.kanban_db_path(board="never-created")
    assert not missing_path.exists()
    with pytest.raises(FileNotFoundError):
        kbc._connect_existing_query_only(board="never-created")
    assert not missing_path.exists()


def test_signal_capabilities_are_discoverable_and_not_cached(signal_client):
    response = signal_client.get("/api/plugins/kanban/capabilities")
    assert response.status_code == 200
    assert set(response.json()["capabilities"]) == CAPABILITIES
    assert response.headers["cache-control"] == "private, no-store"


def test_signal_read_model_get_head_allowlists_and_redacts(signal_client):
    secret, parent, child = _seed_sensitive_graph()
    url = "/api/plugins/kanban/read-model/signal?board=default"

    response = signal_client.get(url)
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) == {
        "schema",
        "capabilities",
        "board",
        "generation",
        "cursor",
        "generated_at",
        "truncated",
        "tasks",
        "links",
        "runs",
    }
    assert data["schema"] == "kanban.read_model.signal.v1"
    assert data["truncated"] is False
    assert set(data["capabilities"]) == CAPABILITIES
    assert data["board"] == "default"
    assert re.fullmatch(r"[0-9a-f]{32}", data["generation"])
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-kanban-event-generation"] == data["generation"]
    assert int(response.headers["x-kanban-event-cursor"]) == data["cursor"]

    text = response.text
    assert secret not in text
    assert DENIED_KEYS.isdisjoint(set(_all_keys(data)))
    assert {task["id"] for task in data["tasks"]} == {parent, child}
    assert set(data["tasks"][0]) == {
        "id",
        "status",
        "priority",
        "assignee",
        "created_at",
        "started_at",
        "completed_at",
        "current_run_id",
        "block_kind",
    }
    assert data["links"] == [
        {"source": parent, "target": child, "kind": "parent-child"},
    ]
    assert set(data["runs"][0]) == {
        "id",
        "task_id",
        "profile",
        "status",
        "started_at",
        "ended_at",
        "outcome",
    }

    head = signal_client.head(url)
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["cache-control"] == "private, no-store"
    assert head.headers["x-kanban-event-generation"] == data["generation"]
    assert int(head.headers["x-kanban-event-cursor"]) == data["cursor"]


@pytest.mark.parametrize(
    "query,status_code",
    [
        ("", 400),
        ("?board=default&board=default", 400),
        ("?board=default&extra=1", 400),
        ("?board=../default", 400),
        ("?board=missing", 404),
    ],
)
def test_signal_read_model_rejects_noncanonical_queries(
    signal_client, query, status_code,
):
    response = signal_client.get(f"/api/plugins/kanban/read-model/signal{query}")
    assert response.status_code == status_code


def test_signal_read_model_bounds_links_and_marks_truncation(
    signal_client, monkeypatch,
):
    module = sys.modules["hermes_dashboard_plugin_kanban_signal_test"]
    monkeypatch.setattr(module.signal_api, "_SIGNAL_LINK_LIMIT", 1)
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="pm")
        first = kb.create_task(conn, title="first", assignee="dev")
        second = kb.create_task(conn, title="second", assignee="qa")
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?), (?, ?)",
            (parent, first, parent, second),
        )
    finally:
        conn.close()

    response = signal_client.get(
        "/api/plugins/kanban/read-model/signal?board=default"
    )
    assert response.status_code == 200
    assert response.json()["truncated"] is True
    assert len(response.json()["links"]) == 1


def test_event_generation_rotates_only_when_gc_deletes_history(signal_home):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="old", assignee="qa")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        conn.execute(
            "UPDATE task_events SET created_at=0 WHERE task_id=?",
            (task_id,),
        )
        before = kb.event_generation(conn)
        assert kb.gc_events(conn, older_than_seconds=1) > 0
        after = kb.event_generation(conn)
        assert after != before
        assert kb.gc_events(conn, older_than_seconds=1) == 0
        assert kb.event_generation(conn) == after
    finally:
        conn.close()


def test_event_generation_is_stable_for_append_only_history(signal_home):
    conn = kbc.connect()
    try:
        before = kb.event_generation(conn)
        kb.create_task(conn, title="one", assignee="qa")
        kb.create_task(conn, title="two", assignee="dev")
        assert kb.event_generation(conn) == before
    finally:
        conn.close()


def test_event_generation_is_materialized_for_legacy_board(signal_home):
    conn = kbc.connect()
    try:
        conn.execute("DROP TABLE kanban_metadata")
    finally:
        conn.close()

    kb.init_db()
    conn = kbc.connect()
    try:
        assert re.fullmatch(r"[0-9a-f]{32}", kb.event_generation(conn))
    finally:
        conn.close()


def test_event_generation_rotates_with_hard_delete(signal_home):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="delete", assignee="qa")
        conn.execute("DELETE FROM task_events WHERE task_id=?", (task_id,))
        before = kb.event_generation(conn)
        assert kb.delete_task(conn, task_id) is True
        after = kb.event_generation(conn)
        assert after != before
        assert kb.delete_task(conn, task_id) is False
        assert kb.event_generation(conn) == after
    finally:
        conn.close()


def test_event_generation_rotates_with_archived_delete(signal_home):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="archive", assignee="qa")
        assert kb.archive_task(conn, task_id) is True
        before = kb.event_generation(conn)
        assert kb.delete_archived_task(conn, task_id) is True
        assert kb.event_generation(conn) != before
    finally:
        conn.close()


def test_signal_websocket_sends_initial_allowlisted_frame(signal_client):
    secret, parent, _child = _seed_sensitive_graph()
    path = (
        "/api/plugins/kanban/events"
        "?view=signal&board=default&since=0&token=test"
    )
    with signal_client.websocket_connect(path) as websocket:
        frame = websocket.receive_json()

    assert set(frame) == {
        "schema",
        "initial",
        "generation",
        "reset",
        "cursor",
        "events",
    }
    assert frame["schema"] == "kanban.events.signal.v1"
    assert frame["initial"] is True
    assert frame["reset"] is False
    assert re.fullmatch(r"[0-9a-f]{32}", frame["generation"])
    assert frame["events"]
    assert {event["task_id"] for event in frame["events"]} >= {parent}
    assert all(
        set(event) == {"id", "task_id", "run_id", "kind", "created_at"}
        for event in frame["events"]
    )
    assert secret not in json.dumps(frame)
    assert "payload" not in set(_all_keys(frame))


@pytest.mark.parametrize(
    "query",
    [
        "?view=signal&board=default&token=test",
        "?view=signal&since=0&token=test",
        "?view=signal&board=default&since=-1&token=test",
        "?view=signal&board=default&since=01&token=test",
        "?view=signal&board=default&since=x&token=test",
        "?view=signal&board=default&since=1&token=test",
        "?view=signal&board=default&since=1&generation=bad&token=test",
        "?view=signal&board=default&since=1&generation=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA&token=test",
        "?view=signal&board=default&since=1&generation=00000000000000000000000000000000&generation=11111111111111111111111111111111&token=test",
        "?view=signal&view=signal&board=default&since=0&token=test",
        "?view=signal&board=default&since=0&token=test&ticket=test",
        "?view=signal&board=default&since=0&extra=1&token=test",
    ],
)
def test_signal_websocket_rejects_noncanonical_queries(signal_client, query):
    with pytest.raises(WebSocketDisconnect) as exc:
        with signal_client.websocket_connect(
            f"/api/plugins/kanban/events{query}",
        ):
            pass
    assert exc.value.code == 1008


def test_signal_ws_delegates_to_canonical_web_auth(signal_home, monkeypatch):
    module = _load_plugin_module()
    observed = []

    from hermes_cli import web_server_chat

    def deny(websocket):
        observed.append(websocket)
        return False

    monkeypatch.setattr(web_server_chat, "_ws_auth_ok", deny)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                "/api/plugins/kanban/events?view=signal&board=default&since=0&token=denied"
            ):
                pass

    assert exc.value.code == 1008
    assert len(observed) == 1


def test_signal_websocket_resets_after_history_rotation(signal_client):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="old", assignee="qa")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        cursor = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id=?", (task_id,),
        ).fetchone()[0]
        old_generation = kb.event_generation(conn)
        conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (task_id,))
        assert kb.gc_events(conn, older_than_seconds=1) > 0
        assert kb.event_generation(conn) != old_generation
    finally:
        conn.close()

    path = (
        "/api/plugins/kanban/events"
        f"?view=signal&board=default&since={cursor}"
        f"&generation={old_generation}&token=test"
    )
    with signal_client.websocket_connect(path) as websocket:
        frame = websocket.receive_json()
    assert frame["initial"] is True
    assert frame["reset"] is True
    assert frame["events"] == []


def test_signal_websocket_pushes_reset_when_generation_changes(signal_client):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="old", assignee="qa")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        cursor = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id=?", (task_id,),
        ).fetchone()[0]
        generation = kb.event_generation(conn)
    finally:
        conn.close()

    path = (
        "/api/plugins/kanban/events"
        f"?view=signal&board=default&since={cursor}"
        f"&generation={generation}&token=test"
    )
    with signal_client.websocket_connect(path) as websocket:
        initial = websocket.receive_json()
        assert initial["reset"] is False

        conn = kbc.connect()
        try:
            conn.execute(
                "UPDATE task_events SET created_at=0 WHERE task_id=?", (task_id,),
            )
            assert kb.gc_events(conn, older_than_seconds=1) > 0
        finally:
            conn.close()

        reset = websocket.receive_json()

    assert reset["initial"] is False
    assert reset["reset"] is True
    assert reset["generation"] != initial["generation"]
    assert reset["events"] == []


def test_signal_reconnect_detects_internal_history_gap(signal_client):
    conn = kbc.connect()
    try:
        task_ids = [
            kb.create_task(conn, title=f"task-{index}", assignee="qa")
            for index in range(3)
        ]
        conn.execute(
            "UPDATE tasks SET status='done' WHERE id IN (?, ?, ?)", task_ids,
        )
        cursor = conn.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
        generation = kb.event_generation(conn)
        conn.execute(
            "UPDATE task_events SET created_at=0 WHERE task_id=?", (task_ids[1],),
        )
        assert kb.gc_events(conn, older_than_seconds=1) == 1
    finally:
        conn.close()

    path = (
        "/api/plugins/kanban/events"
        f"?view=signal&board=default&since={cursor}"
        f"&generation={generation}&token=test"
    )
    with signal_client.websocket_connect(path) as websocket:
        frame = websocket.receive_json()

    assert frame["initial"] is True
    assert frame["reset"] is True
    assert frame["events"] == []


def test_legacy_websocket_still_emits_raw_event_envelope(signal_client):
    _secret, parent, _child = _seed_sensitive_graph()
    path = "/api/plugins/kanban/events?since=0&token=test"
    with signal_client.websocket_connect(path) as websocket:
        frame = websocket.receive_json()

    assert set(frame) == {"events", "cursor"}
    event = next(item for item in frame["events"] if item["task_id"] == parent)
    assert "payload" in event


def test_event_generation_rotates_when_legacy_events_are_renumbered(signal_home):
    conn = kbc.connect()
    try:
        before = kb.event_generation(conn)
        conn.execute("DROP TABLE task_events")
        conn.execute(
            "CREATE TABLE task_events (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, "
            "run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO task_events VALUES ('99', 't_legacy', NULL, 'created', NULL, 1)")
        kbc._rebuild_drifted_tables(conn)
        assert conn.execute("SELECT id FROM task_events").fetchone()[0] == 1
        after = kb.event_generation(conn)
        assert after != before
        kbc._rebuild_drifted_tables(conn)
        assert kb.event_generation(conn) == after
    finally:
        conn.close()


def test_event_rebuild_and_generation_roll_back_together(signal_home):
    conn = kbc.connect()
    try:
        before = kb.event_generation(conn)
        conn.execute("DROP TABLE task_events")
        conn.execute(
            "CREATE TABLE task_events (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, "
            "run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO task_events VALUES ('99', 't_legacy', NULL, 'created', NULL, 1)")
        conn.execute(
            "CREATE TRIGGER reject_generation BEFORE UPDATE ON kanban_metadata "
            "BEGIN SELECT RAISE(ABORT, 'generation unavailable'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="generation unavailable"):
            kbc._rebuild_drifted_tables(conn)
        assert not conn.in_transaction
        assert conn.execute("SELECT id FROM task_events").fetchone()[0] == "99"
        assert kb.event_generation(conn) == before
    finally:
        conn.close()


@pytest.mark.parametrize("mutation", ["delete_task", "delete_archived_task", "gc_events"])
def test_signal_history_deletion_rolls_back_if_generation_cannot_rotate(signal_home, mutation):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="retained if rotation fails")
        kb.archive_task(conn, task_id)
        conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (task_id,))
        before = kb.event_generation(conn)
        count = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        conn.execute(
            "CREATE TRIGGER reject_generation BEFORE UPDATE ON kanban_metadata "
            "BEGIN SELECT RAISE(ABORT, 'generation unavailable'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="generation unavailable"):
            if mutation == "gc_events":
                kb.gc_events(conn, older_than_seconds=1)
            else:
                getattr(kb, mutation)(conn, task_id)
        assert not conn.in_transaction
        assert kb.get_task(conn, task_id) is not None
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == count
        assert kb.event_generation(conn) == before
    finally:
        conn.close()


@pytest.mark.parametrize("limited", ["TASK", "RUN"])
def test_signal_snapshot_bounds_each_entity(signal_client, monkeypatch, limited):
    module = sys.modules["hermes_dashboard_plugin_kanban_signal_test"].signal_api
    monkeypatch.setattr(module, f"_SIGNAL_{limited}_LIMIT", 1)
    _seed_sensitive_graph()
    conn = kbc.connect()
    try:
        task_id = conn.execute("SELECT id FROM tasks ORDER BY created_at, id LIMIT 1").fetchone()[0]
        conn.executemany(
            "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'done', 1)",
            [(task_id,), (task_id,)],
        )
    finally:
        conn.close()
    response = signal_client.get("/api/plugins/kanban/read-model/signal?board=default")
    assert response.status_code == 200
    model = response.json()
    assert model["truncated"] is True
    assert len(model["tasks" if limited == "TASK" else "runs"]) == 1
    visible = {task["id"] for task in model["tasks"]}
    assert all(link["source"] in visible and link["target"] in visible for link in model["links"])
    assert all(run["task_id"] in visible for run in model["runs"])


def test_signal_event_batches_are_bounded_and_empty_batch_keeps_cursor(signal_home):
    module = _load_plugin_module().signal_api
    conn = kbc.connect()
    try:
        conn.executemany(
            "INSERT INTO task_events (task_id, kind, created_at) VALUES ('t_safe', 'created', ?)",
            [(index,) for index in range(201)],
        )
        generation = kb.event_generation(conn)
    finally:
        conn.close()
    first = module._signal_event_frame("default", 0, initial=True)
    second = module._signal_event_frame("default", first["cursor"], initial=False, expected_generation=generation)
    empty = module._signal_event_frame("default", second["cursor"], initial=False, expected_generation=generation)
    assert len(first["events"]) == 200
    assert len(second["events"]) == 1
    assert empty["events"] == []
    assert empty["cursor"] == second["cursor"]
    assert not any(frame["reset"] for frame in (first, second, empty))


@pytest.mark.parametrize("token,origin,accepted", [
    ("test-only", "http://127.0.0.1", True),
    ("incorrect", "http://127.0.0.1", False),
    ("test-only", "https://foreign.invalid", False),
])
def test_signal_real_canonical_token_and_origin_gates(signal_home, monkeypatch, token, origin, accepted):
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "test-only")
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    module = _load_plugin_module()
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    url = f"ws://127.0.0.1/api/plugins/kanban/events?view=signal&board=default&since=0&token={token}"
    monkeypatch.setattr(web_server.app.state, "bound_port", 80, raising=False)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        if accepted:
            with client.websocket_connect(url, headers={"origin": origin}) as websocket:
                frame = websocket.receive_json()
            assert frame["initial"] is True
            assert frame["events"] == []
        else:
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect(url, headers={"origin": origin}):
                    pass
            assert exc.value.code == 1008
