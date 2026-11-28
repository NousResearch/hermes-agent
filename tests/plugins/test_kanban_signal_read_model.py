"""Contract tests for the Kanban sanitized signal read surface."""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
from functools import partial
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tests.hermes_cli import test_plugin_api_consent_gate as _consent_gate

# The multiplexed-gateway stand-in that mints native ``X-Hermes-Gateway-Ticket`` grants.
daemon = _consent_gate.daemon

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
    reader = kbc._connect_existing_query_only(board="default")
    try:
        first = module._signal_event_frame(reader, 0, initial=True)
        second = module._signal_event_frame(reader, first["cursor"], initial=False, expected_generation=generation)
        empty = module._signal_event_frame(reader, second["cursor"], initial=False, expected_generation=generation)
        assert not reader.in_transaction
    finally:
        reader.close()
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


# ---------------------------------------------------------------------------
# Real dashboard mount: the SDK scope parameter and canonical WS credentials
# ---------------------------------------------------------------------------

_DASHBOARD_HOST = {"Host": "127.0.0.1:9119"}


@pytest.fixture
def real_dashboard(signal_home, monkeypatch):
    """``hermes_cli.web_server.app`` as served: the bundled kanban router mounted
    behind ``_plugin_route_secret_scope`` and the plugin WebSocket gate."""
    from hermes_cli import profiles, web_server

    app = web_server.app
    mounted = {getattr(route, "path", "") for route in app.routes}
    assert "/api/plugins/kanban/read-model/signal" in mounted
    assert "/api/plugins/kanban/events" in mounted
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: signal_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: signal_home / "profiles")
    monkeypatch.setattr(app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(app.state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(app.state, "auth_required", False, raising=False)
    client = TestClient(app)
    client.headers.update({
        **_DASHBOARD_HOST, web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
    })
    return client, web_server._SESSION_TOKEN


@pytest.mark.parametrize("profile", ["default", "current"])
def test_signal_http_accepts_sdk_profile_scope_on_real_dashboard_mount(real_dashboard, profile):
    """fetchJSON/authedFetch append ``profile`` to every /api/plugins/ URL (PLG-3)."""
    client, _token = real_dashboard
    url = f"/api/plugins/kanban/read-model/signal?board=default&profile={profile}"

    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.json()["board"] == "default"
    assert "profile" not in response.json()
    assert response.headers["cache-control"] == "private, no-store"
    head = client.head(url)
    assert head.status_code == 200
    assert head.content == b""


@pytest.mark.parametrize("query,status_code", [
    ("?board=default&profile=default&profile=default", 400),
    ("?board=default&profile=default&extra=1", 400),
    ("?profile=default", 400),
])
def test_signal_http_scope_param_keeps_allowlist_strict_on_real_mount(real_dashboard, query, status_code):
    client, _token = real_dashboard
    response = client.get(f"/api/plugins/kanban/read-model/signal{query}")
    assert response.status_code == status_code, response.text


@pytest.mark.parametrize("profile", ["default", "current"])
def test_signal_ws_accepts_sdk_profile_scope_on_real_dashboard_mount(real_dashboard, profile):
    """buildWsUrl adds ``profile`` to every /api/plugins/ socket URL."""
    client, token = real_dashboard
    url = (
        "/api/plugins/kanban/events?view=signal&board=default&since=0"
        f"&token={token}&profile={profile}"
    )
    with client.websocket_connect(url, headers=_DASHBOARD_HOST) as websocket:
        frame = websocket.receive_json()
    assert frame["initial"] is True
    assert frame["reset"] is False
    assert "profile" not in frame


def test_signal_ws_accepts_gateway_ticket_subprotocol_without_query_credential(signal_home, monkeypatch):
    """The canonical gate admits a one-use ticket carried as the ``hermes-gateway-ticket.``
    subprotocol; the signal view must not then demand a second credential in the query, and it
    selects only the public gateway protocol (never the ticket) on accept."""
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests, mint_ticket
    from hermes_cli.web_server_chat import _GATEWAY_WS_PROTOCOL, _GATEWAY_WS_TICKET_PROTOCOL_PREFIX
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    _reset_for_tests()
    clear_providers()
    register_provider(StubAuthProvider())
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 9119, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    try:
        client = TestClient(
            web_server.app, base_url="http://127.0.0.1:9119", client=("127.0.0.1", 50000),
        )
        ticket = mint_ticket(user_id="signal-observer", provider="stub")
        protocols = [_GATEWAY_WS_PROTOCOL, f"{_GATEWAY_WS_TICKET_PROTOCOL_PREFIX}{ticket}"]
        url = "/api/plugins/kanban/events?view=signal&board=default&since=0"
        with client.websocket_connect(url, subprotocols=protocols, headers=_DASHBOARD_HOST) as websocket:
            frame = websocket.receive_json()
            assert websocket.accepted_subprotocol == _GATEWAY_WS_PROTOCOL
        assert frame["initial"] is True
        assert frame["schema"] == "kanban.events.signal.v1"

        # One-use: replaying the same ticket is refused by the canonical gate.
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(url, subprotocols=protocols, headers=_DASHBOARD_HOST):
                pass
        assert exc.value.code == 1008
    finally:
        clear_providers()
        _reset_for_tests()


def test_signal_ws_accepts_native_gateway_ticket_header_without_query_credential(signal_home, daemon):
    """A local Desktop's ``X-Hermes-Gateway-Ticket`` grant is redeemed by the plugin socket gate
    and accepted by ``_ws_auth_ok``; the signal view must open without a query credential."""
    _consent_gate._plugins(daemon.launch)
    from hermes_cli import web_server

    grant = web_server.app.state.gateway_runner.session_ticket_store.mint(
        profile_id=str(daemon.launch), subject="desktop", purpose="native-http",
    )
    url = "/api/plugins/kanban/events?view=signal&board=default&since=0"
    with daemon.client.websocket_connect(url, headers={"X-Hermes-Gateway-Ticket": grant}) as websocket:
        frame = websocket.receive_json()
    assert frame["initial"] is True
    assert frame["schema"] == "kanban.events.signal.v1"

    # A forged grant is refused before plugin code runs; there is no query fallback.
    with pytest.raises(WebSocketDisconnect) as exc:
        with daemon.client.websocket_connect(url, headers={"X-Hermes-Gateway-Ticket": "forged"}):
            pass
    assert exc.value.code == 1008


def _counting_signal_opens(monkeypatch):
    opened: list = []
    real = kbc._connect_existing_query_only

    def counting(*, board):
        conn = real(board=board)
        opened.append(conn)
        return conn

    monkeypatch.setattr(kbc, "_connect_existing_query_only", counting)
    return opened


def test_signal_ws_reuses_one_query_only_connection_per_socket(signal_client, monkeypatch):
    """Polling every 0.3 s must not open (and WAL/SHM-churn) a fresh SQLite connection per poll:
    one query-only connection serves the socket's lifetime and is closed on disconnect."""
    from hermes_cli.sqlite_safe_read import has_live_connection

    module = sys.modules["hermes_dashboard_plugin_kanban_signal_test"]
    monkeypatch.setattr(module, "_EVENT_POLL_SECONDS", 0.01)
    opened = _counting_signal_opens(monkeypatch)
    db_path = kb.kanban_db_path(board="default")
    path = "/api/plugins/kanban/events?view=signal&board=default&since=0&token=test"

    with signal_client.websocket_connect(path) as websocket:
        initial = websocket.receive_json()
        cursor = initial["cursor"]
        for title in ("one", "two", "three"):
            conn = kbc.connect()
            try:
                kb.create_task(conn, title=title, assignee="qa")
            finally:
                conn.close()
            frame = websocket.receive_json()
            assert frame["reset"] is False
            assert frame["cursor"] > cursor
            cursor = frame["cursor"]
        assert has_live_connection(db_path)

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].in_transaction  # the property raises once the connection is closed
    assert not has_live_connection(db_path)


# ---------------------------------------------------------------------------
# Unavailable boards: stable status/close codes, no file creation
# ---------------------------------------------------------------------------

_LEGACY_SCHEMA = (
    "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT, priority INTEGER,"
    " assignee TEXT, created_at INTEGER, started_at INTEGER, completed_at INTEGER,"
    " current_run_id INTEGER, block_kind TEXT);"
    "CREATE TABLE task_links (parent_id TEXT, child_id TEXT);"
    "CREATE TABLE task_runs (id INTEGER PRIMARY KEY, task_id TEXT, profile TEXT, status TEXT,"
    " started_at INTEGER, ended_at INTEGER, outcome TEXT);"
    "CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,"
    " run_id INTEGER, kind TEXT, payload TEXT, created_at INTEGER);"
)


def _snapshot_tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def _ws_close_code(client, query: str) -> int:
    """The close code of a signal socket, whether refused before or after accept."""
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/api/plugins/kanban/events{query}") as websocket:
            websocket.receive_json()
    return exc.value.code


def _archived_board(slug: str) -> None:
    kb.create_board(slug)
    kb.remove_board(slug, archive=True)
    assert kb.board_exists(slug)
    assert kb.read_board_metadata(slug).get("archived")


def _board_without_db(slug: str) -> None:
    kb.write_board_metadata(slug, name=slug)
    assert kb.board_exists(slug)
    assert not kb.kanban_db_path(board=slug).exists()


def _legacy_board(slug: str, *, metadata_table: bool = False) -> None:
    kb.write_board_metadata(slug, name=slug)
    path = kb.kanban_db_path(board=slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(path)
    try:
        legacy.executescript(_LEGACY_SCHEMA + (
            "CREATE TABLE kanban_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            if metadata_table else ""
        ))
        legacy.commit()
    finally:
        legacy.close()


@pytest.mark.parametrize("make,status_code,close_code", [
    (_archived_board, 404, 1008),
    (_board_without_db, 404, 1008),
    (_legacy_board, 503, 1011),
    (partial(_legacy_board, metadata_table=True), 503, 1011),
], ids=["archived-tombstone", "db-missing", "legacy-no-metadata", "legacy-no-generation"])
def test_signal_unavailable_board_answers_stably_without_creating_files(
    signal_client, make, status_code, close_code,
):
    make("observed")
    root = kb.kanban_home()
    before = _snapshot_tree(root)

    for method in ("GET", "HEAD"):
        response = signal_client.request(method, "/api/plugins/kanban/read-model/signal?board=observed")
        assert response.status_code == status_code, response.text
        assert response.headers["cache-control"] == "private, no-store"
    assert _ws_close_code(signal_client, "?view=signal&board=observed&since=0&token=test") == close_code

    assert _snapshot_tree(root) == before


def test_signal_default_board_without_database_is_404_and_not_created(signal_client):
    db_path = kb.kanban_db_path(board="default")
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    root = kb.kanban_home()
    before = _snapshot_tree(root)

    response = signal_client.get("/api/plugins/kanban/read-model/signal?board=default")
    assert response.status_code == 404, response.text
    assert _ws_close_code(signal_client, "?view=signal&board=default&since=0&token=test") == 1008

    assert not db_path.exists()
    assert _snapshot_tree(root) == before


def test_signal_stream_failure_closes_1011_and_releases_connection(signal_client, monkeypatch):
    from hermes_cli.sqlite_safe_read import has_live_connection

    module = sys.modules["hermes_dashboard_plugin_kanban_signal_test"]
    monkeypatch.setattr(module, "_EVENT_POLL_SECONDS", 0.01)
    opened = _counting_signal_opens(monkeypatch)
    real_frame = module.signal_api._signal_event_frame
    calls = []

    def failing_after_initial(conn, cursor, **kwargs):
        calls.append(cursor)
        if len(calls) > 1:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return real_frame(conn, cursor, **kwargs)

    monkeypatch.setattr(module.signal_api, "_signal_event_frame", failing_after_initial)
    path = "/api/plugins/kanban/events?view=signal&board=default&since=0&token=test"
    with pytest.raises(WebSocketDisconnect) as exc:
        with signal_client.websocket_connect(path) as websocket:
            assert websocket.receive_json()["initial"] is True
            websocket.receive_json()
    assert exc.value.code == 1011
    assert len(opened) == 1
    assert not has_live_connection(kb.kanban_db_path(board="default"))


def test_kanban_dashboard_modules_do_not_shadow_the_standard_library():
    """plugin_api imports its sibling by absolute path; a sibling named like a stdlib module
    (``signal``) would shadow it for anything importing from that directory."""
    dashboard = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard"
    stems = {path.stem for path in dashboard.glob("*.py")}
    assert stems.isdisjoint(sys.stdlib_module_names), stems & sys.stdlib_module_names
