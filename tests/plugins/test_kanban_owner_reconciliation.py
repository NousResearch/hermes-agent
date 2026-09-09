"""Owner contract regression gates on the real, split dashboard/auth stack.

Only disposable HOME/boards and synthetic credentials; no running service needed.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import token_auth
from hermes_cli.dashboard_auth.base import DashboardAuthProvider, TokenPrincipal
from plugins.kanban.dashboard import plugin_api as api


class FixtureProvider(DashboardAuthProvider):
    name = "owner-fixture"
    display_name = "Isolated owner test"
    supports_token = True
    supports_session = False

    def start_login(self, **kwargs):
        raise NotImplementedError

    def complete_login(self, **kwargs):
        raise NotImplementedError

    def verify_session(self, **kwargs):
        return None

    def refresh_session(self, **kwargs):
        raise NotImplementedError

    def revoke_session(self, **kwargs):
        return None

    def verify_token(self, *, token):
        if token == "scoped-fixture":
            return TokenPrincipal("reader", self.name, ("kanban:owner:read",))
        if token == "wrong-scope-fixture":
            return TokenPrincipal("reader", self.name, ("unrelated:read",))
        if token == "forged-provider-fixture":
            return TokenPrincipal("reader", "dashboard-internal", ("kanban:owner:read",))
        return None


@pytest.fixture
def host(tmp_path, monkeypatch):
    from hermes_cli import web_server as ws

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ws, "_SESSION_TOKEN", "this-host-fixture")
    monkeypatch.setattr(ws.app.state, "internal_api_token", "this-host-fixture")
    clear_providers()
    token_auth.clear_token_routes()
    api.register_owner_token_routes()
    register_provider(FixtureProvider())
    original_routes = list(ws.app.router.routes)
    ws.app.include_router(api.router, prefix="/api/plugins/kanban")
    kb.init_db()
    try:
        # No lifespan: do not start gateway/background services for an HTTP gate.
        yield TestClient(ws.app), home
    finally:
        ws.app.router.routes[:] = original_routes
        clear_providers()
        token_auth.clear_token_routes()


@pytest.mark.parametrize("route", ["owner-snapshot", "owner-events"])
@pytest.mark.parametrize("token,expected", [
    (None, 401),
    ("foreign-host-fixture", 401),
    ("wrong-scope-fixture", 403),
    ("forged-provider-fixture", 401),
    ("this-host-fixture", 200),
    ("scoped-fixture", 200),
])
def test_real_host_authorizes_before_board_resolution(host, route, token, expected):
    client, home = host
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = client.get(f"/api/plugins/kanban/{route}?board=default", headers=headers)
    assert response.status_code == expected, response.text
    if expected != 200:
        # No board existence oracle or initialization before auth/scope checking.
        unknown = client.get(f"/api/plugins/kanban/{route}?board=unknown-owner", headers=headers)
        assert unknown.status_code == expected
        assert not (home / "kanban" / "boards" / "unknown-owner").exists()
    else:
        assert response.json()["profile_id"] == "default"
        assert response.json()["board_id"] == "default"


def test_cookie_or_session_header_cannot_bypass_owner_bearer(host):
    client, _ = host
    response = client.get(
        "/api/plugins/kanban/owner-snapshot",
        headers={"X-Hermes-Session-Token": "this-host-fixture"},
        cookies={"hermes_session": "this-host-fixture"},
    )
    assert response.status_code == 401


def test_owner_identity_is_server_derived_and_board_is_exact(host):
    client, _ = host
    headers = {"Authorization": "Bearer this-host-fixture"}
    kb.create_board("other")
    with kbc.connect(board="default") as conn:
        first = kb.create_task(conn, title="default only", created_by="owner-a")
    with kbc.connect(board="other") as conn:
        second = kb.create_task(conn, title="other only", created_by="owner-b")
    for board, expected in [("default", first), ("other", second)]:
        response = client.get(
            f"/api/plugins/kanban/owner-snapshot?board={board}&profile_id=forged&board_id=forged",
            headers=headers,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert (data["profile_id"], data["board_id"]) == ("default", board)
        assert [r["task"]["id"] for r in data["receipts"]] == [expected]


def test_snapshot_cursor_and_receipts_share_read_transaction(host, monkeypatch):
    client, _ = host
    headers = {"Authorization": "Bearer this-host-fixture"}
    with kbc.connect() as conn:
        first = kb.create_task(conn, title="before snapshot", created_by="fixture")
    original_cursor = api._owner_event_cursor
    inserted = []

    def concurrent_writer(reader):
        assert reader.in_transaction
        with kbc.connect() as writer:
            inserted.append(kb.create_task(writer, title="after snapshot read", created_by="fixture"))
        return original_cursor(reader)

    with monkeypatch.context() as scoped:
        scoped.setattr(api, "_owner_event_cursor", concurrent_writer)
        response = client.get("/api/plugins/kanban/owner-snapshot", headers=headers)
    assert response.status_code == 200, response.text
    snapshot = response.json()
    assert [r["task"]["id"] for r in snapshot["receipts"]] == [first]
    drained = client.get(
        f"/api/plugins/kanban/owner-events?after={snapshot['event_cursor']}", headers=headers,
    ).json()
    assert [e["task"]["id"] for e in drained["events"]] == inserted


def test_malformed_creation_identity_denies_complete_snapshot(host):
    client, _ = host
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="valid", created_by="fixture")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET title = '' WHERE id = ?", (task,))
    response = client.get(
        "/api/plugins/kanban/owner-snapshot", headers={"Authorization": "Bearer this-host-fixture"},
    )
    assert response.status_code == 409


def test_owner_events_pagination_gaps_and_rollback(host):
    client, _ = host
    headers = {"Authorization": "Bearer this-host-fixture"}
    with kbc.connect() as conn:
        first = kb.create_task(conn, title="first", created_by="fixture")
        second = kb.create_task(conn, title="second", created_by="fixture")
        with pytest.raises(RuntimeError, match="rollback"):
            with kb.write_txn(conn):
                kb._append_owner_event(conn, first, "updated")
                raise RuntimeError("rollback")
        assert kb.delete_task(conn, second)
    cursor, kinds = 0, []
    while True:
        response = client.get(f"/api/plugins/kanban/owner-events?after={cursor}&limit=1", headers=headers)
        assert response.status_code == 200, response.text
        data = response.json()
        assert len(data["events"]) <= 1
        assert data["next_cursor"] > cursor
        kinds.extend(e["kind"] for e in data["events"])
        cursor = data["next_cursor"]
        if not data["has_more"]:
            break
    assert kinds == ["created", "created", "deleted"]
    with kbc.connect() as conn:
        with kb.write_txn(conn):
            conn.execute("DELETE FROM owner_events WHERE id = ?", (cursor,))
    data = client.get(f"/api/plugins/kanban/owner-events?after={cursor}", headers=headers).json()
    assert data["events"] == []
    assert data["next_cursor"] == cursor
    assert data["has_more"] is False


def test_graph_lineage_and_creator_origin_remain_distinct_after_gc(host):
    from hermes_cli.kanban_db_graph import decompose_triage_task

    client, _ = host
    headers = {"Authorization": "Bearer this-host-fixture"}
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="root", triage=True, created_by="fixture")
        peer = kb.create_task(conn, title="peer", creator_task_id=root, created_by="fixture")
        specs = [{"title": "child", "assignee": "worker"}]
        children = decompose_triage_task(
            conn, root, root_assignee="worker", children=specs,
            author="auto-decomposer", auto_promote=False,
        )
        assert children and len(children) == 1
        child = children[0]
        before = client.get("/api/plugins/kanban/owner-snapshot", headers=headers)
        assert before.status_code == 200, before.text
        receipts = {r["task"]["id"]: r for r in before.json()["receipts"]}
        assert receipts[peer]["task"]["created_by"] == "fixture"
        # Native creation records creator_task_id, not a decomposition or by field.
        assert receipts[peer]["created_event"]["payload"] == {}
        assert receipts[child]["created_event"]["payload"] == {
            "by": "auto-decomposer", "from_decompose_of": root,
        }
        # Age history without waiting; exercise the real GC and replay guard.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done'")
            conn.execute("UPDATE task_events SET created_at=0")
        assert kb.gc_events(conn, older_than_seconds=1) > 0
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (root,))
        assert decompose_triage_task(
            conn, root, root_assignee="worker", children=specs,
            author="auto-decomposer", auto_promote=False,
        ) is None
    after = client.get("/api/plugins/kanban/owner-snapshot", headers=headers)
    assert after.status_code == 200, after.text
    assert after.json() == before.json()
    drained = client.get("/api/plugins/kanban/owner-events", headers=headers)
    assert drained.status_code == 200, drained.text
    events = drained.json()["events"]
    assert [e["task"]["id"] for e in events] == [root, peer, child]
    assert [e["payload"] for e in events] == [
        receipts[tid]["created_event"]["payload"] for tid in (root, peer, child)
    ]
