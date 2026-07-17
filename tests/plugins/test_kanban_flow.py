"""Contract tests for the bundled Kanban Flow view and workflow archive APIs."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


PLUGIN_ROOT = (
    Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard"
)


def _load_plugin_module():
    plugin_file = PLUGIN_ROOT / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_user_kanban_flow_plugin_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    return _load_plugin_module()


@pytest.fixture
def client(tmp_path, monkeypatch, plugin_module):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.state.kanban_plugin = plugin_module
    app.include_router(plugin_module.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _task(client: TestClient, title: str, tenant: str | None = None) -> str:
    response = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": title, "tenant": tenant},
    )
    assert response.status_code == 200, response.text
    return response.json()["task"]["id"]


def _link(client: TestClient, parent: str, child: str, board: str | None = None) -> None:
    suffix = f"?board={board}" if board else ""
    response = client.post(
        f"/api/plugins/kanban/links{suffix}",
        json={"parent_id": parent, "child_id": child},
    )
    assert response.status_code == 200, response.text


def _preview(
    client: TestClient,
    seed: str,
    visible: list[str] | None = None,
    board: str | None = None,
) -> dict:
    suffix = f"?board={board}" if board else ""
    response = client.post(
        f"/api/plugins/kanban/workflow-groups/preview{suffix}",
        json={"seed_task_id": seed, "visible_task_ids": visible or [seed]},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _archive(client: TestClient, preview: dict, board: str | None = None):
    suffix = f"?board={board}" if board else ""
    return client.post(
        f"/api/plugins/kanban/workflow-groups/archive{suffix}",
        json={
            "board": preview["board"],
            "seed_task_id": preview["seed_task_id"],
            "preview_id": preview["preview_id"],
            "task_ids": preview["task_ids"],
        },
    )


def _all_tasks(client: TestClient, board: str | None = None) -> dict[str, dict]:
    suffix = "?include_archived=true"
    if board:
        suffix += f"&board={board}"
    payload = client.get(f"/api/plugins/kanban/board{suffix}").json()
    return {
        task["id"]: task
        for column in payload["columns"]
        for task in column["tasks"]
    }


def _make_running(task_id: str, *, board: str | None = None) -> int:
    conn = kb.connect(board=board)
    try:
        with kb.write_txn(conn):
            cursor = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, claim_lock, worker_pid, started_at) "
                "VALUES (?, 'default', 'running', 'test-host:claim', 424242, ?)",
                (task_id, int(time.time())),
            )
            assert cursor.lastrowid is not None
            run_id = int(cursor.lastrowid)
            conn.execute(
                "UPDATE tasks SET status = 'running', current_run_id = ?, "
                "claim_lock = 'test-host:claim', worker_pid = 424242 WHERE id = ?",
                (run_id, task_id),
            )
        return run_id
    finally:
        conn.close()


def _set_status_sql(task_id: str, status: str, *, board: str | None = None) -> None:
    conn = kb.connect(board=board)
    try:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    finally:
        conn.close()


def test_board_snapshot_includes_visible_dependency_edges(client):
    parent = _task(client, "Prepare source", tenant="alpha")
    child = _task(client, "Compile report", tenant="alpha")
    linked = client.post(
        "/api/plugins/kanban/links",
        json={"parent_id": parent, "child_id": child},
    )
    assert linked.status_code == 200, linked.text

    response = client.get("/api/plugins/kanban/board")
    assert response.status_code == 200
    assert response.json()["links"] == [
        {"parent_id": parent, "child_id": child}
    ]


def test_tenant_snapshot_does_not_leak_edges_to_hidden_tasks(client):
    visible = _task(client, "Visible task", tenant="alpha")
    hidden = _task(client, "Hidden task", tenant="beta")
    linked = client.post(
        "/api/plugins/kanban/links",
        json={"parent_id": visible, "child_id": hidden},
    )
    assert linked.status_code == 200, linked.text

    response = client.get("/api/plugins/kanban/board?tenant=alpha")
    assert response.status_code == 200
    assert response.json()["links"] == []


def test_bundle_integrates_flow_as_a_sibling_of_board_columns():
    bundle = (PLUGIN_ROOT / "dist" / "index.js").read_text(encoding="utf-8")
    styles = (PLUGIN_ROOT / "dist" / "style.css").read_text(encoding="utf-8")

    assert 'const [viewMode, setViewMode] = useState("board")' in bundle
    assert 'const [layoutPreset, setLayoutPreset] = useState("balanced-horizontal")' in bundle
    assert 'const [layoutSettingsState, setLayoutSettingsState] = useState("loading")' in bundle
    assert '`${API}/flow-settings`' in bundle
    assert 'method: "PATCH"' in bundle
    assert 'const previous = layoutPreset' not in bundle
    assert 'setLayoutPreset(previous)' not in bundle
    assert '"aria-label": "Flow layout"' in bundle
    assert 'role: "status"' in bundle
    assert '"aria-live": "polite"' in bundle
    assert '"aria-atomic": "true"' in bundle
    assert 'role: "group", "aria-label": "Graph view controls"' in bundle
    assert 'props.layoutSettingsState === "loading" || props.layoutSettingsState === "saving"' in bundle
    assert "Loading layout…" in bundle
    assert "Saving layout…" in bundle
    assert "Layout saved" in bundle
    assert "Layout setting failed" in bundle
    assert "function TaskGraphView(props)" in bundle
    assert 'buildTaskGraphLayout(props.board, matchingBoard, props.layoutPreset)' in bundle
    assert 'layout.nodes.find(function (node) { return node.id === props.focusTaskId; })' in bundle
    assert 'viewMode === "flow" ? h(TaskGraphView' in bundle
    assert "layoutSettingsState," in bundle
    assert "focusTaskId: selectedTaskId" in bundle
    assert "h(TaskDrawer" in bundle
    assert "Graph view loading" not in bundle

    kanban_page = bundle[bundle.index("  function KanbanPage()") : bundle.index("\n  // Attention strip")]
    graph_view = bundle[bundle.index("  function TaskGraphView(props)") : bundle.index("\n  // -------------------------------------------------------------------------\n  // Bulk action bar")]
    assert kanban_page.count('role: "status"') == 1
    assert kanban_page.index('role: "status"') < kanban_page.index('viewMode === "flow"')
    assert 'role: "status"' not in graph_view
    assert 'const isEmpty = layout.nodes.length === 0' in graph_view
    assert 'if (layout.nodes.length === 0)' not in graph_view
    assert graph_view.index('return h("section"') < graph_view.index('isEmpty ? h("div"')
    assert graph_view.index('isEmpty ? h("div"') < graph_view.index('role: "group", "aria-label": "Graph view controls"')

    assert ".hermes-kanban-graph-layout-control select" in styles
    assert ".hermes-kanban-graph-layout-control select:hover:not(:disabled)" in styles
    assert ".hermes-kanban-graph-layout-control select:focus-visible" in styles
    assert ".hermes-kanban-graph-layout-control select:disabled" in styles
    assert ".hermes-kanban-graph-layout-status" in styles
    assert "min-height: 44px" in styles
    assert "height: min(calc(70vh - 48px), 712px)" in styles
    assert "transition: all" not in styles
    assert """  .hermes-kanban-graph-controls {
    position: relative;
    right: auto;
    bottom: auto;
    left: auto;
    width: 100%;
    flex-wrap: wrap;""" in styles
    assert "function WorkflowArchiveDialog(props)" in bundle
    assert '"Archive workflow"' in bundle
    assert '"Restore workflow"' in bundle
    assert '`${API}/workflow-groups/preview`' in bundle
    assert '`${API}/workflow-groups/archive`' in bundle
    assert "island.isUnlinked ? null" in bundle
    assert 'role: "dialog"' in bundle or 'h("dialog"' in bundle
    assert '"aria-label": "Workflow archive activity"' in bundle
    assert ".hermes-kanban-workflow-dialog" in styles
    assert ".hermes-kanban-graph-island.is-archived" in styles
    assert "background: var(--color-primary);" in styles
    assert "color: var(--color-primary-foreground);" in styles


def test_mobile_layout_selector_keeps_full_balanced_horizontal_label():
    bundle = (PLUGIN_ROOT / "dist" / "index.js").read_text(encoding="utf-8")
    styles = (PLUGIN_ROOT / "dist" / "style.css").read_text(encoding="utf-8")
    mobile_styles = styles[
        styles.index("@media (max-width: 760px)") :
        styles.index("@media (prefers-reduced-motion: reduce)")
    ]

    assert '"Balanced horizontal"' in bundle
    assert """  .hermes-kanban-graph-layout-control select {
    min-height: 44px;
    max-width: 11.5rem;
  }""" in mobile_styles
    assert "max-width: 8.5rem" not in mobile_styles


def test_manifest_describes_the_integrated_flow_view():
    manifest = (PLUGIN_ROOT / "manifest.json").read_text(encoding="utf-8")

    assert '"version": "1.1.0"' in manifest
    assert "dependency flow" in manifest.lower()
    assert "archive and restore" in manifest.lower()


def test_flow_settings_default_and_board_scoped_persistence(client):
    default = client.get("/api/plugins/kanban/flow-settings")
    assert default.status_code == 200
    assert default.json() == {"layout_preset": "balanced-horizontal"}

    updated = client.patch(
        "/api/plugins/kanban/flow-settings",
        json={"layout_preset": "balanced-vertical"},
    )
    assert updated.status_code == 200
    assert updated.json() == {"layout_preset": "balanced-vertical"}

    reloaded = client.get("/api/plugins/kanban/flow-settings")
    assert reloaded.json() == {"layout_preset": "balanced-vertical"}


def test_flow_settings_reject_unknown_preset(client):
    response = client.patch(
        "/api/plugins/kanban/flow-settings",
        json={"layout_preset": "manual"},
    )
    assert response.status_code == 422


def test_workflow_preview_uses_canonical_component_not_visible_subset(client):
    a = _task(client, "Research A", tenant="alpha")
    b = _task(client, "Research B", tenant="beta")
    c = _task(client, "Synthesis", tenant="alpha")
    _link(client, a, c)
    _link(client, b, c)

    data = _preview(client, a, [a, c])
    assert set(data["task_ids"]) == {a, b, c}
    assert data["hidden_count"] == 1
    assert data["board"] == "default"
    assert data["counts"] == {
        "total": 3,
        "active": 3,
        "running": 0,
        "review": 0,
        "done": 0,
        "archived": 0,
    }
    assert len(data["preview_id"]) == 64
    assert data["expires_at"] > int(time.time())


def test_workflow_preview_rejects_unlinked_seed(client):
    task_id = _task(client, "Standalone")
    response = client.post(
        "/api/plugins/kanban/workflow-groups/preview",
        json={"seed_task_id": task_id, "visible_task_ids": [task_id]},
    )
    assert response.status_code == 409
    assert "not a linked workflow" in response.json()["detail"]


def test_workflow_archive_rejects_cross_board_token_and_incomplete_scope(client):
    a = _task(client, "A")
    b = _task(client, "B")
    _link(client, a, b)
    preview = _preview(client, a, [a, b])
    created = client.post("/api/plugins/kanban/boards", json={"slug": "other"})
    assert created.status_code == 200, created.text

    cross_board = _archive(client, preview, board="other")
    assert cross_board.status_code == 409
    assert "board" in cross_board.json()["detail"].lower()

    incomplete = client.post(
        "/api/plugins/kanban/workflow-groups/archive",
        json={
            "board": preview["board"],
            "seed_task_id": a,
            "preview_id": preview["preview_id"],
            "task_ids": [a],
        },
    )
    assert incomplete.status_code == 409
    assert "scope" in incomplete.json()["detail"].lower()
    assert all(task["status"] != "archived" for task in _all_tasks(client).values())


def test_workflow_archive_rejects_stale_and_expired_preview(client):
    a = _task(client, "A")
    b = _task(client, "B")
    _link(client, a, b)
    stale = _preview(client, a, [a, b])
    _set_status_sql(b, "done")
    response = _archive(client, stale)
    assert response.status_code == 409
    assert "refresh" in response.json()["detail"].lower()

    fresh = _preview(client, a, [a, b])
    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE workflow_archive_previews SET expires_at = ? WHERE preview_id = ?",
                (int(time.time()) - 1, fresh["preview_id"]),
            )
    finally:
        conn.close()
    expired = _archive(client, fresh)
    assert expired.status_code == 409
    assert "expired" in expired.json()["detail"].lower()


def test_workflow_archive_aborts_before_status_changes_when_run_termination_fails(
    client, plugin_module, monkeypatch
):
    a = _task(client, "A")
    b = _task(client, "B")
    _link(client, a, b)
    run_id = _make_running(a)
    preview = _preview(client, a, [a, b])
    monkeypatch.setattr(
        plugin_module,
        "_terminate_active_run",
        lambda row: {
            "ok": False,
            "run_id": run_id,
            "task_id": row["task_id"],
            "detail": "worker survived termination timeout",
        },
    )

    response = _archive(client, preview)
    assert response.status_code == 409
    assert str(run_id) in response.json()["detail"]
    tasks = _all_tasks(client)
    assert tasks[a]["status"] == "running"
    assert tasks[b]["status"] != "archived"


def test_workflow_archive_rolls_back_metadata_and_statuses_on_sql_failure(client):
    a = _task(client, "A")
    b = _task(client, "B")
    _link(client, a, b)
    preview = _preview(client, a, [a, b])
    conn = kb.connect()
    try:
        conn.execute(
            f"CREATE TRIGGER fail_archive_{b.replace('-', '_')} BEFORE UPDATE OF status ON tasks "
            f"WHEN NEW.id = '{b}' AND NEW.status = 'archived' "
            "BEGIN SELECT RAISE(ABORT, 'forced archive failure'); END"
        )
    finally:
        conn.close()

    response = _archive(client, preview)
    assert response.status_code == 500
    assert "rolled back" in response.json()["detail"].lower()
    tasks = _all_tasks(client)
    assert tasks[a]["status"] != "archived"
    assert tasks[b]["status"] != "archived"
    conn = kb.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM workflow_archives").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM workflow_archive_tasks").fetchone()[0] == 0
    finally:
        conn.close()


def test_workflow_archive_success_preserves_links_and_emits_refresh_events(client):
    a = _task(client, "A")
    b = _task(client, "B")
    _link(client, a, b)
    preview = _preview(client, a, [a, b])
    response = _archive(client, preview)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["archived_count"] == 2
    assert result["refresh_required"] is True
    board = client.get("/api/plugins/kanban/board?include_archived=true").json()
    assert board["links"] == [{"parent_id": a, "child_id": b}]
    tasks = _all_tasks(client)
    assert tasks[a]["status"] == tasks[b]["status"] == "archived"
    assert tasks[a]["workflow_archive_id"] == result["archive_id"]
    assert tasks[b]["workflow_archive_id"] == result["archive_id"]
    conn = kb.connect()
    try:
        kinds = [
            row[0]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id IN (?, ?) ORDER BY id", (a, b)
            )
        ]
        assert kinds.count("workflow_archived") == 1
        assert kinds.count("status") >= 2
    finally:
        conn.close()


def test_workflow_restore_uses_provenance_maps_running_and_is_all_or_nothing(
    client, plugin_module, monkeypatch
):
    a = _task(client, "A")
    b = _task(client, "B")
    c = _task(client, "C")
    _link(client, a, b)
    _link(client, b, c)
    _set_status_sql(c, "done")
    run_id = _make_running(a)
    monkeypatch.setattr(
        plugin_module,
        "_terminate_active_run",
        lambda row: {"ok": True, "run_id": run_id, "task_id": row["task_id"], "detail": "terminated"},
    )
    archived = _archive(client, _preview(client, a, [a, b, c]))
    assert archived.status_code == 200, archived.text
    archive_id = archived.json()["archive_id"]

    restored = client.post(f"/api/plugins/kanban/workflow-groups/{archive_id}/restore")
    assert restored.status_code == 200, restored.text
    tasks = _all_tasks(client)
    assert tasks[a]["status"] in {"todo", "ready"}
    assert tasks[a]["status"] != "running"
    assert tasks[c]["status"] == "done"
    conn = kb.connect()
    try:
        run = conn.execute("SELECT status, ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert run["status"] == "reclaimed"
        assert run["ended_at"] is not None
    finally:
        conn.close()

    second = client.post(f"/api/plugins/kanban/workflow-groups/{archive_id}/restore")
    assert second.status_code == 409


def test_workflow_restore_rejects_changed_scope_without_partial_restore(client):
    a = _task(client, "A")
    b = _task(client, "B")
    _link(client, a, b)
    archived = _archive(client, _preview(client, a, [a, b])).json()
    c = _task(client, "C")
    _link(client, b, c)

    response = client.post(
        f"/api/plugins/kanban/workflow-groups/{archived['archive_id']}/restore"
    )
    assert response.status_code == 409
    assert "scope" in response.json()["detail"].lower()
    tasks = _all_tasks(client)
    assert tasks[a]["status"] == "archived"
    assert tasks[b]["status"] == "archived"


def test_restore_keeps_members_that_were_already_archived_archived(client):
    a = _task(client, "A")
    b = _task(client, "B")
    _link(client, a, b)
    archived_b = client.patch(f"/api/plugins/kanban/tasks/{b}", json={"status": "archived"})
    assert archived_b.status_code == 200, archived_b.text
    result = _archive(client, _preview(client, a, [a, b])).json()
    restored = client.post(
        f"/api/plugins/kanban/workflow-groups/{result['archive_id']}/restore"
    )
    assert restored.status_code == 200, restored.text
    tasks = _all_tasks(client)
    assert tasks[b]["status"] == "archived"
