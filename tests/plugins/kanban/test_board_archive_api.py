"""Tests for the Kanban board archive / restore / hard-delete API.

Covers the 7 new endpoints added in feat/kanban-board-archive-ui:

  * POST   /boards/{slug}/archive            — archive (cascade optional)
  * POST   /boards/{slug}/restore            — un-archive
  * DELETE /boards/{slug}                    — hard delete (zero-task rule)
  * GET    /boards/archived                  — archive viewer (boards)
  * GET    /tasks/archived                   — archive viewer (tasks)
  * POST   /tasks/{task_id}/restore          — un-archive a task
  * DELETE /tasks/{task_id}                  — hard delete (terminal-only rule)

Plus the integrity rules:

  * Default board protected (no archive, no hard delete).
  * Currently-active board can't be archived.
  * Archive with non-archived tasks requires cascade.
  * Hard-delete needs zero tasks total.
  * Hard-delete of a task needs status in {archived, done}.

The fixtures mock the filesystem via ``tmp_path`` + ``HERMES_HOME`` so the
real ``/opt/data/kanban/...`` tree is never touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures — mirror tests/plugins/test_kanban_dashboard_plugin.py so the
# new tests live in the same isolated tmp_path / HERMES_HOME environment.
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[3]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_archive_test",
        plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _create_board(client, slug, switch=False):
    r = client.post(
        "/api/plugins/kanban/boards",
        json={"slug": slug, "switch": switch},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _create_task(client, title, board=None, status=None):
    """Create a task on ``board`` (default if None); optionally re-status it."""
    url = "/api/plugins/kanban/tasks"
    if board:
        url = f"{url}?board={board}"
    r = client.post(url, json={"title": title})
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    if status and status != task["status"]:
        patch_url = f"/api/plugins/kanban/tasks/{task['id']}"
        if board:
            patch_url = f"{patch_url}?board={board}"
        rp = client.patch(patch_url, json={"status": status})
        assert rp.status_code == 200, rp.text
        task = rp.json()["task"]
    return task


# ---------------------------------------------------------------------------
# POST /boards/{slug}/archive  — happy paths + integrity rules
# ---------------------------------------------------------------------------


def test_archive_default_board_refused(client):
    """Default board cannot be archived."""
    r = client.post("/api/plugins/kanban/boards/default/archive")
    assert r.status_code == 409
    body = r.json()
    assert "default" in str(body).lower()


def test_archive_active_board_refused(client):
    """Currently-active board cannot be archived — user must switch first."""
    _create_board(client, "proj-a", switch=True)
    r = client.post("/api/plugins/kanban/boards/proj-a/archive")
    assert r.status_code == 409
    assert "active" in r.json()["detail"]["error"].lower()


def test_archive_empty_board(client):
    """Empty board archives cleanly — moves to boards/_archived/."""
    _create_board(client, "proj-a", switch=False)
    r = client.post("/api/plugins/kanban/boards/proj-a/archive")
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["action"] == "archived"
    assert Path(res["new_path"]).exists()

    # GET /boards/archived now shows it
    r = client.get("/api/plugins/kanban/boards/archived")
    assert r.status_code == 200
    rows = r.json()["archived_boards"]
    slugs = [row["slug"] for row in rows]
    assert "proj-a" in slugs


def test_archive_nonempty_without_cascade_refused(client):
    """Board with live tasks needs cascade=true to archive."""
    _create_board(client, "proj-a", switch=False)
    _create_task(client, "live task", board="proj-a")
    r = client.post("/api/plugins/kanban/boards/proj-a/archive")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["live_task_count"] == 1
    assert "cascade" in detail["error"].lower()


def test_archive_with_cascade_archives_tasks(client):
    """cascade=true archives every non-archived task, then the board."""
    _create_board(client, "proj-a", switch=False)
    t1 = _create_task(client, "live task", board="proj-a")
    _create_task(client, "another live one", board="proj-a")
    r = client.post(
        "/api/plugins/kanban/boards/proj-a/archive?cascade=true",
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["cascade_archived_tasks"]) == 2
    assert t1["id"] in body["cascade_archived_tasks"]
    # Board directory should have moved under _archived/.
    assert Path(body["result"]["new_path"]).exists()


# ---------------------------------------------------------------------------
# POST /boards/{slug}/restore
# ---------------------------------------------------------------------------


def test_restore_archived_board(client):
    """Archived board can be restored back to the active list."""
    _create_board(client, "proj-a", switch=False)
    archive = client.post("/api/plugins/kanban/boards/proj-a/archive").json()
    assert archive["result"]["action"] == "archived"

    r = client.post("/api/plugins/kanban/boards/proj-a/restore")
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["action"] == "restored"
    assert Path(res["new_path"]).exists()

    # Board is back in the active list.
    r = client.get("/api/plugins/kanban/boards")
    slugs = [b["slug"] for b in r.json()["boards"]]
    assert "proj-a" in slugs


def test_restore_when_active_collides_refused(client):
    """Refuse restore if an active board with the same slug exists."""
    _create_board(client, "proj-a", switch=False)
    client.post("/api/plugins/kanban/boards/proj-a/archive")
    # Re-create an active board with the same slug.
    _create_board(client, "proj-a", switch=False)
    r = client.post("/api/plugins/kanban/boards/proj-a/restore")
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]["error"].lower()


def test_restore_unknown_archive_404(client):
    """No archived board with this slug → 404."""
    r = client.post("/api/plugins/kanban/boards/never-existed/restore")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /boards/{slug}  — hard delete
# ---------------------------------------------------------------------------


def test_hard_delete_default_refused(client):
    """Default board cannot be hard-deleted."""
    r = client.delete("/api/plugins/kanban/boards/default?delete=true")
    assert r.status_code == 409


def test_hard_delete_active_board_refused(client):
    """Currently-active board cannot be hard-deleted."""
    _create_board(client, "proj-a", switch=True)
    r = client.delete("/api/plugins/kanban/boards/proj-a?delete=true")
    assert r.status_code == 409
    assert "active" in r.json()["detail"]["error"].lower()


def test_hard_delete_with_archived_tasks_refused(client):
    """Hard delete must refuse (409) when ANY task exists and cascade is absent.

    Why: Default-safety — a single misclick on Delete must not silently
    wipe an entire board (active + archived tasks). Cascade=true is the
    explicit opt-in (see test_hard_delete_with_cascade_destroys_everything).
    """
    _create_board(client, "proj-a", switch=False)
    t = _create_task(client, "to archive", board="proj-a")
    # Archive the task — leaves a row with status='archived'.
    rp = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}?board=proj-a",
        json={"status": "archived"},
    )
    assert rp.status_code == 200, rp.text

    r = client.delete("/api/plugins/kanban/boards/proj-a?delete=true")
    assert r.status_code == 409
    body = r.json()["detail"]
    assert body["task_count"] == 1
    # New wording points the user at the cascade opt-in.
    assert "cascade=true" in body["error"].lower()


def test_hard_delete_with_cascade_destroys_everything(client):
    """Hard delete with cascade=true purges a non-empty board outright.

    Why: User explicitly confirmed (via the dashboard's two-step warning)
    that they want the board AND every task on it gone forever.
    What: ``DELETE /boards/X?delete=true&cascade=true`` succeeds even
    when the board has active + archived tasks. ``hard_delete_board``
    removes the on-disk directory (and any ``_archived/<slug>-*/``),
    so all task rows go with it.
    Test: Create board with 1 active + 1 archived task; cascade-delete;
    assert board dir + every archive dir for the slug are gone.
    """
    _create_board(client, "proj-a", switch=False)
    _create_task(client, "active one", board="proj-a")
    t2 = _create_task(client, "archived one", board="proj-a")
    rp = client.patch(
        f"/api/plugins/kanban/tasks/{t2['id']}?board=proj-a",
        json={"status": "archived"},
    )
    assert rp.status_code == 200, rp.text

    r = client.delete(
        "/api/plugins/kanban/boards/proj-a?delete=true&cascade=true",
    )
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["action"] == "hard-deleted"
    for p in res["removed_paths"]:
        assert not Path(p).exists(), f"path should be gone: {p}"

    # GET /boards must NOT list 'proj-a' any more.
    rlist = client.get("/api/plugins/kanban/boards")
    assert rlist.status_code == 200
    slugs = [b["slug"] for b in rlist.json()["boards"]]
    assert "proj-a" not in slugs


def test_hard_delete_empty_board(client):
    """Empty board hard-deletes cleanly — dir removed, no archive entry."""
    _create_board(client, "proj-a", switch=False)
    r = client.delete("/api/plugins/kanban/boards/proj-a?delete=true")
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["action"] == "hard-deleted"
    assert len(res["removed_paths"]) >= 1
    for p in res["removed_paths"]:
        assert not Path(p).exists(), f"path should be gone: {p}"


def test_list_boards_excludes_archived_after_archive(client):
    """After archive, GET /boards (default include_archived=False) must
    NOT return the archived slug.

    Why: Regression test for the selector-shows-archived UX bug. Drives
    the backend's filter contract that the dashboard switcher relies on.
    What: Archive a board, GET /boards, assert the slug is not in the
    list AND show that include_archived=true still surfaces it via
    /boards/archived (the dedicated archive viewer endpoint).
    Test: Create 'квантс', archive while non-current, list boards →
    no 'квантс'. Then GET /boards/archived → 'квантс' present.
    """
    _create_board(client, "kvants", switch=False)
    r = client.post("/api/plugins/kanban/boards/kvants/archive")
    assert r.status_code == 200, r.text

    rlist = client.get("/api/plugins/kanban/boards")
    assert rlist.status_code == 200
    body = rlist.json()
    slugs = [b["slug"] for b in body["boards"]]
    assert "kvants" not in slugs, (
        f"archived 'kvants' must not appear in /boards (got {slugs})"
    )
    # Sanity: it's still reachable via the archive viewer.
    rarch = client.get("/api/plugins/kanban/boards/archived")
    arch_slugs = [b["slug"] for b in rarch.json()["archived_boards"]]
    assert "kvants" in arch_slugs


def test_archive_active_board_via_switch_then_archive(client):
    """End-to-end: switch off the to-be-archived board, then archive it,
    and verify the selector list reflects the change.

    Why: Real user flow from the field — they archived the currently
    active board (failed with 409), so the UI switches to default
    automatically and re-issues archive.
    Test: 'квантс' is current → archive refused (409). Switch to
    default → archive succeeds. /boards no longer lists it; current
    is 'default'.
    """
    _create_board(client, "kvants", switch=True)

    # Step 1: cannot archive while active.
    r = client.post("/api/plugins/kanban/boards/kvants/archive")
    assert r.status_code == 409

    # Step 2: switch to default, then archive cleanly.
    rs = client.post("/api/plugins/kanban/boards/default/switch")
    assert rs.status_code == 200, rs.text
    r = client.post("/api/plugins/kanban/boards/kvants/archive")
    assert r.status_code == 200, r.text

    # /boards no longer surfaces 'kvants'; current is 'default'.
    rlist = client.get("/api/plugins/kanban/boards").json()
    assert rlist["current"] == "default"
    assert "kvants" not in {b["slug"] for b in rlist["boards"]}


def test_delete_default_archive_path_uses_archive(client):
    """``DELETE /boards/{slug}`` (no ``delete=true``) is the archive path."""
    _create_board(client, "proj-a", switch=False)
    r = client.delete("/api/plugins/kanban/boards/proj-a")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["action"] == "archived"


# ---------------------------------------------------------------------------
# GET /boards/archived  and  GET /tasks/archived
# ---------------------------------------------------------------------------


def test_list_archived_boards_empty(client):
    """No archived boards → empty list, count=0."""
    r = client.get("/api/plugins/kanban/boards/archived")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["archived_boards"] == []


def test_list_archived_boards_after_archive(client):
    """Archive 2 boards, both appear with can_restore=true."""
    _create_board(client, "proj-a", switch=False)
    _create_board(client, "proj-b", switch=False)
    client.post("/api/plugins/kanban/boards/proj-a/archive")
    client.post("/api/plugins/kanban/boards/proj-b/archive")
    r = client.get("/api/plugins/kanban/boards/archived")
    assert r.status_code == 200
    rows = r.json()["archived_boards"]
    slugs = {row["slug"] for row in rows}
    assert {"proj-a", "proj-b"} <= slugs
    for row in rows:
        assert row["can_restore"] is True


def test_list_archived_tasks_current_board(client):
    """``/tasks/archived`` returns archived tasks on the active board."""
    t = _create_task(client, "t-on-default")
    rp = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "archived"},
    )
    assert rp.status_code == 200, rp.text

    r = client.get("/api/plugins/kanban/tasks/archived")
    assert r.status_code == 200
    rows = r.json()["archived_tasks"]
    assert len(rows) == 1
    assert rows[0]["id"] == t["id"]
    assert rows[0]["board_slug"] == "default"


def test_list_archived_tasks_all_boards(client):
    """``all_boards=true`` aggregates across every active board."""
    _create_board(client, "proj-a", switch=False)
    t1 = _create_task(client, "task on a", board="proj-a")
    t2 = _create_task(client, "task on default")
    # Archive both.
    client.patch(
        f"/api/plugins/kanban/tasks/{t1['id']}?board=proj-a",
        json={"status": "archived"},
    )
    client.patch(
        f"/api/plugins/kanban/tasks/{t2['id']}",
        json={"status": "archived"},
    )
    r = client.get("/api/plugins/kanban/tasks/archived?all_boards=true")
    assert r.status_code == 200
    rows = r.json()["archived_tasks"]
    by_board = {row["board_slug"] for row in rows}
    assert by_board == {"default", "proj-a"}


# ---------------------------------------------------------------------------
# POST /tasks/{task_id}/restore
# ---------------------------------------------------------------------------


def test_restore_archived_task(client):
    """Archived task flips back to 'todo'."""
    t = _create_task(client, "to restore")
    client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "archived"},
    )
    r = client.post(f"/api/plugins/kanban/tasks/{t['id']}/restore")
    assert r.status_code == 200, r.text
    restored = r.json()["task"]
    # 'todo' or 'ready' both acceptable — recompute_ready may promote.
    assert restored["status"] in {"todo", "ready"}


def test_restore_non_archived_task_refused(client):
    """Restoring a task that isn't archived → 409."""
    t = _create_task(client, "live task")
    r = client.post(f"/api/plugins/kanban/tasks/{t['id']}/restore")
    assert r.status_code == 409
    assert "not archived" in r.json()["detail"]["error"].lower()


def test_restore_unknown_task_404(client):
    r = client.post("/api/plugins/kanban/tasks/t_doesnt_exist/restore")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /tasks/{task_id}  — terminal-status integrity rule
# ---------------------------------------------------------------------------


def test_hard_delete_task_refused_unless_terminal(client):
    """Hard delete refuses when status not in {archived, done}."""
    t = _create_task(client, "live task")
    # Status is 'ready' (no parents → auto-promoted on create).
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["current_status"] in {"ready", "todo"}


def test_hard_delete_archived_task(client):
    """Hard delete of an archived task succeeds (204)."""
    t = _create_task(client, "to archive then delete")
    client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "archived"},
    )
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 204


def test_hard_delete_done_task(client):
    """Hard delete of a done task succeeds."""
    t = _create_task(client, "complete me")
    client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "done"},
    )
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 204


def test_hard_delete_force_overrides(client):
    """force=true allows hard delete of a live task (escape hatch)."""
    t = _create_task(client, "force-deleted")
    r = client.delete(f"/api/plugins/kanban/tasks/{t['id']}?force=true")
    assert r.status_code == 204


# ---------------------------------------------------------------------------
# Round-trip: archive board → restore → tasks still accessible
# ---------------------------------------------------------------------------


def test_round_trip_archive_then_restore_preserves_tasks(client):
    """Archive a board with cascade, restore, check tasks come back as archived."""
    _create_board(client, "proj-a", switch=False)
    t = _create_task(client, "round-trip", board="proj-a")
    arch = client.post(
        "/api/plugins/kanban/boards/proj-a/archive?cascade=true",
    )
    assert arch.status_code == 200
    assert t["id"] in arch.json()["cascade_archived_tasks"]

    rest = client.post("/api/plugins/kanban/boards/proj-a/restore")
    assert rest.status_code == 200

    # Board is back; the task is still archived because cascade archived
    # it as part of the wrap-up. Restoring the board doesn't auto-restore
    # tasks (deliberate — keeps the inverse asymmetric so users notice).
    r = client.get("/api/plugins/kanban/tasks/archived?board=proj-a")
    assert r.status_code == 200
    rows = r.json()["archived_tasks"]
    assert any(row["id"] == t["id"] for row in rows)
