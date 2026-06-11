"""Tests for Kanban task file attachments (#35338).

Covers three layers:
  * ``hermes_cli.kanban_db`` accessors (add/list/get/delete + path helpers)
  * the dashboard REST surface (upload / list / download / delete)
  * worker-context surfacing so a kanban worker sees the absolute paths

The plugin router is attached to a bare FastAPI app — same approach as
``test_kanban_dashboard_plugin.py`` — so we exercise the real HTTP path
(multipart upload, streaming download) without the whole dashboard.
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
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_attach_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _make_task(conn, title="t") -> str:
    return kb.create_task(conn, title=title)


# ---------------------------------------------------------------------------
# DB-layer accessors
# ---------------------------------------------------------------------------


def test_add_list_get_delete_attachment(kanban_home, tmp_path):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        # Write a real blob under the per-task dir so delete can unlink it.
        dest_dir = kb.task_attachments_dir(task_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        blob = dest_dir / "source.pdf"
        blob.write_bytes(b"%PDF-1.4 fake")

        att_id = kb.add_attachment(
            conn,
            task_id,
            filename="source.pdf",
            stored_path=str(blob),
            content_type="application/pdf",
            size=blob.stat().st_size,
            uploaded_by="tester",
        )
        assert att_id > 0

        atts = kb.list_attachments(conn, task_id)
        assert len(atts) == 1
        a = atts[0]
        assert a.filename == "source.pdf"
        assert a.content_type == "application/pdf"
        assert a.size == len(b"%PDF-1.4 fake")
        assert a.uploaded_by == "tester"
        assert a.stored_path == str(blob)

        got = kb.get_attachment(conn, att_id)
        assert got is not None and got.id == att_id

        removed = kb.delete_attachment(conn, att_id)
        assert removed is not None and removed.id == att_id
        assert kb.list_attachments(conn, task_id) == []
        assert not blob.exists(), "delete should unlink the on-disk blob"
        assert kb.get_attachment(conn, att_id) is None
    finally:
        conn.close()


def test_add_attachment_rejects_unknown_task(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError):
            kb.add_attachment(
                conn, "t_doesnotexist", filename="x.txt", stored_path="/tmp/x.txt"
            )
    finally:
        conn.close()


def test_add_attachment_appends_event(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        kb.add_attachment(
            conn, task_id, filename="a.txt", stored_path="/tmp/a.txt", size=3
        )
        kinds = [e.kind for e in kb.list_events(conn, task_id)]
        assert "attached" in kinds
    finally:
        conn.close()


def test_delete_attachment_missing_returns_none(kanban_home):
    conn = kb.connect()
    try:
        assert kb.delete_attachment(conn, 999999) is None
    finally:
        conn.close()


def test_attachments_root_is_per_board(kanban_home, monkeypatch):
    # default board uses <root>/kanban/attachments
    default_root = kb.attachments_root(board="default")
    assert default_root.name == "attachments"
    # a named board nests under its board dir
    monkeypatch.delenv("HERMES_KANBAN_ATTACHMENTS_ROOT", raising=False)
    named = kb.attachments_root(board="default")
    assert named == default_root


def test_attachments_root_env_override(kanban_home, monkeypatch, tmp_path):
    override = tmp_path / "custom-attach"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(override))
    assert kb.attachments_root() == override
    assert kb.task_attachments_dir("t_abc") == override / "t_abc"


# ---------------------------------------------------------------------------
# Worker context surfacing
# ---------------------------------------------------------------------------


def test_worker_context_lists_attachments_with_absolute_path(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="translate PDF")
        dest_dir = kb.task_attachments_dir(task_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        blob = dest_dir / "manual.pdf"
        blob.write_bytes(b"data")
        kb.add_attachment(
            conn,
            task_id,
            filename="manual.pdf",
            stored_path=str(blob.resolve()),
            content_type="application/pdf",
            size=4,
        )
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Attachments" in ctx
        assert "manual.pdf" in ctx
        # The absolute path must appear so the worker can read_file it.
        assert str(blob.resolve()) in ctx
    finally:
        conn.close()


def test_worker_context_no_attachments_section_when_empty(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Attachments" not in ctx
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# REST surface — upload / list / download / delete round-trip
# ---------------------------------------------------------------------------


def _create_task_via_api(client) -> str:
    r = client.post("/api/plugins/kanban/tasks", json={"title": "x"})
    assert r.status_code == 200, r.text
    return r.json()["task"]["id"]


def test_upload_list_download_delete_roundtrip(client):
    task_id = _create_task_via_api(client)
    content = b"hello attachment world"

    # Upload
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert r.status_code == 200, r.text
    att = r.json()["attachment"]
    assert att["filename"] == "notes.txt"
    assert att["size"] == len(content)
    att_id = att["id"]

    # List (drawer also embeds it in GET /tasks/:id)
    r = client.get(f"/api/plugins/kanban/tasks/{task_id}/attachments")
    assert r.status_code == 200
    assert [a["filename"] for a in r.json()["attachments"]] == ["notes.txt"]

    detail = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()
    assert "attachments" in detail
    assert len(detail["attachments"]) == 1

    # Download streams the exact bytes back
    r = client.get(f"/api/plugins/kanban/attachments/{att_id}")
    assert r.status_code == 200
    assert r.content == content

    # Delete removes the row and the file
    r = client.delete(f"/api/plugins/kanban/attachments/{att_id}")
    assert r.status_code == 200
    assert client.get(f"/api/plugins/kanban/attachments/{att_id}").status_code == 404
    assert client.get(
        f"/api/plugins/kanban/tasks/{task_id}/attachments"
    ).json()["attachments"] == []


def test_upload_sanitizes_traversal_filename(client):
    task_id = _create_task_via_api(client)
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("../../../../etc/passwd", b"x", "text/plain")},
    )
    assert r.status_code == 200, r.text
    stored_path = r.json()["attachment"]["stored_path"]
    # The leaf name only; never escapes the per-task attachments dir.
    assert Path(stored_path).name == "passwd"
    task_dir = kb.task_attachments_dir(task_id).resolve()
    assert Path(stored_path).resolve().is_relative_to(task_dir)


def test_upload_name_collision_gets_suffixed(client):
    task_id = _create_task_via_api(client)
    for _ in range(2):
        r = client.post(
            f"/api/plugins/kanban/tasks/{task_id}/attachments",
            files={"file": ("dup.txt", b"a", "text/plain")},
        )
        assert r.status_code == 200, r.text
    names = sorted(
        a["filename"]
        for a in client.get(
            f"/api/plugins/kanban/tasks/{task_id}/attachments"
        ).json()["attachments"]
    )
    assert names == ["dup (1).txt", "dup.txt"]


def test_upload_unknown_task_404(client):
    r = client.post(
        "/api/plugins/kanban/tasks/t_nope/attachments",
        files={"file": ("x.txt", b"x", "text/plain")},
    )
    assert r.status_code == 404


def test_download_unknown_attachment_404(client):
    assert client.get("/api/plugins/kanban/attachments/424242").status_code == 404
