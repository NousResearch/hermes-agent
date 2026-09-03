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
import json
import sqlite3
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
        assert a.attachment_type == "attachment"

        got = kb.get_attachment(conn, att_id)
        assert got is not None and got.id == att_id

        removed = kb.delete_attachment(conn, att_id)
        assert removed is not None and removed.id == att_id
        assert kb.list_attachments(conn, task_id) == []
        assert not blob.exists(), "delete should unlink the on-disk blob"
        assert kb.get_attachment(conn, att_id) is None
    finally:
        conn.close()


def test_fresh_attachment_type_constraint_tracks_declared_policy(kanban_home, tmp_path):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="fresh-constraint")
        for index, attachment_type in enumerate(sorted(kb.VALID_ATTACHMENT_TYPES), 1):
            conn.execute(
                "INSERT INTO task_attachments "
                "(task_id, filename, stored_path, size, created_at, attachment_type) "
                "VALUES (?, ?, ?, 0, ?, ?)",
                (
                    task_id,
                    f"valid-{index}.txt",
                    str(tmp_path / f"fresh-valid-{index}.txt"),
                    index,
                    attachment_type,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO task_attachments "
                "(task_id, filename, stored_path, size, created_at, attachment_type) "
                "VALUES (?, 'invalid.txt', ?, 0, 99, 'unknown')",
                (task_id, str(tmp_path / "fresh-invalid.txt")),
            )
    finally:
        conn.close()


def test_attachment_type_migration_is_idempotent_and_legacy_rows_remain_inputs(
    kanban_home, tmp_path,
):
    """A pre-type board upgrades in place, and old writers stay compatible."""
    db_path = tmp_path / "legacy-attachments.db"
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        task_id = _make_task(conn, title="legacy")
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_attachments)")
        }
        if "attachment_type" in columns:
            conn.execute("ALTER TABLE task_attachments DROP COLUMN attachment_type")
        conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, 'legacy.txt', ?, 'text/plain', 6, 'old-version', 1)",
            (task_id, str(tmp_path / "legacy.txt")),
        )
        conn.commit()

    kb.init_db(db_path)
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        info = {
            row["name"]: row for row in conn.execute("PRAGMA table_info(task_attachments)")
        }
        assert "attachment_type" in info
        assert info["attachment_type"]["notnull"] == 0
        assert info["attachment_type"]["dflt_value"] is None

        legacy = kb.list_attachments(conn, task_id)
        assert [(item.filename, item.attachment_type) for item in legacy] == [
            ("legacy.txt", "attachment"),
        ]

        # Simulate a downgraded/older Hermes binary that omits the new column.
        conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, 'old-writer.txt', ?, 'text/plain', 1, 'old-version', 2)",
            (task_id, str(tmp_path / "old-writer.txt")),
        )
        conn.commit()
        assert [item.attachment_type for item in kb.list_attachments(conn, task_id)] == [
            "attachment",
            "attachment",
        ]

        # The additive migration and fresh schema share one generated CHECK
        # policy: every declared type must be accepted, while unknown values
        # still fail at SQLite's persistence boundary.
        for index, attachment_type in enumerate(sorted(kb.VALID_ATTACHMENT_TYPES), 3):
            conn.execute(
                "INSERT INTO task_attachments "
                "(task_id, filename, stored_path, size, created_at, attachment_type) "
                "VALUES (?, ?, ?, 0, ?, ?)",
                (
                    task_id,
                    f"valid-{index}.txt",
                    str(tmp_path / f"valid-{index}.txt"),
                    index,
                    attachment_type,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO task_attachments "
                "(task_id, filename, stored_path, size, created_at, attachment_type) "
                "VALUES (?, 'invalid.txt', ?, 0, 99, 'unknown')",
                (task_id, str(tmp_path / "invalid.txt")),
            )


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


def test_worker_context_labels_inputs_and_generated_artifacts_separately(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="render report")
        kb.add_attachment(
            conn,
            task_id,
            filename="brief.txt",
            stored_path="/tmp/brief.txt",
            attachment_type="attachment",
        )
        kb.add_attachment(
            conn,
            task_id,
            filename="report.pdf",
            stored_path="/tmp/report.pdf",
            attachment_type="artifact",
        )

        ctx = kb.build_worker_context(conn, task_id)
        attachments_block, artifacts_block = ctx.split("## Artifacts", 1)
        assert "## Attachments" in attachments_block
        assert "brief.txt" in attachments_block
        assert "report.pdf" not in attachments_block
        assert "Files generated by this task" in artifacts_block
        assert "report.pdf" in artifacts_block
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
    assert att["attachment_type"] == "attachment"
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


def test_upload_list_detail_and_download_artifact_roundtrip(client):
    task_id = _create_task_via_api(client)
    content = b"generated report"

    created = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("report.txt", content, "text/plain")},
        data={"attachment_type": "artifact"},
    )
    assert created.status_code == 200, created.text
    artifact = created.json()["attachment"]
    assert artifact["attachment_type"] == "artifact"

    listed = client.get(
        f"/api/plugins/kanban/tasks/{task_id}/attachments"
    ).json()["attachments"]
    assert [(item["filename"], item["attachment_type"]) for item in listed] == [
        ("report.txt", "artifact"),
    ]
    detail = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()
    assert detail["attachments"][0]["attachment_type"] == "artifact"

    downloaded = client.get(
        f"/api/plugins/kanban/attachments/{artifact['id']}"
    )
    assert downloaded.status_code == 200
    assert downloaded.content == content


def test_upload_rejects_unknown_attachment_type_without_leaving_a_blob(client):
    task_id = _create_task_via_api(client)
    response = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("bad.txt", b"bad", "text/plain")},
        data={"attachment_type": "output"},
    )
    assert response.status_code == 400
    assert "attachment_type" in response.json()["detail"]
    assert client.get(
        f"/api/plugins/kanban/tasks/{task_id}/attachments"
    ).json()["attachments"] == []
    assert not kb.task_attachments_dir(task_id).exists()


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


def test_download_unknown_attachment_404(client):
    assert client.get("/api/plugins/kanban/attachments/424242").status_code == 404


# ---------------------------------------------------------------------------
# Shared helper — store_attachment_bytes (used by dashboard + tool + CLI)
# ---------------------------------------------------------------------------


def test_store_attachment_bytes_roundtrip(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        att_id = kb.store_attachment_bytes(
            conn, task_id, "doc.txt", b"some bytes",
            content_type="text/plain", uploaded_by="tester",
        )
        a = kb.get_attachment(conn, att_id)
        assert a is not None
        assert a.filename == "doc.txt"
        assert a.size == len(b"some bytes")
        assert a.uploaded_by == "tester"
        assert a.attachment_type == "attachment"
        assert Path(a.stored_path).read_bytes() == b"some bytes"
        assert Path(a.stored_path).resolve().is_relative_to(
            kb.task_attachments_dir(task_id).resolve()
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI — hermes kanban attach / attachments / attach-rm
# ---------------------------------------------------------------------------


def test_cli_attach_attachments_and_rm(kanban_home, tmp_path):
    from hermes_cli.kanban import run_slash

    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="cli-attach")
    finally:
        conn.close()

    src = tmp_path / "upload.txt"
    src.write_bytes(b"cli file body")

    out = run_slash(f"attach {task_id} {src}")
    assert "Attached" in out, out

    conn = kb.connect()
    try:
        atts = kb.list_attachments(conn, task_id)
        assert len(atts) == 1
        att_id = atts[0].id
        assert atts[0].filename == "upload.txt"
        assert atts[0].attachment_type == kb.DEFAULT_HUMAN_ATTACHMENT_TYPE
        assert Path(atts[0].stored_path).read_bytes() == b"cli file body"
    finally:
        conn.close()

    listed = run_slash(f"attachments {task_id}")
    assert "upload.txt" in listed

    removed = run_slash(f"attach-rm {att_id}")
    assert "Deleted attachment" in removed
    conn = kb.connect()
    try:
        assert kb.list_attachments(conn, task_id) == []
    finally:
        conn.close()


def test_cli_can_create_and_label_an_artifact(kanban_home, tmp_path):
    from hermes_cli.kanban import run_slash

    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="cli-artifact")
    finally:
        conn.close()

    src = tmp_path / "result.json"
    src.write_text('{"ok": true}', encoding="utf-8")
    created = run_slash(f"attach {task_id} {src} --type artifact")
    assert "artifact" in created.lower()

    conn = kb.connect()
    try:
        artifact = kb.list_attachments(conn, task_id)[0]
        assert artifact.attachment_type == "artifact"
    finally:
        conn.close()

    listed = run_slash(f"attachments {task_id}")
    assert "artifact" in listed.lower()
    assert "result.json" in listed

