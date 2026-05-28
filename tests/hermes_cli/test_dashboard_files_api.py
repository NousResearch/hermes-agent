"""Dashboard file-browser API safety tests."""

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    from hermes_cli import web_server as ws

    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(tmp_path / "files"))
    ws.app.state.bound_host = "127.0.0.1"
    return TestClient(ws.app), ws


def _headers(ws):
    return {"X-Hermes-Session-Token": ws._SESSION_TOKEN, "Host": "127.0.0.1"}


def test_dashboard_files_lists_and_downloads_within_root(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = Path(tmp_path / "files")
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "summary.txt").write_text("hello from hermes", encoding="utf-8")

    listing = client.get("/api/files", params={"path": "reports"}, headers=_headers(ws))
    assert listing.status_code == 200
    body = listing.json()
    assert body["path"] == "reports"
    assert body["parent"] == ""
    assert body["entries"] == [
        {
            "name": "summary.txt",
            "path": "reports/summary.txt",
            "type": "file",
            "size": len("hello from hermes"),
            "modified_at": body["entries"][0]["modified_at"],
            "mime_type": "text/plain",
        }
    ]

    download = client.get(
        "/api/files/download",
        params={"path": "reports/summary.txt"},
        headers=_headers(ws),
    )
    assert download.status_code == 200
    assert download.content == b"hello from hermes"


@pytest.mark.parametrize("bad_path", ["../config.yaml", "/etc/passwd", "reports/../../outside"])
def test_dashboard_files_reject_path_traversal(monkeypatch, tmp_path, bad_path):
    client, ws = _client(monkeypatch, tmp_path)

    resp = client.get("/api/files", params={"path": bad_path}, headers=_headers(ws))

    assert resp.status_code in {400, 403}


def test_dashboard_files_skip_symlinks_outside_root(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = tmp_path / "files"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    (outside / "secret.txt").write_text("nope", encoding="utf-8")
    (root / "safe.txt").write_text("ok", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside / "secret.txt")

    listing = client.get("/api/files", headers=_headers(ws))
    assert listing.status_code == 200
    names = [entry["name"] for entry in listing.json()["entries"]]
    assert names == ["safe.txt"]

    escaped = client.get(
        "/api/files/download",
        params={"path": "escape.txt"},
        headers=_headers(ws),
    )
    assert escaped.status_code == 403


def test_dashboard_files_require_session_token(monkeypatch, tmp_path):
    client, _ws = _client(monkeypatch, tmp_path)

    resp = client.get("/api/files", headers={"Host": "127.0.0.1"})

    assert resp.status_code == 401


def test_dashboard_files_upload_documents_to_uploads_folder(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)

    resp = client.post(
        "/api/files/upload",
        files=[
            ("files", ("notes.txt", b"alpha", "text/plain")),
            ("files", ("../notes.txt", b"beta", "text/plain")),
        ],
        headers=_headers(ws),
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["upload_dir"].endswith("/files/uploads")
    paths = [item["path"] for item in body["files"]]
    assert paths == ["uploads/notes.txt", "uploads/notes-1.txt"]
    absolute_paths = [Path(item["absolute_path"]) for item in body["files"]]
    assert [p.read_bytes() for p in absolute_paths] == [b"alpha", b"beta"]

    listing = client.get("/api/files", params={"path": "uploads"}, headers=_headers(ws))
    assert listing.status_code == 200
    assert [entry["name"] for entry in listing.json()["entries"]] == ["notes-1.txt", "notes.txt"]


def test_dashboard_files_upload_documents_to_requested_folder(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = Path(tmp_path / "files")
    inbox = root / "uploads" / "sample-project" / "inbox"
    inbox.mkdir(parents=True)

    resp = client.post(
        "/api/files/upload",
        data={"path": "uploads/sample-project/inbox"},
        files=[("files", ("briefing.pdf", b"pdf-ish", "application/pdf"))],
        headers=_headers(ws),
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "uploads/sample-project/inbox"
    assert body["upload_dir"].endswith("/files/uploads/sample-project/inbox")
    assert body["files"][0]["path"] == "uploads/sample-project/inbox/briefing.pdf"
    assert (inbox / "briefing.pdf").read_bytes() == b"pdf-ish"


@pytest.mark.parametrize("bad_path", ["../outside", "/tmp", "uploads/../../outside"])
def test_dashboard_files_upload_rejects_path_traversal(monkeypatch, tmp_path, bad_path):
    client, ws = _client(monkeypatch, tmp_path)

    resp = client.post(
        "/api/files/upload",
        data={"path": bad_path},
        files=[("files", ("notes.txt", b"alpha", "text/plain"))],
        headers=_headers(ws),
    )

    assert resp.status_code in {400, 403}


def test_dashboard_files_upload_rejects_symlink_upload_directory(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = tmp_path / "files"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    (root / "uploads").symlink_to(outside, target_is_directory=True)

    resp = client.post(
        "/api/files/upload",
        files=[("files", ("notes.txt", b"alpha", "text/plain"))],
        headers=_headers(ws),
    )

    assert resp.status_code in {400, 403}
    assert not (outside / "notes.txt").exists()


def test_dashboard_files_upload_does_not_follow_target_symlinks(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = tmp_path / "files"
    outside = tmp_path / "outside"
    uploads = root / "uploads"
    uploads.mkdir(parents=True)
    outside.mkdir()
    (uploads / "notes.txt").symlink_to(outside / "notes.txt")

    resp = client.post(
        "/api/files/upload",
        files=[("files", ("notes.txt", b"alpha", "text/plain"))],
        headers=_headers(ws),
    )

    assert resp.status_code == 200
    assert resp.json()["files"][0]["path"] == "uploads/notes-1.txt"
    assert not (outside / "notes.txt").exists()
    assert (uploads / "notes-1.txt").read_bytes() == b"alpha"


def test_profile_files_upload_lands_in_profile_uploads(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    from hermes_cli import profiles

    profiles_root = tmp_path / "profiles"
    profile_home = profiles_root / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    resp = client.post(
        "/api/profiles/worker/files/upload",
        files=[("files", ("briefing.txt", b"profile scoped", "text/plain"))],
        headers=_headers(ws),
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "uploads"
    assert body["files"][0]["path"] == "uploads/briefing.txt"
    assert (profile_home / "files" / "uploads" / "briefing.txt").read_bytes() == b"profile scoped"


def test_profile_files_upload_rejects_missing_profile(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    from hermes_cli import profiles

    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    resp = client.post(
        "/api/profiles/missing/files/upload",
        files=[("files", ("briefing.txt", b"profile scoped", "text/plain"))],
        headers=_headers(ws),
    )

    assert resp.status_code == 404


def test_profile_files_upload_rejects_symlink_files_root(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    from hermes_cli import profiles

    profiles_root = tmp_path / "profiles"
    profile_home = profiles_root / "worker"
    outside = tmp_path / "outside"
    profile_home.mkdir(parents=True)
    outside.mkdir()
    (profile_home / "files").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    resp = client.post(
        "/api/profiles/worker/files/upload",
        files=[("files", ("briefing.txt", b"profile scoped", "text/plain"))],
        headers=_headers(ws),
    )

    assert resp.status_code in {400, 403}
    assert not (outside / "uploads" / "briefing.txt").exists()


def test_dashboard_files_delete_file_within_root(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = tmp_path / "files"
    root.mkdir(parents=True)
    target = root / "ready.txt"
    target.write_text("download me", encoding="utf-8")

    resp = client.delete(
        "/api/files",
        params={"path": "ready.txt"},
        headers=_headers(ws),
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "path": "ready.txt", "type": "file"}
    assert not target.exists()


def test_dashboard_files_delete_rejects_directories_and_escaped_symlinks(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = tmp_path / "files"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    (root / "folder").mkdir()
    (outside / "secret.txt").write_text("nope", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside / "secret.txt")

    directory_resp = client.delete(
        "/api/files",
        params={"path": "folder"},
        headers=_headers(ws),
    )
    escape_resp = client.delete(
        "/api/files",
        params={"path": "escape.txt"},
        headers=_headers(ws),
    )

    assert directory_resp.status_code == 400
    assert escape_resp.status_code == 403
    assert (outside / "secret.txt").exists()


def test_dashboard_files_prunes_ready_downloads_after_72_hours_but_keeps_uploads(monkeypatch, tmp_path):
    client, ws = _client(monkeypatch, tmp_path)
    root = tmp_path / "files"
    uploads = root / "uploads"
    root.mkdir(parents=True)
    uploads.mkdir()
    stale_ready = root / "stale-ready.txt"
    fresh_ready = root / "fresh-ready.txt"
    stale_upload = uploads / "stale-upload.txt"
    for item in (stale_ready, fresh_ready, stale_upload):
        item.write_text(item.name, encoding="utf-8")
    old = time.time() - (73 * 60 * 60)
    os.utime(stale_ready, (old, old))
    os.utime(stale_upload, (old, old))

    resp = client.get("/api/files", headers=_headers(ws))

    assert resp.status_code == 200
    names = [entry["name"] for entry in resp.json()["entries"]]
    assert "stale-ready.txt" not in names
    assert "fresh-ready.txt" in names
    assert "uploads" in names
    assert not stale_ready.exists()
    assert stale_upload.exists()
