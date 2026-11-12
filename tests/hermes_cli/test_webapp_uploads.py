from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from hermes_cli import web_server_profiles
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from hermes_cli import profiles, staged_uploads, web_server
from hermes_cli.web_routers import uploads
from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES, mark_named_profile_deleted


_SESSION_HEADER = "X-Hermes-Session-Token"
_BARRIER_TIMEOUT_SECONDS = 30


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "webapp-test-token")
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    # The loopback Webapp: the only surface that serves the browser's file routes.
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    return TestClient(web_server.app, base_url="http://127.0.0.1")


def _docx_upload(text: str) -> bytes:
    """A minimal OOXML package, without optional document-writing dependencies."""
    package_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    office_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    main_part = "word/document.xml"
    parts = {
        main_part: (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
        ),
        "[Content_Types].xml": (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/{main_part}" ContentType="application/vnd.openxmlformats-officedocument.'
            'wordprocessingml.document.main+xml"/></Types>'
        ),
        "_rels/.rels": (
            f'<Relationships xmlns="{package_ns}"><Relationship Id="rId1" '
            f'Type="{office_ns}/officeDocument" Target="{main_part}"/></Relationships>'
        ),
    }
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for part, xml in parts.items():
            archive.writestr(part, xml)
    return buffer.getvalue()


def test_browser_document_upload_remains_extractable_after_file_attach(
    tmp_path: Path, monkeypatch
):
    """The upload route's receipt, attached by the profile's gateway (the browser's only route to a
    session), leaves a document the file tool still extracts."""
    filename = "отчёт.docx"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with _client(tmp_path, monkeypatch) as client:
        from gateway.runtime_bootstrap import TicketStore
        from hermes_cli import install_identity
        from tests.gateway.test_canonical_file_attach_staged_sources import _attach, _authority_with_old_desktop_chat
        from tools.file_tools import read_file_tool
        from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides
        from tools.terminal_tool_lifecycle import cleanup_vm

        home = tmp_path / "hermes-home"
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sid = "old-desktop"
        monkeypatch.setattr(install_identity, "_INSTALL_ID_CACHE", {"root": None, "value": None})
        text = "Browser document content survives upload and attachment"
        payload = _docx_upload(text)
        response = client.post(
            "/api/chat/file-upload",
            files={"file": (filename, payload, "application/octet-stream")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )
        assert response.status_code == 200, response.text
        source = Path(response.json()["path"])

        async def attach():
            authority = await _authority_with_old_desktop_chat(home, monkeypatch, cwd=workspace)
            host = _attach(authority, TicketStore("first", frozenset({authority.profile_id})), "interactive", "uid:1000")
            await host.dispatch({"id": "resume", "method": "session.resume", "params": {"session_id": sid}})
            return await host.dispatch({"id": "attach", "method": "file.attach", "params": {
                "session_id": sid, "name": filename, "staged_upload": response.json()["staged_upload"]}})

        attached = asyncio.run(attach())
        assert "error" not in attached, attached
        stored = Path(attached["result"]["path"])
        assert source.parent == home / "uploads"
        assert stored.parent == home / "attachments"
        assert attached["result"]["uploaded"] is True
        assert stored.read_bytes() == source.read_bytes() == payload
        register_task_env_overrides(sid, {"env_type": "local", "cwd": str(workspace)})
        try:
            extracted = json.loads(read_file_tool(attached["result"]["ref_path"], task_id=sid))
        finally:
            cleanup_vm(sid)
            clear_task_env_overrides(sid)
        assert extracted.get("extracted_document") is True, extracted
        assert text in extracted["content"]
        assert stored.suffix == source.suffix == Path(filename).suffix


@pytest.mark.parametrize(
    ("filename", "suffix"),
    [
        ("отчёт.docx", ".docx"),
        ("../../таблица.xlsx", ".xlsx"),
        (r"..\..\таблица.xlsx", ".xlsx"),
        ("!!!.PDF", ".PDF"),
        ("long-" * 100 + ".docx", ".docx"),
        ("report.tar.gz", ".gz"),
        ("x." + "a" * 200, None),
        ("../../..", ""),
        (None, ""),
    ],
    ids=["cyrillic", "traversal", "backslashes", "punctuation", "long-stem",
         "compound", "long-suffix", "dots", "missing"],
)
def test_safe_upload_filename_is_bounded_basename_with_document_suffix(filename, suffix):
    cleaned = staged_uploads._safe_filename(filename)
    assert 1 <= len(cleaned) <= 120
    assert re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", cleaned)
    assert Path(cleaned).name == cleaned
    assert Path(cleaned).stem.strip(".-")
    if suffix is not None:
        assert Path(cleaned).suffix == suffix


def test_browser_upload_stages_bytes_under_hermes_home(tmp_path: Path, monkeypatch):
    upload_root = tmp_path / "hermes-home" / "uploads"
    upload_root.mkdir(parents=True)
    abandoned = upload_root / "web-abandoned-old.txt"
    unrelated = upload_root / "keep-me.txt"
    abandoned.write_text("old", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")
    os.utime(abandoned, (1, 1))

    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload",
            files={"file": ("../notes from browser.txt", b"browser bytes", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 200
    payload = response.json()
    staged = Path(payload["path"])
    assert staged.parent == tmp_path / "hermes-home" / "uploads"
    assert staged.name.endswith("notes-from-browser.txt")
    assert staged.read_bytes() == b"browser bytes"
    assert not abandoned.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


@pytest.mark.platforms("linux")
def test_browser_upload_stages_owner_only_file_on_posix(tmp_path: Path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload",
            files={"file": ("private.txt", b"private", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 200
    assert Path(response.json()["path"]).stat().st_mode & 0o777 == 0o600


def test_browser_upload_rejects_oversize_without_leaving_partial_file(
    tmp_path: Path, monkeypatch
):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload",
            files={
                "file": (
                    "large.bin",
                    b"x" * (WEBAPP_ATTACHMENT_MAX_BYTES + 1),
                    "application/octet-stream",
                )
            },
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 413
    upload_root = tmp_path / "hermes-home" / "uploads"
    assert not upload_root.exists() or list(upload_root.iterdir()) == []


@pytest.mark.parametrize("payload", [b"", b"12345678", b"123456789"])
def test_browser_upload_checks_actual_spooled_size_and_closes_upload(
    payload: bytes, tmp_path: Path, monkeypatch
):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(uploads, "_MAX_UPLOAD_BYTES", 8)
    spool = tempfile.SpooledTemporaryFile(max_size=4, mode="w+b")
    spool.write(payload)
    file = UploadFile(file=spool, size=0, filename="notes.txt")

    if len(payload) > 8:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(uploads.upload_chat_file(file=file))
        assert exc_info.value.status_code == 413
        assert not (home / "uploads").exists()
    else:
        result = asyncio.run(uploads.upload_chat_file(file=file))
        assert result["size"] == len(payload)
        assert Path(result["path"]).read_bytes() == payload
    assert spool.closed


def test_browser_file_cleanup_failure_preserves_primary_staging_error(
    tmp_path: Path, monkeypatch
):
    home = tmp_path / "hermes-home"
    home.mkdir()
    real_open = staged_uploads.os.open
    real_unlink = Path.unlink

    def fail_target_open(path, *args, **kwargs):
        if Path(path).parent == home / "uploads":
            raise PermissionError("target open denied")
        return real_open(path, *args, **kwargs)

    def fail_target_cleanup(path: Path, *args, **kwargs):
        if path.parent == home / "uploads":
            raise OSError("target cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(staged_uploads.os, "open", fail_target_open)
    monkeypatch.setattr(Path, "unlink", fail_target_cleanup)

    with pytest.raises(HTTPException) as exc_info:
        staged_uploads.publish_staged_upload(BytesIO(b"bytes"), home, "notes.txt")

    assert exc_info.value.status_code == 500
    assert "target open denied" in exc_info.value.detail


def test_largest_browser_upload_is_readable_by_attachment_flow(
    tmp_path: Path, monkeypatch
):
    payload = b"x" * WEBAPP_ATTACHMENT_MAX_BYTES
    with _client(tmp_path, monkeypatch) as client:
        upload = client.post(
            "/api/chat/file-upload",
            files={"file": ("largest.bin", payload, "application/octet-stream")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )
        assert upload.status_code == 200

        read_back = client.get(
            "/api/fs/read-data-url",
            params={"path": upload.json()["path"]},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert read_back.status_code == 200
    assert read_back.json()["dataUrl"].startswith("data:application/octet-stream;base64,")


def test_browser_upload_requires_the_server_session(tmp_path: Path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload",
            files={"file": ("notes.txt", b"nope", "text/plain")},
            headers={_SESSION_HEADER: "wrong-token"},
        )

    assert response.status_code == 401


def test_browser_file_upload_never_recreates_profile_deleted_after_resolution(
    tmp_path: Path, monkeypatch
):
    profile_home = tmp_path / "hermes-home" / "profiles" / "worker"
    profile_home.mkdir(parents=True)

    real_resolve = staged_uploads.resolve_upload_home

    def resolve_then_delete(profile):
        resolved = real_resolve(profile)
        shutil.rmtree(profile_home)
        return resolved

    monkeypatch.setattr(staged_uploads, "_resolve_profile_dir", lambda _name: profile_home)
    monkeypatch.setattr(uploads, "resolve_upload_home", resolve_then_delete)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload?profile=worker",
            files={"file": ("notes.txt", b"bytes", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 404
    assert not profile_home.exists()


@pytest.mark.parametrize(
    ("write_error", "expected_status"),
    [(PermissionError("image write denied"), 403), (OSError("image write failed"), 500)],
)
def test_browser_image_upload_cleanup_failure_does_not_mask_write_status(
    tmp_path: Path, monkeypatch, write_error, expected_status
):
    image_dir = tmp_path / "hermes-home" / "cache" / "images"
    real_write_bytes = Path.write_bytes
    real_unlink = Path.unlink

    def fail_image_write(path: Path, data: bytes):
        if path.parent == image_dir:
            raise write_error
        return real_write_bytes(path, data)

    def fail_image_cleanup(path: Path, *args, **kwargs):
        if path.parent == image_dir:
            raise OSError("image cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", fail_image_write)
    monkeypatch.setattr(Path, "unlink", fail_image_cleanup)

    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/image-upload",
            json={
                "data_url": "data:image/png;base64,iVBORw0KGgo=",
                "filename": "image.png",
            },
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == expected_status
    if expected_status == 500:
        assert "image write failed" in response.json()["detail"]


def test_browser_uploads_to_a_profile_do_not_queue_behind_the_skills_lock(
    tmp_path: Path, monkeypatch
):
    """Resolving an upload's profile home is a path lookup. A skills request holding the
    process-wide skills lock (and the skills-module globals it retargets) must not stall it."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    profile_home = profiles.create_profile("worker", no_alias=True, no_skills=True)
    holding, release = threading.Event(), threading.Event()

    def hold_skills_lock():
        with web_server_profiles._SKILLS_PROFILE_LOCK:
            holding.set()
            release.wait()

    headers = {_SESSION_HEADER: "webapp-test-token"}
    with _client(tmp_path, monkeypatch) as client, ThreadPoolExecutor(max_workers=3) as pool:
        try:
            pool.submit(hold_skills_lock)
            assert holding.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
            file_upload = pool.submit(
                client.post,
                "/api/chat/file-upload?profile=worker",
                files={"file": ("notes.txt", b"bytes", "text/plain")},
                headers=headers,
            )
            image_upload = pool.submit(
                client.post,
                "/api/chat/image-upload?profile=worker",
                json={"data_url": "data:image/png;base64,iVBORw0KGgo=", "filename": "image.png"},
                headers=headers,
            )
            file_response = file_upload.result(timeout=_BARRIER_TIMEOUT_SECONDS)
            image_response = image_upload.result(timeout=_BARRIER_TIMEOUT_SECONDS)
        finally:
            release.set()

    assert file_response.status_code == 200, file_response.text
    assert Path(file_response.json()["path"]).parent == profile_home / "uploads"
    assert image_response.status_code == 200, image_response.text
    assert Path(image_response.json()["path"]).parent == profile_home / "cache" / "images"


def test_browser_image_upload_never_creates_a_missing_profile_home(
    tmp_path: Path, monkeypatch
):
    profile_home = tmp_path / "hermes-home" / "profiles" / "worker"

    monkeypatch.setattr(staged_uploads, "_resolve_profile_dir", lambda _name: profile_home)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/image-upload?profile=worker",
            json={
                "data_url": "data:image/png;base64,iVBORw0KGgo=",
                "filename": "image.png",
            },
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 404
    assert not profile_home.exists()


def test_browser_uploads_reject_tombstone_before_profile_directory_removal(
    tmp_path: Path, monkeypatch
):
    profile_home = tmp_path / "hermes-home" / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    mark_named_profile_deleted(profile_home)

    monkeypatch.setattr(staged_uploads, "_resolve_profile_dir", lambda _name: profile_home)
    with _client(tmp_path, monkeypatch) as client:
        file_response = client.post(
            "/api/chat/file-upload?profile=worker",
            files={"file": ("notes.txt", b"bytes", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )
        image_response = client.post(
            "/api/chat/image-upload?profile=worker",
            json={
                "data_url": "data:image/png;base64,iVBORw0KGgo=",
                "filename": "image.png",
            },
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert file_response.status_code == 404
    assert image_response.status_code == 404
    assert profile_home.is_dir()
    assert not (profile_home / "uploads").exists()
    assert not (profile_home / "cache" / "images").exists()


def test_browser_file_upload_does_not_report_a_path_after_tombstone_wins(
    tmp_path: Path, monkeypatch
):
    profile_home = tmp_path / "hermes-home" / "profiles" / "worker"
    profile_home.mkdir(parents=True)

    real_fsync = staged_uploads.os.fsync

    def tombstone_after_flush(fd):
        real_fsync(fd)
        mark_named_profile_deleted(profile_home)

    monkeypatch.setattr(staged_uploads, "_resolve_profile_dir", lambda _name: profile_home)
    monkeypatch.setattr(staged_uploads.os, "fsync", tombstone_after_flush)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload?profile=worker",
            files={"file": ("race.txt", b"bytes", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 404
    upload_root = profile_home / "uploads"
    assert not upload_root.exists() or list(upload_root.iterdir()) == []
