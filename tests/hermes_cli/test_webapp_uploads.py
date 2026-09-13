from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
import os
from pathlib import Path
import shutil
import tempfile
import threading

import pytest

from hermes_cli import profile_lifecycle
from hermes_cli.web_routers import files as image_routes
from hermes_cli import web_server_profiles
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from hermes_cli import profile_incarnation, profiles, web_server
from hermes_cli.web_routers import uploads
from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES


_SESSION_HEADER = "X-Hermes-Session-Token"
_BARRIER_TIMEOUT_SECONDS = 30


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "webapp-test-token")
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    return TestClient(web_server.app)


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


@pytest.mark.linux_only
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
    assert response.json()["detail"] == "File is too large; cap is 16 MiB"
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
    real_open = uploads.os.open
    real_unlink = Path.unlink

    def fail_target_open(path, *args, **kwargs):
        if Path(path).parent == home / "uploads":
            raise PermissionError("target open denied")
        return real_open(path, *args, **kwargs)

    def fail_target_cleanup(path: Path, *args, **kwargs):
        if path.parent == home / "uploads":
            raise OSError("target cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(uploads.os, "open", fail_target_open)
    monkeypatch.setattr(Path, "unlink", fail_target_cleanup)

    with pytest.raises(HTTPException) as exc_info:
        uploads._publish_staged_upload(BytesIO(b"bytes"), home, None, "notes.txt")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not stage file: target open denied"


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

    @contextmanager
    def deleting_scope(_profile):
        yield profile_home
        shutil.rmtree(profile_home)

    monkeypatch.setattr(uploads, "_profile_scope", deleting_scope)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload?profile=worker",
            files={"file": ("notes.txt", b"bytes", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 404
    assert not profile_home.exists()


def _assert_browser_file_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    profile_home = profiles.create_profile("worker", no_alias=True, no_skills=True)
    generation_a = profile_incarnation.read_profile_incarnation(profile_home)
    assert generation_a is not None

    publish_ready = threading.Event()
    resume_publish = threading.Event()
    real_lease = profile_incarnation.profile_incarnation_lease

    @contextmanager
    def pause_before_publish(home, expected_incarnation=None, **kwargs):
        if Path(home) == profile_home and expected_incarnation == generation_a:
            publish_ready.set()
            if not resume_publish.wait(timeout=_BARRIER_TIMEOUT_SECONDS):
                raise TimeoutError("upload publish barrier was not released")
        with real_lease(home, expected_incarnation, **kwargs) as leased_home:
            yield leased_home

    monkeypatch.setattr(
        uploads,
        "profile_incarnation_lease",
        pause_before_publish,
        raising=False,
    )

    with _client(tmp_path, monkeypatch) as client, ThreadPoolExecutor(max_workers=1) as pool:
        stale_request = pool.submit(
            client.post,
            "/api/chat/file-upload?profile=worker",
            files={"file": ("stale.txt", b"generation-a", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )
        try:
            assert publish_ready.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
            profiles.delete_profile("worker", yes=True)
            recreated_home = profiles.create_profile(
                "worker",
                no_alias=True,
                no_skills=True,
            )
            generation_b = profile_incarnation.read_profile_incarnation(recreated_home)
            assert generation_b is not None
            assert generation_b != generation_a
        finally:
            resume_publish.set()

        stale_response = stale_request.result(timeout=_BARRIER_TIMEOUT_SECONDS)
        assert stale_response.status_code == 404, stale_response.text
        assert not (recreated_home / "uploads").exists()

        current_response = client.post(
            "/api/chat/file-upload?profile=worker",
            files={"file": ("current.txt", b"generation-b", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert current_response.status_code == 200
    current_path = Path(current_response.json()["path"])
    assert current_path.parent == recreated_home / "uploads"
    assert current_path.read_bytes() == b"generation-b"


def test_browser_file_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    _assert_browser_file_upload_cannot_publish_into_recreated_profile(
        tmp_path,
        monkeypatch,
    )


@pytest.mark.macos_only
def test_macos_browser_file_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    _assert_browser_file_upload_cannot_publish_into_recreated_profile(
        tmp_path,
        monkeypatch,
    )


@pytest.mark.windows_only
def test_windows_browser_file_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    _assert_browser_file_upload_cannot_publish_into_recreated_profile(
        tmp_path,
        monkeypatch,
    )


@pytest.mark.parametrize(
    ("write_error", "expected_status", "expected_detail"),
    [
        (PermissionError("image write denied"), 403, "Image directory is not writable"),
        (OSError("image write failed"), 500, "Could not write image: image write failed"),
    ],
)
def test_browser_image_upload_cleanup_failure_does_not_mask_write_status(
    tmp_path: Path, monkeypatch, write_error, expected_status, expected_detail
):
    image_dir = tmp_path / "hermes-home" / "images"
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
    assert response.json()["detail"] == expected_detail


def _assert_browser_image_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    profile_home = profiles.create_profile("worker", no_alias=True, no_skills=True)
    generation_a = profile_incarnation.read_profile_incarnation(profile_home)
    assert generation_a is not None

    publish_ready = threading.Event()
    resume_publish = threading.Event()
    real_lease = profile_incarnation.profile_incarnation_lease
    lease_call = []

    @contextmanager
    def pause_before_publish(home, expected_incarnation=None, **kwargs):
        lease_call.append((Path(home), expected_incarnation))
        publish_ready.set()
        if not resume_publish.wait(timeout=_BARRIER_TIMEOUT_SECONDS):
            raise TimeoutError("image publish barrier was not released")
        with real_lease(home, expected_incarnation, **kwargs) as leased_home:
            yield leased_home

    monkeypatch.setattr(
        image_routes,
        "profile_incarnation_lease",
        pause_before_publish,
        raising=False,
    )

    payload = {
        "data_url": "data:image/png;base64,iVBORw0KGgo=",
        "filename": "image.png",
    }
    headers = {_SESSION_HEADER: "webapp-test-token"}
    with _client(tmp_path, monkeypatch) as client, ThreadPoolExecutor(max_workers=1) as pool:
        stale_request = pool.submit(
            client.post,
            "/api/chat/image-upload?profile=worker",
            json=payload,
            headers=headers,
        )
        try:
            assert publish_ready.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
            assert lease_call == [(profile_home, generation_a)]
            profiles.delete_profile("worker", yes=True)
            recreated_home = profiles.create_profile(
                "worker",
                no_alias=True,
                no_skills=True,
            )
            generation_b = profile_incarnation.read_profile_incarnation(recreated_home)
            assert generation_b is not None
            assert generation_b != generation_a
        finally:
            resume_publish.set()

        stale_response = stale_request.result(timeout=_BARRIER_TIMEOUT_SECONDS)
        assert stale_response.status_code == 404
        assert not (recreated_home / "images").exists()

        current_response = client.post(
            "/api/chat/image-upload?profile=worker",
            json=payload,
            headers=headers,
        )

    assert current_response.status_code == 200
    current_path = Path(current_response.json()["path"])
    assert current_path.parent == recreated_home / "images"
    assert current_path.read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_browser_image_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    _assert_browser_image_upload_cannot_publish_into_recreated_profile(
        tmp_path,
        monkeypatch,
    )


@pytest.mark.macos_only
def test_macos_browser_image_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    _assert_browser_image_upload_cannot_publish_into_recreated_profile(
        tmp_path,
        monkeypatch,
    )


@pytest.mark.windows_only
def test_windows_browser_image_upload_cannot_publish_into_recreated_profile(
    tmp_path: Path, monkeypatch
):
    _assert_browser_image_upload_cannot_publish_into_recreated_profile(
        tmp_path,
        monkeypatch,
    )


def test_browser_image_upload_never_creates_a_missing_profile_home(
    tmp_path: Path, monkeypatch
):
    profile_home = tmp_path / "hermes-home" / "profiles" / "worker"

    @contextmanager
    def missing_scope(_profile):
        yield profile_home

    monkeypatch.setattr(web_server_profiles, "_profile_scope", missing_scope)
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
    profile_lifecycle.mark_profile_deleting(profile_home)

    @contextmanager
    def deleting_scope(_profile):
        yield profile_home

    monkeypatch.setattr(uploads, "_profile_scope", deleting_scope)
    monkeypatch.setattr(web_server_profiles, "_profile_scope", deleting_scope)
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
    assert not (profile_home / "images").exists()


def test_browser_file_upload_does_not_report_a_path_after_tombstone_wins(
    tmp_path: Path, monkeypatch
):
    profile_home = tmp_path / "hermes-home" / "profiles" / "worker"
    profile_home.mkdir(parents=True)

    @contextmanager
    def profile_scope(_profile):
        yield profile_home

    real_fsync = uploads.os.fsync

    def tombstone_after_flush(fd):
        real_fsync(fd)
        profile_lifecycle.mark_profile_deleting(profile_home)

    monkeypatch.setattr(uploads, "_profile_scope", profile_scope)
    monkeypatch.setattr(uploads.os, "fsync", tombstone_after_flush)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/chat/file-upload?profile=worker",
            files={"file": ("race.txt", b"bytes", "text/plain")},
            headers={_SESSION_HEADER: "webapp-test-token"},
        )

    assert response.status_code == 404
    upload_root = profile_home / "uploads"
    assert not upload_root.exists() or list(upload_root.iterdir()) == []


def test_browser_image_upload_does_not_report_a_path_after_tombstone_wins(
    tmp_path: Path, monkeypatch
):
    import hermes_constants

    profile_home = tmp_path / "hermes-home" / "profiles" / "worker"
    profile_home.mkdir(parents=True)

    @contextmanager
    def profile_scope(_profile):
        yield profile_home

    checks = 0

    def unavailable_after_write(home):
        nonlocal checks
        checks += 1
        if checks >= 3:
            profile_lifecycle.mark_profile_deleting(Path(home))
            return True
        return False

    monkeypatch.setattr(web_server_profiles, "_profile_scope", profile_scope)
    monkeypatch.setattr(
        hermes_constants,
        "named_profile_home_is_unavailable",
        unavailable_after_write,
    )
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
    image_dir = profile_home / "images"
    assert not image_dir.exists() or list(image_dir.iterdir()) == []
