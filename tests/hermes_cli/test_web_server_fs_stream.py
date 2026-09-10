from pathlib import Path

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


def test_fs_stream_serves_video_inline_with_range_support(client, tmp_path: Path):
    target = tmp_path / "movie.mp4"
    target.write_bytes(b"0123456789")

    response = client.get(
        "/api/fs/stream",
        params={"path": str(target)},
        headers={"Range": "bytes=2-5"},
    )

    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.headers["content-disposition"].startswith("inline")
    assert response.headers["x-content-type-options"] == "nosniff"

    head = client.head("/api/fs/stream", params={"path": str(target)})
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == "10"


def test_fs_stream_resolves_relative_path_in_originating_session(client, tmp_path: Path, monkeypatch):
    from hermes_state import SessionDB

    hermes_home = tmp_path / "hermes-home"
    session_cwd = tmp_path / "session-workspace"
    gateway_cwd = tmp_path / "gateway-workspace"
    session_cwd.mkdir()
    gateway_cwd.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.chdir(gateway_cwd)

    expected = session_cwd / "movie.mp4"
    expected.write_bytes(b"session media")
    (gateway_cwd / expected.name).write_bytes(b"wrong gateway media")

    hermes_home.mkdir(parents=True)
    db = SessionDB(db_path=hermes_home / "state.db")
    try:
        db.create_session("origin-session", source="gui", cwd=str(session_cwd))
    finally:
        db.close()

    response = client.get(
        "/api/fs/stream",
        params={"path": "./movie.mp4", "profile": "default", "session_id": "origin-session"},
    )

    assert response.status_code == 200, response.text
    assert response.content == b"session media"
    assert response.headers["content-disposition"].startswith("inline")

    missing = client.get(
        "/api/fs/stream",
        params={"path": str(expected), "profile": "default", "session_id": "missing-session"},
    )
    assert missing.status_code == 404


def test_fs_stream_rejects_sensitive_and_non_media_files(client, tmp_path: Path):
    sensitive = tmp_path / ".env"
    sensitive.write_text("SECRET=1", encoding="utf-8")
    unsupported = tmp_path / "page.html"
    unsupported.write_text("<h1>active content</h1>", encoding="utf-8")

    sensitive_response = client.get("/api/fs/stream", params={"path": str(sensitive)})
    unsupported_response = client.get("/api/fs/stream", params={"path": str(unsupported)})

    assert sensitive_response.status_code == 403
    assert unsupported_response.status_code == 415


def test_fs_stream_requires_auth(tmp_path: Path):
    client = TestClient(web_server.app)
    target = tmp_path / "movie.mp4"
    target.write_bytes(b"video")

    response = client.get("/api/fs/stream", params={"path": str(target)})
    query_token_response = client.get(
        "/api/fs/stream",
        params={"path": str(target), "token": web_server._SESSION_TOKEN},
    )

    assert response.status_code == 401
    assert query_token_response.status_code == 401
