"""Document-scoped image reads do not inherit attachment-only host exceptions."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.web_routers import files as file_routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    session = TestClient(web_server.app)
    session.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(params=["cache/images", "uploads"])
def host_image(request, monkeypatch, tmp_path):
    home = tmp_path / "profile-home"
    folder = home / request.param
    folder.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = folder / "report.md"
    source.write_text("# Report", encoding="utf-8")
    image = folder / "web-chart.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert file_routes._is_host_file(str(image), None)
    return source, image


def test_scoped_cached_image_keeps_the_ssh_workspace_boundary(client, monkeypatch, host_image):
    source, image = host_image

    class RemoteWorkspace:
        def read_bytes(self, *_args, **_kwargs):
            pytest.fail("unsupported scoped images must not invoke the remote reader")

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: RemoteWorkspace())
    ordinary = client.get("/api/fs/read-data-url", params={"path": str(image)})
    assert ordinary.status_code == 200, ordinary.text

    scoped = client.get(
        "/api/fs/read-data-url",
        params={"path": str(image), "relative_to_file": str(source)},
    )
    assert scoped.status_code == 501, scoped.text


def test_scoped_cached_image_keeps_the_hosted_read_boundary(
    client, monkeypatch, tmp_path, host_image,
):
    source, image = host_image
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(managed))
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: None)
    ordinary = client.get("/api/fs/read-data-url", params={"path": str(image)})
    assert ordinary.status_code == 200, ordinary.text

    scoped = client.get(
        "/api/fs/read-data-url",
        params={"path": str(image), "relative_to_file": str(source)},
    )
    assert scoped.status_code == 403, scoped.text


@pytest.mark.parametrize("limit,expected_status", [(4, 413), (16, 200)])
def test_scoped_session_read_keeps_limits_and_one_owner_lookup(
    client, monkeypatch, host_image, limit, expected_status,
):
    from hermes_cli.web_routers import sessions

    source, image = host_image
    lookups = []

    async def session_detail(session_id, profile):
        lookups.append((session_id, profile))
        return {"cwd": str(source.parent)}

    monkeypatch.setattr(sessions, "get_session_detail", session_detail)
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: None)
    response = client.get(
        "/api/fs/read-data-url",
        params={"path": str(image), "relative_to_file": str(source),
                "session_id": "session-a", "max_bytes": limit},
    )
    assert response.status_code == expected_status, response.text
    assert lookups == [("session-a", None)]


def test_scoped_session_is_checked_before_selecting_an_ssh_backend(client, monkeypatch, host_image):
    from fastapi import HTTPException
    from hermes_cli.web_routers import sessions

    source, image = host_image

    async def missing_session(_session_id, _profile):
        raise HTTPException(status_code=404, detail="Session not found")

    def unexpected_backend(_profile):
        pytest.fail("an invalid session must be rejected before selecting a backend")

    monkeypatch.setattr(sessions, "get_session_detail", missing_session)
    monkeypatch.setattr(file_routes, "_fs_backend", unexpected_backend)
    response = client.get(
        "/api/fs/read-data-url",
        params={"path": str(image), "relative_to_file": str(source), "session_id": "missing"},
    )
    assert response.status_code == 404, response.text
