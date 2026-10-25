"""WA-5: a Webapp file route that names a session resolves ``/workspace`` in THAT session's
per-workspace Docker sandbox (#143655's mapping), and refuses a session it cannot see."""

from pathlib import Path

import pytest
from fastapi import HTTPException

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client():
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


def _workspace_sandbox(sandbox: Path, cwd: Path) -> Path:
    from tools.environments.path_utils import sanitize_task_id_for_path
    from tools.terminal_tool import workspace_key_for

    name = sanitize_task_id_for_path(f"default:ws:{workspace_key_for(str(cwd))}")
    workspace = sandbox / "docker" / name / "workspace"
    workspace.mkdir(parents=True)
    return workspace


@pytest.fixture
def sandboxes(monkeypatch, tmp_path):
    """A persistent Docker backend with three ``/workspace`` copies: this session's, another
    session's, and the shared profile sandbox. ``sess-ws`` belongs to the default profile,
    ``sess-work`` to the ``work`` profile, so the default profile's lookup cannot see it."""
    project, other = tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    sandbox = tmp_path / "sandboxes"
    own = _workspace_sandbox(sandbox, project)
    foreign = _workspace_sandbox(sandbox, other)
    shared = sandbox / "docker" / "default" / "workspace"
    shared.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    for root, label in ((own, "session"), (foreign, "foreign"), (shared, "shared")):
        (root / "clip.mp4").write_bytes(f"{label}-clip".encode())
        (root / "note.txt").write_text(f"{label} note", encoding="utf-8")
    (outside / "secret.mp4").write_bytes(b"outside-clip")
    (outside / "secret.txt").write_text("outside note", encoding="utf-8")
    (own / "escape.mp4").symlink_to(outside / "secret.mp4")
    (own / "escape.txt").symlink_to(outside / "secret.txt")
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", raising=False)
    rows = {
        "sess-ws": ("default", str(project)),
        "sess-other": ("default", str(other)),
        "sess-work": ("work", str(project)),
    }

    async def detail(session_id, profile=None):
        # The real lookup reads only the requested profile's state.db.
        row = rows.get(session_id)
        if row is None or row[0] != (profile or "default"):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"cwd": row[1], "id": session_id}

    monkeypatch.setattr("hermes_cli.web_routers.sessions.get_session_detail", detail)
    return {"own": own, "foreign": foreign, "shared": shared, "outside": outside}


# Part 1: media streams


def test_stream_plays_the_sessions_own_workspace_clip(client, sandboxes):
    params = {"path": "/workspace/clip.mp4", "session_id": "sess-ws"}

    full = client.get("/api/files/stream", params=params)
    partial = client.get("/api/files/stream", params=params, headers={"Range": "bytes=0-6"})
    other = client.get("/api/files/stream", params={**params, "session_id": "sess-other"})

    assert (full.status_code, full.content) == (200, b"session-clip")
    assert (partial.status_code, partial.content) == (206, b"session")
    assert (other.status_code, other.content) == (200, b"foreign-clip")
    # No session keeps the shared profile sandbox, as before.
    bare = client.get("/api/files/stream", params={"path": "/workspace/clip.mp4"})
    assert (bare.status_code, bare.content) == (200, b"shared-clip")


def test_a_stream_ticket_for_a_session_plays_that_sessions_clip(client, sandboxes, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    query = {"path": "/workspace/clip.mp4", "session_id": "sess-ws"}
    minted = client.post("/api/files/ticket", json={"route": "stream", **query})
    assert minted.status_code == 200, minted.text
    del client.headers[web_server._SESSION_HEADER_NAME]

    played = client.get("/api/files/stream", params={**query, "ticket": minted.json()["ticket"]})

    assert (played.status_code, played.content) == (200, b"session-clip")


@pytest.mark.parametrize("session_id", ["sess-work", "missing-session", " "])
def test_stream_refuses_a_session_this_profile_does_not_own(client, sandboxes, monkeypatch, session_id):
    params = {"path": "/workspace/clip.mp4", "session_id": session_id}

    assert client.get("/api/files/stream", params=params).status_code == 404
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    assert client.post("/api/files/ticket", json={"route": "stream", **params}).status_code == 404


@pytest.mark.parametrize("path", ["/workspace/escape.mp4", "/workspace/../outside/secret.mp4"])
def test_stream_refuses_a_clip_outside_the_sessions_workspace(client, sandboxes, path):
    response = client.get("/api/files/stream", params={"path": path, "session_id": "sess-ws"})

    assert response.status_code in (400, 403, 404), response.status_code
    assert b"outside-clip" not in response.content


# Part 2: file: paths and text previews


@pytest.mark.parametrize("path", ["/workspace/note.txt", "file:///workspace/note.txt"])
def test_read_text_previews_the_sessions_own_workspace_file(client, sandboxes, path):
    def read(**params):
        return client.get("/api/fs/read-text", params={"path": path, **params})

    own = read(session_id="sess-ws")
    other = read(session_id="sess-other")
    bare = read()

    assert own.status_code == 200, own.text
    assert own.json()["text"] == "session note"
    assert own.json()["path"] == str(sandboxes["own"] / "note.txt")
    assert other.json()["text"] == "foreign note"
    # No session keeps the shared profile sandbox, the same answer for both spellings.
    assert (bare.status_code, bare.json()["text"]) == (200, "shared note")


def test_a_file_url_clip_streams_from_the_sessions_sandbox(client, sandboxes, monkeypatch):
    query = {"path": "file:///workspace/clip.mp4", "session_id": "sess-ws"}

    direct = client.get("/api/files/stream", params=query)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    minted = client.post("/api/files/ticket", json={"route": "stream", **query})
    assert minted.status_code == 200, minted.text
    del client.headers[web_server._SESSION_HEADER_NAME]
    ticketed = client.get("/api/files/stream", params={**query, "ticket": minted.json()["ticket"]})

    assert (direct.status_code, direct.content) == (200, b"session-clip")
    assert (ticketed.status_code, ticketed.content) == (200, b"session-clip")


@pytest.mark.parametrize("session_id", ["sess-work", "missing-session", " "])
def test_read_text_refuses_a_session_this_profile_does_not_own(client, sandboxes, session_id):
    for path in ("/workspace/note.txt", "file:///workspace/note.txt"):
        response = client.get("/api/fs/read-text", params={"path": path, "session_id": session_id})
        assert response.status_code == 404, (path, response.text)


@pytest.mark.parametrize("path", [
    "/workspace/escape.txt",
    "file:///workspace/escape.txt",
    "/workspace/../outside/secret.txt",
    "file:///workspace/sub/../escape.txt",
])
def test_read_text_refuses_a_file_outside_the_sessions_workspace(client, sandboxes, path):
    response = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})

    assert response.status_code in (400, 403, 404), response.text
    assert "outside note" not in response.text
