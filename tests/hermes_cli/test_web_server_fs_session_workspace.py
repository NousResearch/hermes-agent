"""WA-5: a Webapp file route that names a session resolves ``/workspace`` in THAT session's
Docker sandbox (#143655's mapping), and refuses a session it cannot see. A session has its own
per-workspace sandbox only with ``docker_mount_cwd_to_workspace`` on; without it every session
shares the profile sandbox, and a ``*_ws_*`` copy an older build left behind is not its own."""

import shutil
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
    session's, and the shared profile sandbox. The cwd mount is off, so the per-workspace copies
    are leftovers until a test turns it on (:func:`_mount_cwd`). ``sess-ws`` belongs to the default profile,
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
        "sess-container": ("default", "/workspace"),  # no host workspace binding
    }

    lookups = []

    async def detail(session_id, profile=None):
        # The real lookup reads only the requested profile's state.db.
        lookups.append(session_id)
        row = rows.get(session_id)
        if row is None or row[0] != (profile or "default"):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"cwd": row[1], "id": session_id}

    monkeypatch.setattr("hermes_cli.web_routers.sessions.get_session_detail", detail)
    return {"own": own, "foreign": foreign, "shared": shared, "outside": outside, "lookups": lookups}


def test_each_file_route_looks_its_session_up_once(client, sandboxes, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    clip = {"path": "file:///workspace/clip.mp4", "session_id": "sess-ws"}
    note = {"path": "/workspace/note.txt", "session_id": "sess-ws"}
    requests = [
        lambda: client.get("/api/files/stream", params=clip),
        lambda: client.post("/api/files/ticket", json={"route": "stream", **clip}),
        lambda: client.post("/api/files/ticket", json={"route": "download", **note}),
        lambda: client.get("/api/fs/read-text", params=note),
        lambda: client.get("/api/fs/read-data-url", params=note),
        lambda: client.get("/api/fs/download", params=note),
        lambda: client.post("/api/fs/write-text", json={**note, "content": "edited"}),
    ]

    for send in requests:
        sandboxes["lookups"].clear()
        response = send()
        assert response.status_code == 200, response.text
        assert sandboxes["lookups"] == ["sess-ws"], response.request.url


# Part 1: media streams


def _mount_cwd(monkeypatch):
    """Per-workspace sandboxes exist only with the cwd mount on. The session folders here hold no
    files, so ``/workspace`` falls through to the sandbox copies whose ordering these tests pin;
    Part 4 covers a real mounted folder."""
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")


def test_stream_plays_the_sessions_own_workspace_clip(client, sandboxes, monkeypatch):
    # No session keeps the shared profile sandbox, as before.
    bare = client.get("/api/files/stream", params={"path": "/workspace/clip.mp4"})
    assert (bare.status_code, bare.content) == (200, b"shared-clip")
    _mount_cwd(monkeypatch)
    params = {"path": "/workspace/clip.mp4", "session_id": "sess-ws"}

    full = client.get("/api/files/stream", params=params)
    partial = client.get("/api/files/stream", params=params, headers={"Range": "bytes=0-6"})
    other = client.get("/api/files/stream", params={**params, "session_id": "sess-other"})

    assert (full.status_code, full.content) == (200, b"session-clip")
    assert (partial.status_code, partial.content) == (206, b"session")
    assert (other.status_code, other.content) == (200, b"foreign-clip")


def test_a_stream_ticket_for_a_session_plays_that_sessions_clip(client, sandboxes, monkeypatch):
    _mount_cwd(monkeypatch)
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
def test_read_text_previews_the_sessions_own_workspace_file(client, sandboxes, monkeypatch, path):
    def read(**params):
        return client.get("/api/fs/read-text", params={"path": path, **params})

    bare = read()
    _mount_cwd(monkeypatch)
    own = read(session_id="sess-ws")
    other = read(session_id="sess-other")

    assert own.status_code == 200, own.text
    assert own.json()["text"] == "session note"
    assert own.json()["path"] == str(sandboxes["own"] / "note.txt")
    assert other.json()["text"] == "foreign note"
    # No session keeps the shared profile sandbox, the same answer for both spellings.
    assert (bare.status_code, bare.json()["text"]) == (200, "shared note")


def test_a_file_url_clip_streams_from_the_sessions_sandbox(client, sandboxes, monkeypatch):
    _mount_cwd(monkeypatch)
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


# Part 3: Save


def _notes(sandboxes):
    return {name: (sandboxes[name] / "note.txt").read_text(encoding="utf-8") for name in ("own", "foreign", "shared")}


@pytest.mark.parametrize("path", ["/workspace/note.txt", "file:///workspace/note.txt"])
def test_save_writes_the_file_the_preview_read(client, sandboxes, monkeypatch, path):
    _mount_cwd(monkeypatch)
    read = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})
    saved = client.post("/api/fs/write-text", json={"path": path, "content": "edited", "session_id": "sess-ws"})

    assert saved.status_code == 200, saved.text
    assert saved.json()["path"] == read.json()["path"] == str(sandboxes["own"] / "note.txt")
    assert _notes(sandboxes) == {"own": "edited", "foreign": "foreign note", "shared": "shared note"}


@pytest.mark.parametrize("path", ["/workspace/note.txt", "file:///workspace/note.txt"])
def test_save_never_falls_back_to_the_shared_copy_once_the_sessions_copy_is_gone(
    client, sandboxes, monkeypatch, path,
):
    _mount_cwd(monkeypatch)
    (sandboxes["shared"] / "note.txt").write_text("session note", encoding="utf-8")  # the same text
    (sandboxes["own"] / "note.txt").unlink()

    # A read still falls back, so the editor's stale-file check sees the text it opened.
    read = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})
    saved = client.post("/api/fs/write-text", json={"path": path, "content": "edited", "session_id": "sess-ws"})

    assert (read.status_code, read.json()["text"]) == (200, "session note")
    assert (saved.status_code, saved.json()) == (409, {"detail": "File is no longer in this session's workspace"})
    assert (sandboxes["shared"] / "note.txt").read_text(encoding="utf-8") == "session note"
    assert not (sandboxes["own"] / "note.txt").exists()


@pytest.mark.parametrize("path", ["/workspace/note.txt", "file:///workspace/note.txt"])
def test_save_never_falls_back_to_the_shared_copy_once_the_sessions_sandbox_is_gone(
    client, sandboxes, monkeypatch, path,
):
    _mount_cwd(monkeypatch)
    read = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})
    assert read.json()["text"] == "session note"
    shutil.rmtree(sandboxes["own"].parent)  # the session's whole sandbox, not just its file

    reread = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})
    saved = client.post("/api/fs/write-text", json={"path": path, "content": "edited", "session_id": "sess-ws"})

    assert (reread.status_code, reread.json()["text"]) == (200, "shared note")  # reads still fall back
    assert (saved.status_code, saved.json()) == (409, {"detail": "File is no longer in this session's workspace"})
    assert (sandboxes["shared"] / "note.txt").read_text(encoding="utf-8") == "shared note"
    assert not sandboxes["own"].parent.exists()


def test_save_without_a_session_writes_where_a_bare_read_reads(client, sandboxes):
    read = client.get("/api/fs/read-text", params={"path": "/workspace/note.txt"})
    saved = client.post("/api/fs/write-text", json={"path": "/workspace/note.txt", "content": "edited"})

    assert saved.status_code == 200, saved.text
    assert saved.json()["path"] == read.json()["path"] == str(sandboxes["shared"] / "note.txt")
    assert _notes(sandboxes) == {"own": "session note", "foreign": "foreign note", "shared": "edited"}


@pytest.mark.parametrize("flag", [None, "on"])  # "on" is not a value the terminal mounts on
@pytest.mark.parametrize("path", ["/workspace/note.txt", "file:///workspace/note.txt"])
def test_without_the_cwd_mount_a_session_reads_and_saves_the_profile_sandbox(
    client, sandboxes, monkeypatch, path, flag,
):
    """No cwd mount: the session's container is the profile's, so a leftover ``*_ws_*`` copy is
    neither previewed nor counted as its own (a Save there would be refused or lost)."""
    if flag:
        monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", flag)
    read = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})
    saved = client.post("/api/fs/write-text", json={"path": path, "content": "edited", "session_id": "sess-ws"})

    assert (read.status_code, read.json()["text"]) == (200, "shared note"), read.text
    assert saved.status_code == 200, saved.text
    assert saved.json()["path"] == read.json()["path"] == str(sandboxes["shared"] / "note.txt")
    assert _notes(sandboxes) == {"own": "session note", "foreign": "foreign note", "shared": "edited"}


@pytest.mark.parametrize("session_id", ["sess-work", "missing-session", " "])
def test_save_refuses_a_session_this_profile_does_not_own(client, sandboxes, session_id):
    for path in ("/workspace/note.txt", "file:///workspace/note.txt"):
        response = client.post(
            "/api/fs/write-text", json={"path": path, "content": "edited", "session_id": session_id},
        )
        assert response.status_code == 404, (path, response.text)
    assert _notes(sandboxes) == {"own": "session note", "foreign": "foreign note", "shared": "shared note"}


# Part 4: docker_mount_cwd_to_workspace


@pytest.fixture
def cwd_mount(sandboxes, monkeypatch, tmp_path):
    """The cwd mount on: the profile's ``terminal.cwd`` is P (``projmount``), while ``sess-ws`` is
    bound to its own workspace W (``project``), which its container mounts at ``/workspace``."""
    profile_cwd, workspace = tmp_path / "projmount", tmp_path / "project"
    profile_cwd.mkdir()
    (profile_cwd / "note.txt").write_text("profile note", encoding="utf-8")
    (workspace / "note.txt").write_text("workspace note", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    monkeypatch.setenv("TERMINAL_CWD", str(profile_cwd))
    return {**sandboxes, "profile_cwd": profile_cwd, "workspace": workspace}


@pytest.fixture(params=["session row", "live binding"])
def bound(request, cwd_mount):
    """``sess-ws``'s workspace from its stored cwd, or from the binding its running turn registered."""
    if request.param == "session row":
        yield cwd_mount
        return
    from gateway.session_policy import unbind_turn_workspace
    from tools.terminal_tool import register_task_env_overrides, workspace_key_for

    workspace = str(cwd_mount["workspace"])
    register_task_env_overrides("sess-ws", {
        "cwd": workspace, "cwd_source": "session", "workspace_key": workspace_key_for(workspace),
    })
    try:
        yield cwd_mount
    finally:
        unbind_turn_workspace("sess-ws")


def _mount_notes(roots):
    return {name: (roots[name] / "note.txt").read_text(encoding="utf-8") for name in ("workspace", "profile_cwd")}


@pytest.mark.parametrize("path", ["/workspace/note.txt", "file:///workspace/note.txt"])
def test_a_cwd_mounted_session_reads_and_saves_its_own_workspace(client, bound, path):
    read = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})
    saved = client.post("/api/fs/write-text", json={"path": path, "content": "edited", "session_id": "sess-ws"})

    assert (read.status_code, read.json()["text"]) == (200, "workspace note"), read.text
    assert saved.status_code == 200, saved.text
    assert saved.json()["path"] == read.json()["path"] == str(bound["workspace"] / "note.txt")
    assert _mount_notes(bound) == {"workspace": "edited", "profile_cwd": "profile note"}


@pytest.mark.parametrize("path", ["/workspace/note.txt", "file:///workspace/note.txt"])
def test_a_session_on_a_shared_container_never_reads_or_saves_its_unmounted_workspace(
    client, bound, monkeypatch, path,
):
    """``terminal.docker_shared_container_key``: every session reuses ONE ``shared:<key>`` container,
    which mounts the cwd of the session that created it (P), so ``sess-ws``'s own W is not its
    ``/workspace``. Nothing records P, so reads keep the profile cwd and Save refuses."""
    from gateway.session_policy import unbind_turn_workspace
    from tools.terminal_tool import _resolve_container_task_id, register_task_env_overrides, workspace_key_for

    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team")
    first = str(bound["profile_cwd"])
    register_task_env_overrides("sess-first", {
        "cwd": first, "cwd_source": "session", "workspace_key": workspace_key_for(first),
    })
    try:
        assert _resolve_container_task_id("sess-first") == _resolve_container_task_id("sess-ws") == "shared:team"
        read = client.get("/api/fs/read-text", params={"path": path, "session_id": "sess-ws"})
        saved = client.post("/api/fs/write-text", json={"path": path, "content": "edited", "session_id": "sess-ws"})
    finally:
        unbind_turn_workspace("sess-first")

    assert (read.status_code, read.json()["text"]) == (200, "profile note"), read.text
    assert read.json()["path"] == str(bound["profile_cwd"] / "note.txt")
    assert (saved.status_code, saved.json()) == (409, {"detail": "File is no longer in this session's workspace"})
    assert _mount_notes(bound) == {"workspace": "workspace note", "profile_cwd": "profile note"}


def test_a_cwd_mounted_session_without_a_workspace_binding_keeps_the_profile_cwd(client, cwd_mount):
    for params in ({"session_id": "sess-container"}, {}):
        read = client.get("/api/fs/read-text", params={"path": "/workspace/note.txt", **params})
        assert (read.status_code, read.json()["text"]) == (200, "profile note"), (params, read.text)
        assert read.json()["path"] == str(cwd_mount["profile_cwd"] / "note.txt")


@pytest.mark.parametrize("session_id", ["sess-work", "missing-session", " "])
def test_a_cwd_mounted_session_this_profile_does_not_own_is_refused(client, cwd_mount, session_id):
    note = {"path": "/workspace/note.txt", "session_id": session_id}

    read = client.get("/api/fs/read-text", params=note)
    saved = client.post("/api/fs/write-text", json={**note, "content": "edited"})

    assert (read.status_code, saved.status_code) == (404, 404)
    assert _mount_notes(cwd_mount) == {"workspace": "workspace note", "profile_cwd": "profile note"}


@pytest.mark.parametrize("path", [
    "/workspace/escape.txt",
    "file:///workspace/escape.txt",
    "/workspace/../outside/secret.txt",
])
def test_save_refuses_a_file_outside_the_sessions_workspace(client, sandboxes, path):
    response = client.post(
        "/api/fs/write-text", json={"path": path, "content": "edited", "session_id": "sess-ws"},
    )

    assert response.status_code in (400, 403, 404), response.text
    assert (sandboxes["outside"] / "secret.txt").read_text(encoding="utf-8") == "outside note"
    assert (sandboxes["own"] / "escape.txt").is_symlink()
