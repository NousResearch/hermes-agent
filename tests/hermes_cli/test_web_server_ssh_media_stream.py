"""WA-5 part 4: an SSH profile's audio/video streams from the SSH workspace with HTTP Range, behind
the same ticket, profile and file checks as the host path. Fake SSH adapters only."""

import shlex
from pathlib import Path

import pytest

from hermes_cli import ssh_workspace_fs, web_server
from hermes_cli.web_routers import files as file_routes

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient

CLIP = "/srv/media/clip one.mp4"
BYTES = b"0123456789abcdef"
UNAVAILABLE = (503, {"detail": "SSH workspace is unavailable"})


class FakeSshWorkspace:
    """The SSH machine: a few files, one link, and a record of every byte range read."""

    def __init__(self):
        self.files = {CLIP: BYTES, "/srv/media/notes.txt": b"notes"}
        self.links = {"/srv/media/linked.mp4": "/home/dev/.hermes/mcp-tokens/token.mp4"}
        self.reads = []
        self.writes = []

    def read_bytes(self, path, max_bytes=None, read_limit=None):
        data = self.files[path]
        return data[:read_limit] if read_limit else data, len(data), path

    def write_text(self, path, text, max_bytes=None):
        self.writes.append((path, text))
        return path, len(text.encode("utf-8"))

    def inspect_file(self, path):
        target = self.links.get(path, path)
        if target not in self.files and target not in self.links.values():
            raise ssh_workspace_fs.SshWorkspaceFsError("ENOENT", f"{path} is not on the SSH machine")
        return target, len(self.files.get(target, b"secret"))

    def stream_file(self, path, chunk_size=64 * 1024, *, offset=0, length=None):
        self.reads.append((path, offset, length))
        data = self.files[path][offset:None if length is None else offset + length]
        yield data[:3]
        yield data[3:]


@pytest.fixture
def client():
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        test_client.close()
        if previous is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


@pytest.fixture
def ssh(monkeypatch, tmp_path):
    """A default profile on this host and a ``remote-dev`` profile whose workspace is the fake
    SSH machine; the browser's ticket routes are on (the Webapp)."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "remote-dev"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    workspace = FakeSshWorkspace()
    monkeypatch.setattr(
        file_routes, "_fs_backend", lambda profile=None: workspace if profile == "remote-dev" else None,
    )
    return profile, workspace


def _stream(client, *, headers=None, method="GET", **params):
    return client.request(method, "/api/files/stream", params={"path": CLIP, "profile": "remote-dev", **params},
                          headers=headers or {})


def test_ssh_stream_serves_the_whole_clip_inline(client, ssh):
    _profile, workspace = ssh

    full = _stream(client)
    head = _stream(client, method="HEAD")

    assert (full.status_code, full.content) == (200, BYTES)
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["content-length"] == str(len(BYTES))
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-disposition"].startswith("inline;")
    assert full.headers["x-content-type-options"] == "nosniff"
    assert (head.status_code, head.content, head.headers["content-length"]) == (200, b"", str(len(BYTES)))
    assert workspace.reads == [(CLIP, 0, None)]  # the HEAD read nothing


@pytest.mark.parametrize(("header", "first", "last"), [
    ("bytes=2-5", 2, 5),
    ("bytes=10-", 10, 15),
    ("bytes=-4", 12, 15),
    ("bytes=14-99", 14, 15),
])
def test_ssh_stream_answers_a_range_with_206(client, ssh, header, first, last):
    _profile, workspace = ssh

    partial = _stream(client, headers={"Range": header})

    assert (partial.status_code, partial.content) == (206, BYTES[first:last + 1])
    assert partial.headers["content-range"] == f"bytes {first}-{last}/{len(BYTES)}"
    assert partial.headers["content-length"] == str(last - first + 1)
    assert partial.headers["accept-ranges"] == "bytes"
    assert workspace.reads == [(CLIP, first, last - first + 1)]


@pytest.mark.parametrize("header", ["bytes=16-", "bytes=99-120", "bytes=-0"])
def test_ssh_stream_answers_an_unsatisfiable_range_with_416(client, ssh, header):
    _profile, workspace = ssh

    response = _stream(client, headers={"Range": header})

    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(BYTES)}"
    assert workspace.reads == []


LONG = "9" * 5000  # past Python's int() digit limit


@pytest.mark.parametrize(("header", "first", "last"), [
    (f"bytes=2-{LONG}", 2, 15),  # a last byte past the end is the end
    (f"bytes=-{LONG}", 0, 15),  # a suffix longer than the file is all of it
    (f"bytes={'0' * 5000}3-{'0' * 5000}5", 3, 5),  # leading zeros are still a small number
])
def test_ssh_stream_reads_an_oversized_range_number_without_converting_it(client, ssh, header, first, last):
    _profile, workspace = ssh

    partial = _stream(client, headers={"Range": header})

    assert (partial.status_code, partial.content) == (206, BYTES[first:last + 1])
    assert partial.headers["content-range"] == f"bytes {first}-{last}/{len(BYTES)}"
    assert workspace.reads == [(CLIP, first, last - first + 1)]


@pytest.mark.parametrize("header", [f"bytes={LONG}-", f"bytes={LONG}-{LONG}"])
def test_ssh_stream_answers_an_oversized_first_byte_with_416(client, ssh, header):
    _profile, workspace = ssh

    response = _stream(client, headers={"Range": header})

    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(BYTES)}"
    assert workspace.reads == []


def test_ssh_stream_ignores_an_oversized_range_that_ends_before_it_starts(client, ssh):
    _profile, workspace = ssh

    whole = _stream(client, headers={"Range": f"bytes={LONG}-3"})

    assert (whole.status_code, whole.content) == (200, BYTES)
    assert workspace.reads == [(CLIP, 0, None)]


@pytest.mark.parametrize(("path", "status"), [
    ("/srv/media/linked.mp4", 403),  # resolves into a credential folder on the SSH machine
    ("/srv/media/notes.txt", 415),
    ("/srv/media/missing.mp4", 404),
])
def test_ssh_stream_refuses_a_file_download_or_media_rules_refuse(client, ssh, path, status):
    _profile, workspace = ssh

    direct = _stream(client, path=path)
    minted = client.post("/api/files/ticket", json={"route": "stream", "path": path, "profile": "remote-dev"})

    assert direct.status_code == status, direct.text
    assert minted.status_code == status, minted.text
    assert workspace.reads == []


def test_an_ssh_stream_ticket_plays_ranges_and_only_its_own_profile_and_path(client, ssh):
    from hermes_constants import mark_named_profile_deleted

    profile, workspace = ssh
    query = {"path": CLIP, "profile": "remote-dev"}
    minted = client.post("/api/files/ticket", json={"route": "stream", **query})
    assert minted.status_code == 200, minted.text
    ticket = minted.json()["ticket"]
    player = TestClient(web_server.app)  # a media element sends no session header
    try:
        for _ in range(2):  # a stream ticket serves every Range request of one source
            partial = player.get("/api/files/stream", params={**query, "ticket": ticket},
                                 headers={"Range": "bytes=1-3"})
            assert (partial.status_code, partial.content) == (206, BYTES[1:4])
        for foreign in (
            {**query, "profile": "default"},
            {"path": CLIP},
            {**query, "path": "/srv/media/other.mp4"},
        ):
            assert player.get("/api/files/stream", params={**foreign, "ticket": ticket}).status_code == 401, foreign
        # The default profile's own ticket names no SSH workspace: this host has no such clip.
        assert client.post("/api/files/ticket", json={"route": "stream", "path": CLIP}).status_code == 404

        mark_named_profile_deleted(profile)
        gone = player.get("/api/files/stream", params={**query, "ticket": ticket})
        assert gone.status_code in (401, 403, 404), gone.text
        assert workspace.reads == [(CLIP, 1, 3), (CLIP, 1, 3)]
    finally:
        player.close()


def test_a_host_staged_clip_of_an_ssh_profile_streams_from_this_host(client, ssh):
    _profile, workspace = ssh
    staged = client.post("/api/chat/file-upload", params={"profile": "remote-dev"},
                         files={"file": ("clip.mp4", b"host clip", "video/mp4")})
    assert staged.status_code == 200, staged.text
    query = {"path": staged.json()["path"], "profile": "remote-dev"}

    direct = client.get("/api/files/stream", params=query, headers={"Range": "bytes=0-3"})
    minted = client.post("/api/files/ticket", json={"route": "stream", **query})
    assert minted.status_code == 200, minted.text
    del client.headers[web_server._SESSION_HEADER_NAME]
    ticketed = client.get("/api/files/stream", params={**query, "ticket": minted.json()["ticket"]})

    assert (direct.status_code, direct.content) == (206, b"host")
    assert (ticketed.status_code, ticketed.content) == (200, b"host clip")
    assert workspace.reads == []


@pytest.mark.parametrize("ssh_down", [False, True])
def test_a_host_staged_clip_spelled_as_a_file_url_streams_from_this_host(client, ssh, monkeypatch, ssh_down):
    from fastapi import HTTPException

    _profile, workspace = ssh
    staged = client.post("/api/chat/file-upload", params={"profile": "remote-dev"},
                         files={"file": ("clip.mp4", b"host clip", "video/mp4")})
    assert staged.status_code == 200, staged.text
    if ssh_down:
        def down(profile=None):
            raise HTTPException(status_code=503, detail="SSH workspace is unavailable")

        monkeypatch.setattr(file_routes, "_fs_backend", down)
    query = {"path": Path(staged.json()["path"]).as_uri(), "profile": "remote-dev"}

    direct = client.get("/api/files/stream", params=query, headers={"Range": "bytes=0-3"})
    data_url = client.get("/api/fs/read-data-url", params=query)
    download = client.get("/api/fs/download", params=query)
    minted = client.post("/api/files/ticket", json={"route": "stream", **query})
    assert minted.status_code == 200, minted.text
    del client.headers[web_server._SESSION_HEADER_NAME]
    ticketed = client.get("/api/files/stream", params={**query, "ticket": minted.json()["ticket"]})

    assert (direct.status_code, direct.content) == (206, b"host"), direct.text
    assert data_url.json() == {"dataUrl": "data:video/mp4;base64,aG9zdCBjbGlw"}
    assert (download.status_code, download.content) == (200, b"host clip")
    assert (ticketed.status_code, ticketed.content) == (200, b"host clip")
    assert workspace.reads == []


def test_a_file_url_off_this_hosts_staging_still_goes_to_the_ssh_workspace(client, ssh, monkeypatch):
    _profile, workspace = ssh
    asked = []
    monkeypatch.setattr(workspace, "inspect_file", lambda path: asked.append(path) or (CLIP, len(BYTES)))
    staged = client.post("/api/chat/file-upload", params={"profile": "remote-dev"},
                         files={"file": ("clip.mp4", b"host clip", "video/mp4")})
    staged_path = Path(staged.json()["path"])
    beside = staged_path.parent.parent / "clip.mp4"  # not directly in the staging folder
    beside.write_bytes(b"host file")
    encoded = staged_path.as_uri().replace(f"/{staged_path.name}", f"%2f{staged_path.name}")

    for url in (beside.as_uri(), encoded):
        response = _stream(client, path=url)
        assert (response.status_code, response.content) == (200, BYTES), url
    # Neither is a host-staged file, so the SSH workspace was asked for both.
    assert len(asked) == 2


@pytest.mark.parametrize("profile", ["remote-dev", "default"])
def test_a_file_url_with_an_encoded_nul_is_an_invalid_path(client, ssh, monkeypatch, profile):
    _profile, workspace = ssh

    def refuse(path, *args, **kwargs):  # as the real adapter's path check refuses every file: URL
        raise ssh_workspace_fs.SshWorkspaceFsError("EINVAL", "Invalid path")

    monkeypatch.setattr(workspace, "read_bytes", refuse)
    monkeypatch.setattr(workspace, "inspect_file", refuse)
    query = {"path": "file:///tmp/clip%00.mp4", "profile": profile}

    responses = [
        client.get("/api/files/stream", params=query),
        client.get("/api/fs/read-data-url", params=query),
        client.get("/api/fs/download", params=query),
    ]

    for response in responses:
        assert (response.status_code, response.json()) == (400, {"detail": "Invalid path"}), response.request.url
    assert workspace.reads == []


def test_a_down_ssh_host_is_unavailable_to_the_stream(client, ssh, monkeypatch):
    from tools.environments.base import EnvironmentConnectionError

    class DownHost:
        def __init__(self, **_options):
            raise EnvironmentConnectionError("SSH connection failed: Connection refused")

    monkeypatch.undo()
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(ssh_workspace_fs, "SSHEnvironment", DownHost)
    monkeypatch.setattr(ssh_workspace_fs, "_BACKENDS", {})
    monkeypatch.setattr(
        file_routes, "load_config",
        lambda: {"terminal": {"backend": "ssh", "ssh_host": "127.0.0.1", "ssh_user": "dev", "ssh_port": 1}},
    )
    monkeypatch.setattr(file_routes, "load_env", dict)

    direct = client.get("/api/files/stream", params={"path": CLIP}, headers={"Range": "bytes=0-3"})
    minted = client.post("/api/files/ticket", json={"route": "stream", "path": CLIP})

    assert (direct.status_code, direct.json()) == UNAVAILABLE
    assert (minted.status_code, minted.json()) == UNAVAILABLE


# A session_id is checked against the profile before the SSH backend is chosen.


@pytest.fixture
def sessions(monkeypatch):
    """``ssh-sess`` belongs to the SSH profile, ``host-sess`` to the default one; the real lookup
    reads only the requested profile's state.db."""
    from fastapi import HTTPException

    rows = {"ssh-sess": "remote-dev", "host-sess": "default"}

    async def detail(session_id, profile=None):
        if rows.get(session_id) != (profile or "default"):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"cwd": "/srv", "id": session_id}

    monkeypatch.setattr("hermes_cli.web_routers.sessions.get_session_detail", detail)
    return rows


NOT_OWNED = ["host-sess", "missing-session", " ", ""]


@pytest.mark.parametrize("session_id", NOT_OWNED)
def test_ssh_stream_and_tickets_refuse_a_session_the_profile_does_not_own(client, ssh, sessions, session_id):
    _profile, workspace = ssh
    query = {"path": CLIP, "profile": "remote-dev", "session_id": session_id}

    direct = _stream(client, session_id=session_id, headers={"Range": "bytes=0-3"})
    head = _stream(client, method="HEAD", session_id=session_id)
    minted = client.post("/api/files/ticket", json={"route": "stream", **query})
    download = client.post("/api/files/ticket", json={"route": "download", **query})

    assert [r.status_code for r in (direct, head, minted, download)] == [404] * 4, minted.text
    assert workspace.reads == []


@pytest.mark.parametrize("session_id", NOT_OWNED)
def test_an_ssh_stream_ticket_never_plays_for_a_session_the_profile_does_not_own(client, ssh, sessions, session_id):
    _profile, workspace = ssh
    query = {"path": CLIP, "profile": "remote-dev"}
    player = TestClient(web_server.app)  # a media element sends no session header
    try:
        minted = client.post("/api/files/ticket", json={"route": "stream", **query, "session_id": session_id})
        assert minted.status_code == 404, minted.text
        bare = client.post("/api/files/ticket", json={"route": "stream", **query})
        assert bare.status_code == 200, bare.text

        redeemed = player.get(
            "/api/files/stream", params={**query, "session_id": session_id, "ticket": bare.json()["ticket"]},
        )

        assert redeemed.status_code in (401, 404), redeemed.text
        assert workspace.reads == []
    finally:
        player.close()


def test_an_ssh_stream_ticket_stops_playing_once_its_session_is_gone(client, ssh, sessions, monkeypatch):
    _profile, workspace = ssh
    query = {"path": CLIP, "profile": "remote-dev", "session_id": "ssh-sess"}
    minted = client.post("/api/files/ticket", json={"route": "stream", **query})
    assert minted.status_code == 200, minted.text
    ticketed = {**query, "ticket": minted.json()["ticket"]}
    player = TestClient(web_server.app)
    try:
        played = player.get("/api/files/stream", params=ticketed, headers={"Range": "bytes=0-3"})
        assert (played.status_code, played.content) == (206, BYTES[:4])

        monkeypatch.delitem(sessions, "ssh-sess")
        gone = player.get("/api/files/stream", params=ticketed, headers={"Range": "bytes=0-3"})

        assert gone.status_code == 404, gone.text
        assert workspace.reads == [(CLIP, 0, 4)]
    finally:
        player.close()


def test_an_ssh_profile_streams_reads_and_saves_for_its_own_session(client, ssh, sessions):
    _profile, workspace = ssh

    streamed = _stream(client, session_id="ssh-sess")
    read = client.get("/api/fs/read-text", params={"path": "/srv/media/notes.txt", "profile": "remote-dev",
                                                   "session_id": "ssh-sess"})
    saved = client.post("/api/fs/write-text", params={"profile": "remote-dev"},
                        json={"path": "/srv/media/notes.txt", "content": "edited", "session_id": "ssh-sess"})

    assert (streamed.status_code, streamed.content) == (200, BYTES)
    assert (read.status_code, read.json()["text"]) == (200, "notes")
    assert saved.status_code == 200, saved.text
    assert workspace.writes == [("/srv/media/notes.txt", "edited")]


@pytest.mark.parametrize("session_id", NOT_OWNED)
def test_ssh_reads_and_saves_refuse_a_session_the_profile_does_not_own(client, ssh, sessions, session_id):
    _profile, workspace = ssh
    note = {"path": "/srv/media/notes.txt", "profile": "remote-dev", "session_id": session_id}

    read = client.get("/api/fs/read-text", params=note)
    data_url = client.get("/api/fs/read-data-url", params=note)
    download = client.get("/api/fs/download", params=note)
    saved = client.post("/api/fs/write-text", params={"profile": "remote-dev"},
                        json={"path": note["path"], "content": "edited", "session_id": session_id})

    assert [r.status_code for r in (read, data_url, download, saved)] == [404] * 4, read.text
    assert workspace.writes == []


def test_the_adapter_reads_one_range_with_the_path_quoted(monkeypatch):
    """The SSH command reads exactly the asked bytes, and a hostile file name stays one word."""
    launched = []

    class Process:
        def __init__(self, command, **_options):
            launched.append(command)
            import io

            self.stdout = io.BytesIO(b"slice")

        def poll(self):
            return 0

        def wait(self):
            return 0

    class Env:
        cwd, _remote_home = "/srv", "/home/dev"

        def _build_ssh_command(self):
            return ["ssh", "dev@host"]

    monkeypatch.setattr(ssh_workspace_fs.subprocess, "Popen", Process)
    adapter = ssh_workspace_fs.SshWorkspaceFs(Env())
    hostile = "/srv/media/a b $(touch pwned) ';x'.mp4"

    assert b"".join(adapter.stream_file(hostile, offset=10, length=4)) == b"slice"
    assert b"".join(adapter.stream_file(hostile)) == b"slice"

    ranged, whole = (shlex.split(command[-1])[0] for command in launched)
    assert launched[0][:4] == ["ssh", "dev@host", "bash", "-c"]
    assert shlex.split(ranged) == ["tail", "-c", "+11", "<", hostile, "|", "head", "-c", "4"]
    assert shlex.split(whole) == ["exec", "cat", "<", hostile]
