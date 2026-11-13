"""On-demand terminal reads never substitute a host namesake or create an environment."""

import base64
import shlex
import subprocess


import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from tools.environments.base import BaseEnvironment


class ShellNamespace(BaseEnvironment):
    cwd = "/workspace"
    timeout = 30
    _remote_home = "/workspace"
    _hermes_backend_name = "test_remote"

    def __init__(self, root, visible):
        self.root = root
        self.visible = visible
        self.calls = []

    def execute(self, command, **kwargs):
        self.calls.append(command)
        # Execute the production shell transport, but in a distinct filesystem namespace.
        command = command.replace(shlex.quote(self.visible), shlex.quote(str(self.root / "file")))
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)
        output = result.stdout.replace(str(self.root / "file"), self.visible)
        # realpath metadata is base64 encoded, so translate that namespace too.
        output = output.replace(base64.b64encode(str(self.root / "file").encode()).decode(),
                                base64.b64encode(self.visible.encode()).decode())
        return {"output": output, "returncode": result.returncode}

    def cleanup(self):
        pass


@pytest.fixture
def remote_client(tmp_path, monkeypatch):
    from hermes_cli.web_routers import files

    from tools import terminal_tool
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    from contextlib import nullcontext

    # Bind the actual terminal policy and registry; never replace the filesystem backend.
    token = set_terminal_scope({"TERMINAL_ENV": "test_remote"})
    monkeypatch.setattr(files, "_profile_scope", lambda profile=None: nullcontext())
    monkeypatch.setattr(files, "load_config", lambda: {"terminal": {"backend": "test_remote"}})
    monkeypatch.setattr(files, "load_env", dict)
    async def owned(session_id, profile):
        if session_id not in {None, "one", "two", "gone"}:
            from fastapi import HTTPException
            raise HTTPException(404, "Session not found")
        return {"id": session_id} if session_id else None
    monkeypatch.setattr(files, "_fs_owned_session", owned)
    host = tmp_path / "artifact.mp4"
    host.write_bytes(b"HOST NAMESAKE")
    envs = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        (root / "file").write_bytes(name.encode() + b"\x00\xff\n")
        env = ShellNamespace(root, str(host))
        envs.append(env)
        monkeypatch.setitem(terminal_tool._active_environments, name, env)
    def forbidden(*args, **kwargs):
        pytest.fail("A file request must not create a terminal environment")
    monkeypatch.setattr(terminal_tool, "ensure_task_env", forbidden)
    from tools import terminal_tool_backends
    monkeypatch.setattr(terminal_tool_backends, "_create_environment", forbidden)
    app = FastAPI()
    app.include_router(files.router)
    with TestClient(app) as client:
        yield client, host, envs
    reset_terminal_scope(token)


@pytest.mark.platforms("posix")
def test_terminal_preview_and_download_are_on_demand_and_session_bound(remote_client):
    client, host, envs = remote_client
    assert not any(env.calls for env in envs)
    for name in ("one", "two"):
        query = {"path": str(host), "session_id": name}
        preview = client.get("/api/fs/read-data-url", params=query)
        assert preview.status_code == 200, preview.text
        assert base64.b64decode(preview.json()["dataUrl"].split(",", 1)[1]) == name.encode() + b"\x00\xff\n"
        download = client.get("/api/fs/download", params=query)
        assert download.status_code == 200, download.text
        assert download.content == name.encode() + b"\x00\xff\n"
    assert host.read_bytes() == b"HOST NAMESAKE"


@pytest.mark.platforms("posix")
def test_text_and_media_range_use_the_same_terminal(remote_client):
    client, host, _envs = remote_client
    query = {"path": str(host), "session_id": "two"}
    text = client.get("/api/fs/read-text", params=query)
    assert text.status_code == 200, text.text
    assert text.json()["text"] == "two\x00\ufffd\n"
    media = client.get("/api/files/stream", params=query, headers={"Range": "bytes=1-3"})
    assert media.status_code == 206, media.text
    assert media.content == b"wo\x00"
    assert media.headers["Content-Range"] == "bytes 1-3/6"
    head = client.head("/api/files/stream", params=query)
    assert head.status_code == 200
    assert head.content == b""


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("start,end", [(0, 200000), (65530, 140000), (131073, 200000)])
def test_terminal_range_preserves_bytes_across_transfer_blocks(remote_client, start, end):
    client, host, envs = remote_client
    payload = bytes(range(256)) * 1024
    (envs[0].root / "file").write_bytes(payload)
    response = client.get(
        "/api/files/stream", params={"path": str(host), "session_id": "one"},
        headers={"Range": f"bytes={start}-{end}"},
    )
    assert response.status_code == 206, response.text
    assert response.content == payload[start:end + 1]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("route", ["read-text", "read-data-url", "download"])
def test_missing_terminal_files_and_environments_never_fall_back_to_host(remote_client, route):
    client, host, envs = remote_client
    (envs[0].root / "file").unlink()
    missing = client.get(f"/api/fs/{route}", params={"path": str(host), "session_id": "one"})
    assert missing.status_code == 404, missing.text
    unavailable = client.get(f"/api/fs/{route}", params={"path": str(host), "session_id": "gone"})
    assert unavailable.status_code == 503, unavailable.text
    assert "Terminal environment is unavailable" in unavailable.text
    unknown = client.get(f"/api/fs/{route}", params={"path": str(host), "session_id": "not-owned"})
    assert unknown.status_code == 404
    assert not envs[1].calls


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("route", ["read-text", "read-data-url", "download"])
def test_resolved_terminal_credentials_are_denied(remote_client, route):
    client, host, envs = remote_client
    source = envs[0].root / "file"
    source.unlink()
    credential = envs[0].root / ".env"
    credential.write_bytes(b"test fixture, not a real credential")
    source.symlink_to(credential)
    response = client.get(f"/api/fs/{route}", params={"path": str(host), "session_id": "one"})
    assert response.status_code == 403, response.text


@pytest.mark.platforms("posix")
def test_generic_attachment_reader_does_not_enable_remote_edits(remote_client):
    client, host, envs = remote_client
    response = client.post("/api/fs/write-text", json={
        "path": str(host), "session_id": "one", "content": "replacement",
    })
    assert response.status_code == 400, response.text
    assert "read-only" in response.text
    assert (envs[0].root / "file").read_bytes() == b"one\x00\xff\n"


@pytest.mark.platforms("posix")
def test_lost_terminal_connection_has_a_non_ssh_error(remote_client, monkeypatch):
    client, host, envs = remote_client
    def disconnected(*args, **kwargs):
        raise ConnectionError("test transport disconnected")
    monkeypatch.setattr(envs[0], "execute", disconnected)
    response = client.get("/api/fs/read-data-url", params={"path": str(host), "session_id": "one"})
    assert response.status_code == 503, response.text
    assert "Terminal" in response.text
    assert "SSH" not in response.text


@pytest.mark.platforms("posix")
def test_live_registry_lookup_is_profile_qualified_without_raw_id_fallback(tmp_path, monkeypatch):
    from hermes_cli.terminal_workspace_fs import get_terminal_workspace_fs
    from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError
    from hermes_cli.web_server_files import _agent_path_scope
    from tools import terminal_tool
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope

    visible = str(tmp_path / "same.txt")
    raw_env = ShellNamespace(tmp_path, visible)
    monkeypatch.setitem(terminal_tool._active_environments, "same-session", raw_env)
    homes = []
    for profile in ("alpha", "beta"):
        home = tmp_path / profile
        home.mkdir()
        (home / "config.yaml").write_text("terminal:\n  backend: test_remote\n")
        root = home / "remote"
        root.mkdir()
        (root / "file").write_bytes(profile.encode())
        homes.append((home, ShellNamespace(root, visible)))
    token = set_terminal_scope(None)
    try:
        for home, env in homes:
            with _agent_path_scope(home):
                key = terminal_tool._qualify_task_key("session:same-session")
                assert key != "session:same-session"
                with pytest.raises(SshWorkspaceFsError, match="unavailable"):
                    get_terminal_workspace_fs({"id": "same-session"})
                monkeypatch.setitem(terminal_tool._active_environments, key, env)
        for home, env in homes:
            with _agent_path_scope(home):
                reader = get_terminal_workspace_fs({"id": "same-session"})
                assert reader.read_bytes(visible, max_bytes=100)[0] == home.name.encode()
        assert not raw_env.calls
    finally:
        reset_terminal_scope(token)
