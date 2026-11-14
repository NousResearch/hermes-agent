"""POST /api/fs/write-text keeps the guards the spot editor's reads apply (#95306).

The endpoint is the save half of read-text, and it serves remote dashboard sessions too. It used to
write any resolvable path: a credential store read-text refuses, a file outside a locked
managed-files root, or ``~/.ssh/authorized_keys``.
"""

import os

import pytest

from hermes_cli import web_server
from hermes_cli.ssh_workspace_fs import SshWorkspaceFs
from hermes_cli.web_routers import files as file_routes

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
        test_client.close()
        if previous is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


def _save(client, path, content="tampered", **params):
    return client.post("/api/fs/write-text", params=params, json={"path": str(path), "content": content})


@pytest.mark.parametrize("relative", [".env", ".ENV", ".env.local", "auth.json", "config.yaml", "mcp-tokens/github.json"])
def test_write_text_refuses_the_files_read_text_refuses(client, tmp_path, relative):
    existing = tmp_path / "existing" / relative
    existing.parent.mkdir(parents=True)
    existing.write_text("original")
    created = tmp_path / "created" / relative
    created.parent.mkdir(parents=True)

    assert client.get("/api/fs/read-text", params={"path": str(existing)}).status_code == 403
    assert _save(client, existing).status_code == 403
    assert existing.read_text() == "original"
    assert _save(client, created).status_code == 403
    assert not created.exists()
    assert not [p for p in existing.parent.iterdir() if p.name != existing.name]  # no staged temp left


@pytest.mark.platforms("posix")
def test_write_text_judges_a_link_by_the_file_it_names(client, tmp_path):
    store = tmp_path / "home" / "auth.json"
    store.parent.mkdir()
    store.write_text("{}")
    link = tmp_path / "notes.md"
    link.symlink_to(store)

    assert _save(client, link).status_code == 403
    assert store.read_text() == "{}"
    assert link.is_symlink()


def test_write_text_refuses_the_agents_write_denied_credentials(client, tmp_path, monkeypatch):
    home = tmp_path / "user"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    keys = home / ".ssh" / "authorized_keys"

    assert _save(client, keys, "ssh-ed25519 AAAA attacker").status_code == 403
    assert not keys.exists()


def test_write_text_stays_inside_a_locked_files_root(client, tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(managed))

    refused = _save(client, outside / "run.sh", "#!/bin/sh\n")
    saved = _save(client, managed / "notes.md", "kept")

    assert refused.status_code == 403
    assert not (outside / "run.sh").exists()
    assert saved.status_code == 200, saved.text
    assert (managed / "notes.md").read_text() == "kept"


def test_write_text_never_hands_a_credential_path_to_the_ssh_workspace(client, monkeypatch):
    """The guard judges the path the SSH adapter writes, so padding it with whitespace (which the
    adapter strips) or ``..`` still lands on the refusal."""
    sent = []

    class SshMachine:
        cwd = "/srv/app"
        timeout = 30
        _remote_home = "/home/dev"

        def execute(self, command, **_kwargs):
            sent.append(command)
            return {"returncode": 0, "output": ""}

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: SshWorkspaceFs(SshMachine()))

    for path in (
        "/srv/app/.env",
        "/srv/app/sub/../mcp-tokens/x.json",
        "/srv/app/config.yaml",
        "/srv/app/.env ",
        " /srv/app/auth.json\n",
        "/srv/app/.env\t",
        "mcp-tokens/x.json ",
        # ``..`` exposes whitespace a first normalization left inside a component.
        "/srv/app/.env /x/..",
        "/srv/app/.env\t/a/..",
        "/srv/app/auth.json\n/b/..",
        "/srv/app/config.yaml\x1f/q/..",
    ):
        assert _save(client, path, profile="remote-dev").status_code == 403, repr(path)
    assert sent == []
    saved = _save(client, "/srv/app/README.md ", profile="remote-dev")
    assert (saved.status_code, saved.json()["path"]) == (200, "/srv/app/README.md")
    assert len(sent) == 1


def test_write_text_still_saves_an_ordinary_file(client, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "notes.md"
    target.write_text("old")

    response = _save(client, target, "new")

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "path": str(target), "byteSize": 3}
    assert target.read_text() == "new"
    assert os.listdir(project) == ["notes.md"]


def test_write_text_judges_a_named_profile_save_by_that_profiles_protected_state(client, tmp_path, monkeypatch):
    """``?profile=work`` saves under work's scope: its ``state.db`` and ``sessions/`` are protected
    exactly as the launch profile's are, not just ``<root>/state.db``."""
    root = tmp_path / ".hermes"
    work = root / "profiles" / "work"
    (work / "sessions").mkdir(parents=True)
    (work / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    state_db = work / "state.db"
    state_db.write_text("original")
    transcript = work / "sessions" / "s1.jsonl"

    assert _save(client, state_db, profile="work").status_code == 403
    assert state_db.read_text() == "original"
    assert _save(client, transcript, profile="work").status_code == 403
    assert not transcript.exists()
    notes = work / "notes.md"
    saved = _save(client, notes, "kept", profile="work")
    assert saved.status_code == 200, saved.text
    assert notes.read_text() == "kept"


def test_write_text_refuses_the_approval_gated_ssh_config(client, tmp_path, monkeypatch):
    """The file tools gate ~/.ssh/config on an approval prompt; the editor save has none."""
    home = tmp_path / "user"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    config = home / ".ssh" / "config"

    assert _save(client, config, "Host *\n  ProxyCommand sh -c id\n").status_code == 403
    assert not config.exists()


@pytest.mark.parametrize("path", [
    r"\??\UNC\example.invalid\share\notes.txt",
    r"\\?\UNC\example.invalid\share\notes.txt",
    r"\\?\GLOBALROOT\Device\HarddiskVolumeShadowCopy1\notes.txt",
    r"\\.\PhysicalDrive0",
    r" \\?\UNC\example.invalid\share\notes.txt",
])
def test_write_text_refuses_nt_namespaces_before_any_path_resolution(client, monkeypatch, path):
    """Raw NT namespace input must be rejected before a Windows filesystem path is touched."""
    def unexpected_fs_access(*_args, **_kwargs):
        raise AssertionError("NT namespace input reached filesystem path inspection")

    monkeypatch.setattr(file_routes, "_remote_fs_backend", unexpected_fs_access)
    response = _save(client, path)
    assert response.status_code == 403, response.text


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("url", [
    "file:////%3F/UNC/example.invalid/share/notes.txt",
    "file:////./PhysicalDrive0",
])
def test_write_text_refuses_a_file_url_naming_an_nt_namespace(client, monkeypatch, url):
    """On Windows these URLs decode to NT namespace paths: judge the decoded path, not the URL."""
    def unexpected_fs_access(*_args, **_kwargs):
        raise AssertionError("NT namespace input reached filesystem path inspection")

    monkeypatch.setattr(file_routes, "_remote_fs_backend", unexpected_fs_access)
    response = _save(client, url)
    assert response.status_code == 403, response.text
