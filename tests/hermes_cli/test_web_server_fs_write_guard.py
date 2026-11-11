"""POST /api/fs/write-text keeps the guards the spot editor's reads apply (#95306).

The endpoint is the save half of read-text, and it serves remote dashboard sessions too. It used to
write any resolvable path: a credential store read-text refuses, a file outside a locked
managed-files root, or ``~/.ssh/authorized_keys``.
"""

import os

import pytest

from hermes_cli import web_server
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
    written = []

    class SshWorkspace:
        def write_text(self, path, content, *, max_bytes):
            written.append(path)
            return path, len(content.encode("utf-8"))

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: SshWorkspace())

    for path in ("/srv/app/.env", "/srv/app/sub/../mcp-tokens/x.json", "/srv/app/config.yaml"):
        assert _save(client, path, profile="remote-dev").status_code == 403, path
    assert _save(client, "/srv/app/README.md", profile="remote-dev").status_code == 200
    assert written == ["/srv/app/README.md"]


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
