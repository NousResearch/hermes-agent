"""Credential stores reached through a symlink stay hidden from the Files API (#95311).

The name checks see only the names a path is spelled with; ``~/.ssh -> ~/keys`` gives the
store's resolved form an innocuous name, so the guard also compares the resolved path with
the resolved credential-store locations of the guard homes.
"""

import base64

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def home_client(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield client, home
    finally:
        client.close()
        if previous is None:
            delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


def _link(link, target, *, is_dir):
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("filesystem does not allow symlinks")


def test_symlinked_credential_dir_and_file_stay_hidden(home_client):
    client, home = home_client
    (home / "keys").mkdir()
    (home / "keys" / "id_rsa").write_text("secret")
    (home / "dotfiles").mkdir()
    (home / "dotfiles" / "netrc").write_text("secret")
    (home / "notes.txt").write_text("safe")
    _link(home / ".ssh", home / "keys", is_dir=True)
    _link(home / ".netrc", home / "dotfiles" / "netrc", is_dir=False)
    _link(home / "alias", home / ".ssh", is_dir=True)  # a scan must still walk an innocuous link

    names = [e["name"] for e in client.get("/api/files", params={"path": str(home)}).json()["entries"]]
    assert "notes.txt" in names and "dotfiles" in names
    assert "keys" not in names and "alias" not in names
    dotfiles = client.get("/api/files", params={"path": str(home / "dotfiles")}).json()["entries"]
    assert [e["name"] for e in dotfiles] == []

    for rel in (".ssh/id_rsa", "keys/id_rsa", ".netrc", "dotfiles/netrc"):
        for route in ("/api/files/read", "/api/files/download"):
            resp = client.get(route, params={"path": str(home / rel)})
            assert resp.status_code == 403, (route, rel, resp.status_code)
        resp = client.get("/api/fs/read-text", params={"path": str(home / rel)})
        assert resp.status_code == 403, ("/api/fs/read-text", rel, resp.status_code)

    fs_names = [e["name"] for e in client.get("/api/fs/list", params={"path": str(home)}).json()["entries"]]
    assert "keys" not in fs_names and "alias" not in fs_names and "notes.txt" in fs_names

    data_url = "data:text/plain;base64," + base64.b64encode(b"ssh-ed25519 AAAA").decode()
    resp = client.post("/api/files/upload", json={"path": str(home / "keys" / "authorized_keys"), "data_url": data_url})
    assert resp.status_code == 403, resp.status_code
    assert not (home / "keys" / "authorized_keys").exists()


@pytest.mark.platforms("macos")
def test_case_variant_of_symlinked_store_is_refused(home_client):
    """APFS opens ``~/KEYS/id_rsa`` for ``~/keys/id_rsa``; realpath keeps the typed case."""
    client, home = home_client
    (home / "keys").mkdir()
    (home / "keys" / "id_rsa").write_text("secret")
    _link(home / ".ssh", home / "keys", is_dir=True)
    if not (home / "KEYS" / "id_rsa").exists():
        pytest.skip("case-sensitive volume")

    for route in ("/api/files/read", "/api/fs/read-text"):
        resp = client.get(route, params={"path": str(home / "KEYS" / "id_rsa")})
        assert resp.status_code == 403, (route, resp.status_code)
