from fastapi.testclient import TestClient

from hermes_cli import web_server


def test_ssh_ownership_endpoint_requires_token_and_returns_exact_nonce(monkeypatch):
    token = "t" * 64
    nonce = "0123456789abcdef"
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", nonce)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)

    assert client.get("/api/ssh/ownership").status_code == 401
    response = client.get(
        "/api/ssh/ownership",
        headers={"X-Hermes-Session-Token": token},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "sshOwnerNonce": nonce,
        "protocolVersion": 1,
        "runtimeIntact": True,
    }


def test_ssh_ownership_reports_replaced_runtime(monkeypatch):
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", "0123456789abcdef")
    monkeypatch.setattr(web_server, "_SSH_RUNTIME_MARKER", None)
    monkeypatch.setattr(web_server, "_SSH_RUNTIME_PURELIB", ("/venv/site-packages", 10, 20))
    monkeypatch.setattr(web_server.os, "stat", lambda _path: type("Stat", (), {"st_dev": 10, "st_ino": 21})())
    client = TestClient(web_server.app)

    response = client.get("/api/ssh/ownership", headers={"X-Hermes-Session-Token": token})

    assert response.status_code == 200
    assert response.json()["runtimeIntact"] is False


def test_ssh_runtime_marker_detects_recreated_venv_even_with_reused_inode(
    tmp_path, monkeypatch
):
    """The exact #82429 repro: rm -rf venv && recreate. On ext4 the new
    site-packages directory routinely REUSES the old inode (proven live
    during salvage), so the stat snapshot alone reports intact. The marker
    file is the deterministic tier: it dies with the old tree."""
    purelib = tmp_path / "venv" / "lib" / "site-packages"
    purelib.mkdir(parents=True)
    monkeypatch.setattr(
        web_server.sysconfig,
        "get_paths",
        lambda *a, **k: {"purelib": str(purelib)},
    )

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        assert web_server._ssh_runtime_intact() is True

        # Replace the venv; the recreated directory may reuse the inode.
        import shutil

        shutil.rmtree(tmp_path / "venv")
        purelib.mkdir(parents=True)

        assert web_server._ssh_runtime_intact() is False, (
            "marker tier must catch a recreated venv regardless of inode reuse"
        )
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_runtime_marker_survives_in_place_installs(tmp_path, monkeypatch):
    """pip/uv installs INTO the live venv must not read as a replacement."""
    purelib = tmp_path / "venv" / "lib" / "site-packages"
    purelib.mkdir(parents=True)
    monkeypatch.setattr(
        web_server.sysconfig,
        "get_paths",
        lambda *a, **k: {"purelib": str(purelib)},
    )

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        (purelib / "newpkg").mkdir()  # a package landing in the live venv
        assert web_server._ssh_runtime_intact() is True
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_runtime_readonly_purelib_falls_back_to_stat(tmp_path, monkeypatch):
    """When site-packages is unwritable the marker can't be placed; the
    stat-snapshot fallback still arms (weaker, never a false stale)."""
    purelib = tmp_path / "venv" / "lib" / "site-packages"
    purelib.mkdir(parents=True)
    monkeypatch.setattr(
        web_server.sysconfig,
        "get_paths",
        lambda *a, **k: {"purelib": str(purelib)},
    )
    real_open = open

    def refuse_marker(path, *a, **k):
        if ".hermes-ssh-runtime-" in str(path):
            raise OSError(30, "Read-only file system")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", refuse_marker)

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        assert web_server._SSH_RUNTIME_MARKER is None
        assert web_server._SSH_RUNTIME_PURELIB is not None
        assert web_server._ssh_runtime_intact() is True
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_ownership_endpoint_is_absent_without_owner_nonce(monkeypatch):
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", None)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)

    response = client.get(
        "/api/ssh/ownership",
        headers={"X-Hermes-Session-Token": token},
    )
    assert response.status_code == 404
