import hashlib
import hmac
import os
import types

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


def test_ssh_ownership_valid_challenge_returns_verifiable_protocol_2_proof(monkeypatch):
    token = "t" * 64
    nonce = "0123456789abcdef"
    challenge = "a" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", nonce)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)

    response = client.get("/api/ssh/ownership", params={"challenge": challenge})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"ok", "protocolVersion", "proof"}
    assert payload["ok"] is True
    assert payload["protocolVersion"] == 2
    canonical = f"{challenge}:{nonce}:{os.getpid()}:2".encode()
    proof_key = hmac.new(
        token.encode(), b"hermes-ssh-ownership-v2", hashlib.sha256
    ).digest()
    assert hmac.compare_digest(
        payload["proof"], hmac.new(proof_key, canonical, hashlib.sha256).hexdigest()
    )


def test_ssh_ownership_endpoint_requires_token_and_returns_exact_nonce(
    tmp_path, monkeypatch
):
    token = "t" * 64
    nonce = "0123456789abcdef"
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *args, **kwargs: {"purelib": str(purelib)}),
    )
    web_server.app.state.auth_required = False
    web_server._apply_ssh_owner_nonce(nonce)
    try:
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
            "protocolVersion": 2,
            "runtimeIntact": True,
            "pid": os.getpid(),
        }
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_ownership_reports_replaced_runtime(tmp_path, monkeypatch):
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", "0123456789abcdef")
    monkeypatch.setattr(web_server, "_SSH_RUNTIME_MARKER", None)
    # A REAL purelib file whose recorded inode deliberately mismatches what
    # os.stat now reports — never patch os.stat globally here: web_server.os
    # is the os module itself, and swapping its stat() poisons every other
    # thread in this worker process (daemon threads from earlier tests crash
    # in their excepthooks → nondeterministic teardown errors across the
    # whole suite, the Aug 2026 CI flake).
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    st = purelib.stat()
    monkeypatch.setattr(
        web_server, "_SSH_RUNTIME_PURELIB", (str(purelib), st.st_dev, st.st_ino + 1)
    )
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
    # Swap the MODULE ATTRIBUTE on web_server, not sysconfig.get_paths itself:
    # sysconfig is process-global, and mutating it races every other thread
    # in the worker (same cross-thread poisoning class as the os.stat patch
    # this file used to have).
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
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
    # Swap the MODULE ATTRIBUTE on web_server, not sysconfig.get_paths itself:
    # sysconfig is process-global, and mutating it races every other thread
    # in the worker (same cross-thread poisoning class as the os.stat patch
    # this file used to have).
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
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
    # Swap the MODULE ATTRIBUTE on web_server, not sysconfig.get_paths itself:
    # sysconfig is process-global, and mutating it races every other thread
    # in the worker (same cross-thread poisoning class as the os.stat patch
    # this file used to have).
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )
    # Make the directory REALLY unwritable instead of patching builtins.open:
    # a global open() patch races every other thread in the worker process
    # (daemon threads crash in their excepthooks → nondeterministic teardown
    # errors file-wide, the Aug 2026 CI flake). chmod is thread-safe and
    # exercises the genuine OSError path.
    if os.geteuid() == 0:  # pragma: no cover - root ignores mode bits
        pytest.skip("directory write bits are not enforced for root")
    purelib.chmod(0o555)
    try:
        web_server._apply_ssh_owner_nonce("0123456789abcdef")
        try:
            assert web_server._SSH_RUNTIME_MARKER is None
            assert web_server._SSH_RUNTIME_PURELIB is not None
            assert web_server._ssh_runtime_intact() is True
        finally:
            web_server._apply_ssh_owner_nonce(None)
    finally:
        purelib.chmod(0o755)  # let tmp_path cleanup succeed


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


def test_ssh_owner_nonce_rejects_path_traversal(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *args, **kwargs: {"purelib": str(purelib)}),
    )

    web_server._apply_ssh_owner_nonce("../../../../evil")
    try:
        assert web_server._SSH_OWNER_NONCE is None
        assert web_server._SSH_RUNTIME_MARKER is None
        assert not (tmp_path / "evil").exists()
        assert list(purelib.iterdir()) == []
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_owner_nonce_sweeps_dead_runtime_markers_only(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    dead = purelib / ".hermes-ssh-runtime-fedcba9876543210"
    dead.write_text("pid=4194304\n", encoding="utf-8")
    live = purelib / ".hermes-ssh-runtime-aa00000000000001"
    live.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *args, **kwargs: {"purelib": str(purelib)}),
    )

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        current = purelib / ".hermes-ssh-runtime-0123456789abcdef"
        assert current.is_file()
        assert not dead.exists()
        assert live.is_file()
        assert web_server._SSH_RUNTIME_MARKER == str(current)
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_ownership_fails_closed_without_runtime_identity_baseline(monkeypatch):
    token = "t" * 64

    def missing_purelib(*args, **kwargs):
        raise OSError("runtime path unavailable")

    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(
        web_server, "sysconfig", types.SimpleNamespace(get_paths=missing_purelib)
    )
    web_server.app.state.auth_required = False
    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        response = TestClient(web_server.app).get(
            "/api/ssh/ownership",
            headers={"X-Hermes-Session-Token": token},
        )

        assert response.status_code == 200
        assert response.json()["runtimeIntact"] is False
    finally:
        web_server._apply_ssh_owner_nonce(None)
