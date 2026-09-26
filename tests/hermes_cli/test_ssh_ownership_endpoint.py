import os
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.process_identity import _process_create_time


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


def test_clearing_ssh_owner_nonce_removes_its_runtime_marker(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    marker = purelib / ".hermes-ssh-runtime-0123456789abcdef"
    assert marker.is_file()

    web_server._apply_ssh_owner_nonce(None)

    assert not marker.exists()


def test_clearing_ssh_owner_nonce_keeps_a_replacement_runtime_marker(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    marker = purelib / ".hermes-ssh-runtime-0123456789abcdef"
    marker.write_text("pid=999999\n", encoding="utf-8")

    web_server._apply_ssh_owner_nonce(None)

    assert marker.read_text(encoding="utf-8") == "pid=999999\n"


def test_ssh_runtime_marker_is_removed_when_process_exits(tmp_path):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    script = f"""
import types
from hermes_cli import web_server
web_server.sysconfig = types.SimpleNamespace(
    get_paths=lambda: {{"purelib": {str(purelib)!r}}}
)
web_server._apply_ssh_owner_nonce("0123456789abcdef")
"""

    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)

    marker = purelib / ".hermes-ssh-runtime-0123456789abcdef"
    assert not marker.exists()


def test_ssh_runtime_marker_sweep_ignores_scandir_iteration_errors(monkeypatch):
    class BrokenScandir:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            raise OSError("site-packages changed during iteration")

    monkeypatch.setattr(
        web_server, "os", types.SimpleNamespace(scandir=lambda _path: BrokenScandir())
    )

    web_server._sweep_dead_ssh_runtime_markers("/unused")


def test_ssh_runtime_marker_sweep_keeps_a_path_replaced_during_liveness_check(
    tmp_path, monkeypatch
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    marker = purelib / ".hermes-ssh-runtime-fedcba9876543210"
    marker.write_text("pid=999999\n", encoding="utf-8")
    replacement = f"pid={os.getpid()}\n"

    def replace_before_reporting_dead(*_args, **_kwargs):
        marker.write_text(replacement, encoding="utf-8")
        return False

    monkeypatch.setattr(web_server, "_pid_alive_matches", replace_before_reporting_dead)
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        assert marker.read_text(encoding="utf-8") == replacement
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_owner_nonce_sweeps_reused_pid_marker(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    stale = purelib / ".hermes-ssh-runtime-fedcba9876543210"
    stale.write_text(f"pid={os.getpid()}\ncreate_time=0.0\n", encoding="utf-8")
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        assert not stale.exists()
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_owner_nonce_sweeps_dead_runtime_markers_only(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=10)
    assert exited.returncode == 0
    dead = purelib / ".hermes-ssh-runtime-fedcba9876543210"
    dead.write_text(f"pid={exited.pid}\n", encoding="utf-8")
    malformed = purelib / ".hermes-ssh-runtime-bb00000000000002"
    malformed.write_text("pid=\n", encoding="utf-8")
    unmatched = purelib / ".hermes-ssh-runtime-NOT-A-NONCE"
    unmatched.write_text(f"pid={exited.pid}\n", encoding="utf-8")
    live_process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    live_create_time = _process_create_time(live_process.pid)
    assert live_create_time is not None
    live = purelib / ".hermes-ssh-runtime-aa00000000000001"
    live.write_text(
        f"pid={live_process.pid}\ncreate_time={live_create_time}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )

    try:
        web_server._apply_ssh_owner_nonce("0123456789abcdef")
        current = purelib / ".hermes-ssh-runtime-0123456789abcdef"
        assert current.is_file()
        assert not dead.exists()
        assert malformed.is_file()
        assert unmatched.is_file()
        assert live.is_file()
    finally:
        web_server._apply_ssh_owner_nonce(None)
        live_process.terminate()
        live_process.wait(timeout=10)


@pytest.mark.platforms("linux")
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
