import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.process_identity import _process_create_time


class _DelegatingOsProxy:
    def __init__(self, **overrides):
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(os, name)


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
    marker_value = web_server._SSH_RUNTIME_MARKER
    assert marker_value is not None
    marker = Path(marker_value)
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
    marker_value = web_server._SSH_RUNTIME_MARKER
    assert marker_value is not None
    marker = Path(marker_value)
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
assert web_server._SSH_RUNTIME_MARKER is not None
"""

    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)

    assert not list(purelib.glob(".hermes-ssh-runtime-0123456789abcdef-*"))


def test_second_process_with_same_nonce_does_not_replace_live_owner(
    tmp_path, monkeypatch
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )
    nonce = "0123456789abcdef"
    script = f"""
import types
from hermes_cli import web_server
web_server.sysconfig = types.SimpleNamespace(
    get_paths=lambda: {{"purelib": {str(purelib)!r}}}
)
web_server._apply_ssh_owner_nonce({nonce!r})
assert web_server._ssh_runtime_intact()
"""

    web_server._apply_ssh_owner_nonce(nonce)
    try:
        marker_value = web_server._SSH_RUNTIME_MARKER
        assert marker_value is not None
        marker = Path(marker_value)
        original_payload = marker.read_text(encoding="utf-8")

        subprocess.run([sys.executable, "-c", script], check=True, timeout=30)

        assert marker.read_text(encoding="utf-8") == original_payload
        assert list(purelib.glob(f".hermes-ssh-runtime-{nonce}-*")) == [marker]
        assert web_server._ssh_runtime_intact() is True
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_new_owner_does_not_borrow_legacy_same_nonce_marker(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    owner_create_time = _process_create_time(owner.pid)
    assert owner_create_time is not None
    nonce = "0123456789abcdef"
    marker = purelib / f".hermes-ssh-runtime-{nonce}"
    marker.write_text(
        f"pid={owner.pid}\ncreate_time={owner_create_time}\n", encoding="utf-8"
    )

    try:
        web_server._apply_ssh_owner_nonce(nonce)
        current_value = web_server._SSH_RUNTIME_MARKER
        assert current_value is not None
        current_marker = Path(current_value)
        assert current_marker != marker
        assert web_server._SSH_RUNTIME_MARKER_OWNER_PID == os.getpid()
        assert web_server._ssh_runtime_intact() is True

        marker.unlink()

        assert current_marker.is_file()
        assert web_server._ssh_runtime_intact() is True
    finally:
        web_server._apply_ssh_owner_nonce(None)
        owner.terminate()
        owner.wait(timeout=10)


def test_ssh_runtime_marker_sweep_ignores_scandir_iteration_errors(
    tmp_path, monkeypatch
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    other = tmp_path / "other"
    (other / "package").mkdir(parents=True)

    class BrokenScandir:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            raise OSError("site-packages changed during iteration")

    real_scandir = os.scandir

    def scoped_scandir(path):
        if os.fspath(path) == str(purelib):
            return BrokenScandir()
        return real_scandir(path)

    monkeypatch.setattr(
        web_server,
        "os",
        _DelegatingOsProxy(scandir=scoped_scandir),
    )

    assert web_server.os.path is os.path
    with web_server.os.scandir(other) as entries:
        entry = next(iter(entries))
        assert entry.is_dir()
        assert isinstance(entry.inode(), int)
    web_server._sweep_dead_ssh_runtime_markers(str(purelib))


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


def test_ssh_runtime_marker_sweep_keeps_marker_when_liveness_lookup_raises(
    tmp_path, monkeypatch
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    marker = purelib / ".hermes-ssh-runtime-fedcba9876543210"
    marker.write_text("pid=999999\n", encoding="utf-8")

    def lookup_failed(*_args, **_kwargs):
        raise RuntimeError("process provider failed")

    monkeypatch.setattr(web_server, "_pid_alive_matches", lookup_failed)

    web_server._sweep_dead_ssh_runtime_markers(str(purelib))

    assert marker.is_file()


def test_ssh_runtime_marker_sweep_removes_empty_crash_residue_for_dead_named_pid(
    tmp_path,
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=10)
    assert exited.returncode == 0
    marker = purelib / (
        f".hermes-ssh-runtime-fedcba9876543210-{exited.pid}-0123456789abcdef"
    )
    marker.touch()

    web_server._sweep_dead_ssh_runtime_markers(str(purelib))

    assert not marker.exists()


def test_ssh_runtime_marker_sweep_removes_torn_prefixes_for_dead_named_pid(
    tmp_path,
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=10)
    assert exited.returncode == 0
    payloads = (
        "pid=",
        f"pid={exited.pid}",
        f"pid={exited.pid}\ncreate_time=",
        f"pid={exited.pid}\ncreate_time=17",
        f"pid={exited.pid}\ncreate_time=17.",
    )
    markers = []
    for index, payload in enumerate(payloads):
        marker = purelib / (
            f".hermes-ssh-runtime-fedcba9876543210-{exited.pid}-{index:016x}"
        )
        marker.write_text(payload, encoding="utf-8")
        markers.append(marker)

    web_server._sweep_dead_ssh_runtime_markers(str(purelib))

    assert not any(marker.exists() for marker in markers)


def test_ssh_runtime_marker_sweep_keeps_live_torn_create_time_prefixes(tmp_path):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    pid = os.getpid()
    create_time = web_server._process_create_time(pid)
    assert create_time is not None
    create_time_text = str(create_time)
    integer_prefix, dot, fraction = create_time_text.partition(".")
    assert dot == "."
    assert len(fraction) > 1
    prefixes = (integer_prefix, create_time_text[:-1])
    markers = []
    for index, prefix in enumerate(prefixes):
        marker = purelib / (
            f".hermes-ssh-runtime-fedcba9876543210-{pid}-{index:016x}"
        )
        marker.write_text(
            f"pid={pid}\ncreate_time={prefix}",
            encoding="utf-8",
        )
        markers.append(marker)

    web_server._sweep_dead_ssh_runtime_markers(str(purelib))

    assert all(marker.is_file() for marker in markers)


def test_ssh_runtime_marker_sweep_skips_pid_too_large_to_parse(tmp_path):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    marker = purelib / ".hermes-ssh-runtime-fedcba9876543210"
    marker.write_text("pid=" + "1" * 5000 + "\n", encoding="utf-8")

    web_server._sweep_dead_ssh_runtime_markers(str(purelib))

    assert marker.is_file()


def test_ssh_runtime_marker_sweep_keeps_nonempty_unparseable_named_marker(tmp_path):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=10)
    assert exited.returncode == 0
    marker = purelib / (
        f".hermes-ssh-runtime-fedcba9876543210-{exited.pid}-0123456789abcdef"
    )
    marker.write_text("pid=" + "1" * 5000 + "\n", encoding="utf-8")

    web_server._sweep_dead_ssh_runtime_markers(str(purelib))

    assert marker.is_file()


def test_ssh_runtime_marker_sweep_uses_os_stat_for_both_snapshots(
    tmp_path, monkeypatch
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    marker = purelib / ".hermes-ssh-runtime-fedcba9876543210"
    marker.write_text("pid=999999\n", encoding="utf-8")

    class WindowsShapedEntry:
        def __init__(self, entry):
            self._entry = entry
            self.name = entry.name
            self.path = entry.path

        def is_file(self, *, follow_symlinks=True):
            return self._entry.is_file(follow_symlinks=follow_symlinks)

        def stat(self, *, follow_symlinks=True):
            real = self._entry.stat(follow_symlinks=follow_symlinks)
            return os.stat_result((real.st_mode, real.st_ino * 0, real.st_dev * 0) + real[3:])

    class WindowsShapedScandir:
        def __init__(self, path):
            self._entries = os.scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._entries.__exit__(*args)

        def __iter__(self):
            return (WindowsShapedEntry(entry) for entry in self._entries)

    real_scandir = os.scandir

    def windows_shaped_scandir(path):
        if os.fspath(path) == str(purelib):
            return WindowsShapedScandir(path)
        return real_scandir(path)

    monkeypatch.setattr(web_server, "_pid_alive_matches", lambda *_a, **_k: False)
    monkeypatch.setattr(
        web_server,
        "os",
        _DelegatingOsProxy(scandir=windows_shaped_scandir),
    )

    web_server._sweep_dead_ssh_runtime_markers(str(purelib))

    assert not marker.exists()


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
        current_value = web_server._SSH_RUNTIME_MARKER
        assert current_value is not None
        current = Path(current_value)
        assert current.is_file()
        assert not dead.exists()
        assert malformed.is_file()
        assert unmatched.is_file()
        assert live.is_file()
    finally:
        web_server._apply_ssh_owner_nonce(None)
        live_process.terminate()
        live_process.wait(timeout=10)


def test_marker_write_failure_does_not_leave_blocking_marker(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )
    nonce = "0123456789abcdef"
    marker_prefix = f".hermes-ssh-runtime-{nonce}-"
    real_open = open

    class FailingMarkerWrite:
        def __init__(self, path, *args, **kwargs):
            self._fh = real_open(path, *args, **kwargs)

        def __enter__(self):
            self._fh.__enter__()
            return self

        def __exit__(self, *args):
            return self._fh.__exit__(*args)

        def fileno(self):
            return self._fh.fileno()

        def write(self, _payload):
            raise OSError(28, "No space left on device")

    def scoped_open(path, *args, **kwargs):
        if Path(path).name.startswith(marker_prefix):
            return FailingMarkerWrite(path, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(web_server, "open", scoped_open, raising=False)

    try:
        web_server._apply_ssh_owner_nonce(nonce)

        assert web_server._SSH_RUNTIME_MARKER is None
        assert not list(purelib.glob(f"{marker_prefix}*"))
    finally:
        web_server._apply_ssh_owner_nonce(None)
        for marker in purelib.glob(f"{marker_prefix}*"):
            marker.unlink()


def test_preexisting_empty_marker_does_not_disable_marker_tier(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )
    nonce = "0123456789abcdef"
    legacy_marker = purelib / f".hermes-ssh-runtime-{nonce}"
    legacy_marker.touch()

    web_server._apply_ssh_owner_nonce(nonce)
    try:
        marker = web_server._SSH_RUNTIME_MARKER
        assert marker is not None
        assert marker != str(legacy_marker)
        assert os.path.isfile(marker)
        assert web_server._ssh_runtime_intact() is True
    finally:
        web_server._apply_ssh_owner_nonce(None)

    assert legacy_marker.is_file()


def test_random_marker_name_collision_keeps_existing_file_and_uses_stat_fallback(
    tmp_path, monkeypatch
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )
    nonce = "0123456789abcdef"
    token = "fedcba9876543210"
    marker = purelib / f".hermes-ssh-runtime-{nonce}-{os.getpid()}-{token}"
    marker.touch()
    monkeypatch.setattr(
        web_server,
        "secrets",
        types.SimpleNamespace(token_hex=lambda _size: token),
    )

    web_server._apply_ssh_owner_nonce(nonce)
    try:
        assert web_server._SSH_RUNTIME_MARKER is None
        assert marker.read_bytes() == b""
        assert web_server._SSH_RUNTIME_PURELIB is not None
        assert web_server._ssh_runtime_intact() is True
    finally:
        web_server._apply_ssh_owner_nonce(None)


def test_ssh_owner_nonce_arms_without_create_time_when_lookup_raises(
    tmp_path, monkeypatch
):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    monkeypatch.setattr(
        web_server,
        "sysconfig",
        types.SimpleNamespace(get_paths=lambda *a, **k: {"purelib": str(purelib)}),
    )

    def lookup_failed(*_args, **_kwargs):
        raise RuntimeError("process provider failed")

    monkeypatch.setattr(web_server, "_process_create_time", lookup_failed)

    web_server._apply_ssh_owner_nonce("0123456789abcdef")
    try:
        marker_value = web_server._SSH_RUNTIME_MARKER
        assert marker_value is not None
        marker = Path(marker_value)
        assert marker.read_text(encoding="utf-8") == f"pid={os.getpid()}\n"
        assert web_server._ssh_runtime_intact() is True
    finally:
        web_server._apply_ssh_owner_nonce(None)


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
