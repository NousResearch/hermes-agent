"""Synthetic filesystem-boundary tests; these never launch Pi or a proxy."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import landlock
from hermes_cli.landlock import LandlockError, probe_landlock_abi, run_landlocked


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")


def _runtime_roots() -> list[Path]:
    roots = [Path(sys.prefix), Path(sys.base_prefix), Path(__file__).resolve().parents[2]]
    roots.extend(Path(path) for path in ("/usr", "/bin", "/lib", "/lib64"))
    return [path for path in roots if path.exists()]


def test_landlock_runner_denies_external_collection_and_preserves_its_symlink(tmp_path):
    assert probe_landlock_abi() >= 5
    workspace = tmp_path / "workspace"
    home = tmp_path / "sandbox-home"
    collection = tmp_path / "protected-collection"
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    collection.mkdir()
    marker = collection / "marker.txt"
    marker.write_text("synthetic-only")
    marker_stat = marker.stat()
    data_link = workspace / ".binder-data"
    data_link.symlink_to(collection, target_is_directory=True)

    child = r'''import json, subprocess, sys
from pathlib import Path
workspace = Path.cwd()
external = Path(__PROTECTED_MARKER__)
def denied(operation):
    try:
        operation()
    except OSError:
        return True
    return False
read_denied = denied(lambda: external.read_text())
write_denied = denied(lambda: external.write_text("changed"))
symlink_read_denied = denied(lambda: (workspace / ".binder-data" / "marker.txt").read_text())
symlink_write_denied = denied(lambda: (workspace / ".binder-data" / "marker.txt").write_text("changed"))
(workspace / "new-source.txt").write_text("allowed")
runtime_read = bool(Path(sys.executable).read_bytes())
import ctypes
no_new_privs = ctypes.CDLL(None).prctl(39, 0, 0, 0, 0) == 1
grandchild = ("import json; from pathlib import Path; p=Path(" + repr(str(external)) + ");\n"
    "def denied(op):\n try: op()\n except OSError: return True\n return False\n"
    "print(json.dumps({'read':denied(lambda:p.read_text()), "
    "'write':denied(lambda:p.write_text('changed'))}))")
subprocess_result = subprocess.run([sys.executable, "-c", grandchild],
    check=False, capture_output=True, text=True)
print(json.dumps({"read_denied": read_denied, "write_denied": write_denied,
    "symlink_read_denied": symlink_read_denied,
    "symlink_write_denied": symlink_write_denied,
    "runtime_read": runtime_read, "no_new_privs": no_new_privs,
    "subprocess_read_denied": subprocess_result.returncode == 0 and
        json.loads(subprocess_result.stdout) == {"read": True, "write": True}}))
'''
    child = child.replace("__PROTECTED_MARKER__", repr(str(marker)))
    result = run_landlocked(
        [sys.executable, "-c", child],
        workspace=workspace,
        home=home,
        read_paths=_runtime_roots(),
        protected_paths=[marker],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "read_denied": True,
        "write_denied": True,
        "symlink_read_denied": True,
        "symlink_write_denied": True,
        "runtime_read": True,
        "no_new_privs": True,
        "subprocess_read_denied": True,
    }
    assert (workspace / "new-source.txt").read_text() == "allowed"
    assert data_link.is_symlink()
    assert marker.read_text() == "synthetic-only"
    after = marker.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        marker_stat.st_ino, marker_stat.st_size, marker_stat.st_mtime_ns
    )


def test_landlock_runner_drops_parent_credentials_and_closes_parent_descriptors(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    home = tmp_path / "sandbox-home"
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", "/synthetic/real-home-must-not-pass")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-parent-secret")
    fd_path = tmp_path / "parent-fd.txt"
    fd_path.write_text("synthetic-fd")
    original_fd = os.open(fd_path, os.O_RDONLY)
    parent_fd = os.dup2(original_fd, 200)
    os.close(original_fd)
    os.set_inheritable(parent_fd, True)
    child = f"import json,os; fd={parent_fd}; " \
        "open_fd=True; " \
        "exec(\"try:\\n os.fstat(fd)\\nexcept OSError:\\n open_fd=False\"); " \
        "print(json.dumps({'HERMES_HOME':os.getenv('HERMES_HOME'), " \
        "'OPENAI_API_KEY':os.getenv('OPENAI_API_KEY'), " \
        "'OPENAI_BASE_URL':os.getenv('OPENAI_BASE_URL'), 'parent_fd_open':open_fd}))"
    try:
        result = run_landlocked(
            [sys.executable, "-c", child],
            workspace=workspace,
            home=home,
            read_paths=_runtime_roots(),
            proxy_url="http://127.0.0.1:8645/v1",
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        os.close(parent_fd)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "HERMES_HOME": None,
        "OPENAI_API_KEY": "hermes-local-proxy",
        "OPENAI_BASE_URL": "http://127.0.0.1:8645/v1",
        "parent_fd_open": False,
    }


def test_landlock_runner_never_executes_after_sandbox_initialization_failure(monkeypatch):
    payload = landlock.base64.urlsafe_b64encode(
        json.dumps({"read_paths": [], "write_paths": [], "protected_paths": [], "child_env": {}}).encode()
    ).decode()
    executed = False

    def fail_to_apply(**_kwargs):
        raise LandlockError("synthetic initialization failure")

    def record_exec(*_args, **_kwargs):
        nonlocal executed
        executed = True

    monkeypatch.setattr(landlock, "apply_landlock", fail_to_apply)
    monkeypatch.setattr(landlock.os, "execve", record_exec)

    assert landlock._helper_main(["--internal-landlock-exec", payload, "--", "/synthetic/child"]) == 125
    assert executed is False


def test_landlock_runner_refuses_a_workspace_that_contains_protected_data(tmp_path):
    workspace = tmp_path / "workspace"
    home = tmp_path / "sandbox-home"
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    protected = workspace / "collection"
    protected.mkdir()

    with pytest.raises(LandlockError, match="protected path"):
        run_landlocked(
            [sys.executable, "-c", "pass"],
            workspace=workspace,
            home=home,
            read_paths=_runtime_roots(),
            protected_paths=[protected],
        )


def test_landlock_runner_rejects_a_grant_overlapping_default_protected_paths(tmp_path, monkeypatch):
    """Default protection works without caller input: HOME subpaths are blocked."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(mode=0o700)
    monkeypatch.setattr(landlock.Path, "home", staticmethod(lambda: fake_home))
    workspace = tmp_path / "workspace"
    home = tmp_path / "sandbox-home"
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)

    with pytest.raises(LandlockError, match="protected path"):
        run_landlocked(
            [sys.executable, "-c", "pass"],
            workspace=workspace,
            home=home,
            read_paths=[fake_home],
        )


def test_landlock_runner_rejects_a_grant_covering_hermes_home_material(tmp_path, monkeypatch):
    """HERMES_HOME auth/config material is protected by default."""
    hermes_root = tmp_path / "hermes-data"
    hermes_root.mkdir()
    (hermes_root / "auth.json").write_text("synthetic")
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))
    workspace = tmp_path / "workspace"
    home = tmp_path / "sandbox-home"
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)

    with pytest.raises(LandlockError, match="protected path"):
        run_landlocked(
            [sys.executable, "-c", "pass"],
            workspace=workspace,
            home=home,
            read_paths=[hermes_root],
        )


def test_landlock_workspace_rejects_credential_directories(tmp_path):
    workspace = tmp_path / "workspace"
    home = tmp_path / "sandbox-home"
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    aws = workspace / ".aws"
    aws.mkdir()
    (aws / "credentials").write_text("synthetic")

    with pytest.raises(LandlockError, match="credential-like directory"):
        run_landlocked(
            [sys.executable, "-c", "pass"],
            workspace=workspace,
            home=home,
            read_paths=_runtime_roots(),
        )


def test_landlock_runner_rejects_group_or_world_writable_workspace_and_home(tmp_path):
    workspace = tmp_path / "workspace"
    home = tmp_path / "sandbox-home"
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    workspace.chmod(0o777)

    with pytest.raises(LandlockError, match="writable by group or others"):
        run_landlocked(
            [sys.executable, "-c", "pass"],
            workspace=workspace,
            home=home,
            read_paths=_runtime_roots(),
        )

    workspace.chmod(0o700)
    home.chmod(0o777)
    with pytest.raises(LandlockError, match="writable by group or others"):
        run_landlocked(
            [sys.executable, "-c", "pass"],
            workspace=workspace,
            home=home,
            read_paths=_runtime_roots(),
        )


def test_landlock_runner_refuses_a_workspace_containing_the_helper_implementation(tmp_path):
    """A child-writable workspace must never be able to shadow the helper code."""
    module_root = Path(landlock.__file__).resolve().parent.parent
    home = tmp_path / "sandbox-home"
    home.mkdir(mode=0o700)

    with pytest.raises(LandlockError, match="helper implementation"):
        run_landlocked(
            [sys.executable, "-c", "pass"],
            workspace=module_root,
            home=home,
            read_paths=_runtime_roots(),
        )
