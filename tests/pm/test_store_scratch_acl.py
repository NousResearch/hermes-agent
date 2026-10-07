"""Windows mkdtemp applies a protected DACL on Python 3.12.4+.

A same-volume publish preserves that descriptor. The current user's ACE
must therefore be inherited before staging, rather than merely checking
that some inherited ACE exists on the published file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import pm.store as store_module
import pm.install as install_module
from pm.package import InstallError, Package
from pm.store import Store, _reset_scratch_dacl


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return None


@pytest.mark.platforms("windows")
def test_reset_runs_icacls_reset_on_windows(tmp_path, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(store_module.subprocess, "run", recorder)

    _reset_scratch_dacl(tmp_path)

    assert len(recorder.calls) == 1
    command, kwargs = recorder.calls[0]
    assert "icacls" in command[0].lower()
    assert command[1] == str(tmp_path)
    assert "/reset" in command
    assert "/T" not in command
    assert kwargs.get("check") is True
    assert kwargs.get("timeout") == 60


@pytest.mark.platforms("windows")
def test_reset_resolves_icacls_through_system_root(tmp_path, monkeypatch):
    recorder = _Recorder()
    system_root = tmp_path / "Windows"
    system32 = system_root / "System32"
    system32.mkdir(parents=True)
    (system32 / "icacls.exe").write_bytes(b"")
    monkeypatch.setenv("SystemRoot", str(system_root))
    monkeypatch.setattr(store_module.subprocess, "run", recorder)

    _reset_scratch_dacl(tmp_path)

    assert recorder.calls[0][0][0] == str(system32 / "icacls.exe")


@pytest.mark.platforms("linux", "macos")
def test_reset_is_a_noop_off_windows(tmp_path, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(store_module.subprocess, "run", recorder)

    _reset_scratch_dacl(tmp_path)

    assert recorder.calls == []


@pytest.mark.platforms("windows")
def test_scratch_resets_before_yield(tmp_path, monkeypatch):
    order = []
    real_mkdtemp = store_module.tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        order.append(("mkdtemp", path))
        return path

    def record_icacls(command, **kwargs):
        order.append(("icacls", command[1]))
        return None

    monkeypatch.setattr(store_module.tempfile, "mkdtemp", spy_mkdtemp)
    monkeypatch.setattr(store_module.subprocess, "run", record_icacls)

    store = Store(tmp_path)
    with store.scratch() as scratch:
        assert Path(scratch).is_dir()
        assert Path(scratch).name.startswith(".staging-")
        assert [event[0] for event in order] == ["mkdtemp", "icacls"]
        assert order[0][1] == order[1][1] == str(scratch)

    assert not Path(scratch).exists()


@pytest.mark.platforms("windows")
def test_scratch_rejects_icacls_failure_and_cleans_up(tmp_path, monkeypatch):
    def failing(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr=b"Access is denied.")

    monkeypatch.setattr(store_module.subprocess, "run", failing)

    store = Store(tmp_path)
    with pytest.raises(OSError, match="Access is denied"):
        with store.scratch():
            pytest.fail("scratch must not yield after an ACL reset failure")

    assert list(tmp_path.glob(".staging-*")) == []


@pytest.mark.platforms("windows")
def test_install_reports_acl_reset_failure_without_publishing(tmp_path, monkeypatch):
    def failing(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr=b"Access is denied.")

    monkeypatch.setattr(store_module.subprocess, "run", failing)
    monkeypatch.setattr(install_module, "_settle_previous_entry", lambda *args: None)
    monkeypatch.setattr(install_module, "_entry_current", lambda *args: False)
    staged = []

    def prepare(_package, _store, scratch, *_args, **_kwargs):
        staged.append(scratch / "tree")
        staged[-1].mkdir()
        return staged[-1]

    monkeypatch.setattr(install_module, "_prepare_artifacts", prepare)
    monkeypatch.setattr(install_module, "_remove_downloads", lambda *args: None)

    package = Package()
    package.name = "acl-probe"
    lockfile = SimpleNamespace(
        version=lambda _name: "1.0",
        artifacts=lambda *_args: [
            {"url": "https://example.invalid/tool.zip", "sha256": "a" * 64}
        ],
    )
    store = Store(tmp_path)

    with pytest.raises(InstallError, match="Access is denied"):
        install_module._install(
            package, lockfile, None, store, "win32-x64", _lock_held=True
        )

    assert staged == []
    assert not store.entry("acl-probe-1.0-win32-x64").exists()
    assert list(tmp_path.glob(".staging-*")) == []


@pytest.mark.platforms("windows")
def test_scratch_rejects_icacls_timeout_and_cleans_up(tmp_path, monkeypatch):
    def hanging(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(store_module.subprocess, "run", hanging)

    store = Store(tmp_path)
    with pytest.raises(OSError, match="timed out after 60 seconds"):
        with store.scratch():
            pytest.fail("scratch must not yield after an ACL reset timeout")
    assert list(tmp_path.glob(".staging-*")) == []


@pytest.mark.platforms("linux", "macos")
def test_scratch_has_no_dacl_reset_payload_off_windows(tmp_path, monkeypatch):
    seen = []
    recorder = _Recorder()
    real_mkdtemp = store_module.tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        seen.append(path)
        return path

    monkeypatch.setattr(store_module.tempfile, "mkdtemp", spy_mkdtemp)
    monkeypatch.setattr(store_module.subprocess, "run", recorder)

    store = Store(tmp_path)
    with store.scratch() as scratch:
        pass

    assert seen == [str(scratch)]
    assert recorder.calls == []


@pytest.mark.platforms("windows")
def test_scratch_children_inherit_the_current_users_ace(tmp_path):
    if sys.version_info < (3, 12, 4):
        pytest.skip("Windows tempfile DACL hardening requires Python 3.12.4+")

    icacls = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
    if not icacls.is_file():
        pytest.skip("System32 icacls.exe is required for the inheritance probe")
    powershell = (
        Path(os.environ["SystemRoot"])
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip("Windows PowerShell is required for the SID inheritance probe")

    store_root = tmp_path / "store"
    store_root.mkdir()
    env = os.environ.copy()
    env["HERMES_ACL_TEST_STORE"] = str(store_root)
    grant_user_ace = r"""
$ErrorActionPreference = 'Stop'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$inherit = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
$rule = [System.Security.AccessControl.FileSystemAccessRule]::new($identity, [System.Security.AccessControl.FileSystemRights]::FullControl, $inherit, [System.Security.AccessControl.PropagationFlags]::None, [System.Security.AccessControl.AccessControlType]::Allow)
$acl = Get-Acl -LiteralPath $env:HERMES_ACL_TEST_STORE
$acl.AddAccessRule($rule)
Set-Acl -LiteralPath $env:HERMES_ACL_TEST_STORE -AclObject $acl
"""
    subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", grant_user_ace],
        check=True,
        capture_output=True,
        timeout=60,
        env=env,
    )

    def assert_current_user_ace(path: Path, inherited: bool) -> None:
        env["HERMES_ACL_TEST_CHILD"] = str(path)
        env["HERMES_ACL_EXPECT_INHERITED"] = "1" if inherited else "0"
        subprocess.run(
            [
                str(powershell),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                r"""
$ErrorActionPreference = 'Stop'
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$acl = Get-Acl -LiteralPath $env:HERMES_ACL_TEST_CHILD
$inherited = $false
foreach ($ace in $acl.Access) {
    $aceSid = $ace.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
    if ($aceSid -eq $sid -and $ace.IsInherited) { $inherited = $true }
}
$expected = $env:HERMES_ACL_EXPECT_INHERITED -eq '1'
if ($inherited -ne $expected) { throw "current-user ACE inheritance mismatch" }
""",
            ],
            capture_output=True,
            check=True,
            timeout=60,
            env=env,
        )

    store = Store(store_root)
    with store.scratch() as scratch:
        staged = Path(scratch) / "tree"
        staged.mkdir()
        child = staged / "child.bin"
        child.write_bytes(b"x")
        published = store.publish(staged, "with-reset")
        assert_current_user_ace(published / child.name, inherited=True)

    reset_scratch_dacl = store_module._reset_scratch_dacl
    try:
        store_module._reset_scratch_dacl = lambda _path: None
        with store.scratch() as scratch:
            staged = Path(scratch) / "tree"
            staged.mkdir()
            child = staged / "child.bin"
            child.write_bytes(b"x")
            published = store.publish(staged, "without-reset")
            assert_current_user_ace(published / child.name, inherited=False)
    finally:
        store_module._reset_scratch_dacl = reset_scratch_dacl
