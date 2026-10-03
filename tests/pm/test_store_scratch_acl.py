"""Scratch dirs must publish with the store root's inherited Windows ACL.

``tempfile.mkdtemp()`` gained a protected DACL on Windows in Python 3.12.4
(CVE-2024-4030: SYSTEM, Administrators and OWNER RIGHTS, inheritance
disabled), and ``Store.publish`` keeps that descriptor through its
same-volume rename — leaving installed tool entries without the user's
ACE after an elevated update (#122935). The scratch context manager
re-enables inheritance from the store root before anything is staged.

The subprocess call itself is observed through a stub: the descriptor
semantics only exist on Windows, which this suite does not require. The
platform flag is swapped only inside the store module's namespace — a
global ``os.name`` patch would flip pathlib's class dispatch and break
the host's own cleanup paths. One ``platforms("windows")`` test at the
end exercises the real ACL semantics on the native Windows lanes.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import pm.store as store_module
from pm.store import Store, _reset_scratch_dacl


def _windows_store(monkeypatch):
    # environ stays the real mapping so the SystemRoot probe inside the
    # argv builder keeps working under the module swap.
    monkeypatch.setattr(
        store_module, "os", SimpleNamespace(name="nt", environ=os.environ)
    )


def _posix_store(monkeypatch):
    monkeypatch.setattr(store_module, "os", SimpleNamespace(name="posix"))


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return None


def test_reset_runs_icacls_reset_on_windows(tmp_path, monkeypatch):
    """One bounded icacls call on the scratch dir itself — no /T, no other targets."""
    recorder = _Recorder()
    _windows_store(monkeypatch)
    monkeypatch.setattr(store_module.subprocess, "run", recorder)

    _reset_scratch_dacl(tmp_path)

    assert len(recorder.calls) == 1
    command, kwargs = recorder.calls[0]
    assert "icacls" in command[0].lower()  # bare name or the System32 resolve
    assert command[1] == str(tmp_path)
    assert "/reset" in command
    # Resetting the single directory is what lets every staged byte inherit
    # the store root's ACEs afterwards; a recursive /T would also touch
    # thousands of already-inherited entries for nothing.
    assert "/T" not in command
    assert kwargs.get("check") is True
    # scratch() yields into download/unpack — a hung icacls must not wedge it.
    assert kwargs.get("timeout") == 60


def test_reset_resolves_icacls_through_system_root(tmp_path, monkeypatch):
    """A post-install PATH may not carry icacls; System32 wins when present."""
    recorder = _Recorder()
    system_root = tmp_path / "Windows"
    system32 = system_root / "System32"
    system32.mkdir(parents=True)
    (system32 / "icacls.exe").write_bytes(b"")
    monkeypatch.setenv("SystemRoot", str(system_root))
    _windows_store(monkeypatch)
    monkeypatch.setattr(store_module.subprocess, "run", recorder)

    _reset_scratch_dacl(tmp_path)

    assert recorder.calls[0][0][0] == str(system32 / "icacls.exe")


def test_reset_is_a_noop_off_windows(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("icacls must not run off Windows")

    _posix_store(monkeypatch)
    monkeypatch.setattr(store_module.subprocess, "run", fail)

    _reset_scratch_dacl(tmp_path)  # returns without raising


def test_scratch_resets_before_yield(tmp_path, monkeypatch):
    """mkdtemp -> ACL reset -> yield, so no staged byte sees the hardened DACL."""
    order = []
    _windows_store(monkeypatch)
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
    assert not Path(scratch).exists()  # context exit cleans the scratch dir


def test_scratch_survives_icacls_failure(tmp_path, monkeypatch, capsys):
    """A failed reset degrades to the pre-fix install path, never to a crash,
    and the warning names the icacls refusal instead of a bare exit status."""

    def failing(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr=b"Access is denied.")

    _windows_store(monkeypatch)
    monkeypatch.setattr(store_module.subprocess, "run", failing)

    store = Store(tmp_path)
    with store.scratch() as scratch:
        assert Path(scratch).is_dir()
    assert not Path(scratch).exists()
    err = capsys.readouterr().err
    assert "could not restore inheritable ACL" in err
    assert "Access is denied." in err


def test_scratch_survives_icacls_hang(tmp_path, monkeypatch):
    """A wedged icacls times out; the install path still completes."""

    def hanging(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=kwargs.get("timeout"))

    _windows_store(monkeypatch)
    monkeypatch.setattr(store_module.subprocess, "run", hanging)

    store = Store(tmp_path)
    with store.scratch() as scratch:
        assert Path(scratch).is_dir()
    assert not Path(scratch).exists()


def test_scratch_has_no_dacl_reset_payload_off_windows(tmp_path, monkeypatch):
    """Off Windows the scratch flow stays exactly mkdtemp + cleanup."""
    seen = []
    real_mkdtemp = store_module.tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        seen.append(path)
        return path

    _posix_store(monkeypatch)
    monkeypatch.setattr(store_module.tempfile, "mkdtemp", spy_mkdtemp)
    monkeypatch.setattr(store_module.subprocess, "run", _Recorder())

    store = Store(tmp_path)
    with store.scratch() as scratch:
        pass

    assert seen == [str(scratch)]


def test_tempfile_reference_is_module_level():
    """The suite above patches the module tempfile; keep that import honest."""
    assert store_module.tempfile is tempfile


@pytest.mark.platforms("windows")
def test_scratch_children_reinherit_the_store_root_dacl(tmp_path):
    """Real ACL semantics on native Windows: a file staged after the reset
    carries the store root's inheritable ACE (the icacls "(I)" flag)."""
    icacls = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
    if not icacls.is_file():
        pytest.skip("System32 icacls.exe is required for the inheritance probe")
    store = Store(tmp_path)
    with store.scratch() as scratch:
        child = Path(scratch) / "child.bin"
        child.write_bytes(b"x")
        probe = subprocess.run(
            [str(icacls), str(child)],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout.decode("utf-8", "replace")
    assert "(I)" in probe
