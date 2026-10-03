"""Regression for the Ubuntu 22.04 bootstrap interpreter: python-build-standalone
tarballs carry relative terminfo symlinks (``share/terminfo/1/1178 -> ../a/adm1178``)."""

import io
import os
import tarfile

import pytest

from pm.store import extract


def _tar(tmp_path, members):
    archive = tmp_path / "pkg.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, linkname in members:
            info = tarfile.TarInfo(name)
            if linkname is None:
                data = b"x"
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = linkname
                tf.addfile(info)
    return archive


def test_relative_symlinks_resolve_from_their_own_directory(tmp_path):
    archive = _tar(tmp_path, [("python/share/terminfo/a/adm1178", None), ("python/share/terminfo/1/1178", "../a/adm1178")])
    dest = tmp_path / "out"
    extract(archive, dest)
    link = dest / "python/share/terminfo/1/1178"
    assert os.readlink(link) == "../a/adm1178"
    assert link.resolve() == (dest / "python/share/terminfo/a/adm1178").resolve()


@pytest.mark.parametrize("linkname", ["../../../etc/passwd", "/etc/passwd"])
def test_symlinks_escaping_the_destination_are_rejected(tmp_path, linkname):
    archive = _tar(tmp_path, [("python/bin/evil", linkname)])
    with pytest.raises(tarfile.FilterError):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "python/bin/evil").is_symlink()


def _portable_git(tmp_path):
    archive = tmp_path / "fetch" / "PortableGit-2.53.0.3-64-bit.7z.exe"
    archive.parent.mkdir()
    archive.write_bytes(b"pinned bytes; the fake run never executes them")
    return archive


class _FakeProc:
    """Popen stand-in: ``wait`` returns or hangs, ``kill`` is recorded."""

    def __init__(self, argv, returncode=0, hang=False):
        self.argv = argv
        self.returncode = returncode
        self.hang = hang
        self.killed = False

    def wait(self, timeout=None):
        import subprocess

        if self.hang:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode

    def kill(self):
        self.killed = True


def _fake_popen(monkeypatch, returncode=0, calls=None, hang=False):
    """Patch ``subprocess.Popen``; returns the fake processes it handed out."""
    import subprocess

    procs = []

    def popen(argv, **kwargs):
        proc = _FakeProc(argv, returncode=returncode, hang=hang)
        procs.append(proc)
        if calls is not None:
            calls.append((argv, kwargs))
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    return procs


def test_git_unpack_executes_a_scratch_copy_never_the_cached_bytes(tmp_path, monkeypatch):
    """An executed PE can stay handle-held past pm's download-cleanup retry
    (WinError 32), so the extractor runs from a disposable copy beside the
    staging tree, never from the cached fetch-<sha> entry."""
    import subprocess
    from pathlib import Path
    import pm.packages
    from pm.packages import Git

    monkeypatch.setattr(pm.packages, "_HOST_IS_WINDOWS", True)
    calls = []
    _fake_popen(monkeypatch, calls=calls)
    archive = _portable_git(tmp_path)
    staged = tmp_path / "scratch" / "tree"
    Git().unpack(archive, staged, "win32-x64")
    (argv, kwargs), = calls
    assert argv[1:] == [f"-o{staged}", "-y"]
    exe = Path(argv[0])
    assert archive.parent not in exe.parents
    assert staged.parent in exe.parents
    assert not exe.exists(), "the scratch copy is removed after the run"
    # Inherited pipes would let the RunProgram children hold run() open.
    assert all(kwargs[k] is subprocess.DEVNULL for k in ("stdin", "stdout", "stderr"))


def test_git_unpack_names_the_extractor_exit_code(tmp_path, monkeypatch):
    """Under -y the stub reports nothing, so the error carries the exit code."""
    import pm.packages
    from pm.package import InstallError
    from pm.packages import Git

    monkeypatch.setattr(pm.packages, "_HOST_IS_WINDOWS", True)
    _fake_popen(monkeypatch, returncode=7)
    with pytest.raises(InstallError, match="exited 7"):
        Git().unpack(_portable_git(tmp_path), tmp_path / "scratch" / "tree", "win32-x64")


def test_git_unpack_timeout_is_a_package_error(tmp_path, monkeypatch):
    import pm.packages
    from pm.package import InstallError
    from pm.packages import Git

    monkeypatch.setattr(pm.packages, "_HOST_IS_WINDOWS", True)
    _fake_popen(monkeypatch, hang=True)
    with pytest.raises(InstallError, match="did not finish"):
        Git().unpack(_portable_git(tmp_path), tmp_path / "scratch" / "tree", "win32-x64")


def test_git_unpack_timeout_tears_down_the_whole_tree(tmp_path, monkeypatch):
    """A timed-out extractor must not leave its descendants running.

    The stub exits while its RunProgram children keep going, so ``kill()`` on the
    pid alone leaves them holding the staged tree and the scratch dir. The
    timeout path therefore routes through the shared portable teardown
    (``taskkill /F /T`` on Windows), exactly like ``install.ps1``.
    """
    import hermes_cli._subprocess_compat as compat
    import pm.packages
    from pm.package import InstallError
    from pm.packages import Git

    monkeypatch.setattr(pm.packages, "_HOST_IS_WINDOWS", True)
    procs = _fake_popen(monkeypatch, hang=True)
    torn_down = []
    monkeypatch.setattr(compat, "kill_process_tree", lambda proc: torn_down.append(proc))
    with pytest.raises(InstallError, match="did not finish"):
        Git().unpack(_portable_git(tmp_path), tmp_path / "scratch" / "tree", "win32-x64")
    assert torn_down == procs, "the whole extractor tree must be torn down on timeout"


def test_git_unpack_requires_a_windows_host(tmp_path, monkeypatch):
    """Off Windows the PE extractor cannot run: refuse before executing
    anything, with a remedy that is not "retry"."""
    import pm.packages
    from pm.package import InstallError
    from pm.packages import Git

    calls = []
    _fake_popen(monkeypatch, calls=calls)
    monkeypatch.setattr(pm.packages, "_HOST_IS_WINDOWS", False)
    with pytest.raises(InstallError, match="Windows host") as excinfo:
        Git().unpack(_portable_git(tmp_path), tmp_path / "out", "win32-x64")
    assert "retry" not in str(excinfo.value)
    assert not calls, "the guard must refuse before any execution"


def test_target_uses_shared_native_arch(monkeypatch):
    from hermes_platform.host import facts
    from pm import store

    monkeypatch.setattr(facts, "native_arch", lambda: "arm64")
    assert store._native_machine() == "arm64"
