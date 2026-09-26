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

def test_git_pin_is_the_self_extracting_archive():
    """Both win32 targets pin the self-extracting PortableGit archive: it
    carries its own extractor, so no tar/bzip2 is needed anywhere (#122512)."""
    from pm.packages import Git

    for target in ("win32-x64", "win32-arm64"):
        assert Git().fetch_url("2.53.0+3", target).endswith(".7z.exe")


def test_git_unpack_executes_a_scratch_copy_never_the_cached_bytes(tmp_path, monkeypatch):
    """Executing the cached fetch-<sha> artifact in place kept it handle-held
    (Defender on-execute scan, the stub's RunProgram child chain) past pm's
    ~2 s download-cleanup retry and failed the install with WinError 32
    (CI run 36189416163). The extractor must run from a disposable copy
    beside the staging tree, where the .staging-* teardown owns it."""
    import subprocess
    from pathlib import Path
    from pm.packages import Git

    calls = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv) or _Result())
    archive = tmp_path / "fetch" / "PortableGit-2.53.0.3-64-bit.7z.exe"
    archive.parent.mkdir()
    archive.write_bytes(b"pinned bytes; the fake run never executes them")
    staged = tmp_path / "scratch" / "tree"
    Git().unpack(archive, staged, "win32-x64")
    argv = calls[0]
    assert argv[1:] == [f"-o{staged}", "-y"]
    assert argv[0] != str(archive)
    assert archive.parent not in Path(argv[0]).parents
    assert staged.parent in Path(argv[0]).parents


def test_git_unpack_names_the_extractor_exit_code(tmp_path, monkeypatch):
    """Under -y the GUI stub fails silently (no stdout, no stderr, no error
    box), so the message must carry the exit code and the usual causes
    instead of promising captured output (#123094 review)."""
    import subprocess
    from pm.packages import Git

    class _Result:
        returncode = 7
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Result())
    archive = tmp_path / "fetch" / "PortableGit-2.53.0.3-64-bit.7z.exe"
    archive.parent.mkdir()
    archive.write_bytes(b"pinned bytes")
    with pytest.raises(RuntimeError) as excinfo:
        Git().unpack(archive, tmp_path / "scratch" / "tree", "win32-x64")
    message = str(excinfo.value)
    assert "exit code 7" in message
    assert "silent" in message


def test_git_unpack_requires_a_windows_host(tmp_path, monkeypatch):
    """Cross-host staging of win32 git worked with the old tar.bz2 pin; the
    self-extracting pin must refuse off-Windows with that trade-off stated
    instead of a raw exec error (#123094 review)."""
    import subprocess
    import pm.packages
    from pm.packages import Git

    calls = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv) or _Result())
    monkeypatch.setattr(pm.packages, "_HOST_IS_WINDOWS", False, raising=False)
    archive = tmp_path / "x.7z.exe"
    archive.write_bytes(b"pinned bytes")
    with pytest.raises(RuntimeError, match="Windows host"):
        Git().unpack(archive, tmp_path / "out", "win32-x64")
    assert not calls, "the guard must refuse before any execution"


def test_target_uses_shared_native_arch(monkeypatch):
    from hermes_platform.host import facts
    from pm import store

    monkeypatch.setattr(facts, "native_arch", lambda: "arm64")
    assert store._native_machine() == "arm64"
