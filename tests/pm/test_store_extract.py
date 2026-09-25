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

def test_git_staging_needs_no_external_decompressor(tmp_path):
    """Git staging on either side must not route through tar/bzip2 (#122512).

    The old .tar.bz2 pin died on Windows 10 boxes whose System32 tar.exe
    cannot run the bzip2 filter ("unable to run program bzip2 -d").
    """
    from pathlib import Path
    from pm.packages import Git

    # pm pins the self-extracting PortableGit archive for every Windows target.
    for target in ("win32-x64", "win32-arm64"):
        assert Git().fetch_url("2.53.0+3", target).endswith(".7z.exe")
    # The pre-PM bootstrap stages the same pin and never shells to tar.exe.
    installer = Path(__file__).resolve().parents[2] / "scripts" / "install.ps1"
    text = installer.read_text(encoding="utf-8-sig")
    assert "$inboxTar" not in text
    assert "System32\\tar.exe" not in text
    # A corrupt payload fails loudly instead of being silently unpacked.
    junk = tmp_path / "git.7z.exe"
    junk.write_bytes(b"not an archive")
    with pytest.raises(OSError):
        Git().unpack(junk, tmp_path / "out", "win32-x64")


def test_target_uses_shared_native_arch(monkeypatch):
    from hermes_platform.host import facts
    from pm import store

    monkeypatch.setattr(facts, "native_arch", lambda: "arm64")
    assert store._native_machine() == "arm64"
