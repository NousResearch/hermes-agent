"""Regression for the Ubuntu 22.04 bootstrap interpreter: python-build-standalone
tarballs carry relative terminfo symlinks (``share/terminfo/1/1178 -> ../a/adm1178``)."""

import io
import os
import tarfile

import pytest

from pm.store import extract, tarfile_with_filters


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
    with pytest.raises(tarfile_with_filters().FilterError):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "python/bin/evil").is_symlink()

def test_git_tar_ignores_msys_mount_table_link(tmp_path):
    from pm.packages import Git

    archive = tmp_path / "git.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tf:
        mtab = tarfile.TarInfo("etc/mtab")
        mtab.type, mtab.linkname = tarfile.SYMTYPE, "/proc/mounts"
        tf.addfile(mtab)
        binary = tarfile.TarInfo("cmd/git.exe")
        binary.size = 1
        tf.addfile(binary, io.BytesIO(b"x"))
    dest = tmp_path / "out"
    Git().unpack(archive, dest, "win32-x64")
    assert (dest / "cmd/git.exe").read_bytes() == b"x"
    assert not (dest / "etc/mtab").is_symlink()


def test_git_tar_rejects_unrelated_mount_table_links(tmp_path):
    from pm.packages import Git

    archive = tmp_path / "git.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tf:
        mtab = tarfile.TarInfo("etc/mtab")
        mtab.type, mtab.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(mtab)
    with pytest.raises(tarfile_with_filters().FilterError):
        Git().unpack(archive, tmp_path / "out", "win32-x64")


def test_git_tar_skips_only_msys_proc_links_and_rejects_other_unsafe_entries(tmp_path):
    from pm.packages import Git

    archive = tmp_path / "git.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tf:
        proc = tarfile.TarInfo("dev/fd")
        proc.type, proc.linkname = tarfile.SYMTYPE, "/proc/self/fd"
        tf.addfile(proc)
        good = tarfile.TarInfo("cmd/git.exe")
        good.size = 1
        tf.addfile(good, io.BytesIO(b"x"))
        escape = tarfile.TarInfo("../../escaped")
        escape.size = 1
        tf.addfile(escape, io.BytesIO(b"z"))
    with pytest.raises(tarfile_with_filters().FilterError):
        Git().unpack(archive, tmp_path / "out", "win32-x64")
    assert (tmp_path / "out/cmd/git.exe").read_bytes() == b"x"
    assert not (tmp_path / "out/dev/fd").exists()
    assert not (tmp_path / "escaped").exists()


def test_install_ps1_bootstrap_skips_the_same_msys_links_as_pm(tmp_path):
    """The pre-PM bootstrap's tar.exe excludes must stay the links PM skips."""
    import re
    from pathlib import Path
    from pm.packages import Git

    installer = Path(__file__).resolve().parents[2] / "scripts" / "install.ps1"
    listed = re.search(r"\$msysProcLinks = @\(([^)]*)\)", installer.read_text(encoding="utf-8")).group(1)
    excluded = set(re.findall(r"'([^']+)'", listed))
    # The pinned Git-for-Windows archives' symlinks (tar -tvf, 2.53.0.windows.3).
    links = {"dev/fd": "/proc/self/fd", "dev/stdin": "/proc/self/fd/0", "dev/stdout": "/proc/self/fd/1",
             "dev/stderr": "/proc/self/fd/2", "etc/mtab": "/proc/mounts"}
    archive = tmp_path / "git.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tf:
        for name, target in links.items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tf.addfile(info)
        binary = tarfile.TarInfo("cmd/git.exe")
        binary.size = 1
        tf.addfile(binary, io.BytesIO(b"x"))
    dest = tmp_path / "out"
    Git().unpack(archive, dest, "win32-x64")
    skipped = {name for name in links if not os.path.lexists(dest / name)}
    assert excluded == skipped == set(links)


def test_target_uses_shared_native_arch(monkeypatch):
    from hermes_platform.host import facts
    from pm import store

    monkeypatch.setattr(facts, "native_arch", lambda: "arm64")
    assert store._native_machine() == "arm64"


def test_tarfile_with_filters_picks_vendored_module_when_stdlib_lacks_data_filter(monkeypatch):
    """CPython 3.11.0-3.11.3 ship no extraction-filter API (PEP 706 landed in
    3.11.4). The selector must hand back the vendored 3.11.16 tarfile there."""
    from pm.store import tarfile_with_filters

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    selected = tarfile_with_filters()
    assert selected is not tarfile
    assert hasattr(selected, "data_filter")
    assert hasattr(selected, "FilterError")
    assert hasattr(selected.TarInfo, "replace")


def test_extract_tar_works_on_legacy_tarfile(monkeypatch, tmp_path):
    """The #125086 regression: on 3.11.0-3.11.3 the old code raised
    ``TarFile.extractall() got an unexpected keyword argument 'filter'``."""
    from pm.store import extract, tarfile_with_filters

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = _tar(tmp_path, [
        ("python/bin/hermes", None),
        ("python/share/terminfo/a/adm1178", None),
        ("python/share/terminfo/1/1178", "../a/adm1178"),
    ])
    dest = tmp_path / "out"
    extract(archive, dest)
    assert (dest / "python/bin/hermes").read_bytes() == b"x"
    link = dest / "python/share/terminfo/1/1178"
    assert os.readlink(link) == "../a/adm1178"


@pytest.mark.parametrize(
    ("member", "exc_name"),
    [
        (("python/bin/evil", "../../../etc/passwd"), "LinkOutsideDestinationError"),
        (("python/bin/evil", "/etc/passwd"), "AbsoluteLinkError"),
    ],
)
def test_extract_tar_rejects_unsafe_symlinks_on_legacy_tarfile(monkeypatch, tmp_path, member, exc_name):
    from pm.store import extract, tarfile_with_filters

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = _tar(tmp_path, [member])
    with pytest.raises(getattr(tarfile_with_filters(), exc_name)):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "python/bin/evil").is_symlink()


def test_extract_tar_rejects_traversal_and_special_files_on_legacy_tarfile(monkeypatch, tmp_path):
    from pm.store import extract, tarfile_with_filters

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    escape = tmp_path / "pkg.tar.gz"
    with tarfile.open(escape, "w:gz") as tf:
        info = tarfile.TarInfo("../../escaped")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"z"))
    with pytest.raises(tarfile_with_filters().OutsideDestinationError):
        extract(escape, tmp_path / "out")

    special = tmp_path / "special.tar.gz"
    with tarfile.open(special, "w:gz") as tf:
        dev = tarfile.TarInfo("dev/mal")
        dev.type = tarfile.CHRTYPE
        tf.addfile(dev)
    with pytest.raises(tarfile_with_filters().SpecialFileError):
        extract(special, tmp_path / "out2")


def test_deb_payload_filter_errors_become_install_error_on_legacy_tarfile(monkeypatch, tmp_path):
    """pm.package caught ``tarfile.FilterError``, which does not even exist on
    3.11.0-3.11.3 -- the except-clause itself would raise AttributeError."""
    from pm.package import DebPackage, InstallError
    from pm.store import tarfile_with_filters

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tf:
        info = tarfile.TarInfo("python/bin/evil")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(info)
    pkg = DebPackage.__new__(DebPackage)
    with pytest.raises(InstallError) as excinfo:
        pkg._untar_payload(payload.getvalue(), tmp_path / "staged")
    assert "python/bin/evil" in str(excinfo.value)
