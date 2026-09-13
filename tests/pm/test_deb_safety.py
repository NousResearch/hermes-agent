"""DebPackage._safe_untar containment regressions.

A safe extractor must forbid ANY write or chmod outside the staged tree.
The malicious .deb fixtures are real ar+tars built in a temp sandbox and
pointed at an outside SENTINEL file/dir inside the sandbox — never real
user or /etc data. Absolute-symlink rejection runs everywhere; the
symlink-privilege-dependent shapes are host-gated because Windows may lack
symlink privilege (validator-checked before the test bodies create links).
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from pm.package import DebPackage, InstallError


class _P(DebPackage):
    name = "evil-deb"


def _ar_member(name: str, data: bytes) -> bytes:
    hdr = (
        name.ljust(16).encode()
        + b"0".ljust(12)
        + b"0".ljust(6)
        + b"0".ljust(6)
        + b"100644".ljust(8)
        + str(len(data)).encode().ljust(10)
        + b"`\n"
    )
    pad = b"\n" if len(data) % 2 else b""
    return hdr + data + pad


def _build_deb(path: Path, members: list[tarfile.TarInfo | tuple[str, bytes]]) -> None:
    """A real ar archive with one data.tar (uncompressed) holding `members`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for item in members:
            if isinstance(item, tuple):
                name, content = item
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))
            else:
                tf.addfile(item)
    path.write_bytes(
        b"!<arch>\n"
        + _ar_member("debian-binary", b"2.0\n")
        + _ar_member("data.tar", buf.getvalue())
    )


def _symlink_member(name: str, linkname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = linkname
    info.mode = 0o777
    return info


def test_chained_aliases_survive_without_symlink_support(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", unavailable)
    deb = tmp_path / "aliases.deb"
    lib = "data/data/com.termux/files/usr/lib"
    _build_deb(deb, [
        _symlink_member(f"{lib}/libexample.so", "libexample.so.1"),
        _symlink_member(f"{lib}/libexample.so.1", "libexample.so.1.2"),
        (f"{lib}/libexample.so.1.2", b"library payload"),
    ])
    staged = tmp_path / "staged"
    staged.mkdir()
    _P().unpack(deb, staged, "linux-arm64-bionic")
    for name in ("libexample.so", "libexample.so.1", "libexample.so.1.2"):
        assert (staged / lib / name).read_bytes() == b"library payload"


def test_absolute_symlink_target_rejected(tmp_path: Path):
    """A member whose symlink target is ABSOLUTE must be refused outright:
    the link points outside the staged tree the moment it is created."""
    deb = tmp_path / "abs-link.deb"
    _build_deb(
        deb,
        [
            _symlink_member("data/data/com.termux/files/usr/bin/tool", "/etc/shadow"),
        ],
    )
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(InstallError):
        _P().unpack(deb, staged, "linux-arm64-bionic")
    assert not (staged / "data").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "shape",
    ["write-through", "chmod-follow"],
)
def test_symlink_escape_forbidden(tmp_path: Path, shape: str):
    """A symlink planted in the staged tree must never become a path OUT:
    not for a later data member written through it (write-through), and not
    for the final mode-normalization walk following it (chmod-follow)."""
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("do not touch")
    os.chmod(sentinel, 0o600)

    linkname = str(sentinel if shape == "chmod-follow" else outside)
    members: list = [_symlink_member("link", linkname)]
    if shape == "write-through":
        members.append(("link/evil.txt", b"escaped"))
    else:
        members.append(("regular.txt", b"payload"))
    deb = tmp_path / f"escape-{shape}.deb"
    _build_deb(deb, members)

    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(InstallError):
        _P().unpack(deb, staged, "linux-arm64-bionic")

    assert sentinel.read_text() == "do not touch", (
        f"{shape}: a staged symlink was followed outside the tree"
    )
    if shape == "chmod-follow":
        assert os.stat(sentinel).st_mode & 0o777 == 0o600, (
            "the mode walk chmod'd a file outside the staged tree"
        )
    if shape == "write-through":
        assert not (outside / "evil.txt").exists()
