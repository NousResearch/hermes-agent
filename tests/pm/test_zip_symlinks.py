"""Zip extraction must preserve symlinks — and refuse hostile ones.

Mac chromium .framework zips carry a layout that is mostly symlinks
(Versions/Current, Resources, ...). zipfile.extract() writes a symlink
entry's TARGET PATH out as a regular file, which destroys the bundle:
codesign then refuses even a fresh re-sign. So pm's _extract_zip
recreates link entries with os.symlink — but only the safe ones, and
only after every regular file has landed.

A zip symlink entry is a file whose content is the target path and
whose external_attr mode says S_ISLNK. The fixtures here build REAL
entries of that shape, the same bytes a hostile server would send.

Ported from restack's provisioner-zip-symlinks suite, rewritten against
pm.store.extract. Containment arms (hostile links skipped, zip-slip
ordering) are host-independent and run on Windows; the arms asserting
links RESOLVE are POSIX-gated; a separate arm pins pm's truthful win32
degradation (symlink if the host allows it, else a text placeholder —
never an escape).
"""

from __future__ import annotations

import io
import stat
import sys
import zipfile
from pathlib import Path

import pytest

from pm.store import extract

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink resolution semantics are POSIX"
)


def _add_symlink(zf: zipfile.ZipFile, member: str, target: str) -> None:
    """Append a symlink entry the way zip tools actually encode one."""
    info = zipfile.ZipInfo(member)
    info.external_attr = (stat.S_IFLNK | 0o755) << 16
    zf.writestr(info, target)


def _add_file(
    zf: zipfile.ZipFile, member: str, data: bytes = b"x", mode: int = 0o644
) -> None:
    info = zipfile.ZipInfo(member)
    info.external_attr = (stat.S_IFREG | mode) << 16
    zf.writestr(info, data)


def _write_zip(path: Path, build) -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        build(zf)
    path.write_bytes(buf.getvalue())
    return path


@posix_only
def test_framework_shaped_links_round_trip(tmp_path: Path) -> None:
    """The CfT .framework shape: links out of the versioned dir, intact."""

    def build(zf: zipfile.ZipFile) -> None:
        _add_file(zf, "app/F.framework/Versions/A/F", b"machO", mode=0o755)
        _add_file(zf, "app/F.framework/Versions/A/Resources/Info.plist", b"<plist/>")
        _add_symlink(zf, "app/F.framework/Versions/Current", "A")
        _add_symlink(zf, "app/F.framework/Resources", "Versions/Current/Resources")
        _add_symlink(zf, "app/F.framework/F", "Versions/Current/F")

    dest = tmp_path / "out"
    extract(_write_zip(tmp_path / "a.zip", build), dest)

    current = dest / "app/F.framework/Versions/Current"
    assert current.is_symlink() and current.readlink() == Path("A")
    resources = dest / "app/F.framework/Resources"
    assert resources.is_symlink()
    # The links RESOLVE: the whole point of preserving them.
    assert (resources / "Info.plist").read_bytes() == b"<plist/>"
    assert (dest / "app/F.framework/F").read_bytes() == b"machO"
    # And the binary kept its exec bit through the same pass.
    assert (dest / "app/F.framework/Versions/A/F").stat().st_mode & 0o111


@posix_only
def test_intree_dotdot_target_is_kept(tmp_path: Path) -> None:
    """`..` in a target is fine while it stays under dest — real layouts
    (lib/foo -> ../share/foo) depend on it."""

    def build(zf: zipfile.ZipFile) -> None:
        _add_file(zf, "share/data.txt", b"d")
        _add_symlink(zf, "lib/data", "../share/data.txt")

    dest = tmp_path / "out"
    extract(_write_zip(tmp_path / "a.zip", build), dest)
    link = dest / "lib/data"
    assert link.is_symlink() and link.read_bytes() == b"d"


@pytest.mark.parametrize(
    ("member", "target"),
    [
        ("lib/evil", "../../../victim"),  # escapes via target walk-up
        ("lib/evil", "/etc/passwd"),  # absolute target
        ("../evil", "whatever"),  # link path itself escapes
    ],
)
def test_hostile_links_are_skipped_not_written(
    tmp_path: Path, member: str, target: str
) -> None:
    """Containment is host-independent: pm decides to skip a hostile
    link BEFORE any symlink creation, so this must hold on Windows too."""

    def build(zf: zipfile.ZipFile) -> None:
        _add_file(zf, "lib/ok.txt", b"ok")
        _add_symlink(zf, member, target)

    dest = tmp_path / "deep" / "out"
    extract(_write_zip(tmp_path / "a.zip", build), dest)

    # The good entry landed; the hostile link exists NOWHERE — not under
    # dest, not at the escape destination.
    assert (dest / "lib/ok.txt").read_bytes() == b"ok"
    assert not (dest / "lib/evil").exists() and not (dest / "lib/evil").is_symlink()
    assert not (tmp_path / "victim").exists()
    assert not (tmp_path / "evil").exists()


def test_file_written_through_earlier_link_entry_cannot_escape(
    tmp_path: Path,
) -> None:
    """Zip-slip ordering: a link entry followed by a file entry routed
    through it. Links are recreated last, so the file lands as a real
    path and nothing is written outside dest. Host-independent."""

    def build(zf: zipfile.ZipFile) -> None:
        _add_symlink(zf, "outdir", "../..")
        _add_file(zf, "outdir/pwned.txt", b"pwned")

    dest = tmp_path / "deep" / "out"
    extract(_write_zip(tmp_path / "a.zip", build), dest)

    assert not (tmp_path / "pwned.txt").exists()
    written = dest / "outdir" / "pwned.txt"
    assert written.is_file() and not (dest / "outdir").is_symlink()


@pytest.mark.skipif(sys.platform != "win32", reason="pins the win32 degradation")
def test_win32_safe_link_degrades_without_escaping(tmp_path: Path) -> None:
    """On win32 pm attempts a real symlink (works with Developer Mode /
    admin) and otherwise degrades to a text placeholder holding the
    target path. Either way the entry stays inside dest — pin exactly
    that, without faking the host."""

    def build(zf: zipfile.ZipFile) -> None:
        _add_file(zf, "share/data.txt", b"d")
        _add_symlink(zf, "lib/data", "../share/data.txt")

    dest = tmp_path / "out"
    extract(_write_zip(tmp_path / "a.zip", build), dest)

    link = dest / "lib" / "data"
    if link.is_symlink():
        assert link.readlink() == Path("../share/data.txt")
    else:
        assert link.is_file()
        assert link.read_text(encoding="utf-8") == "../share/data.txt"
    # Nothing anywhere else.
    assert set(p.name for p in (dest / "lib").iterdir()) == {"data"}


def test_plain_zip_unchanged(tmp_path: Path) -> None:
    """No links: behavior identical to before, exec bits included."""

    def build(zf: zipfile.ZipFile) -> None:
        _add_file(zf, "bin/tool", b"#!", mode=0o755)
        _add_file(zf, "README", b"r")

    dest = tmp_path / "out"
    extract(_write_zip(tmp_path / "a.zip", build), dest)
    if sys.platform != "win32":
        assert (dest / "bin/tool").stat().st_mode & 0o111
    assert (dest / "bin/tool").read_bytes() == b"#!"
    assert (dest / "README").read_bytes() == b"r"
