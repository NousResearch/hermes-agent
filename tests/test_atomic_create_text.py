"""``utils.atomic_create_text`` / ``unlink_if_unchanged``: create-only publication that never replaces.

``atomic_write_text`` replaces whatever is at the path; ``atomic_create_text`` is for the callers that
must not (a new skill's SKILL.md racing another writer). These tests pin it at the filesystem level:
exclusivity for every kind of occupant, exact bytes and mode, temp-file hygiene, the fallback for
filesystems without hard links (still exclusive), and that cleanup only ever removes the caller's file.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from utils import atomic_create_text, unlink_if_unchanged

TEXT = "내용 ✓ line\nsecond line\n"


@pytest.fixture
def box(tmp_path):
    """An empty directory of its own (tmp_path also holds the hermetic HERMES_HOME)."""
    directory = tmp_path / "box"
    directory.mkdir()
    return directory


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_creates_exact_utf8_bytes_and_leaves_no_temp_file(box):
    target = box / "SKILL.md"
    signature = atomic_create_text(target, TEXT, mode=0o644)
    assert target.read_bytes() == TEXT.encode("utf-8")
    assert _names(box) == ["SKILL.md"]
    st = target.stat()
    assert signature == (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


@pytest.mark.platforms("posix")
def test_mode_is_applied_before_publication_whatever_the_umask(box):
    old = os.umask(0o077)
    try:
        atomic_create_text(box / "a", "x", mode=0o644)
        atomic_create_text(box / "b", "x", mode=0o600)
        atomic_create_text(box / "c", "x")  # no mode: what open(path, "w") would give
    finally:
        os.umask(old)
    assert stat.S_IMODE((box / "a").stat().st_mode) == 0o644
    assert stat.S_IMODE((box / "b").stat().st_mode) == 0o600
    assert stat.S_IMODE((box / "c").stat().st_mode) == 0o600  # umask 077 applies


def test_missing_parent_is_an_error_and_nothing_is_created(box):
    with pytest.raises(FileNotFoundError):
        atomic_create_text(box / "absent" / "SKILL.md", TEXT)
    assert _names(box) == []


def test_existing_regular_file_is_never_replaced(box):
    target = box / "SKILL.md"
    target.write_bytes(b"\xffnot utf-8")
    with pytest.raises(FileExistsError):
        atomic_create_text(target, TEXT)
    assert target.read_bytes() == b"\xffnot utf-8"
    assert _names(box) == ["SKILL.md"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("kind", ["live-symlink", "dangling-symlink", "directory"])
def test_symlinks_and_directories_are_refused_not_followed(box, kind):
    outside = box.parent / "outside.md"
    outside.write_bytes(b"outside")
    target = box / "SKILL.md"
    if kind == "live-symlink":
        target.symlink_to(outside)
    elif kind == "dangling-symlink":
        target.symlink_to(box.parent / "nowhere.md")
    else:
        target.mkdir()
    with pytest.raises(FileExistsError):
        atomic_create_text(target, TEXT)
    assert outside.read_bytes() == b"outside"
    assert not (box.parent / "nowhere.md").exists()
    assert _names(box) == ["SKILL.md"]


def test_a_writer_that_wins_the_last_instant_is_not_replaced(box, monkeypatch):
    target = box / "SKILL.md"
    real_link = os.link

    def plant_then_link(src, dst, *a, **kw):
        Path(dst).write_bytes(b"winner")
        return real_link(src, dst, *a, **kw)

    monkeypatch.setattr(os, "link", plant_then_link)
    with pytest.raises(FileExistsError):
        atomic_create_text(target, TEXT)
    assert target.read_bytes() == b"winner"
    assert _names(box) == ["SKILL.md"]


def test_unexpected_link_errors_propagate_and_clean_up(box, monkeypatch):
    def deny(*a, **kw):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(os, "link", deny)
    with pytest.raises(PermissionError):
        atomic_create_text(box / "SKILL.md", TEXT)
    assert _names(box) == []  # no target, no temp file: a refused create is not turned into a plain write


@pytest.fixture
def no_hard_links(monkeypatch):
    def unsupported(*a, **kw):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(os, "link", unsupported)


@pytest.mark.platforms("posix")
def test_without_hard_links_the_create_is_still_exclusive_and_exact(box, no_hard_links):
    target = box / "SKILL.md"
    signature = atomic_create_text(target, TEXT, mode=0o644)
    assert target.read_bytes() == TEXT.encode("utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert _names(box) == ["SKILL.md"]
    assert unlink_if_unchanged(target, signature) is True
    target.write_bytes(b"human")
    with pytest.raises(FileExistsError):
        atomic_create_text(target, TEXT)
    assert target.read_bytes() == b"human"


@pytest.mark.platforms("posix")
def test_without_hard_links_a_writer_in_the_gap_is_still_not_replaced(box, monkeypatch):
    target = box / "SKILL.md"

    def plant_then_unsupported(src, dst, *a, **kw):
        Path(dst).write_bytes(b"winner")
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(os, "link", plant_then_unsupported)
    with pytest.raises(FileExistsError):
        atomic_create_text(target, TEXT)
    assert target.read_bytes() == b"winner"
    assert _names(box) == ["SKILL.md"]


@pytest.mark.platforms("posix")
def test_without_hard_links_a_failed_write_leaves_no_partial_file(box, no_hard_links, monkeypatch):
    real_fsync = os.fsync
    calls: list[int] = []

    def flaky_fsync(fd):
        calls.append(fd)
        if len(calls) == 2:  # 1st: the staged temp file; 2nd: the in-place create
            raise OSError(errno.EIO, "disk error")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", flaky_fsync)
    with pytest.raises(OSError):
        atomic_create_text(box / "SKILL.md", TEXT)
    assert len(calls) == 2
    assert _names(box) == []


def test_unlink_if_unchanged_removes_only_the_file_it_published(box):
    target = box / "SKILL.md"
    signature = atomic_create_text(target, TEXT)
    assert unlink_if_unchanged(target, signature) is True
    assert not target.exists()
    assert unlink_if_unchanged(target, signature) is False  # already gone

    signature = atomic_create_text(target, TEXT)
    replacement = box / "editor.tmp"
    replacement.write_bytes(b"human save")
    os.replace(replacement, target)  # an editor's atomic save
    assert unlink_if_unchanged(target, signature) is False
    assert target.read_bytes() == b"human save"

    signature = atomic_create_text(box / "other", TEXT)
    with open(box / "other", "a", encoding="utf-8") as f:  # appended in place: same inode, new size
        f.write("more")
    assert unlink_if_unchanged(box / "other", signature) is False
    assert (box / "other").read_text(encoding="utf-8") == TEXT + "more"


@pytest.mark.platforms("posix")
def test_unlink_if_unchanged_never_removes_a_symlink_or_directory(box):
    victim = box.parent / "victim"
    victim.write_bytes(b"v")
    signature = atomic_create_text(box / "SKILL.md", TEXT)
    (box / "SKILL.md").unlink()
    (box / "SKILL.md").symlink_to(victim)
    assert unlink_if_unchanged(box / "SKILL.md", signature) is False
    assert (box / "SKILL.md").is_symlink() and victim.read_bytes() == b"v"
    (box / "SKILL.md").unlink()
    (box / "SKILL.md").mkdir()
    assert unlink_if_unchanged(box / "SKILL.md", signature) is False
    assert (box / "SKILL.md").is_dir()
