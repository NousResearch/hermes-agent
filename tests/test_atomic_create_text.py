"""``utils.atomic_create_text`` / ``unlink_if_unchanged``: create-only publication that never replaces.

``atomic_write_text`` replaces whatever is at the path; ``atomic_create_text`` is for the callers that
must not (a new skill's SKILL.md racing another writer). These tests pin it at the filesystem level:
exclusivity for every kind of occupant, exact bytes and mode, temp-file hygiene, the fallback for
filesystems without hard links (still exclusive), and that cleanup only ever removes the caller's file.
"""

from __future__ import annotations

import errno
import hashlib
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
    assert tuple(signature)[:4] == (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    # size+mtime can be forged by an in-place edit; the digest of the published bytes cannot
    assert signature.sha256 == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()


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
        target.write_bytes(b"winner")  # (dst may be a bare name relative to a directory fd)
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
    assert unlink_if_unchanged(target, signature)
    target.write_bytes(b"human")
    with pytest.raises(FileExistsError):
        atomic_create_text(target, TEXT)
    assert target.read_bytes() == b"human"


@pytest.mark.platforms("posix")
def test_without_hard_links_a_writer_in_the_gap_is_still_not_replaced(box, monkeypatch):
    target = box / "SKILL.md"

    def plant_then_unsupported(src, dst, *a, **kw):
        target.write_bytes(b"winner")
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
    assert unlink_if_unchanged(target, signature)
    assert not target.exists()
    assert not unlink_if_unchanged(target, signature)  # already gone

    signature = atomic_create_text(target, TEXT)
    replacement = box / "editor.tmp"
    replacement.write_bytes(b"human save")
    os.replace(replacement, target)  # an editor's atomic save
    assert not unlink_if_unchanged(target, signature)
    assert target.read_bytes() == b"human save"

    signature = atomic_create_text(box / "other", TEXT)
    with open(box / "other", "a", encoding="utf-8") as f:  # appended in place: same inode, new size
        f.write("more")
    assert not unlink_if_unchanged(box / "other", signature)
    assert (box / "other").read_text(encoding="utf-8-sig") == TEXT + "more"


@pytest.mark.platforms("posix")
def test_unlink_if_unchanged_never_removes_a_symlink_or_directory(box):
    victim = box.parent / "victim"
    victim.write_bytes(b"v")
    signature = atomic_create_text(box / "SKILL.md", TEXT)
    (box / "SKILL.md").unlink()
    (box / "SKILL.md").symlink_to(victim)
    assert not unlink_if_unchanged(box / "SKILL.md", signature)
    assert (box / "SKILL.md").is_symlink() and victim.read_bytes() == b"v"
    (box / "SKILL.md").unlink()
    (box / "SKILL.md").mkdir()
    assert not unlink_if_unchanged(box / "SKILL.md", signature)
    assert (box / "SKILL.md").is_dir()


# --- cleanup never deletes data it cannot prove is its own -------------------------------------------

HUMAN = b"HUMAN EDIT - must survive\n"


def _hook_removal_primitives(monkeypatch, action) -> list:
    """Run ``action(name)`` before every remove/rename/replace/unlink whose source is a ``SKILL.md``."""
    fired: list = []
    for name in ("unlink", "remove", "rename", "replace"):
        def hook(src, *a, _real=getattr(os, name), _name=name, **kw):
            if os.path.basename(os.fspath(src)) == "SKILL.md":
                fired.append(_name)
                action(_name)
            return _real(src, *a, **kw)
        monkeypatch.setattr(os, name, hook)
    return fired


def test_removal_of_our_own_file_leaves_nothing_behind_and_reports_removed(box):
    target = box / "SKILL.md"
    result = unlink_if_unchanged(target, atomic_create_text(target, TEXT))
    assert result and result.status == "removed" and result.preserved_at is None
    assert _names(box) == []  # no quarantine / temp residue


def test_in_place_edit_with_same_size_and_mtime_is_not_removed(box):
    target = box / "SKILL.md"
    signature = atomic_create_text(target, TEXT)
    before = target.stat()
    target.write_bytes(b"H" * before.st_size)  # same inode, same size ...
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))  # ... same mtime
    result = unlink_if_unchanged(target, signature)
    assert target.read_bytes() == b"H" * before.st_size and _names(box) == ["SKILL.md"]
    assert not result and result.status == "changed"


@pytest.mark.platforms("posix")
def test_replacement_landing_between_the_check_and_the_removal_is_kept(box, monkeypatch):
    target = box / "SKILL.md"
    signature = atomic_create_text(target, TEXT)
    swapped: list = []
    real_replace = os.replace

    def swap_once(_name):
        if not swapped:
            swapped.append(True)
            tmp = box / "human.tmp"
            tmp.write_bytes(HUMAN)
            real_replace(tmp, target)

    fired = _hook_removal_primitives(monkeypatch, swap_once)
    result = unlink_if_unchanged(target, signature)
    assert fired and swapped, "no removal primitive ran, this test proved nothing"
    assert target.read_bytes() == HUMAN and _names(box) == ["SKILL.md"]
    assert not result and result.status == "changed"


@pytest.mark.platforms("posix")
def test_a_third_writer_in_the_restore_gap_is_not_clobbered_and_the_kept_file_is_reported(box, monkeypatch):
    target = box / "SKILL.md"
    signature = atomic_create_text(target, TEXT)
    real_replace, real_rename = os.replace, os.rename
    state: list = []

    def swap_in_human_then_rename_then_plant_third(src, dst, *a, **kw):
        if os.path.basename(os.fspath(src)) == "SKILL.md" and not state:
            state.append(True)
            tmp = box / "human.tmp"
            tmp.write_bytes(HUMAN)
            real_replace(tmp, target)  # foreign data takes our place ...
            out = real_rename(src, dst, *a, **kw)  # ... gets moved aside by the cleanup ...
            target.write_bytes(b"THIRD WRITER")  # ... and somebody else claims the name meanwhile
            return out
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "rename", swap_in_human_then_rename_then_plant_third)
    result = unlink_if_unchanged(target, signature)
    assert state, "the cleanup never renamed the file aside, this test proved nothing"
    assert target.read_bytes() == b"THIRD WRITER"
    assert not result and result.status == "changed" and result.preserved_at
    assert Path(result.preserved_at).read_bytes() == HUMAN  # nobody's bytes were deleted
    assert sorted(_names(box)) == sorted(["SKILL.md", Path(result.preserved_at).name])


def test_blocked_removal_is_reported_as_failed_and_the_file_stays(box, monkeypatch):
    target = box / "SKILL.md"
    signature = atomic_create_text(target, TEXT)

    def deny(name):
        raise PermissionError(errno.EACCES, f"simulated: {name} denied")

    fired = _hook_removal_primitives(monkeypatch, deny)
    result = unlink_if_unchanged(target, signature)
    assert fired, "no removal was attempted, this test proved nothing"
    assert target.read_bytes() == TEXT.encode("utf-8")
    assert not result and result.status == "failed" and "denied" in result.error


# --- the no-hard-link fallback must not turn a failure into data loss or a false success -------------

def _is_target(fd: int, target: Path) -> bool:
    try:
        return os.fstat(fd).st_ino == os.stat(target).st_ino
    except OSError:
        return False


@pytest.mark.platforms("posix")
def test_fallback_fsync_failure_after_an_in_place_human_edit_keeps_the_human_file(box, no_hard_links, monkeypatch):
    target = box / "SKILL.md"
    real_fsync = os.fsync
    edited: list = []

    def fsync_fails_after_a_human_edits_in_place(fd):
        if _is_target(fd, target) and not edited:
            edited.append(True)
            target.write_bytes(HUMAN)  # same inode: the person saved over our half-created file
            raise OSError(errno.EIO, "simulated fsync error")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync_fails_after_a_human_edits_in_place)
    with pytest.raises(OSError) as raised:
        atomic_create_text(target, TEXT)
    assert edited, "the in-place create never fsynced, this test proved nothing"
    # Without hard links a displaced file cannot be put back under its name without a clobber risk, so it
    # may sit under an aside name: either way exactly one file remains, it is the human's, and the error says where.
    (kept,) = list(box.iterdir())
    assert kept.read_bytes() == HUMAN
    assert kept.name == "SKILL.md" or str(kept) in "\n".join(getattr(raised.value, "__notes__", []))


@pytest.mark.platforms("posix")
def test_fallback_chmod_failure_is_an_error_not_a_success_with_the_wrong_mode(box, no_hard_links, monkeypatch):
    target = box / "SKILL.md"
    real_fchmod = os.fchmod
    refused: list = []

    def fchmod_refused_on_the_target(fd, mode):
        if _is_target(fd, target):
            refused.append(True)
            raise OSError(errno.EPERM, "simulated chmod failure")
        return real_fchmod(fd, mode)

    monkeypatch.setattr(os, "fchmod", fchmod_refused_on_the_target)
    with pytest.raises(OSError):
        atomic_create_text(target, TEXT, mode=0o644)
    assert refused, "the in-place create never set its mode, this test proved nothing"
    assert _names(box) == []  # our own empty file is removed; no success was claimed


# --- the parent directory is pinned: a symlink in its place is never written through -----------------

@pytest.mark.skipif(not (hasattr(os, "O_NOFOLLOW") and os.link in os.supports_dir_fd),
                    reason="needs O_NOFOLLOW and dir_fd support (not Windows)")
def test_a_symlinked_parent_directory_is_refused_and_not_written_through(box):
    outside = box.parent / "outside"
    outside.mkdir()
    (box / "skill").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        atomic_create_text(box / "skill" / "SKILL.md", TEXT)
    assert list(outside.iterdir()) == []
    assert (box / "skill").is_symlink()
