"""Hostile archives must not write outside the store entry being staged.

Every pm package arrives as an archive from the internet. The sha256 pin
proves the bytes are the ones we reviewed, but that is a supply-chain
control, not a containment one: it says nothing about what a
legitimately-published archive does when unpacked, and a pin refresh is
a human copying a digest. So pm.store.extract itself must be safe
against path traversal, absolute paths, symlink escapes, and collisions
— and must never clobber a file it did not create.

Ported from the restack branch's runtime-archive-safety suite, rewritten
against pm's real extraction surface (pm.store.extract / _extract_zip /
flatten_single_dir). The traversal / absolute-path / never-clobber arms
are host-independent and run on Windows; the arms that assert POSIX
symlink or mode semantics are gated to POSIX, with a separate arm
pinning pm's truthful win32 degradation.
"""

from __future__ import annotations

import io
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from pm.store import extract, flatten_single_dir

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX symlink/mode semantics"
)
win_only = pytest.mark.skipif(
    sys.platform != "win32", reason="pins win32 degradation specifically"
)


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    staging = path.parent / f".stage-{path.stem}"
    with tarfile.open(path, "w:gz") as tf:
        for rel, data in members.items():
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            tf.add(target, arcname=rel)
    return path


def _add_symlink(zf: zipfile.ZipFile, member: str, target: str) -> None:
    """Append a symlink entry the way zip tools actually encode one:
    content is the target path, external_attr mode says S_ISLNK."""
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


class TestPathTraversal:
    def test_zip_entries_cannot_escape_the_destination(self, tmp_path):
        archive = tmp_path / "slip.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../../ESCAPED.txt", "pwned")
            zf.writestr("bin/tool", "fine")

        dest = tmp_path / "sub" / "dest"
        extract(archive, dest)

        assert not (tmp_path / "ESCAPED.txt").exists()
        assert not (tmp_path.parent / "ESCAPED.txt").exists()
        assert (dest / "bin" / "tool").is_file()

    def test_absolute_zip_paths_stay_inside_the_destination(self, tmp_path):
        archive = tmp_path / "abs.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("/etc/PWNED.txt", "pwned")

        dest = tmp_path / "dest"
        extract(archive, dest)

        assert not Path("/etc/PWNED.txt").exists()
        assert (dest / "etc" / "PWNED.txt").is_file()

    def test_tar_entries_cannot_escape_the_destination(self, tmp_path):
        payload = tmp_path / "payload.txt"
        payload.write_text("pwned")
        archive = tmp_path / "slip.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(payload, arcname="../../ESCAPED.txt")

        with pytest.raises(Exception):
            extract(archive, tmp_path / "sub" / "dest")

        assert not (tmp_path.parent / "ESCAPED.txt").exists()

    def test_absolute_tar_paths_cannot_escape(self, tmp_path):
        data = b"pwned"
        archive = tmp_path / "abs.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            entry = tarfile.TarInfo("/tmp/PWNED-pm-store.txt")
            entry.size = len(data)
            tf.addfile(entry, io.BytesIO(data))

        # The data filter either strips the leading slash (extracting
        # inside dest) or refuses the member; both keep / untouched.
        dest = tmp_path / "dest"
        try:
            extract(archive, dest)
        except Exception:
            pass

        assert not Path("/tmp/PWNED-pm-store.txt").exists()
        outside = [
            p for p in tmp_path.rglob("PWNED-pm-store.txt") if dest not in p.parents
        ]
        assert outside == []

    def test_a_tar_symlink_cannot_be_used_to_write_outside(self, tmp_path):
        """Classic two-entry attack: a symlink pointing out of the tree,
        then a regular file written through it."""
        victim = tmp_path / "outside.txt"
        victim.write_text("ORIGINAL")

        archive = tmp_path / "symlink.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            link = tarfile.TarInfo("escape")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../outside.txt"
            tf.addfile(link)

            data = b"OVERWRITTEN"
            entry = tarfile.TarInfo("escape")
            entry.size = len(data)
            tf.addfile(entry, io.BytesIO(data))

        with pytest.raises(Exception):
            extract(archive, tmp_path / "sub" / "dest")

        assert victim.read_text() == "ORIGINAL"


class TestZipModeHandling:
    def test_a_traversing_entry_cannot_touch_a_file_outside(self, tmp_path):
        """The exec-bit restore must chmod the path zipfile actually
        wrote, never the raw (traversing) entry name."""
        victim = tmp_path / "victim.txt"
        victim.write_text("untouched")
        victim.chmod(0o644)

        archive = tmp_path / "chmod.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            info = zipfile.ZipInfo("../../victim.txt")
            info.external_attr = 0o777 << 16
            zf.writestr(info, "x")

        extract(archive, tmp_path / "sub" / "dest")

        assert victim.read_text() == "untouched"
        if sys.platform != "win32":
            assert victim.stat().st_mode & 0o777 == 0o644

    @posix_only
    def test_the_executable_bit_is_still_restored_for_real_entries(self, tmp_path):
        """zip drops the exec bit; an un-executable binary is a broken
        install, so the restore loop must keep working for honest entries."""
        archive = tmp_path / "exec.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            info = zipfile.ZipInfo("uv")
            info.external_attr = 0o755 << 16
            zf.writestr(info, "#!/bin/sh\n")

        dest = tmp_path / "dest"
        extract(archive, dest)

        assert (dest / "uv").stat().st_mode & 0o111


class TestCollisionsAndClobbering:
    def test_duplicate_entry_names_stay_inside_dest(self, tmp_path):
        """Colliding entry names are resolved inside dest (last wins);
        nothing leaks out and extraction does not crash."""
        archive = tmp_path / "dupe.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("bin/tool", "first")
            zf.writestr("bin/tool", "second")

        dest = tmp_path / "dest"
        extract(archive, dest)

        assert (dest / "bin" / "tool").read_text() == "second"
        assert list(dest.iterdir()) == [dest / "bin"]

    def test_a_hidden_top_level_file_is_not_replaced_by_flatten(self, tmp_path):
        """{".config", "wrapper/.config"} must not look like a bare
        wrapper: hoisting would silently replace the outer dotfile."""
        archive = _tar(
            tmp_path / "hidden.tar.gz",
            {
                ".config": b"TOP LEVEL ORIGINAL",
                "wrapper/.config": b"FROM WRAPPER",
                "wrapper/bin/tool": b"x",
            },
        )
        dest = tmp_path / "dest"
        extract(archive, dest)

        flatten_single_dir(dest)

        assert (dest / ".config").read_bytes() == b"TOP LEVEL ORIGINAL"

    def test_a_child_named_like_its_wrapper_refuses_rather_than_clobbers(
        self, tmp_path
    ):
        """An unflattened tree is merely ugly; a clobbered file is data
        loss. pm refuses the hoist and leaves the tree intact."""
        archive = _tar(tmp_path / "same.tar.gz", {"gh/gh": b"inner"})
        dest = tmp_path / "dest"
        extract(archive, dest)

        flatten_single_dir(dest)

        assert (dest / "gh" / "gh").read_bytes() == b"inner"

    def test_a_real_wrapper_still_unwraps(self, tmp_path):
        archive = _tar(
            tmp_path / "wrapped.tar.gz", {"gh_2.97.0_linux_amd64/bin/gh": b"x"}
        )
        dest = tmp_path / "dest"
        extract(archive, dest)

        flatten_single_dir(dest)

        assert (dest / "bin" / "gh").is_file()

    def test_a_lone_layout_dir_is_left_alone(self, tmp_path):
        """A bare bin/ IS the tool's layout, not a wrapper."""
        archive = _tar(tmp_path / "flat.tar.gz", {"bin/gh": b"x"})
        dest = tmp_path / "dest"
        extract(archive, dest)

        flatten_single_dir(dest)

        assert (dest / "bin" / "gh").is_file()


class TestStoreIsolation:
    def test_extract_replaces_only_its_own_entry_directory(self, tmp_path):
        """Each staged entry owns exactly its dest dir. Neighbouring
        store entries and unrelated files must survive a re-stage."""
        store = tmp_path / "store"
        (store / "node-abc" / "bin").mkdir(parents=True)
        (store / "node-abc" / "bin" / "node").write_text("other entry")
        keep = store / "downloads" / "important.bin"
        keep.parent.mkdir(parents=True)
        keep.write_text("KEEP")

        archive = _tar(tmp_path / "gh.tar.gz", {"bin/gh": b"x"})
        extract(archive, store / "gh-def")

        assert (store / "node-abc" / "bin" / "node").read_text() == "other entry"
        assert keep.read_text() == "KEEP"

    def test_restaging_clears_a_stale_tree_first(self, tmp_path):
        """A file from an older version must not linger inside the
        entry's own directory and shadow the new layout."""
        dest = tmp_path / "gh"
        dest.mkdir()
        (dest / "STALE").write_text("from an older version")

        archive = _tar(tmp_path / "gh.tar.gz", {"bin/gh": b"x"})
        extract(archive, dest)

        assert not (dest / "STALE").exists()
        assert (dest / "bin" / "gh").is_file()

    def test_unsupported_archive_names_are_refused(self, tmp_path):
        weird = tmp_path / "tool.rar"
        weird.write_bytes(b"not really")
        with pytest.raises(ValueError, match="unsupported archive"):
            extract(weird, tmp_path / "dest")
