"""File sharing refuses oversized, linked and private inputs before staging."""
import os
from contextlib import contextmanager

import pytest

from tools.group_file_tool import _read_bounded


@pytest.mark.parametrize("restore_mtime", [False, True])
def test_same_size_mutation_during_read_requires_retry(tmp_path, monkeypatch, restore_mtime):
    file = tmp_path / "report.txt"
    file.write_bytes(b"first version")
    before = file.stat()
    fdopen = os.fdopen

    @contextmanager
    def changing_reader(*args, **kwargs):
        with fdopen(*args, **kwargs) as handle:
            yield handle
            file.write_bytes(b"other version")
            if restore_mtime:
                os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))

    monkeypatch.setattr(os, "fdopen", changing_reader)
    with pytest.raises(ValueError, match="changed.*Try again"):
        _read_bounded(str(file), 32)


@pytest.mark.parametrize("size", [0, 17])
def test_file_size_is_checked_before_reading(tmp_path, size):
    file = tmp_path / "report.bin"
    file.write_bytes(b"x" * size)
    with pytest.raises(ValueError, match="non-empty file of at most"):
        _read_bounded(str(file), 16)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("link", ["hard", "file-symlink", "directory-symlink"])
def test_open_rejects_links_even_after_path_validation(tmp_path, link):
    source = tmp_path / "private"
    source.mkdir()
    (source / "key.txt").write_bytes(b"secret")
    target = tmp_path / "target"
    if link == "directory-symlink":
        target.symlink_to(source, target_is_directory=True)
        target = target / "key.txt"
    elif link == "file-symlink":
        target.symlink_to(source / "key.txt")
    else:
        os.link(source / "key.txt", target)
    with pytest.raises((OSError, ValueError)):
        _read_bounded(str(target), 16)
