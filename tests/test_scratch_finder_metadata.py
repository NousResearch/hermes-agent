"""Finder metadata rewrites must not extend scratch idle retention."""

import os
import time

import pytest

from hermes_constants import get_scratch_dir, prune_scratch_dir


@pytest.mark.parametrize("relative", [".DS_Store", "task/.DS_Store", "task/nested/.DS_Store"])
def test_existing_metadata_rewrite_does_not_keep_idle_entry(tmp_path, relative):
    scratch = get_scratch_dir(tmp_path, prune=False)
    metadata = scratch / relative
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_bytes(b"old metadata")
    old = time.time() - 30 * 3600
    for path in [metadata, *metadata.parents]:
        if path == scratch:
            break
        os.utime(path, (old, old))
    directory_mtime = metadata.parent.stat().st_mtime_ns

    metadata.write_bytes(b"rewritten metadata")
    assert metadata.parent.stat().st_mtime_ns == directory_mtime
    assert prune_scratch_dir(scratch) == 1
    assert not (scratch / relative.split("/")[0]).exists()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("activity", [
    "metadata-created", "metadata-removed", "real-write", "directory-touch", "same-name-directory",
    pytest.param("same-name-symlink", marks=pytest.mark.platforms("posix")),
])
def test_metadata_does_not_hide_task_or_directory_activity(tmp_path, nested, activity):
    scratch = get_scratch_dir(tmp_path, prune=False)
    task = scratch / "task"
    directory = task / "nested" if nested else task
    directory.mkdir(parents=True)
    metadata = directory / ".DS_Store"
    work = directory / "work.txt"
    work.write_bytes(b"old task work")
    if activity == "same-name-directory":
        metadata.mkdir()
        (metadata / "work.txt").write_bytes(b"recent task work")
    elif activity == "same-name-symlink":
        metadata.symlink_to(work)
    elif activity != "metadata-created":
        metadata.write_bytes(b"old metadata")
    old = time.time() - 30 * 3600
    for path in (work, directory, task):
        os.utime(path, (old, old))
    if metadata.exists() and not metadata.is_symlink():
        os.utime(metadata, (old, old))

    def rewrite_task():
        metadata.write_bytes(b"rewritten metadata")
        work.write_bytes(b"recent task work")

    updates = {
        "metadata-created": lambda: metadata.write_bytes(b"new metadata"),
        "metadata-removed": metadata.unlink,
        "real-write": rewrite_task,
        "directory-touch": lambda: os.utime(directory),
    }
    update = updates.get(activity)
    if update is not None:
        update()
    # Same-named directories are scanned; symlinks retain their own recent mtime.
    assert prune_scratch_dir(scratch) == 0
    assert task.exists()
