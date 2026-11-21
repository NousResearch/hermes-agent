"""``utils.atomic_write_bytes``: a failed file sync never publishes, and a retry succeeds.

Contract: when the temp file's ``fsync`` fails (a delayed write error such as ENOSPC), the
``OSError`` reaches the caller, an existing destination keeps its exact bytes (or a new one is never
created), no temp file is left in the directory, and a retry of the same write with a working sync
publishes exactly the new bytes.
"""

from __future__ import annotations

import errno

import pytest

import utils

_OLD = b"\x00old\xff\xfe payload\x00"
_NEW = b"\xff\x00new\x80\xc3\x28 payload\x00\x01"


@pytest.mark.parametrize("existing", [_OLD, None], ids=["existing-destination", "first-creation"])
def test_fsync_failure_keeps_destination_and_retry_publishes(tmp_path, monkeypatch, existing):
    workdir = tmp_path / "work"  # tmp_path also holds the autouse HERMES_HOME
    workdir.mkdir()
    target = workdir / "blob.bin"
    if existing is not None:
        target.write_bytes(existing)

    def failing_fsync(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as faulty:
        faulty.setattr(utils.os, "fsync", failing_fsync)
        with pytest.raises(OSError) as excinfo:
            utils.atomic_write_bytes(target, _NEW)

    assert excinfo.value.errno == errno.ENOSPC
    if existing is None:
        assert not target.exists()
        assert list(workdir.iterdir()) == []
    else:
        assert target.read_bytes() == existing
        assert list(workdir.iterdir()) == [target]

    utils.atomic_write_bytes(target, _NEW)

    assert target.read_bytes() == _NEW
    assert list(workdir.iterdir()) == [target]
