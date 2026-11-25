"""Reopening a worker log owned by another account in a shared-group kanban home (#63594)."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_worker_log as kwl

pytestmark = pytest.mark.platforms("posix")


def _as_non_owner(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make chmod behave as for a file another account owns; track descriptors opened."""
    opened: list[int] = []
    real_open = os.open

    def _open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def _chmod(*_args, **_kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(kwl.os, "open", _open)
    monkeypatch.setattr(kwl.os, "chmod", _chmod)
    monkeypatch.setattr(kwl, "worker_log_rotation_config", lambda: (1 << 20, 1))
    return opened


def _is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def test_private_group_writable_log_of_another_owner_reopens(tmp_path: Path, monkeypatch):
    log = tmp_path / "t.log"
    log.write_bytes(b"earlier run\n")
    log.chmod(0o660)
    _as_non_owner(monkeypatch)

    with kwl.open_worker_log(log) as f:
        f.write(b"next run\n")

    assert log.read_bytes() == b"earlier run\nnext run\n"


def test_world_readable_log_that_cannot_be_tightened_fails_closed_without_leaking(
    tmp_path: Path, monkeypatch
):
    log = tmp_path / "t.log"
    log.write_bytes(b"earlier run\n")
    log.chmod(0o664)
    opened = _as_non_owner(monkeypatch)

    with pytest.raises(PermissionError):
        kwl.open_worker_log(log)

    assert opened and not _is_open(opened[-1])
    assert log.read_bytes() == b"earlier run\n"
