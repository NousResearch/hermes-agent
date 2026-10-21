"""An abandoned lease file must be reclaimable, on every host.

The pruner unlinks a lease while it still holds the lock on it, so the handle
it opens has to permit the delete. On Windows a handle opened without
``FILE_SHARE_DELETE`` refuses the deletion of its own file (WinError 32), so
the unlink failed, the lease was never reclaimed and the ``.leases`` directory
grew without bound (#142528).
"""
import os
from pathlib import Path

import pytest

from hermes_cli import runtime_state
from hermes_cli.runtime_state import _prune_unlocked_leases


def _abandoned_lease(leases: Path, name: str = "deadbeef") -> Path:
    """A lease file whose creator died: nobody holds its lock."""
    leases.mkdir(parents=True, exist_ok=True)
    lease = leases / name
    fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(fd)
    return lease


def test_prune_reclaims_an_abandoned_lease_file(tmp_path):
    """The regression: the abandoned file is GONE after a prune, not still there."""
    leases = tmp_path / ".leases"
    stale = _abandoned_lease(leases)

    assert _prune_unlocked_leases(leases) is False, "an unlocked lease is not a held lease"
    assert not stale.exists(), f"the abandoned lease survived the prune: {list(leases.iterdir())}"
    assert list(leases.iterdir()) == []


def test_prune_keeps_a_lease_a_live_holder_still_locks(tmp_path):
    """Fail closed: a lease somebody still holds is reported as held, not removed."""
    leases = tmp_path / ".leases"
    live = _abandoned_lease(leases, "held")

    from pm.filesystem import lock_fd

    fd = os.open(live, os.O_RDWR)
    try:
        assert lock_fd(fd, wait=False) is True
        assert _prune_unlocked_leases(leases) is True, "a live lock must read as held"
        assert live.exists()
    finally:
        os.close(fd)


@pytest.mark.platforms("windows")
def test_the_pruner_handle_permits_deleting_its_own_file(tmp_path):
    """The Windows-specific half: the handle the pruner opens must not block
    the delete of its own file, or the lease is never reclaimed."""
    from pm.filesystem import open_deletable

    lease = tmp_path / "lease"
    fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(fd)

    deletable = open_deletable(lease)
    try:
        lease.unlink()
    finally:
        os.close(deletable)
    assert not lease.exists()
