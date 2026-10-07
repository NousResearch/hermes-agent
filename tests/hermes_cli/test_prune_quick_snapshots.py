"""Regression tests for quick-snapshot pruning.

`_prune_quick_snapshots` must keep the N most *recently modified* snapshots,
not the N that sort newest by directory name. A directory whose name sorts
above the timestamp-prefix convention (e.g. an archived snapshot named
``20260831T000011Z-pre-dashboard-loopback``) previously defeated name-based
ordering and caused the just-created pre-update safety-net snapshot to be
deleted under ``keep=1``.
"""

import os

import pytest

from hermes_cli.backup import _prune_quick_snapshots


def _make_snapshot(root, name: str, mtime_ns: int):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    # Set both atime and mtime so the ordering is deterministic.
    os.utime(d, ns=(mtime_ns, mtime_ns))
    return d


def test_prune_keeps_newest_by_mtime_not_by_name(tmp_path):
    """An out-of-convention name must not shadow the real safety-net snapshot."""
    root = tmp_path / "state-snapshots"
    root.mkdir()

    old = _make_snapshot(root, "20260831-010000-old", 1_000_000)
    # Name sorts ABOVE the timestamp convention ('T' > '-'), but it is OLD.
    archive = _make_snapshot(root, "20260831T000011Z-pre-dashboard-loopback", 2_000_000)
    # The just-created pre-update safety net is the newest by modification time.
    safety = _make_snapshot(root, "20260831-042336-p0-strict-preflight", 3_000_000)

    deleted = _prune_quick_snapshots(root, keep=1)

    # The newest-by-mtime snapshot (the safety net) must survive.
    assert safety.exists()
    assert not old.exists()
    assert not archive.exists()
    assert deleted == 2


def test_prune_preserves_just_created_snapshot_under_keep_one(tmp_path):
    """Invariant: the snapshot a process just wrote is never pruned at keep=1."""
    root = tmp_path / "state-snapshots"
    root.mkdir()

    stale = _make_snapshot(root, "20260830-220509-stale", 1_000_000)
    just_created = _make_snapshot(root, "20260831-042336-p0-strict-preflight", 9_000_000)

    _prune_quick_snapshots(root, keep=1)

    assert just_created.exists()
    assert not stale.exists()


def test_prune_removes_oldest_timestamped_snapshots(tmp_path):
    """Normal convention-prefixed names still prune oldest-first."""
    root = tmp_path / "state-snapshots"
    root.mkdir()

    newest = _make_snapshot(root, "20260831-030000-newest", 3_000_000)
    middle = _make_snapshot(root, "20260831-020000-middle", 2_000_000)
    oldest = _make_snapshot(root, "20260831-010000-oldest", 1_000_000)

    _prune_quick_snapshots(root, keep=2)

    assert newest.exists()
    assert middle.exists()
    assert not oldest.exists()


def test_prune_is_noop_when_under_keep(tmp_path):
    root = tmp_path / "state-snapshots"
    root.mkdir()

    a = _make_snapshot(root, "20260831-010000-a", 1_000_000)
    b = _make_snapshot(root, "20260831-020000-b", 2_000_000)

    deleted = _prune_quick_snapshots(root, keep=5)

    assert deleted == 0
    assert a.exists()
    assert b.exists()


def test_prune_skips_dotfiles_and_partials(tmp_path):
    root = tmp_path / "state-snapshots"
    root.mkdir()

    _make_snapshot(root, ".hidden", 1_000_000)
    partial = root / "20260831-010000-partial.partial"
    partial.mkdir()
    os.utime(partial, ns=(1_000_000, 1_000_000))

    deleted = _prune_quick_snapshots(root, keep=0)

    assert deleted == 0
    assert (root / ".hidden").exists()
    assert partial.exists()
