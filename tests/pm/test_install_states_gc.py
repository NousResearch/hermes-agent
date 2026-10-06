"""Dependency state of a deleted checkout is reclaimed; state anything live could still read is
kept (``pm/install_states.py``)."""

from __future__ import annotations

import os

from pm.filesystem import lock_fd
from pm.install_states import collect_orphan_install_states, orphan_install_states


def _state(installs, key, project_root):
    state = installs / key
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(project_root), encoding="utf-8")
    (state / "environments" / "gen" / "venv").mkdir(parents=True)
    return state


def test_only_states_of_deleted_checkouts_are_reclaimed(tmp_path):
    installs = tmp_path / "installs"
    live_root = tmp_path / "checkout-live"
    live_root.mkdir()
    live = _state(installs, "aaaa", live_root)
    gone = _state(installs, "bbbb", tmp_path / "checkout-deleted")
    # Pre-record state (no .project-root) has no evidence either way: never touched.
    unknown = installs / "cccc"
    (unknown / "environments").mkdir(parents=True)

    assert orphan_install_states(installs) == [gone]
    assert collect_orphan_install_states(installs) == [gone]
    assert live.is_dir() and unknown.is_dir() and not gone.exists()


def test_held_orphan_is_kept(tmp_path):
    installs = tmp_path / "installs"
    locked = _state(installs, "dddd", tmp_path / "gone-a")
    leased = _state(installs, "eeee", tmp_path / "gone-b")

    fd = os.open(locked / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    assert lock_fd(fd, wait=False)
    lease_dir = leased / "environments" / "gen" / ".leases"
    lease_dir.mkdir()
    lease_fd = os.open(lease_dir / "reader", os.O_CREAT | os.O_RDWR, 0o600)
    assert lock_fd(lease_fd, wait=False)
    try:
        assert collect_orphan_install_states(installs) == []
        assert locked.is_dir() and leased.is_dir()
    finally:
        os.close(fd)
        os.close(lease_fd)
    # Locks released (owner exited): both are reclaimed on the next pass.
    assert sorted(p.name for p in collect_orphan_install_states(installs)) == ["dddd", "eeee"]
