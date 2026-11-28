"""A dashboard PTY child dies with the dashboard, however the dashboard dies.

``PtyBridge.close()`` is the bridge's only teardown, and it does not run on SIGKILL, an OOM kill or
a crash. Closing the master does send SIGHUP, but that reaches only members which keep the default
action — the reason ``ProcessOwnershipMixin._own_pty_job`` hands its PTY jobs to the owner-death
supervisor instead of trusting the hangup. The bridge now registers the same way.

The supervisor itself is not exercised here (``test_process_identity_release_once_empty.py`` owns
that): these tests pin the registration contract, including the case where registering would be
actively harmful.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("ptyprocess", reason="ptyprocess not installed")

from hermes_cli import process_identity
from hermes_cli.pty_bridge import PtyBridge


@pytest.fixture
def owned(monkeypatch):
    """Records (owned, released) pgids instead of starting a real supervisor."""
    owned: list = []
    released: list = []
    monkeypatch.setattr(process_identity, "own_process_groups", lambda pgids: owned.extend(pgids))
    monkeypatch.setattr(process_identity, "release_process_group_once_empty",
                        lambda pgid, **kw: released.append(pgid) or True)
    return owned, released


def _bridge() -> PtyBridge:
    return PtyBridge.spawn([sys.executable, "-c", "import time; time.sleep(60)"])


@pytest.mark.platforms("posix")
def test_spawn_hands_the_childs_group_to_the_owner_death_supervisor(owned):
    owned_pgids, _ = owned
    bridge = _bridge()
    try:
        # ptyprocess puts the child in its own session, so it leads its own group.
        assert os.getpgid(bridge.pid) == bridge.pid
        assert owned_pgids == [bridge.pid], "the PTY child's group never reached the supervisor"
    finally:
        bridge.close()


@pytest.mark.platforms("posix")
def test_close_releases_the_group_it_registered(owned):
    owned_pgids, released_pgids = owned
    bridge = _bridge()
    pid = bridge.pid
    bridge.close()
    assert owned_pgids == [pid]
    # Released through release_process_group_once_empty: a member the close sweep could not end
    # keeps the coverage, so owner death still reaps it.
    assert released_pgids == [pid]


@pytest.mark.platforms("posix")
def test_a_child_sharing_our_group_is_never_registered(owned, monkeypatch):
    """``pgid == pid`` is the proof the child leads its own group. Without that guard a child which
    shares ours would register THIS process's group, and the supervisor would kill us on exit."""
    owned_pgids, _ = owned
    ours = os.getpgrp()
    monkeypatch.setattr(os, "getpgid", lambda _pid: ours)
    bridge = _bridge()
    try:
        assert owned_pgids == [], f"registered a shared group ({owned_pgids}); the supervisor would kill us"
    finally:
        monkeypatch.undo()
        bridge.close()
