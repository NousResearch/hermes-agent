"""Tests for dispatcher lease + heartbeat takeover (t_9e151ee8).

Covers the three executable scenarios from the task body at the decision-logic
level (pure helpers in gateway.kanban_watchers):

1. Kill the owner's loop (stale heartbeat) -> another gateway takes over.
2. Healthy owner (fresh heartbeat) -> no takeover succeeds.
3. Root preference -> with a root and a profile gateway both waiting, the
   root acquires (a non-root claimant yields to a live root gateway).
"""
import time
from pathlib import Path

from gateway.kanban_watchers import (
    _dispatcher_heartbeat_is_stale,
    _should_seize_dispatcher,
    _touch_dispatcher_heartbeat,
)


def _root(owner_stale=None):
    """Build a takeover inputs dict; override only the fields a test varies."""
    return {
        "am_root": False,
        "owner_stale": False,
        "root_live": False,
    }


def test_healthy_owner_never_seized(tmp_path):
    """A fresh heartbeat means a contender must NOT take over."""
    for am_root in (True, False):
        for root_live in (True, False):
            assert _should_seize_dispatcher(
                am_root=am_root, owner_stale=False, root_live=root_live,
            ) is False, f"am_root={am_root} root_live={root_live}"


def test_stale_owner_seized_by_any_when_no_root(tmp_path):
    """Stale owner + no live root gateway -> anyone (root or profile) may seize."""
    assert _should_seize_dispatcher(
        am_root=False, owner_stale=True, root_live=False,
    ) is True
    assert _should_seize_dispatcher(
        am_root=True, owner_stale=True, root_live=False,
    ) is True


def test_root_preference_nonroot_yields(tmp_path):
    """Stale owner but a live root gateway -> a non-root claimant yields."""
    assert _should_seize_dispatcher(
        am_root=False, owner_stale=True, root_live=True,
    ) is False
    # ...but the root itself takes over the stale owner.
    assert _should_seize_dispatcher(
        am_root=True, owner_stale=True, root_live=True,
    ) is True


def test_heartbeat_absent_counts_stale(tmp_path):
    heartbeat = tmp_path / ".dispatcher.heartbeat"
    assert _dispatcher_heartbeat_is_stale(heartbeat) is True


def test_heartbeat_fresh_after_touch(tmp_path):
    heartbeat = tmp_path / ".dispatcher.heartbeat"
    _touch_dispatcher_heartbeat(heartbeat)
    assert heartbeat.exists()
    assert _dispatcher_heartbeat_is_stale(heartbeat, now=time.time()) is False


def test_heartbeat_becomes_stale_after_window(tmp_path):
    heartbeat = tmp_path / ".dispatcher.heartbeat"
    _touch_dispatcher_heartbeat(heartbeat)
    # 301s later (> 300s stale window) the same file is judged stale.
    assert _dispatcher_heartbeat_is_stale(heartbeat, now=time.time() + 301) is True