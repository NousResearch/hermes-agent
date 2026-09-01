"""Tests for dispatcher lease + heartbeat takeover (t_9e151ee8).

Covers the three executable scenarios from the task body at the
decision-logic level (pure helpers in gateway.kanban_watchers):

1. Kill the owner's loop (stale heartbeat) -> another gateway takes over.
2. Healthy owner (fresh heartbeat) -> no takeover succeeds.
3. Root preference -> with a root and a profile gateway both waiting, the
   root acquires (a non-root claimant yields to a freshly-restarted live root
   gateway) — but the yield is BOUNDED: once the root has been up past the
   grace period, a non-root seizes a stale owner even if the root process is
   alive (process-alive / dispatcher-dead on the root must not freeze the
   board, which is the exact failure this card exists to eliminate).
"""
import time
from pathlib import Path
from unittest import mock

import pytest

from gateway.kanban_watchers import (
    _DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS,
    _dispatcher_heartbeat_is_stale,
    _root_gateway_in_grace,
    _should_seize_dispatcher,
    _touch_dispatcher_heartbeat,
)


# ── _should_seize_dispatcher (pure decision) ──────────────────────────────


def test_healthy_owner_never_seized():
    """A fresh heartbeat means a contender must NOT take over."""
    for am_root in (True, False):
        for defer_to_root in (True, False):
            assert _should_seize_dispatcher(
                am_root=am_root, owner_stale=False, defer_to_root=defer_to_root,
            ) is False, f"am_root={am_root} defer_to_root={defer_to_root}"


def test_stale_owner_seized_when_no_deferring_root():
    """Stale owner + no freshly-restarted root gateway -> anyone seizes."""
    assert _should_seize_dispatcher(
        am_root=False, owner_stale=True, defer_to_root=False,
    ) is True
    assert _should_seize_dispatcher(
        am_root=True, owner_stale=True, defer_to_root=False,
    ) is True


def test_root_preference_bounded_grace_nonroot_yields():
    """Stale owner + freshly-restarted live root -> non-root yields (in grace),
    and the root itself takes over. Once grace elapses (defer_to_root=False) the
    non-root seizes the stale owner even though the root process is alive."""
    # In grace: non-root claimant yields to the freshly-restarted root.
    assert _should_seize_dispatcher(
        am_root=False, owner_stale=True, defer_to_root=True,
    ) is False
    # ...but the root itself takes over the stale owner (root never defers).
    assert _should_seize_dispatcher(
        am_root=True, owner_stale=True, defer_to_root=True,
    ) is True
    # Grace elapsed: a non-root seizes the stale owner despite a live root.
    assert _should_seize_dispatcher(
        am_root=False, owner_stale=True, defer_to_root=False,
    ) is True


# ── _root_gateway_in_grace (bounded, anchored to root process start) ──────


def _fake_identity(pid, start):
    return (pid, start)


def test_root_unprobeable_never_defer(tmp_path):
    """Any probe error / identity-unavailable -> return False (do not defer)."""
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        side_effect=RuntimeError("boom"),
    ):
        assert _root_gateway_in_grace(tmp_path) is False
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=None,
    ):
        assert _root_gateway_in_grace(tmp_path) is False


def test_root_in_grace_within_window(tmp_path):
    """A freshly (re)started root is in grace and gets priority."""
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=_fake_identity(42, time.time()),
    ):
        assert _root_gateway_in_grace(tmp_path) is True


def test_root_out_of_grace_after_window(tmp_path):
    """A root up longer than the grace period is NOT deferred to."""
    # Start from just inside the window so the boundary is meaningful, then a
    # start_time older than the grace marks it settled.
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=_fake_identity(
            42,
            time.time() - (_DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS + 1),
        ),
    ):
        assert _root_gateway_in_grace(tmp_path) is False


# ── Integration-style: frozen ROOT owner is still taken over ──────────────
#
# The acceptance gate from review round 1: process-alive / dispatcher-dead on
# the ROOT must NOT freeze the board. Simulate a non-root claimant racing a
# root gateway whose process is alive but whose dispatcher loop has frozen
# (heartbeat stale). Once the root has been up past the grace period, the
# non-root must seize the lease despite the live root.

def test_nonroot_seizes_stale_owner_when_root_frozen_out_of_grace(tmp_path):
    """A live-but-frozen ROOT owner is taken over by a non-root after grace."""
    # Heartbeat last touched 10 minutes ago -> stale (> 300s window).
    heartbeat = tmp_path / ".dispatcher.heartbeat"
    _touch_dispatcher_heartbeat(heartbeat)
    old_stale = time.time() - 600
    heartbeat.touch()
    import os as _os
    _os.utime(heartbeat, (old_stale, old_stale))
    assert _dispatcher_heartbeat_is_stale(heartbeat, now=time.time()) is True

    # Root process is alive (pidfile resolves and returns an identity) but it
    # has been up LONGER than the grace period — i.e. it settled and its loop
    # silently died (the original outage relocated to root).
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=_fake_identity(
            42,
            time.time() - (_DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS + 1),
        ),
    ):
        defer_to_root = _root_gateway_in_grace(tmp_path)
    assert defer_to_root is False

    # The non-root claimant seizes the stale owner.
    assert _should_seize_dispatcher(
        am_root=False, owner_stale=True, defer_to_root=defer_to_root,
    ) is True


def test_nonroot_defers_only_during_frozen_root_grace(tmp_path):
    """The bounded grace DOES yield to a freshly-restarted root that has just
    gone stale, but only within the grace window."""
    heartbeat = tmp_path / ".dispatcher.heartbeat"
    _touch_dispatcher_heartbeat(heartbeat)
    stale_at = time.time() - 600
    import os as _os
    _os.utime(heartbeat, (stale_at, stale_at))
    assert _dispatcher_heartbeat_is_stale(heartbeat, now=time.time()) is True

    # Root freshly restarted (within grace) -> non-root defers so root can win.
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=_fake_identity(42, time.time()),
    ):
        defer_to_root = _root_gateway_in_grace(tmp_path)
    assert defer_to_root is True
    assert _should_seize_dispatcher(
        am_root=False, owner_stale=True, defer_to_root=defer_to_root,
    ) is False


# ── heartbeat helpers ─────────────────────────────────────────────────────


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