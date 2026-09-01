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

Grace arithmetic (``_root_gateway_in_grace``) computes uptime from a TRUE epoch
creation timestamp (``gateway.status.get_process_create_time_epoch`` ->
``psutil.create_time()``), NOT from ``get_running_pid_identity_strict``'s
platform-dependent PID-reuse fingerprint. A regression test
(``test_fingerprint_units_never_leak_into_grace``) asserts that even when the
strict-identity slot carries a macOS-style centisecond-epoch fingerprint, the
grace probe stays correct — feeding the fingerprint into uptime arithmetic
would silently defer a non-root to a live root forever (the round-2 Critical).
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
#
# The grace probe needs TWO mocks: the strict-identity slot (proves the root
# process is a live gateway and yields its PID) plus the true-epoch creation
# time (the only number comparable to time.time()). The strict-identity `start`
# value is deliberately given a non-epoch value in the regression test below to
# prove it is never used for uptime arithmetic.


def _mock_root_probe(*, create_time: float, identity_start=None,
                     identity=None):
    """Patch both status probes used by ``_root_gateway_in_grace``.

    ``identity`` (default ``(42, identity_start)``) feeds the strict-identity
    slot; ``create_time`` feeds the true-epoch creation time.
    """
    if identity is None:
        identity = (42, identity_start if identity_start is not None else (time.time() * 100))
    return mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=identity,
    ), mock.patch(
        "gateway.status.get_process_create_time_epoch",
        return_value=create_time,
    )


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
    # Creation-time unavailable (psutil failed) -> never defer.
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=(42, time.time()),
    ), mock.patch(
        "gateway.status.get_process_create_time_epoch",
        return_value=None,
    ):
        assert _root_gateway_in_grace(tmp_path) is False
    # Invalid / non-positive PID -> never defer.
    with mock.patch(
        "gateway.status.get_running_pid_identity_strict",
        return_value=(0, time.time()),
    ), mock.patch(
        "gateway.status.get_process_create_time_epoch",
        return_value=time.time(),
    ):
        assert _root_gateway_in_grace(tmp_path) is False


def test_root_in_grace_within_window(tmp_path):
    """A freshly (re)started root is in grace and gets priority."""
    id_patch, create_patch = _mock_root_probe(create_time=time.time())
    with id_patch, create_patch:
        assert _root_gateway_in_grace(tmp_path) is True


def test_root_out_of_grace_after_window(tmp_path):
    """A root up longer than the grace period is NOT deferred to."""
    id_patch, create_patch = _mock_root_probe(
        create_time=time.time() - (_DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS + 1),
    )
    with id_patch, create_patch:
        assert _root_gateway_in_grace(tmp_path) is False


def test_fingerprint_units_never_leak_into_grace(tmp_path):
    """Regression (round-2 Critical): the PID-reuse fingerprint from
    ``get_running_pid_identity_strict`` must NOT be used in uptime arithmetic.

    On macOS the fingerprint is a centisecond-epoch int (~1.78e11); on Linux it
    is ticks-since-boot. Feeding either into ``time.time() - start`` is wildly
    wrong (always-negative on macOS -> grace always True -> non-root defers to
    a live root forever). The grace probe must read the separate true-epoch
    ``get_process_create_time_epoch`` source. Here we pass a macOS-style
    centisecond fingerprint into the identity slot while the true-epoch probe
    says the root is up longer than the grace -> must be out-of-grace.
    """
    macos_fingerprint = int(round(time.time() * 100))  # e.g. 178828243616
    id_patch, create_patch = _mock_root_probe(
        create_time=time.time() - (_DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS + 1),
        identity=(42, macos_fingerprint),
    )
    with id_patch, create_patch:
        # Old root: not deferred to, even though the fingerprint is a huge
        # number that a naive time.time() - fingerprint would read as in-grace.
        assert _root_gateway_in_grace(tmp_path) is False
    # And the same fingerprint with a FRESH true-epoch creation time is in-grace
    # (proving the fingerprint value itself is inert, only the epoch source
    # drives the decision).
    id_patch2, create_patch2 = _mock_root_probe(
        create_time=time.time(),
        identity=(42, macos_fingerprint),
    )
    with id_patch2, create_patch2:
        assert _root_gateway_in_grace(tmp_path) is True


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
    import os as _os
    _os.utime(heartbeat, (old_stale, old_stale))
    assert _dispatcher_heartbeat_is_stale(heartbeat, now=time.time()) is True

    # Root process is alive (pidfile resolves) but it has been up LONGER than
    # the grace period — it settled and its loop silently died (the original
    # outage relocated to root). The strict-identity slot yields a live-root
    # PID; the true-epoch source says it is old.
    id_patch, create_patch = _mock_root_probe(
        create_time=time.time() - (_DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS + 1),
    )
    with id_patch, create_patch:
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

    # Root freshly restarted (within grace) -> non-root defers so root wins.
    id_patch, create_patch = _mock_root_probe(create_time=time.time())
    with id_patch, create_patch:
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