"""The review fork's durable turn-lease holder is qualified by PID namespace like every other.

``hermes_state_pidns`` never probes an UNSTAMPED TTL holder on a host with PID namespaces (STRICT
policy: a false reclaim ends a live turn), so a review holder without the stamp would leave a
crashed fork's row to its TTL under a waiting user turn there, while the stamped foreground and
compression holders are reclaimed at once. Pinning the local namespace makes the contract
host-independent.
"""

from __future__ import annotations

import os
import types

import pytest

import hermes_state
import hermes_state_pidns
from agent import background_review as background_review_module
from agent import review_admission
from hermes_state import SessionDB
from hermes_state_compression import BACKGROUND_REVIEW_LEASE_HOLDER_MARK
from hermes_state_pidns import LocalPidNamespace


def test_dead_review_fork_in_this_pid_namespace_is_reclaimed_like_any_holder(
    tmp_path, monkeypatch, caplog
):
    """A dead fork in our namespace is reclaimed by the next foreground acquire by the same rule
    as any other holder, and logged as the review owner."""
    monkeypatch.setattr(
        hermes_state_pidns, "_LOCAL_PID_NS", LocalPidNamespace("111", True)
    )
    path = tmp_path / "state.db"
    review_db = SessionDB(path)
    foreground_db = SessionDB(path)
    review_db.create_session("shared-session", source="test")
    review_agent = types.SimpleNamespace()
    run = background_review_module._BackgroundReviewRun()
    dead_fork_pid = 424242

    class _ForkProcessOs:
        """The fork's view of ``os``: its own pid, everything else the real module."""

        @staticmethod
        def getpid() -> int:
            return dead_fork_pid

        def __getattr__(self, name):
            return getattr(os, name)

    with pytest.MonkeyPatch.context() as fork_process:
        fork_process.setattr(background_review_module, "os", _ForkProcessOs())
        lease, reason = background_review_module._try_acquire_durable_review_lease(
            types.SimpleNamespace(_session_db=review_db),
            review_agent,
            "shared-session",
            run,
        )
    assert reason is None and lease is not None
    holder = review_agent._active_session_turn_lease_holder
    assert holder.startswith(f"pid={dead_fork_pid}:pidns=111")
    assert BACKGROUND_REVIEW_LEASE_HOLDER_MARK in holder

    probed: list[int] = []

    def pid_exists(pid: int) -> bool:
        probed.append(pid)
        return False

    monkeypatch.setattr(
        hermes_state, "psutil", types.SimpleNamespace(pid_exists=pid_exists)
    )
    foreground_holder = f"pid={os.getpid()}:pidns=111:turn=foreground"
    with caplog.at_level("INFO"):
        assert foreground_db.try_acquire_session_turn_lease(
            "shared-session", foreground_holder, ttl_seconds=5
        )
    assert probed == [dead_fork_pid]
    reclaimed = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_LEASE_EXPIRED_RECLAIMED in record.getMessage()
    ]
    assert len(reclaimed) == 1 and "owner=" in reclaimed[0]
    lease.release()  # holder-fenced: the row is the foreground's now
    foreground_db.release_session_turn_lease("shared-session", foreground_holder)
