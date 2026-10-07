"""A multi-entry deferred refresh sweep must leave waiter windows between holds (#124533).

``_refresh_entry`` legitimately holds the cross-process auth-store lock across its refresh POST
(single-use-token safety is test-pinned), but ``_refresh_pending_entries`` used to chain N holds
back-to-back, so a waiter with a shorter timeout (a Desktop assistant start waiting on the same
profile auth.json, AUTH_LOCK_TIMEOUT_SECONDS = 15s) starved behind the whole chain. The sweep must
leave a bounded lock-free window between consecutive holds so a waiter can interleave.
"""

from __future__ import annotations

import time

import hermes_cli.auth as auth
from agent.credential_pool import CredentialPool


def test_pending_refresh_sweep_leaves_waiter_windows_between_holds(tmp_path):
    auth_path = tmp_path / "profiles" / "coder" / "auth.json"
    auth_path.parent.mkdir(parents=True)

    pool = object.__new__(CredentialPool)
    hold_events = []

    def fake_refresh(entry, *, force):
        # Same shape as a real single-use refresh: the cross-process lock is held
        # across the "POST" (the sleep), released only when the entry is done.
        with auth._auth_store_lock(timeout_seconds=30.0, target_path=auth_path):
            hold_events.append(("acquired", time.monotonic()))
            time.sleep(0.05)
            hold_events.append(("released", time.monotonic()))

    pool._refresh_entry = fake_refresh
    CredentialPool._refresh_pending_entries(pool, [object(), object(), object()])

    acquires = [t for kind, t in hold_events if kind == "acquired"]
    releases = [t for kind, t in hold_events if kind == "released"]
    assert len(acquires) == 3  # every entry refreshed
    inter_hold_windows = [a - r for r, a in zip(releases, acquires[1:])]
    # A waiter polling every 50ms (the _file_lock retry cadence) must get a real
    # window between two consecutive holds of the sweep, not a scheduling artifact.
    assert max(inter_hold_windows) >= 0.4, (
        f"deferred refresh sweep left no waiter window between holds: {inter_hold_windows}"
    )
