"""A turn end's active-session lease release never runs on the caller (the gateway loop).

The release takes the registry's file lock and rewrites its JSON; at every turn end that was a
measured loop stall. It is queued to one ordered worker instead, and still lands.
"""

import threading
import time

from gateway.active_lease_io import release_soon
from hermes_cli import active_sessions


def test_a_queued_release_returns_while_the_registry_is_locked_and_lands_after(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    lease, refusal = active_sessions.try_acquire_active_session(
        session_id='s1', surface='gateway:local', config={}, registry_home=tmp_path)
    assert refusal is None and lease is not None
    assert lease.lock_path is not None and lease.state_path is not None
    held = active_sessions._FileLock(lease.lock_path)  # another process mid-rewrite
    held.__enter__()
    try:
        caller = threading.Thread(target=release_soon, args=(lease,), daemon=True)
        caller.start()
        caller.join(timeout=5)
        # The registry lock is still held, so a caller that returned did not release inline.
        assert not caller.is_alive(), "the release waited on the registry lock in the caller"
        assert not lease.released
    finally:
        held.__exit__(None, None, None)
    deadline = time.monotonic() + 5
    while not lease.released:
        assert time.monotonic() < deadline, "the queued release never landed"
        time.sleep(0.02)
    assert active_sessions._read_entries(lease.state_path) == []
