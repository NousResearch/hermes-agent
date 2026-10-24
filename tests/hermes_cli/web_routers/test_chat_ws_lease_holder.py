"""The dashboard reads a resumed chat's lease holder from the HOME that chat runs in.

Regression for #131172: a profile-scoped chat's TUI child claims its single-writer lease
under the profile's home, so a lookup against the dashboard's launch home finds nothing and
the stranded PTY holding that lease is never released.
"""

import os

from hermes_cli.active_sessions import try_acquire_active_session
from hermes_cli.web_routers.chat_ws import _lease_holder_pid


def _claim(home, session_id):
    lease, refusal = try_acquire_active_session(
        session_id=session_id, surface="tui", config={}, registry_home=home,
        metadata={"live_session_id": "x"},
    )
    assert lease is not None, refusal
    return lease


def test_lease_holder_is_read_from_the_chats_profile_home(tmp_path):
    profile_home = tmp_path / "profiles" / "work"
    launch_home = tmp_path / "default"
    profile_home.mkdir(parents=True)
    launch_home.mkdir()
    lease = _claim(profile_home, "sess-a")
    try:
        assert _lease_holder_pid("sess-a", registry_home=str(profile_home)) == os.getpid()
        assert _lease_holder_pid("sess-a", registry_home=str(launch_home)) is None
        assert _lease_holder_pid("sess-b", registry_home=str(profile_home)) is None
        assert _lease_holder_pid(None, registry_home=str(profile_home)) is None
    finally:
        lease.release()


def test_unreadable_registry_yields_no_holder(tmp_path):
    bogus = tmp_path / "runtime" / "active_sessions.json"
    bogus.parent.mkdir(parents=True)
    bogus.write_text("{not json")
    assert _lease_holder_pid("sess-a", registry_home=str(tmp_path)) is None
