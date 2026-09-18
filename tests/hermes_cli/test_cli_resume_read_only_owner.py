from pathlib import Path
import json
import pytest

from cli import HermesCLI
from hermes_cli import active_sessions
from hermes_cli.active_sessions import (
    ActiveSessionLease,
    RegistryUnreadableError,
    active_session_registry_snapshot,
    live_session_owner,
    transfer_active_session,
    try_acquire_active_session,
)


@pytest.fixture(autouse=True)
def setup_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_writer_acquire_and_observer_allowed():
    """Writer A + observer B for the same session is allowed."""
    lease_a, msg_a = try_acquire_active_session(
        session_id="session-shared",
        surface="cli",
        config={"max_concurrent_sessions": 5},
        mode="writer",
    )
    assert lease_a is not None
    assert msg_a is None
    assert lease_a.mode == "writer"

    lease_b, msg_b = try_acquire_active_session(
        session_id="session-shared",
        surface="desktop",
        config={"max_concurrent_sessions": 5},
        mode="observer",
    )
    assert lease_b is not None
    assert msg_b is None
    assert lease_b.mode == "observer"

    snapshot = active_session_registry_snapshot()
    assert len(snapshot) == 2
    modes = {entry["lease_id"]: entry.get("mode") for entry in snapshot}
    assert modes[lease_a.lease_id] == "writer"
    assert modes[lease_b.lease_id] == "observer"

    lease_a.release()
    lease_b.release()


def test_duplicate_writer_refused():
    """Writer A + writer B for the same session is rejected."""
    lease_a, msg_a = try_acquire_active_session(
        session_id="session-exclusive",
        surface="cli",
        config={},
        mode="writer",
    )
    assert lease_a is not None
    assert msg_a is None

    lease_b, msg_b = try_acquire_active_session(
        session_id="session-exclusive",
        surface="tui",
        config={},
        mode="writer",
    )
    assert lease_b is None
    assert msg_b is not None
    assert "already owned by another active writer" in msg_b

    lease_a.release()

    # After writer A releases, writer B can acquire
    lease_b2, msg_b2 = try_acquire_active_session(
        session_id="session-exclusive",
        surface="tui",
        config={},
        mode="writer",
    )
    assert lease_b2 is not None
    assert msg_b2 is None
    lease_b2.release()


def test_transfer_refused_when_target_session_owned_by_foreign_writer():
    """Transfer cannot steal a session already owned by another active writer."""
    lease_a, _ = try_acquire_active_session(
        session_id="target-session",
        surface="cli",
        config={},
        mode="writer",
    )
    assert lease_a is not None

    lease_b, _ = try_acquire_active_session(
        session_id="source-session",
        surface="gateway",
        config={},
        mode="writer",
    )
    assert lease_b is not None

    # Attempt to transfer lease_b onto target-session (held by lease_a)
    transferred = transfer_active_session(lease_b, session_id="target-session")
    assert transferred is False
    assert lease_b.session_id == "source-session"

    snapshot = active_session_registry_snapshot()
    sessions = {entry["lease_id"]: entry["session_id"] for entry in snapshot}
    assert sessions[lease_a.lease_id] == "target-session"
    assert sessions[lease_b.lease_id] == "source-session"

    lease_a.release()
    lease_b.release()


def test_dead_owner_pruned_and_reacquired():
    """Dead owners are pruned before evaluating writer collisions."""
    state_path = active_sessions._state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    dead_entry = {
        "lease_id": "dead-lease",
        "session_id": "stale-session",
        "surface": "cli",
        "pid": 99999999,
        "mode": "writer",
        "started_at": 1000.0,
        "updated_at": 1000.0,
    }
    active_sessions._write_entries(state_path, [dead_entry])

    lease, msg = try_acquire_active_session(
        session_id="stale-session",
        surface="cli",
        config={},
        mode="writer",
    )
    assert lease is not None
    assert msg is None

    snapshot = active_session_registry_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["lease_id"] == lease.lease_id
    lease.release()


def test_unreadable_registry_fails_closed(tmp_path):
    """Corrupt or unreadable registry fails closed rather than assuming unowned."""
    state_path = active_sessions._state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{ corrupt json ... not valid", encoding="utf-8")

    with pytest.raises(RegistryUnreadableError):
        live_session_owner("any-session")

    lease, msg = try_acquire_active_session(
        session_id="any-session",
        surface="cli",
        config={},
        mode="writer",
    )
    assert lease is None
    assert msg is not None
    assert "unreadable or corrupt" in msg

    dummy_lease = ActiveSessionLease(
        lease_id="dummy",
        session_id="dummy-session",
        surface="cli",
        enabled=True,
    )
    assert transfer_active_session(dummy_lease, session_id="new-session") is False


def test_live_session_owner_lookup():
    """live_session_owner returns the active writer or None."""
    assert live_session_owner("non-existent") is None

    lease, _ = try_acquire_active_session(
        session_id="owner-check",
        surface="cli",
        config={},
        mode="writer",
    )
    assert lease is not None

    owner = live_session_owner("owner-check")
    assert owner is not None
    assert owner["lease_id"] == lease.lease_id
    assert owner["session_id"] == "owner-check"

    lease.release()
    assert live_session_owner("owner-check") is None


def test_cli_resume_opens_read_only_when_foreign_writer_active():
    """CLI interactive resume on an actively-owned session opens in read-only observer mode."""
    writer_lease, _ = try_acquire_active_session(
        session_id="resumed-session",
        surface="tui",
        config={},
        mode="writer",
    )
    assert writer_lease is not None

    cli = object.__new__(HermesCLI)
    cli.session_id = "resumed-session"
    cli.config = {}
    cli._active_session_lease = None
    cli._resumed = True
    cli._single_query_mode = False
    printed = []
    cli._console_print = lambda text: printed.append(text)

    assert cli._claim_active_session("cli") is True
    assert cli._active_session_lease is not None
    assert cli._active_session_lease.mode == "observer"
    assert getattr(cli, "is_read_only", False) is True
    assert any("read-only observer mode" in p for p in printed)

    assert cli.chat("Attempted prompt") is None

    cli._release_active_session()
    writer_lease.release()
