"""A pid that OpenProcess denies while psutil proves it gone must not strand its lease.

Windows reports ``ERROR_ACCESS_DENIED`` from ``OpenProcess`` for a dead pid whose kernel
object has not been reaped yet, so ``gateway.status._pid_exists`` calls it alive. psutil
raises ``NoSuchProcess`` for the same pid. The two answers used to resolve to "liveness
unknowable", and ``_prune_dead`` refuses a claim whose *target* owner is unknowable — so
every chat leased by the replaced backend stayed unsendable, behind a message that told
the user to delete a registry file that read perfectly well.
"""

import os

import pytest

from hermes_cli import active_sessions


DEAD_PID = 999_999  # no such process on any supported host


def _seed(home, session_id="dead-owner-chat"):
    state_path = home / "runtime" / "active_sessions.json"
    active_sessions._write_entries(state_path, [{
        "lease_id": "stranded-lease", "session_id": session_id, "surface": "desktop",
        "pid": DEAD_PID, "process_start_time": 1_793_684_221.3976328,
        "track_liveness": True,
    }])
    return state_path


def _as_windows_access_denied(monkeypatch):
    """The pid exists to OpenProcess, and psutil cannot read its start time."""
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr(active_sessions, "_process_start_time", lambda pid: None)


def test_pid_proven_gone_is_not_unknowable():
    assert active_sessions._pid_proven_gone(DEAD_PID) is True
    assert active_sessions._pid_proven_gone(os.getpid()) is False  # a live pid is never "gone"


def test_proven_dead_owner_is_pruned_and_the_chat_can_be_claimed(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    state_path = _seed(home)
    _as_windows_access_denied(monkeypatch)

    assert active_sessions._pid_liveness(DEAD_PID, 1_793_684_221.3976328) is False

    lease, message = active_sessions.try_acquire_active_session(
        session_id="dead-owner-chat", surface="desktop", config={}, track_liveness=True,
    )
    assert message is None and lease is not None
    remaining = active_sessions._read_entries(state_path)
    assert [entry["lease_id"] for entry in remaining] == [lease.lease_id]
    lease.release()


def test_untracked_claim_does_not_blame_the_registry_file(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    state_path = _seed(home)
    _as_windows_access_denied(monkeypatch)

    lease, message = active_sessions.try_acquire_active_session(
        session_id="dead-owner-chat", surface="cli", config={},
    )
    assert message is None and lease is not None
    assert active_sessions._read_entries(state_path)[0]["lease_id"] == lease.lease_id
    lease.release()


def test_unreadable_start_time_without_proof_still_fails_closed(tmp_path, monkeypatch):
    """No proof of death: the live writer might be another user's process — keep refusing."""
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed(home)
    _as_windows_access_denied(monkeypatch)
    monkeypatch.setattr(active_sessions, "_pid_proven_gone", lambda pid: False)

    assert active_sessions._pid_liveness(DEAD_PID, 1_793_684_221.3976328) is None
    with pytest.raises(active_sessions.ActiveSessionRegistryError, match="liveness is unknown"):
        active_sessions.try_acquire_active_session(
            session_id="dead-owner-chat", surface="desktop", config={}, track_liveness=True,
        )
