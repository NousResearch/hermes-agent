"""Sibling owner-liveness checks tolerate a drifted start-time fingerprint (#117505).

``cron.executions`` is not the only reconciler that compares a recorded owner fingerprint with
a fresh ``get_process_start_time`` reading: the delivery ledger, API-server durable runs and the
async-delegation ledger did the same exact-equality test and reconciled a live owner to
dead/unknown under the same ~1 s same-host drift. All of them now share
``gateway.status.start_time_fingerprints_match``.
"""

from __future__ import annotations

import pytest

from gateway import status
from gateway.status import START_TIME_DRIFT_TOLERANCE

RECORDED = 178864182760


def _conflict(recorded_start, current_start) -> bool:
    """``gateway.status._start_times_conflict`` under test, named for readability below."""
    return status._start_times_conflict(recorded_start, current_start)


@pytest.fixture(autouse=True)
def _live_pid_with_drift(monkeypatch):
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: RECORDED + 100)


def _delivery_ledger_alive() -> bool:
    from gateway import delivery_ledger

    return delivery_ledger._owner_alive(4242, RECORDED)


def _api_server_run_alive() -> bool:
    from gateway.platforms import api_server_runs

    return api_server_runs._owner_alive(4242, RECORDED)


def _async_delegation_alive(monkeypatch, tmp_path) -> bool:
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, updated_at, "
            "delivery_state, delivery_attempts, owner_pid, owner_started_at, task_json) "
            "VALUES ('d-drift', 's', 'running', 1, 1, 'pending', 0, 4242, ?, '{}')",
            (RECORDED,),
        )
    return ad.recover_abandoned_delegations() == 0


@pytest.mark.parametrize("site", ["delivery_ledger", "api_server_runs", "async_delegation"])
def test_one_second_drift_keeps_owner_live(site, monkeypatch, tmp_path):
    if site == "delivery_ledger":
        assert _delivery_ledger_alive() is True
    elif site == "api_server_runs":
        assert _api_server_run_alive() is True
    else:
        assert _async_delegation_alive(monkeypatch, tmp_path) is True


def test_far_fingerprint_is_still_a_recycled_pid(monkeypatch):
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: RECORDED + 3600)
    assert _delivery_ledger_alive() is False
    assert _api_server_run_alive() is False


# ── gateway/status.py::_start_times_conflict ────────────────────────────────
#
# The sibling exact-compare fixed in the same commit, on the path that decides whether a LIVE
# gateway's scoped lock is stale and may be taken over. It previously had no test coverage
# anywhere in tests/, so the tolerance change -- and its junk-handling side effect -- were
# unpinned. Each case below is deliberate, not incidental.


def test_conflict_none_on_either_side_is_not_a_conflict():
    """No reading on one side = no evidence either way, so not a conflict.

    Pins ``None`` semantics: an unreadable start time must never be read as PID reuse.
    """
    assert _conflict(RECORDED, None) is False
    assert _conflict(None, RECORDED) is False
    assert _conflict(None, None) is False


def test_conflict_within_drift_tolerance_is_not_a_conflict():
    """Drift inside START_TIME_DRIFT_TOLERANCE is the same incarnation (macOS kern.boottime)."""
    for drift in (0, 1, 100, START_TIME_DRIFT_TOLERANCE):
        assert _conflict(RECORDED, RECORDED + drift) is False, f"{drift}cs of drift"
        assert _conflict(RECORDED, RECORDED - drift) is False, f"{-drift}cs of drift"


def test_conflict_beyond_tolerance_is_a_conflict():
    """Past the window it is a different process again -- PID reuse still refused."""
    assert _conflict(RECORDED, RECORDED + START_TIME_DRIFT_TOLERANCE + 1) is True
    assert _conflict(RECORDED, RECORDED - (START_TIME_DRIFT_TOLERANCE + 1)) is True
    assert _conflict(RECORDED, RECORDED + 3600_00) is True


def test_conflict_junk_on_either_side_is_not_a_conflict():
    """Unparseable input is not evidence of reuse, so it is not a conflict.

    This is a deliberate behaviour change from the old exact ``!=`` (``"abc"`` vs ``"def"`` was a
    conflict). Pinned here so it cannot drift silently: junk means "unknown", and the callers fall
    back to their other evidence (cmdline / record argv) rather than treating junk as proof.
    """
    assert _conflict("abc", "def") is False
    assert _conflict("abc", RECORDED) is False
    assert _conflict(RECORDED, "def") is False
    assert _conflict("", RECORDED) is False
    assert _conflict(RECORDED, "") is False


def test_conflict_non_numeric_types_are_not_a_conflict():
    """Non-numeric types raise inside the comparator and are swallowed the same way."""
    assert _conflict(RECORDED, object()) is False
    assert _conflict(RECORDED, [RECORDED]) is False


@pytest.mark.parametrize(
    "drift_cs, expect_live",
    [
        (0, True),
        (START_TIME_DRIFT_TOLERANCE, True),   # same incarnation despite drift
        (START_TIME_DRIFT_TOLERANCE + 1, False),  # recycled PID -> record is not live
    ],
)
def test_live_pid_from_record_tolerates_drift_but_refuses_reuse(monkeypatch, drift_cs, expect_live):
    """Call site 1 (status.py:1015): a drifted-but-live PID stays live; a recycled one does not."""
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: RECORDED + drift_cs)

    assert status._live_pid_from_record({"pid": 4242, "start_time": RECORDED}) is (
        4242 if expect_live else None
    )


def test_scoped_lock_record_is_not_stale_when_owner_only_drifted(monkeypatch):
    """Call site 2 (status.py:1675): the live gateway's lock must NOT be declared stale.

    Drift means the recorded owner is still running, so another gateway must not take the lock
    over it -- the same duplicate-writer shape as the board bug, one layer down.
    """
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: RECORDED + START_TIME_DRIFT_TOLERANCE)
    monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)
    monkeypatch.setattr(status, "_process_is_stopped", lambda pid: False)
    record = {"pid": 4242, "start_time": RECORDED, "kind": status._GATEWAY_KIND}

    assert status._scoped_lock_record_is_stale(record, 4242) is False


def test_scoped_lock_record_is_stale_when_owner_pid_was_reused(monkeypatch):
    """The guard's purpose is preserved: a genuinely reused PID still makes the record stale."""
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: RECORDED + 3600_00)
    monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)
    monkeypatch.setattr(status, "_process_is_stopped", lambda pid: False)
    record = {"pid": 4242, "start_time": RECORDED, "kind": status._GATEWAY_KIND}

    assert status._scoped_lock_record_is_stale(record, 4242) is True
