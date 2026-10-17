"""Owner-liveness is fail-safe: a live pid is never proved dead by a fingerprint mismatch.

``cron.executions`` is not the only reconciler that compared a recorded owner start fingerprint
with a fresh ``get_process_start_time`` reading: the delivery ledger, API-server durable runs and
the async-delegation ledger treated a mismatch as death and reconciled a live owner to
dead/unknown. Same-host readings of a LIVE process drift without bound (measured 2026-10-17:
1064 of 1073 live-owner rows mismatched the 2 s tolerance), so a fixed tolerance can never
separate "drifted" from "recycled". Every one of these liveness checks now fails safe: the owner
pid's continued existence is the only positive signal, and a mismatched fingerprint never proves
death. A genuinely wedged claim is released by each caller's stale bound, never by the
fingerprint.
"""

from __future__ import annotations

import pytest

from gateway import status

RECORDED = 178864182760


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


def test_unbounded_drift_never_proves_death(monkeypatch):
    """A live owner whose fingerprint drifted far past the old tolerance is still alive.

    The drift has no fixed bound, so no magnitude of mismatch can prove PID reuse. A far mismatch
    on a still-existing pid therefore stays owned; the stale bound is the reclaim path.
    """
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: RECORDED + 3600)
    assert _delivery_ledger_alive() is True
    assert _api_server_run_alive() is True


def test_gone_owner_is_still_dead(monkeypatch):
    """A pid that no longer exists remains the one way liveness proves death."""
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
    assert _delivery_ledger_alive() is False
    assert _api_server_run_alive() is False
