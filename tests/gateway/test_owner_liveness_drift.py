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


def test_far_fingerprint_is_still_a_recycled_pid(monkeypatch):
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: RECORDED + 3600)
    assert _delivery_ledger_alive() is False
    assert _api_server_run_alive() is False
