"""Session-list rows carry the session's live background work, so the sidebar can show Working
for a session this window never opened (started on the phone, or before an app restart)."""

from __future__ import annotations

import pytest

from hermes_cli.session_background_work import background_work_counts, stamp_background_work


class _Registry:
    def __init__(self, processes):
        self.processes = processes

    def list_sessions(self):
        return self.processes


@pytest.fixture()
def processes(monkeypatch):
    import tools.process_registry as registry_mod

    reg = _Registry([])
    monkeypatch.setattr(registry_mod, "process_registry", reg)
    return reg


@pytest.fixture()
def delegations(monkeypatch):
    import tools.async_delegation as deleg

    records: dict = {}
    monkeypatch.setattr(deleg, "_records", records)
    return records


def test_counts_running_processes_and_live_delegations_per_session(processes, delegations):
    processes.processes = [
        {"owner_task_id": "s1", "status": "running"},
        {"owner_task_id": "s1", "status": "exited"},
        {"owner_task_id": "s2", "status": "running"},
    ]
    delegations.update({
        "d1": {"status": "running", "parent_session_id": "s1"},
        "d2": {"status": "completed", "parent_session_id": "s2"},
        "d3": {"status": "stalling", "session_key": "s3"},
    })

    assert background_work_counts() == {"s1": 2, "s2": 1, "s3": 1}


def test_stamps_rows_by_id_or_lineage_and_leaves_idle_rows_bare(processes, delegations):
    processes.processes = [
        {"owner_task_id": "tip", "status": "running"},
        {"owner_task_id": "root", "status": "running"},
    ]
    rows = [{"id": "tip", "_lineage_root_id": "root"}, {"id": "idle"}]

    stamp_background_work(rows)

    assert rows[0]["background_work"] == 2
    assert "background_work" not in rows[1]


def test_registry_failure_leaves_rows_unstamped(monkeypatch, delegations):
    import tools.process_registry as registry_mod

    class _Broken:
        def list_sessions(self):
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry_mod, "process_registry", _Broken())
    rows = [{"id": "s1"}]

    stamp_background_work(rows)

    assert rows == [{"id": "s1"}]
