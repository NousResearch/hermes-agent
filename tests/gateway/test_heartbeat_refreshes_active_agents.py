"""The 1-minute runtime heartbeat also refreshes ``active_agents``."""
from types import SimpleNamespace

import gateway.run as gateway_run


def test_heartbeat_writes_active_agents(monkeypatch):
    writes = []
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **f: writes.append(f))
    gateway_run._heartbeat_runtime_status(SimpleNamespace(_active_work_count=lambda: 3))
    gateway_run._heartbeat_runtime_status(None)
    assert writes == [{"active_agents": 3}, {}]


def test_heartbeat_survives_count_failure(monkeypatch):
    writes = []
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **f: writes.append(f))

    def boom():
        raise RuntimeError

    gateway_run._heartbeat_runtime_status(SimpleNamespace(_active_work_count=boom))
    assert writes == [{}]
