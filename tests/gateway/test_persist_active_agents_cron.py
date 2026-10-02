"""#122813: the gateway registers a cron lifecycle persist hook so
``gateway_state.json``'s ``active_agents`` refreshes the moment a cron job
starts or ends — not only at inbound-turn boundaries (until now the file read
0 during a whole cron job with no inbound traffic, and a count written mid-job
stayed stuck until the next inbound message)."""
import cron.scheduler as sched

from tests.gateway.restart_test_helpers import make_restart_runner


def test_gateway_persists_on_cron_register_and_release(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "gateway.run._write_runtime_status_quiet",
        lambda **fields: writes.append(fields))
    runner, _adapter = make_restart_runner()

    runner._register_cron_persist_hook()
    job_id = "persist-gw-test-1"
    try:
        assert sched.try_register_running_job(job_id) is True
        assert writes and writes[-1].get("active_agents") == 1
        sched.release_running_job(job_id)
        assert writes[-1].get("active_agents") == 0
    finally:
        runner._unregister_cron_persist_hook()
        sched.release_running_job(job_id)  # idempotent safety net


def test_unregister_stops_gateway_persist(monkeypatch):
    writes = []
    monkeypatch.setattr("gateway.run._write_runtime_status_quiet",
                        lambda **fields: writes.append(fields))
    runner, _adapter = make_restart_runner()
    runner._register_cron_persist_hook()
    runner._unregister_cron_persist_hook()

    job_id = "persist-gw-test-2"
    try:
        assert sched.try_register_running_job(job_id) is True
    finally:
        sched.release_running_job(job_id)
    assert writes == []


def test_register_is_idempotent(monkeypatch):
    writes = []
    monkeypatch.setattr("gateway.run._write_runtime_status_quiet",
                        lambda **fields: writes.append(fields))
    runner, _adapter = make_restart_runner()
    runner._register_cron_persist_hook()
    try:
        runner._register_cron_persist_hook()  # double register: still one callback
        job_id = "persist-gw-test-3"
        try:
            assert sched.try_register_running_job(job_id) is True
            # One notification -> one write carrying the count (not two).
            assert len([w for w in writes if "active_agents" in w]) == 1
        finally:
            sched.release_running_job(job_id)
    finally:
        runner._unregister_cron_persist_hook()
