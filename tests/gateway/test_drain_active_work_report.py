"""A draining gateway names the work it is waiting on, and the updater's wait prints it.

Before: ``hermes update`` printed "draining (up to 1875s)..." and then nothing for up to 30 minutes
while the gateway waited on one cron job; neither surface said WHAT was being waited on.
"""

import json
import time

import cron.scheduler as sched
from gateway.run import GatewayRunner
from gateway.session_state import SessionState
from hermes_cli.update_cmd_drain_report import drain_progress_reporter


class _FakeAgent:
    model = "test/model"

    def get_activity_summary(self):
        return {"current_tool": "terminal", "seconds_since_activity": 1.0}


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._restart_requested = True
    runner.adapters = {}
    return runner


def test_draining_status_names_chat_and_cron_units_and_clears_when_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = _runner()
    state = SessionState()
    state.turn.agent = _FakeAgent()
    state.turn.started_ts = time.time() - 10
    runner._sessions_map()["telegram:dm:1"] = state
    assert sched.try_register_running_job("job-a")
    try:
        with sched._running_lock:
            sched._running_worker_pids["job-a"] = 4242
        runner._update_runtime_status("draining")
        record = json.loads((tmp_path / "gateway_state.json").read_text())
        by_kind = {unit["kind"]: unit for unit in record["active_work"]}
        assert by_kind["chat"]["session"] == "telegram:dm:1" and by_kind["chat"]["current_tool"] == "terminal"
        assert by_kind["cron"]["job_id"] == "job-a" and by_kind["cron"]["pid"] == 4242 and by_kind["cron"]["external"]
        assert record["active_agents"] == len(record["active_work"])
    finally:
        sched.release_running_job("job-a")
    assert sched.get_running_job_details() == []
    runner._update_runtime_status("running")
    assert json.loads((tmp_path / "gateway_state.json").read_text())["active_work"] is None


def test_drain_progress_reporter_prints_holder_and_config_knob(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "cron").mkdir()
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{"id": "job-a", "name": "nightly-scout"}]}))
    (tmp_path / "gateway_state.json").write_text(json.dumps({
        "pid": 1, "gateway_state": "draining",
        "active_work": [{"kind": "cron", "job_id": "job-a", "pid": 4242, "external": True, "elapsed_s": 95}],
    }))
    out = []
    tick = drain_progress_reporter(tmp_path, budget_s=600, interval_s=0.0, emit=out.append)
    tick()
    report = out[0]
    assert "nightly-scout" in report and "job-a" in report and "pid 4242" in report and "1m35s" in report
    assert "restart_after_turn_timeout" in report

    # Pre-fix gateway (no active_work field): the wait still explains itself instead of going silent.
    (tmp_path / "gateway_state.json").write_text(json.dumps({"pid": 1, "gateway_state": "draining"}))
    out.clear()
    tick()
    assert "did not report" in out[0]
