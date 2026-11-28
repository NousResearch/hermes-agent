"""The exit-poll acknowledgement race must not turn into a false handoff timeout."""

import json
from pathlib import Path


def test_exit_poll_crossing_full_grace_still_accepts_ack(tmp_path, monkeypatch, caplog):
    """An ack published during poll must be consumed even if the grace expires there.

    #154901: rechecking an ack and continuing can jump past the loop deadline.
    The ordinary handoff path must record the worker PID, not warn about timeout.
    """
    from cron import scheduler
    from cron.executions import HANDOFF_ADOPTION_GRACE_SECONDS
    from tools.process_registry import GatewayChildDispatch

    job_id, execution_id = "job-deadline-ack", "exec-deadline-ack"
    ack_path = tmp_path / "cron" / "external-workers" / f"{execution_id}.ready"
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv",
        lambda command, **_: GatewayChildDispatch("degraded", list(command)),
    )
    monkeypatch.setattr(
        scheduler, "mark_execution_handoff_pending",
        lambda _eid: {"id": execution_id, "handoff_pending": 1},
    )
    monkeypatch.setattr(
        scheduler, "get_execution",
        lambda _eid: {"id": execution_id, "status": "completed"},
    )
    monkeypatch.setattr(scheduler, "_running_worker_pids", {})

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

    clock = Clock()

    class AckThenExit:
        pid = 4321
        returncode = None

        def poll(self):
            if self.returncode is None:
                # The parent already observed a missing ack; descheduling here
                # lets the completed worker publish its ack and pass the deadline.
                ack_path.write_text(
                    json.dumps({"pid": self.pid, "execution_id": execution_id}),
                    encoding="utf-8",
                )
                self.returncode = 0
                clock.now = HANDOFF_ADOPTION_GRACE_SECONDS + 1.0
            return self.returncode

        def wait(self, timeout=None):
            return self.poll()

    def popen(command, **_kwargs):
        assert Path(command[command.index("--ack-file") + 1]) == ack_path
        return AckThenExit()

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    monkeypatch.setattr(scheduler.time, "monotonic", clock.monotonic)

    with caplog.at_level("WARNING", logger=scheduler.logger.name):
        assert scheduler._launch_external_cron_worker(
            {"id": job_id, "execution_id": execution_id, "prompt": "work"}
        ) is True

    assert scheduler._running_worker_pids == {scheduler._inflight_key(job_id): 4321}
    assert "did not acknowledge" not in caplog.text
    assert not ack_path.exists()
