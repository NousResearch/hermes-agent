"""Gateway restart must not sweep jobs that explicitly survive owner release."""

import time

from gateway.run_shutdown import GatewayShutdownMixin
from tools import process_registry as registry_module
from tools.process_registry import ProcessRegistry, ProcessSession


def test_restart_cleanup_keeps_persisted_job_but_host_shutdown_reaps_it(monkeypatch):
    """Exercise the gateway's sweep entry point, not just kill_all's source filter."""
    registry = ProcessRegistry()
    volatile = ProcessSession(
        id="proc_volatile", command="sleep 60", task_id="owner",
        started_at=time.time(), persist_on_release=False,
    )
    persisted = ProcessSession(
        id="proc_persisted", command="sleep 60", task_id="owner",
        started_at=time.time(), persist_on_release=True,
    )
    registry._running.update({volatile.id: volatile, persisted.id: persisted})
    monkeypatch.setattr(registry_module, "process_registry", registry)

    killed = []

    def fake_kill(session_id, *, source, consume_output=False):
        killed.append((session_id, source))
        registry._running[session_id].exited = True
        return {"status": "killed"}

    monkeypatch.setattr(registry, "kill_process", fake_kill)

    def run_registry_step_only(label, fn):
        # Skip unrelated cron/delegation/browser/environment teardown; exercise
        # the real registry and the gateway's selection of its kill source.
        return fn() if label.startswith("process_registry.kill_all") else None

    monkeypatch.setattr(
        GatewayShutdownMixin, "_quiet_step", staticmethod(run_registry_step_only)
    )

    GatewayShutdownMixin._stop_kill_tool_subprocesses("post-interrupt", restart=True)
    assert killed == [("proc_volatile", "gateway_restart")]
    assert persisted.exited is False

    GatewayShutdownMixin._stop_kill_tool_subprocesses("final-cleanup", restart=False)
    assert killed == [
        ("proc_volatile", "gateway_restart"),
        ("proc_persisted", "gateway_shutdown"),
    ]
    assert persisted.exited is True
