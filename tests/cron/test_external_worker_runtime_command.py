"""External cron worker must launch through the installation-bound runtime
bootstrap, not the bare gateway interpreter (#124279).

A managed-store gateway runs on the bare store Python with the committed
dependency generation activated *in-process* (``hermes_bootstrap`` ->
``pm.environments.activate_dependencies``).  ``sys.executable`` therefore
does not own the runtime dependencies, and the sanitizer-built worker env
deliberately drops runtime site-packages (``cron/scheduler_worker_env.py``),
so a worker spawned as ``sys.executable -m cron.scheduler`` re-resolves
imports from scratch and dies with ``ModuleNotFoundError`` before its
ownership ack.

The launch site must use the same installation-bound launcher every other
Hermes entry point uses (``hermes_cli._launchers.runtime_command``): store
Python + ``-I`` + bootstrap that leases the committed generation in the
child, with the existing ``--external-worker-file``/``--ack-file`` argv
preserved after it.
"""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest


def _stub_external_worker_launch(scheduler, monkeypatch):
    """Fake Popen that acks the handoff and reports running -> completed.

    Returns ``(spawned, payloads, handoff, get)`` for the caller's assertions.
    """

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
            return self.returncode

    spawned = []
    payloads = []

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        payload_index = command.index("--external-worker-file") + 1
        payloads.append(json.loads(Path(command[payload_index]).read_text()))
        ack_index = command.index("--ack-file") + 1
        Path(command[ack_index]).write_text(
            json.dumps({"pid": 4321, "execution_id": "exec-1"}),
            encoding="utf-8",
        )
        return FakeProcess()

    handoff = Mock(return_value={"id": "exec-1", "handoff_pending": 1})
    monkeypatch.setattr(scheduler, "mark_execution_handoff_pending", handoff)
    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    observed_statuses = iter(
        [
            {"id": "exec-1", "status": "running"},
            {"id": "exec-1", "status": "completed"},
        ]
    )
    get = Mock(side_effect=lambda _execution_id: next(observed_statuses))
    monkeypatch.setattr(scheduler, "get_execution", get)
    return spawned, payloads, handoff, get


def _launch_worker(scheduler, monkeypatch, tmp_path):
    job = {"id": "job-1", "execution_id": "exec-1", "prompt": "work"}
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    from tools.process_registry import GatewayChildDispatch

    def wrap(command, *, unit_suffix, require_restart_safe_scope=False):
        return GatewayChildDispatch("scoped", list(command))

    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv", wrap
    )
    spawned, payloads, handoff, get = _stub_external_worker_launch(
        scheduler, monkeypatch
    )
    assert scheduler._launch_external_cron_worker(job) is True
    return spawned[0][0]


def test_worker_command_is_installation_bound_bootstrap(tmp_path, monkeypatch):
    """The spawned argv is runtime_command-shaped: -I, -c bootstrap importing
    hermes_bootstrap, launching cron.scheduler with the handoff argv intact."""
    import cron.scheduler as scheduler

    command = _launch_worker(scheduler, monkeypatch, tmp_path)

    assert command[1:3] == ["-I", "-c"], (
        "worker must launch via the installation-bound runtime command "
        "(store python -I -c <bootstrap>), got: %r" % (command[:3],)
    )
    bootstrap = command[3]
    assert "import hermes_bootstrap" in bootstrap
    assert "cron.scheduler" in bootstrap


def test_worker_preserves_handoff_argv(tmp_path, monkeypatch):
    """--external-worker-file/--ack-file survive after the bootstrap prefix."""
    import cron.scheduler as scheduler

    command = _launch_worker(scheduler, monkeypatch, tmp_path)

    payload_index = command.index("--external-worker-file") + 1
    assert Path(command[payload_index]).name == "exec-1.json"
    ack_index = command.index("--ack-file") + 1
    assert Path(command[ack_index]).name == "exec-1.ready"


def test_worker_command_uses_resolved_store_python(tmp_path, monkeypatch):
    """runtime_command resolves the store python; the bare sys.executable of the
    gateway process is only the last-resort fallback inside runtime_command
    itself, never the reason a managed worker loses its dependencies."""
    import cron.scheduler as scheduler
    import hermes_cli._launchers as launchers

    resolved = []
    real_runtime_command = launchers.runtime_command

    def spy(repo_root, args=(), **kwargs):
        result = real_runtime_command(repo_root, args, **kwargs)
        resolved.append(result[0])
        return result

    monkeypatch.setattr(
        "hermes_cli._launchers.runtime_command", spy
    )
    command = _launch_worker(scheduler, monkeypatch, tmp_path)
    assert resolved, "scheduler must build the worker command via runtime_command"
    assert command[0] == resolved[0]
