"""The cron external worker must be launched through this install's launcher.

The worker is spawned as ``sys.executable -m cron.scheduler``. Under the PM layout
``sys.executable`` is the *store* interpreter, which carries no third-party distributions: the
dependency generations live apart and are leased by ``hermes_bootstrap``. A bare argv inherits no
bootstrap, so the worker dies on its first third-party import (``ruamel``, via ``hermes_yaml``)
before it can publish its ownership acknowledgement — every job then fails with

    Restart-safe cron worker dispatch failed: cron external worker exited before ownership
    acknowledgement (exit 1); worker stderr: ... ModuleNotFoundError: No module named 'ruamel'

The neighbouring ``test_restart_safe_worker.py`` stubs ``restart_safe_gateway_child_argv`` and never
executes a real spawn, which is why that shipped. These two tests close that gap: one pins the argv
contract in CI, the other actually runs it and requires a clean import.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from cron import scheduler as sched
from tools import process_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


def _captured_worker_argv(monkeypatch) -> list[str]:
    """The argv ``_launch_external_cron_worker`` builds, without running a job.

    The dispatch hook is the last thing the function does before any handoff state is written, so
    patching it to report ``in_process`` returns immediately after the command is built — no
    payload, acknowledgement or worker process.
    """
    captured: dict[str, list[str]] = {}

    def fake_dispatch(command, **kwargs):
        captured["command"] = list(command)
        return process_registry.GatewayChildDispatch("in_process", command)

    monkeypatch.setattr(process_registry, "restart_safe_gateway_child_argv", fake_dispatch)
    ran = sched._launch_external_cron_worker({"id": "test-job", "execution_id": "test-execution"})
    assert ran is False, "the in_process dispatch must not claim to have launched a worker"
    return captured["command"]


def test_worker_command_uses_the_launcher_bootstrap(monkeypatch):
    """argv must be ``[interpreter, "-I", "-c", <bootstrap>, *args]``, never a bare ``-m``.

    A bare ``-m cron.scheduler`` child has no bootstrap, so it cannot see this install's dependency
    generation — the regression this file exists for.
    """
    argv = _captured_worker_argv(monkeypatch)

    assert argv[1:3] == ["-I", "-c"], f"worker argv is not launcher-shaped: {argv!r}"
    bootstrap = argv[3]
    assert "hermes_bootstrap" in bootstrap, "the launcher bootstrap must lease the dependencies"
    assert "cron.scheduler" in bootstrap, "the bootstrap must run the worker module"
    assert "--external-worker-file" in argv and "--ack-file" in argv, argv


def test_worker_command_survives_a_third_party_import(monkeypatch, tmp_path):
    """Run the real argv: it must reach the payload, not die importing a third-party package.

    A missing payload file is the *expected* failure here — it proves the module imported and ran.
    What must never appear is an import error.
    """
    argv = _captured_worker_argv(monkeypatch)
    payload_index = argv.index("--external-worker-file") + 1
    argv[payload_index] = str(tmp_path / "absent-payload.json")

    proc = subprocess.run(
        argv, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "ModuleNotFoundError" not in combined, combined[-2000:]
    assert "No module named" not in combined, combined[-2000:]
