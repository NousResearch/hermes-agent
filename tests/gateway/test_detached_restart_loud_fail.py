"""Detached POSIX restart watcher loud-fail (#134728).

The POSIX branch of ``_launch_detached_restart_command`` used to spawn its
bash watcher with stdout/stderr on DEVNULL and no OSError guard: a spawn
failure (bash missing, fd exhaustion) escaped as a bare traceback into the
dying gateway's stderr, and a restart child that could not run (binary
replaced mid-restart, venv torn down, PATH loss) died invisibly — the parent
was already exiting, so the gateway never came back and nothing recorded why.
Both failure surfaces now land somewhere loud: the spawn failure in the
gateway log (mirroring the Windows branch's loud handling), the child's
streams plus exit code in ``logs/gateway-restart-watcher.log``.
"""

import logging
import os
import subprocess
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.restart_test_helpers import make_restart_runner


def _hermetic_runner(monkeypatch, tmp_path):
    """A runner whose watcher spawn is fully hermetic: no real env resolution,
    no real home — the log path pins to tmp_path."""
    runner, _adapter = make_restart_runner()
    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["hermes"])
    monkeypatch.setattr(
        GatewayShutdownMixin,
        "_restart_watcher_env",
        staticmethod(
            lambda: {"PATH": os.environ["PATH"], "HERMES_HOME": str(tmp_path)}
        ),
    )
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: tmp_path)
    return runner


@pytest.mark.platforms("macos", "linux")
@pytest.mark.asyncio
async def test_posix_detached_restart_spawn_failure_is_loud(
    monkeypatch, tmp_path, caplog
):
    """A watcher spawn that cannot start (OSError from Popen) is logged on the
    gateway log and swallowed — never raised into the dying process and never
    silently discarded (#134728)."""
    runner = _hermetic_runner(monkeypatch, tmp_path)

    def _refusing_popen(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "Popen", _refusing_popen)
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        # Must not raise: the caller is mid-shutdown; a bare OSError here would
        # abort the rest of the stop path.
        await runner._launch_detached_restart_command()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        # Mirrors the Windows watcher's privacy mode: interpreter basename +
        # numeric errno only, never argv/env/str(exc) (may carry a full path).
        "restart watcher was not started" in message and "errno=2" in message
        for message in messages
    ), f"expected a loud spawn-failure record, got: {messages}"


@pytest.mark.platforms("macos", "linux")
@pytest.mark.asyncio
async def test_posix_detached_restart_child_output_is_not_discarded(
    monkeypatch, tmp_path
):
    """The watcher's child streams must land in the restart diag log (with a
    header naming the attempt and an exit-code marker in the watcher shell),
    not DEVNULL — a restart child dying invisible is the #134728 alert-loss
    shape."""
    runner = _hermetic_runner(monkeypatch, tmp_path)
    captured = []

    def _recording_popen(argv, **kwargs):
        captured.append((argv, kwargs))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)
    await runner._launch_detached_restart_command()

    assert len(captured) == 1
    argv, kwargs = captured[0]
    log_path = tmp_path / "logs" / "gateway-restart-watcher.log"

    # Child streams ride the diag log, never DEVNULL.
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["stdout"] is not subprocess.DEVNULL
    assert getattr(kwargs["stdout"], "name", "").endswith("gateway-restart-watcher.log")

    # The header names the attempt before any child output.
    header = log_path.read_bytes()
    assert b"detached restart watcher" in header
    assert b"gateway restart" in header

    # The watcher shell records the restart child's exit code into the same log.
    bash_at = argv.index("bash")
    shell_cmd = argv[-1]
    assert argv[bash_at + 1] == "-lc"
    assert "rc=" in shell_cmd and "gateway-restart-watcher.log" in shell_cmd


@pytest.mark.platforms("macos", "linux")
@pytest.mark.asyncio
async def test_posix_detached_restart_keeps_detached_session_semantics(
    monkeypatch, tmp_path
):
    """Behavior guard: loud-fail must not change the launch contract — the watcher
    still detaches into its own session (setsid preferred, bare bash otherwise),
    still waits out the drain deadline, and still runs ``hermes gateway restart``."""
    runner = _hermetic_runner(monkeypatch, tmp_path)
    captured = []

    def _recording_popen(argv, **kwargs):
        captured.append((argv, kwargs))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)
    await runner._launch_detached_restart_command()

    assert len(captured) == 1
    argv, kwargs = captured[0]
    assert kwargs["start_new_session"] is True
    bash_at = argv.index("bash")
    assert bash_at in (0, 1)  # setsid-prefixed where setsid exists, bare bash otherwise
    assert argv[bash_at + 1] == "-lc"
    shell_cmd = argv[-1]
    assert f"kill -0 {os.getpid()}" in shell_cmd
    assert "gateway restart" in shell_cmd
