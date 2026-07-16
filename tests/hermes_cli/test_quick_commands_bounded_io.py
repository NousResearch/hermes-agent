"""Streaming output-limit tests for deterministic argv quick commands."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from hermes_cli.quick_commands import (
    QUICK_COMMAND_OUTPUT_MAX_BYTES,
    QuickCommandOutputError,
    communicate_bounded_async,
    run_bounded_argv,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _minimal_test_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "")}


def test_minimal_environment_preserves_context_resolved_hermes_home(tmp_path):
    from hermes_cli.quick_commands import build_argv_environment

    named_home = tmp_path / "profiles" / "coder"
    token = set_hermes_home_override(named_home)
    try:
        child_env = build_argv_environment()
    finally:
        reset_hermes_home_override(token)

    assert child_env["HERMES_HOME"] == str(named_home)


@pytest.mark.live_system_guard_bypass
def test_sync_streaming_combined_cap_terminates_and_reaps(monkeypatch):
    spawned: list[subprocess.Popen[bytes]] = []
    spawn_kwargs: list[dict[str, object]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if args and args[0][0] == sys.executable:
            spawned.append(proc)
            spawn_kwargs.append(kwargs)
        return proc

    monkeypatch.setattr(
        "hermes_cli.quick_commands.subprocess.Popen", recording_popen
    )
    command = [
        sys.executable,
        "-c",
        "import sys,time;"
        "sys.stdout.buffer.write(b'o'*40000);sys.stdout.flush();"
        "sys.stderr.buffer.write(b'e'*40000);sys.stderr.flush();"
        "time.sleep(60)",
    ]

    with pytest.raises(QuickCommandOutputError, match="65536"):
        run_bounded_argv(command, env=_minimal_test_env(), timeout=5)

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert spawn_kwargs[0]["stdin"] is subprocess.DEVNULL
    assert spawn_kwargs[0]["stdout"] is subprocess.PIPE
    assert spawn_kwargs[0]["stderr"] is subprocess.PIPE
    assert spawn_kwargs[0]["env"] == _minimal_test_env()
    assert "shell" not in spawn_kwargs[0]


@pytest.mark.live_system_guard_bypass
def test_sync_streaming_timeout_terminates_and_reaps(monkeypatch):
    spawned: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if args and args[0][0] == sys.executable:
            spawned.append(proc)
        return proc

    monkeypatch.setattr(
        "hermes_cli.quick_commands.subprocess.Popen", recording_popen
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded_argv(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=_minimal_test_env(),
            timeout=0.1,
        )

    assert len(spawned) == 1
    assert spawned[0].poll() is not None


@pytest.mark.live_system_guard_bypass
def test_sync_reader_hang_terminates_forked_descendant(monkeypatch, tmp_path):
    pid_path = tmp_path / "descendant.pid"
    heartbeat_path = tmp_path / "descendant.heartbeat"
    descendant = (
        "import pathlib,sys,time\n"
        "path=pathlib.Path(sys.argv[1])\n"
        "while True:\n"
        " path.write_text(str(time.time_ns()))\n"
        " time.sleep(0.01)\n"
    )
    script = (
        "import pathlib,subprocess,sys,time\n"
        "heartbeat=pathlib.Path(sys.argv[2])\n"
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],str(heartbeat)])\n"
        "deadline=time.monotonic()+2\n"
        "while not heartbeat.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
    )
    monkeypatch.setattr("hermes_cli.quick_commands._PROCESS_STOP_GRACE_SECONDS", 0.1)

    with pytest.raises(QuickCommandOutputError, match="streams did not close"):
        run_bounded_argv(
            [
                sys.executable,
                "-c",
                script,
                str(pid_path),
                str(heartbeat_path),
                descendant,
            ],
            env=_minimal_test_env(),
            timeout=5,
        )

    descendant_pid = int(pid_path.read_text())
    before = heartbeat_path.read_text()
    import time

    time.sleep(0.1)
    assert heartbeat_path.read_text() == before, (
        f"forked argv descendant {descendant_pid} survived reader hang"
    )


def test_sync_streaming_allows_exact_combined_cap():
    half = QUICK_COMMAND_OUTPUT_MAX_BYTES // 2
    result = run_bounded_argv(
        [
            sys.executable,
            "-c",
            "import sys;"
            f"sys.stdout.buffer.write(b'o'*{half});sys.stdout.flush();"
            f"sys.stderr.buffer.write(b'e'*{half});sys.stderr.flush()",
        ],
        env=_minimal_test_env(),
        timeout=5,
    )

    assert result.returncode == 0
    assert len(result.stdout) + len(result.stderr) == QUICK_COMMAND_OUTPUT_MAX_BYTES


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_async_streaming_combined_cap_terminates_and_reaps():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys,time;"
        "sys.stdout.buffer.write(b'o'*40000);sys.stdout.flush();"
        "sys.stderr.buffer.write(b'e'*40000);sys.stderr.flush();"
        "time.sleep(60)",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_minimal_test_env(),
    )

    with pytest.raises(QuickCommandOutputError, match="65536"):
        await communicate_bounded_async(proc, timeout=5)

    assert proc.returncode is not None


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_async_streaming_timeout_terminates_and_reaps():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_minimal_test_env(),
    )

    with pytest.raises(asyncio.TimeoutError):
        await communicate_bounded_async(proc, timeout=0.1)

    assert proc.returncode is not None
