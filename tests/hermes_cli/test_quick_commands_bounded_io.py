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


def _minimal_test_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "")}


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
