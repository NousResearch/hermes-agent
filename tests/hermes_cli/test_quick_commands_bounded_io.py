"""Streaming output-limit tests for deterministic argv quick commands."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.quick_commands import (
    QUICK_COMMAND_OUTPUT_MAX_BYTES,
    QuickCommandOutputError,
    _posix_group_exists,
    _verified_process,
    communicate_bounded_async,
    run_bounded_argv,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _minimal_test_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "")}


def test_posix_group_probe_is_safe_when_killpg_is_unavailable(monkeypatch):
    monkeypatch.delattr(os, "killpg", raising=False)
    assert _posix_group_exists(12345) is False


def _successful_forking_command(
    tmp_path, *, escape_session: bool = False
) -> tuple[list[str], Path, Path]:
    suffix = "escaped" if escape_session else "grouped"
    pid_path = tmp_path / f"successful-{suffix}-descendant.pid"
    heartbeat_path = tmp_path / f"successful-{suffix}-descendant.heartbeat"
    descendant = (
        "import pathlib,sys,time\n"
        "path=pathlib.Path(sys.argv[1])\n"
        "while True:\n"
        " path.write_text(str(time.time_ns()))\n"
        " time.sleep(0.01)\n"
    )
    leader = (
        "import pathlib,subprocess,sys,time\n"
        "heartbeat=pathlib.Path(sys.argv[2])\n"
        "child=subprocess.Popen(\n"
        " [sys.executable,'-c',sys.argv[3],str(heartbeat)],\n"
        " stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,\n"
        " stderr=subprocess.DEVNULL,close_fds=True,\n"
        " start_new_session=(sys.argv[4]=='escaped'))\n"
        "deadline=time.monotonic()+2\n"
        "while not heartbeat.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "time.sleep(0.3)\n"
        "print('leader complete')\n"
    )
    return (
        [
            sys.executable,
            "-c",
            leader,
            str(pid_path),
            str(heartbeat_path),
            descendant,
            "escaped" if escape_session else "grouped",
        ],
        pid_path,
        heartbeat_path,
    )


def _timing_out_escaped_command(tmp_path) -> tuple[list[str], Path, Path]:
    pid_path = tmp_path / "timeout-escaped-descendant.pid"
    heartbeat_path = tmp_path / "timeout-escaped-descendant.heartbeat"
    descendant = (
        "import pathlib,sys,time\n"
        "path=pathlib.Path(sys.argv[1])\n"
        "while True:\n"
        " path.write_text(str(time.time_ns()))\n"
        " time.sleep(0.01)\n"
    )
    leader = (
        "import pathlib,subprocess,sys,time\n"
        "heartbeat=pathlib.Path(sys.argv[2])\n"
        "child=subprocess.Popen(\n"
        " [sys.executable,'-c',sys.argv[3],str(heartbeat)],\n"
        " stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,\n"
        " stderr=subprocess.DEVNULL,close_fds=True,start_new_session=True)\n"
        "deadline=time.monotonic()+2\n"
        "while not heartbeat.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    return (
        [
            sys.executable,
            "-c",
            leader,
            str(pid_path),
            str(heartbeat_path),
            descendant,
        ],
        pid_path,
        heartbeat_path,
    )


def _kill_test_descendant(pid_path) -> None:
    if not pid_path.exists():
        return
    try:
        os.kill(int(pid_path.read_text()), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, ValueError):
        pass


def test_minimal_environment_preserves_context_resolved_hermes_home(tmp_path):
    from hermes_cli.quick_commands import build_argv_environment

    named_home = tmp_path / "profiles" / "coder"
    token = set_hermes_home_override(named_home)
    try:
        child_env = build_argv_environment()
    finally:
        reset_hermes_home_override(token)

    assert child_env["HERMES_HOME"] == str(named_home)


def test_descendant_identity_guard_rejects_reused_pid(monkeypatch):
    terminated = False

    class ReusedProcess:
        pid = 4242

        @staticmethod
        def create_time():
            return 200.0

        @staticmethod
        def is_running():
            return True

        @staticmethod
        def terminate():
            nonlocal terminated
            terminated = True

    monkeypatch.setattr(
        "hermes_cli.quick_commands.psutil.Process", lambda pid: ReusedProcess()
    )

    process = _verified_process(4242, create_time=100.0)

    assert process is None
    assert terminated is False


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


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real fork regression is POSIX-only")
def test_sync_successful_leader_reaps_closed_stdio_descendant(tmp_path):
    command, pid_path, heartbeat_path = _successful_forking_command(tmp_path)
    try:
        result = run_bounded_argv(command, env=_minimal_test_env(), timeout=5)

        assert result.returncode == 0
        assert result.stdout.strip() == b"leader complete"
        before = heartbeat_path.read_text()
        time.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        _kill_test_descendant(pid_path)


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real setsid regression is POSIX-only")
def test_sync_successful_leader_reaps_escaped_session_descendant(tmp_path):
    command, pid_path, heartbeat_path = _successful_forking_command(
        tmp_path, escape_session=True
    )
    try:
        result = run_bounded_argv(command, env=_minimal_test_env(), timeout=5)

        assert result.returncode == 0
        assert result.stdout.strip() == b"leader complete"
        before = heartbeat_path.read_text()
        time.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        _kill_test_descendant(pid_path)


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real setsid regression is POSIX-only")
def test_sync_timeout_reaps_escaped_session_descendant(tmp_path):
    command, pid_path, heartbeat_path = _timing_out_escaped_command(tmp_path)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_bounded_argv(command, env=_minimal_test_env(), timeout=0.3)

        before = heartbeat_path.read_text()
        time.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        _kill_test_descendant(pid_path)


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real process-group regression is POSIX-only")
def test_sync_unexpected_exception_terminates_leader_group(monkeypatch, tmp_path):
    heartbeat_path = tmp_path / "interrupted-leader.heartbeat"
    script = (
        "import pathlib,sys,time\n"
        "path=pathlib.Path(sys.argv[1])\n"
        "while True:\n"
        " path.write_text(str(time.time_ns()))\n"
        " time.sleep(0.01)\n"
    )
    original_popen = subprocess.Popen
    spawned = []

    def interrupt_after_start(*args, **kwargs):
        proc = original_popen(*args, **kwargs)
        spawned.append(proc)
        original_poll = proc.poll
        interrupted = False

        def poll():
            nonlocal interrupted
            if heartbeat_path.exists() and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("synthetic caller interruption")
            return original_poll()

        proc.poll = poll
        return proc

    monkeypatch.setattr(subprocess, "Popen", interrupt_after_start)
    try:
        with pytest.raises(KeyboardInterrupt, match="synthetic caller interruption"):
            run_bounded_argv(
                [sys.executable, "-c", script, str(heartbeat_path)],
                env=_minimal_test_env(),
                timeout=5,
            )

        assert spawned[0].poll() is not None
        before = heartbeat_path.read_text()
        time.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        if spawned and spawned[0].poll() is None:
            os.killpg(spawned[0].pid, signal.SIGKILL)
            spawned[0].wait()


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


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real process-group regression is POSIX-only")
async def test_async_tracker_start_failure_terminates_running_group(
    monkeypatch, tmp_path
):
    heartbeat_path = tmp_path / "tracker-start-failure.heartbeat"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import pathlib,sys,time\n"
            "path=pathlib.Path(sys.argv[1])\n"
            "while True:\n"
            " path.write_text(str(time.time_ns()))\n"
            " time.sleep(0.01)\n"
        ),
        str(heartbeat_path),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_minimal_test_env(),
        start_new_session=True,
    )

    for _ in range(100):
        if heartbeat_path.exists():
            break
        await asyncio.sleep(0.01)
    assert heartbeat_path.exists()

    def fail_tracker_start(_tracker):
        raise RuntimeError("synthetic tracker start failure")

    monkeypatch.setattr(
        "hermes_cli.quick_commands._DescendantTracker.start",
        fail_tracker_start,
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic tracker start failure"):
            await communicate_bounded_async(proc, timeout=5)

        assert proc.returncode is not None
        before = heartbeat_path.read_text()
        await asyncio.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()


@pytest.mark.asyncio
async def test_async_success_without_descendants_is_unchanged():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print('ordinary success')",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_minimal_test_env(),
        start_new_session=True,
    )

    stdout, stderr = await communicate_bounded_async(proc, timeout=5)

    assert proc.returncode == 0
    assert stdout.strip() == b"ordinary success"
    assert stderr == b""


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real fork regression is POSIX-only")
async def test_async_successful_leader_reaps_closed_stdio_descendant(tmp_path):
    command, pid_path, heartbeat_path = _successful_forking_command(tmp_path)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_minimal_test_env(),
        start_new_session=True,
    )
    try:
        stdout, stderr = await communicate_bounded_async(proc, timeout=5)

        assert proc.returncode == 0
        assert stdout.strip() == b"leader complete"
        assert stderr == b""
        before = heartbeat_path.read_text()
        await asyncio.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        _kill_test_descendant(pid_path)


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real setsid regression is POSIX-only")
async def test_async_successful_leader_reaps_escaped_session_descendant(tmp_path):
    command, pid_path, heartbeat_path = _successful_forking_command(
        tmp_path, escape_session=True
    )
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_minimal_test_env(),
        start_new_session=True,
    )
    try:
        stdout, stderr = await communicate_bounded_async(proc, timeout=5)

        assert proc.returncode == 0
        assert stdout.strip() == b"leader complete"
        assert stderr == b""
        before = heartbeat_path.read_text()
        await asyncio.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        _kill_test_descendant(pid_path)


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="real setsid regression is POSIX-only")
async def test_async_timeout_reaps_escaped_session_descendant(tmp_path):
    command, pid_path, heartbeat_path = _timing_out_escaped_command(tmp_path)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_minimal_test_env(),
        start_new_session=True,
    )
    try:
        with pytest.raises(asyncio.TimeoutError):
            await communicate_bounded_async(proc, timeout=0.3)

        before = heartbeat_path.read_text()
        await asyncio.sleep(0.1)
        assert heartbeat_path.read_text() == before
    finally:
        _kill_test_descendant(pid_path)
