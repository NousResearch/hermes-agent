"""Process-tree containment tests for :mod:`agent.lsp.client`."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.lsp import client as client_module
from agent.lsp.client import LSPClient


requires_linux_groups = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="owned process-group integration tests require Linux /proc",
)

_CHILD_CODE = "import time; time.sleep(60)"
_TERM_RESISTANT_CHILD_CODE = (
    "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "open(os.environ['LSP_TEST_CHILD_READY_FILE'], 'w').close(); time.sleep(60)"
)
_SANITIZED_TERM_RESISTANT_CHILD_CODE = """
import os
import sys

ready_file = os.environ["LSP_TEST_CHILD_READY_FILE"]
code = (
    "import signal,time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    f"open({ready_file!r}, 'w').close(); "
    "time.sleep(60)"
)
os.execve(sys.executable, [sys.executable, "-c", code], {})
"""
_LEADER_CODE = """
import os
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", os.environ["LSP_TEST_CHILD_CODE"]],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
with open(os.environ["LSP_TEST_CHILD_PID_FILE"], "w", encoding="ascii") as fh:
    fh.write(str(child.pid))
if os.environ.get("LSP_TEST_LEADER_EXIT") == "1":
    raise SystemExit(0)
time.sleep(60)
"""


def _client(tmp_path: Path, *, child_code: str = _CHILD_CODE, leader_exit: bool = False) -> LSPClient:
    env = {
        "LSP_TEST_CHILD_CODE": child_code,
        "LSP_TEST_CHILD_PID_FILE": str(tmp_path / "child.pid"),
        "LSP_TEST_CHILD_READY_FILE": str(tmp_path / "child.ready"),
        "LSP_TEST_LEADER_EXIT": "1" if leader_exit else "0",
    }
    return LSPClient(
        server_id="process-tree-test",
        workspace_root=str(tmp_path),
        command=[sys.executable, "-c", _LEADER_CODE, "sensitive-test-token"],
        env=env,
        cwd=str(tmp_path),
    )


async def _wait_for_child_pid(tmp_path: Path) -> int:
    path = tmp_path / "child.pid"
    for _ in range(200):
        try:
            return int(path.read_text(encoding="ascii"))
        except (FileNotFoundError, ValueError):
            await asyncio.sleep(0.01)
    raise AssertionError("test child did not start")


async def _wait_for_path(path: Path) -> None:
    for _ in range(200):
        if path.exists():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{path.name} was not created")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text(encoding="ascii").split()[2] != "Z"
        except (OSError, IndexError):
            pass
    return True


async def _assert_pid_exits(pid: int) -> None:
    for _ in range(200):
        if not _pid_alive(pid):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"process {pid} survived LSP cleanup")


async def _cleanup_test_group(client: LSPClient) -> None:
    """Keep a red regression test from leaking its test-owned process tree."""
    proc = client._proc
    pgid = getattr(client, "_process_group_id", None)
    if pgid is None and proc is not None:
        pgid = proc.pid
    if pgid is not None and pgid > 1 and pgid != os.getpgrp():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc is not None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


def _write_proc_member(
    proc_root: Path,
    *,
    pid: int,
    pgid: int,
    session_id: int,
    start_time: int = 100,
) -> None:
    member = proc_root / str(pid)
    member.mkdir()
    member.joinpath("stat").write_text(
        f"{pid} (lsp child) S 1 {pgid} {session_id} "
        + "0 " * 15
        + f"{start_time} 0",
        encoding="ascii",
    )


@pytest.mark.asyncio
@requires_linux_groups
async def test_normal_shutdown_terminates_descendants(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.2)
    client = _client(tmp_path)
    try:
        await client._spawn()
        child_pid = await _wait_for_child_pid(tmp_path)
        client._state = "running"
        client._send_request = AsyncMock(return_value=None)
        client._send_notification = AsyncMock(return_value=None)

        await client.shutdown()

        await _assert_pid_exits(child_pid)
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_shutdown_timeout_still_terminates_descendants(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.2)
    client = _client(tmp_path)
    try:
        await client._spawn()
        child_pid = await _wait_for_child_pid(tmp_path)
        client._state = "running"
        client._send_request = AsyncMock(side_effect=asyncio.TimeoutError)

        await client.shutdown()

        await _assert_pid_exits(child_pid)
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_cleanup_kills_descendant_after_leader_exits(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.2)
    client = _client(tmp_path, leader_exit=True)
    try:
        await client._spawn()
        child_pid = await _wait_for_child_pid(tmp_path)
        assert client._proc is not None
        await asyncio.wait_for(client._proc.wait(), timeout=2.0)

        await client._cleanup_process(reason="leader_exited")

        await _assert_pid_exits(child_pid)
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_spawn_owns_group_when_leader_exits_before_subprocess_returns(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.2)
    client = _client(tmp_path, leader_exit=True)
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def return_after_leader_exit(*args, **kwargs):
        proc = await create_subprocess_exec(*args, **kwargs)
        await asyncio.wait_for(proc.wait(), timeout=2.0)
        with pytest.raises(ProcessLookupError):
            os.getpgid(proc.pid)
        with pytest.raises(ProcessLookupError):
            os.getsid(proc.pid)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", return_after_leader_exit)
    try:
        await client._spawn()
        child_pid = await _wait_for_child_pid(tmp_path)

        await client._cleanup_process(reason="post_spawn_capture_unavailable")

        await _assert_pid_exits(child_pid)
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_cleanup_escalates_term_resistant_descendant(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.1)
    client = _client(tmp_path, child_code=_TERM_RESISTANT_CHILD_CODE)
    try:
        await client._spawn()
        child_pid = await _wait_for_child_pid(tmp_path)
        await _wait_for_path(tmp_path / "child.ready")

        with caplog.at_level(logging.INFO, logger="agent.lsp.client"):
            await client._cleanup_process(reason="test_timeout")

        await _assert_pid_exits(child_pid)
        assert "SIGKILL" in caplog.text
        assert "sensitive-test-token" not in caplog.text
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_cleanup_escalates_sanitized_descendant_after_leader_exit(
    tmp_path: Path, monkeypatch, caplog
):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.1)
    client = _client(
        tmp_path,
        child_code=_SANITIZED_TERM_RESISTANT_CHILD_CODE,
        leader_exit=True,
    )
    try:
        await client._spawn()
        child_pid = await _wait_for_child_pid(tmp_path)
        await _wait_for_path(tmp_path / "child.ready")
        assert client._proc is not None
        await asyncio.wait_for(client._proc.wait(), timeout=2.0)

        with caplog.at_level(logging.INFO, logger="agent.lsp.client"):
            await client._cleanup_process(reason="sanitized_descendant")

        await _assert_pid_exits(child_pid)
        assert "SIGKILL" in caplog.text
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_start_failure_cleans_process_group(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.2)
    client = _client(tmp_path)

    async def fail_initialize() -> None:
        await _wait_for_child_pid(tmp_path)
        raise RuntimeError("initialize failed")

    client._initialize = fail_initialize
    try:
        with pytest.raises(RuntimeError, match="initialize failed"):
            await client.start()
        await _assert_pid_exits(await _wait_for_child_pid(tmp_path))
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_start_cancellation_cleans_process_group(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.2)
    client = _client(tmp_path)

    async def blocked_initialize() -> None:
        await _wait_for_child_pid(tmp_path)
        await asyncio.Event().wait()

    client._initialize = blocked_initialize
    task = asyncio.create_task(client.start())
    try:
        child_pid = await _wait_for_child_pid(tmp_path)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _assert_pid_exits(child_pid)
    finally:
        await _cleanup_test_group(client)


@pytest.mark.asyncio
@requires_linux_groups
async def test_cleanup_never_signals_current_process_group(monkeypatch, tmp_path: Path):
    client = _client(tmp_path)
    proc = MagicMock()
    proc.pid = os.getpid()
    proc.returncode = None
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    client._proc = proc
    client._process_group_id = os.getpgrp()
    killpg = MagicMock()
    monkeypatch.setattr(os, "killpg", killpg)

    await client._cleanup_process(reason="safety_test")

    killpg.assert_not_called()
    proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_does_not_signal_reused_numeric_process_group(
    monkeypatch, tmp_path: Path
):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    pgid = 41000
    _write_proc_member(
        proc_root,
        pid=pgid,
        pgid=pgid,
        session_id=pgid,
        start_time=200,
    )
    _write_proc_member(
        proc_root,
        pid=41001,
        pgid=pgid,
        session_id=pgid,
        start_time=201,
    )
    client = _client(tmp_path)
    proc = MagicMock(pid=pgid, returncode=0)
    client._proc = proc
    client._process_group_id = pgid
    client._session_id = pgid
    client._leader_start_time = 100
    monkeypatch.setattr(client_module, "_LINUX_PROCESS_GROUPS", True)
    monkeypatch.setattr(client_module, "_PROC_ROOT", proc_root)
    monkeypatch.setattr(client_module, "_PIDFD_OPEN", MagicMock(return_value=71))
    pidfd_send_signal = MagicMock()
    monkeypatch.setattr(client_module, "_PIDFD_SEND_SIGNAL", pidfd_send_signal)
    close = MagicMock()
    monkeypatch.setattr(os, "close", close)

    await client._cleanup_process(reason="reused_pgid")

    pidfd_send_signal.assert_not_called()
    close.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_pins_identity_before_numeric_pid_is_replaced(
    monkeypatch, tmp_path: Path
):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    pgid = 42000
    _write_proc_member(
        proc_root,
        pid=42001,
        pgid=pgid,
        session_id=pgid,
    )
    client = _client(tmp_path)
    proc = MagicMock(pid=pgid, returncode=0)
    client._proc = proc
    client._process_group_id = pgid
    client._session_id = pgid

    def pidfd_open(pid, flags):
        assert (pid, flags) == (42001, 0)
        return 81

    def pidfd_send_signal(pidfd, sig):
        assert (pidfd, sig) == (81, signal.SIGTERM)

    open_members = client._open_owned_process_group_members

    def open_then_replace_numeric_identity(member_pgid):
        opened = open_members(member_pgid)
        member = proc_root / "42001"
        member.joinpath("stat").write_text(
            "42001 (replacement) S 1 99999 99999 " + "0 " * 16,
            encoding="ascii",
        )
        return opened

    monkeypatch.setattr(client_module, "_LINUX_PROCESS_GROUPS", True)
    monkeypatch.setattr(client_module, "_PROC_ROOT", proc_root)
    monkeypatch.setattr(client_module, "_PIDFD_OPEN", pidfd_open)
    monkeypatch.setattr(
        client, "_open_owned_process_group_members", open_then_replace_numeric_identity
    )
    pidfd_send_signal_mock = MagicMock(side_effect=pidfd_send_signal)
    monkeypatch.setattr(
        client_module, "_PIDFD_SEND_SIGNAL", pidfd_send_signal_mock
    )
    close = MagicMock()
    monkeypatch.setattr(os, "close", close)
    numeric_signal = MagicMock()
    monkeypatch.setattr(os, "killpg", numeric_signal)

    await client._cleanup_process(reason="leader_exited")

    pidfd_send_signal_mock.assert_called_once_with(81, signal.SIGTERM)
    close.assert_called_once_with(81)
    numeric_signal.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_closes_every_pidfd_when_signaling_fails(
    monkeypatch, tmp_path: Path
):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    pgid = 43000
    for pid in (43001, 43002):
        _write_proc_member(
            proc_root,
            pid=pid,
            pgid=pgid,
            session_id=pgid,
        )
    client = _client(tmp_path)
    proc = MagicMock(pid=pgid, returncode=0)
    client._proc = proc
    client._process_group_id = pgid
    client._session_id = pgid
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.01)
    monkeypatch.setattr(client_module, "SHUTDOWN_VERIFY_TIMEOUT", 0.01)
    monkeypatch.setattr(client_module, "_LINUX_PROCESS_GROUPS", True)
    monkeypatch.setattr(client_module, "_PROC_ROOT", proc_root)
    opened_fds = []

    def pidfd_open(pid, _flags):
        fd = pid + 100 + len(opened_fds) * 1000
        opened_fds.append(fd)
        return fd

    monkeypatch.setattr(client_module, "_PIDFD_OPEN", pidfd_open)
    monkeypatch.setattr(
        client_module,
        "_PIDFD_SEND_SIGNAL",
        MagicMock(side_effect=PermissionError),
    )
    close = MagicMock()
    monkeypatch.setattr(os, "close", close)

    await client._cleanup_process(reason="signal_failure")

    assert [call.args[0] for call in close.call_args_list] == opened_fds


@pytest.mark.asyncio
async def test_unavailable_pidfd_support_never_invokes_killpg(
    monkeypatch, tmp_path: Path
):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    pgid = 44000
    _write_proc_member(
        proc_root,
        pid=44001,
        pgid=pgid,
        session_id=pgid,
    )
    client = _client(tmp_path)
    proc = MagicMock(pid=pgid, returncode=0)
    client._proc = proc
    client._process_group_id = pgid
    client._session_id = pgid
    killpg = MagicMock()
    monkeypatch.setattr(client_module, "_LINUX_PROCESS_GROUPS", True)
    monkeypatch.setattr(client_module, "_PROC_ROOT", proc_root)
    monkeypatch.setattr(client_module, "_PIDFD_OPEN", None)
    monkeypatch.setattr(client_module, "_PIDFD_SEND_SIGNAL", None)
    monkeypatch.setattr(os, "killpg", killpg)

    await client._cleanup_process(reason="pidfd_unavailable")

    killpg.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_falls_back_to_direct_process_without_groups(monkeypatch, tmp_path: Path):
    client = _client(tmp_path)
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = None
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    client._proc = proc
    client._process_group_id = 12345
    monkeypatch.setattr(client_module, "_LINUX_PROCESS_GROUPS", False)

    await client._cleanup_process(reason="fallback_test")

    proc.terminate.assert_called_once()
    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_direct_process_fallback_escalates_only_direct_process(
    monkeypatch, tmp_path: Path
):
    client = _client(tmp_path)
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = None
    proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError, 0])
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    client._proc = proc
    monkeypatch.setattr(client_module, "_LINUX_PROCESS_GROUPS", False)
    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.01)

    await client._cleanup_process(reason="fallback_escalation_test")

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_cancels_and_awaits_server_request_tasks(tmp_path: Path):
    client = _client(tmp_path)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_handler(_params):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client._request_handlers["test/blocked"] = blocked_handler
    client._start_server_request_task(
        1, {"method": "test/blocked", "params": None}
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await client._cleanup_process(reason="request_task_test")

    assert cancelled.is_set()
    assert not client._server_request_tasks


@pytest.mark.asyncio
async def test_wait_for_diagnostics_cancellation_awaits_child_tasks(tmp_path: Path):
    client = _client(tmp_path)
    pull_started = asyncio.Event()
    push_started = asyncio.Event()
    pull_cancelled = asyncio.Event()
    push_cancelled = asyncio.Event()

    async def blocked_pull(_path):
        pull_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            pull_cancelled.set()

    async def blocked_push(_path, _version, _timeout):
        push_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            push_cancelled.set()

    client._pull_document_diagnostics = blocked_pull
    client._wait_for_fresh_push = blocked_push
    task = asyncio.create_task(client.wait_for_diagnostics("x.py", 1))
    await asyncio.wait_for(
        asyncio.gather(pull_started.wait(), push_started.wait()), timeout=1.0
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert pull_cancelled.is_set()
    assert push_cancelled.is_set()
