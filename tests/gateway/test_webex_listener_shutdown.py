"""Webex shutdown retains ownership beyond the gateway disconnect deadline."""

import asyncio
import sys
from unittest.mock import AsyncMock

import aiohttp
import psutil
import pytest

from gateway import status
from gateway.config import Platform, PlatformConfig
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from plugins.platforms.webex.adapter import WebexAdapter


def _locked_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "_get_lock_dir", lambda: tmp_path / "locks")
    adapter = WebexAdapter(PlatformConfig(token="webex-shutdown-test-token"))
    assert adapter._acquire_platform_lock(
        "webex-bot-token", adapter._token, "Webex test token"
    )
    return adapter, status._get_scope_lock_path("webex-bot-token", adapter._token)


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_gateway_disconnect_deadline_reaps_stubborn_child_and_closes_resources(
    tmp_path, monkeypatch
):
    adapter, lock_path = _locked_adapter(tmp_path, monkeypatch)
    session = aiohttp.ClientSession()
    adapter._session = session
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        "import signal,threading;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "print('ready',flush=True);threading.Event().wait()",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    adapter._listener_process = process
    finished = asyncio.Event()
    disconnect = adapter.disconnect

    async def tracked_disconnect():
        try:
            await disconnect()
        finally:
            finished.set()

    monkeypatch.setattr(adapter, "disconnect", tracked_disconnect)
    lifecycle = GatewayAdapterLifecycleMixin()
    monkeypatch.setattr(lifecycle, "_adapter_disconnect_timeout_secs", lambda: 0.05)
    try:
        assert await asyncio.wait_for(process.stdout.readline(), timeout=10) == b"ready\n"
        adapter._listener_stdout_task = asyncio.create_task(
            adapter._listener_stdout_loop(process)
        )
        adapter._listener_stderr_task = asyncio.create_task(
            adapter._listener_stderr_loop(process)
        )
        adapter._listener_exit_task = asyncio.create_task(
            adapter._listener_exit_loop(process)
        )

        await lifecycle._safe_adapter_disconnect(adapter, Platform("webex"))
        await asyncio.wait_for(finished.wait(), timeout=10)

        assert process.returncode is not None
        assert not psutil.pid_exists(process.pid)
        assert adapter._listener_process is None
        assert adapter._listener_exit_task is None
        assert adapter._listener_stdout_task is None
        assert adapter._listener_stderr_task is None
        assert session.closed
        assert adapter._session is None
        assert not await asyncio.to_thread(lock_path.exists)
    finally:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.wait(), timeout=10)
        await disconnect()
        await session.close()
        adapter._release_platform_lock()


@pytest.mark.asyncio
async def test_runner_cleanup_failure_still_closes_session_and_releases_token(
    tmp_path, monkeypatch
):
    adapter, lock_path = _locked_adapter(tmp_path, monkeypatch)
    session = aiohttp.ClientSession()
    adapter._session = session
    adapter._runner = type(
        "FailingRunner", (), {"cleanup": AsyncMock(side_effect=RuntimeError("close failed"))}
    )()
    try:
        with pytest.raises(RuntimeError, match="close failed"):
            await adapter.disconnect()

        assert session.closed
        assert adapter._session is None
        assert not await asyncio.to_thread(lock_path.exists)
    finally:
        await session.close()
        adapter._release_platform_lock()
