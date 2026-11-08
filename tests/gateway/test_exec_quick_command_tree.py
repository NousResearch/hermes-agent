"""A ``type: exec`` quick command that times out or is cancelled leaves nothing running.

Killing only the shell left ``sleep 120 & wait``'s sleep alive, and a cancelled await (gateway
shutdown on the WebSocket path) skipped cleanup entirely.
"""
import asyncio
from pathlib import Path

import psutil
import pytest

from gateway import run_inbound_hooks
from gateway.run_inbound_hooks import GatewayInboundHooksMixin
from tests.waiting import async_wait_until

_TREE = 'echo $$ > shell.pid; sleep 120 & echo $! > child.pid; wait'


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _pid(path: Path):
    text = path.read_text().strip() if path.exists() else ''
    return int(text) if text else None


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_timed_out_exec_command_kills_its_background_child(tmp_path, monkeypatch):
    monkeypatch.setattr(run_inbound_hooks, '_EXEC_QUICK_COMMAND_TIMEOUT', 2)
    reply = await GatewayInboundHooksMixin()._hm_run_exec_quick_command('probe', _TREE, cwd=str(tmp_path))
    assert reply == run_inbound_hooks.t('gateway.quick_command.timed_out')
    child = _pid(tmp_path / 'child.pid')
    assert child is not None
    await async_wait_until(lambda: not _alive(child), timeout=10, what='the backgrounded sleep to stop')


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_cancelled_exec_command_kills_shell_and_child(tmp_path):
    task = asyncio.create_task(
        GatewayInboundHooksMixin()._hm_run_exec_quick_command('probe', _TREE, cwd=str(tmp_path)))
    child = await async_wait_until(lambda: _pid(tmp_path / 'child.pid'), timeout=10, what='the child pid')
    shell = _pid(tmp_path / 'shell.pid')
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await async_wait_until(lambda: not _alive(shell) and not _alive(child), timeout=10,
                           what='the shell and its child to stop')
