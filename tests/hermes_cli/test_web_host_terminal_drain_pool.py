"""Retained host shells must not hold the event loop's default executor.

Every live shell keeps one PTY read in flight. On the loop's shared default pool, the shells a
Webapp retains for ~15 minutes after disconnect would queue every ``asyncio.to_thread`` hop the
process serves behind them: keystroke fences, the reaper's ownership checks, uploads, SessionDB.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import secrets
import threading

import pytest

from hermes_cli.web_host_terminal_sessions import HostTerminalRegistry


class QuietShell:
    """A shell with nothing to print. A real idle shell's read re-enters select every read
    timeout, so it holds a thread almost continuously; this one holds it until closed."""

    def __init__(self, readers: list):
        self._readers = readers
        self.closed = threading.Event()

    def read(self, timeout):
        self._readers.append(threading.current_thread())
        self.closed.wait()
        return None

    def close(self):
        self.closed.set()


@pytest.mark.asyncio
async def test_idle_host_shells_leave_the_default_executor_free():
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    registry = HostTerminalRegistry()
    readers: list[threading.Thread] = []
    shells: list[QuietShell] = []

    def spawn():
        shells.append(QuietShell(readers))
        return shells[-1]

    try:
        async with asyncio.timeout(5):
            # A full registry: every retained shell reads at once.
            for _ in range(registry._max):
                await registry.attach_or_spawn(secrets.token_urlsafe(32), spawn=spawn)
            while len(readers) < registry._max:
                await asyncio.sleep(0.01)
            assert await asyncio.to_thread(lambda: "ran") == "ran"
    finally:
        # Release reads first so a starved default executor can still run teardown.
        for shell in shells:
            shell.close()
        await registry.close_all()
    # Closing the registry releases its reader threads; a restarted lifespan gets new ones.
    async with asyncio.timeout(5):
        while any(thread.is_alive() for thread in readers):
            await asyncio.sleep(0.01)
