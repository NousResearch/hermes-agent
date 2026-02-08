"""
Monkey patches for making hermes-agent tools work inside async frameworks (Atropos).

Problem:
    Some tools use asyncio.run() internally (e.g., mini-swe-agent's Modal backend,
    web_extract). This crashes when called from inside Atropos's event loop because
    asyncio.run() can't be nested.

Solution:
    Replace the problematic methods with versions that use a dedicated background
    thread with its own event loop. The calling code sees the same sync interface --
    call a function, get a result -- but internally the async work happens on a
    separate thread that doesn't conflict with Atropos's loop.

    These patches are safe for normal CLI use too: when there's no running event
    loop, the behavior is identical (the background thread approach works regardless).

What gets patched:
    - SwerexModalEnvironment.__init__ -- creates Modal deployment on a background thread
    - SwerexModalEnvironment.execute -- runs commands on the same background thread
    - SwerexModalEnvironment.stop -- stops deployment on the background thread

Usage:
    Call apply_patches() once at import time (done automatically by hermes_base_env.py).
    This is idempotent -- calling it multiple times is safe.
"""

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_patches_applied = False


class _AsyncWorker:
    """
    A dedicated background thread with its own event loop.

    Allows sync code to submit async coroutines and block for results,
    even when called from inside another running event loop. Used to
    bridge sync tool interfaces with async backends (Modal, SWE-ReX).
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._started = threading.Event()

    def start(self):
        """Start the background event loop thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=30)

    def _run_loop(self):
        """Background thread entry point -- runs the event loop forever."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run_coroutine(self, coro, timeout=600):
        """
        Submit a coroutine to the background loop and block until it completes.

        Safe to call from any thread, including threads that already have
        a running event loop.
        """
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("AsyncWorker loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self):
        """Stop the background event loop and join the thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)


def _patch_swerex_modal():
    """
    Monkey patch SwerexModalEnvironment to use a background thread event loop
    instead of asyncio.run(). This makes it safe to call from inside Atropos's
    async event loop.

    The patched methods have the exact same interface and behavior -- the only
    difference is HOW the async work is executed internally.
    """
    try:
        from minisweagent.environments.extra.swerex_modal import (
            SwerexModalEnvironment,
            SwerexModalEnvironmentConfig,
        )
        from swerex.deployment.modal import ModalDeployment
        from swerex.runtime.abstract import Command as RexCommand
    except ImportError:
        # mini-swe-agent or swe-rex not installed -- nothing to patch
        logger.debug("mini-swe-agent Modal backend not available, skipping patch")
        return

    # Save original methods so we can refer to config handling
    _original_init = SwerexModalEnvironment.__init__

    def _patched_init(self, **kwargs):
        """Patched __init__: creates Modal deployment on a background thread."""
        self.config = SwerexModalEnvironmentConfig(**kwargs)

        # Start a dedicated event loop thread for all Modal async operations
        self._worker = _AsyncWorker()
        self._worker.start()

        # Create AND start the deployment entirely on the worker's loop/thread
        # so all gRPC channels and async state are bound to that loop
        async def _create_and_start():
            deployment = ModalDeployment(
                image=self.config.image,
                startup_timeout=self.config.startup_timeout,
                runtime_timeout=self.config.runtime_timeout,
                deployment_timeout=self.config.deployment_timeout,
                install_pipx=self.config.install_pipx,
                modal_sandbox_kwargs=self.config.modal_sandbox_kwargs,
            )
            await deployment.start()
            return deployment

        self.deployment = self._worker.run_coroutine(_create_and_start())

    def _patched_execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Patched execute: runs commands on the background thread's loop."""
        async def _do_execute():
            return await self.deployment.runtime.execute(
                RexCommand(
                    command=command,
                    shell=True,
                    check=False,
                    cwd=cwd or self.config.cwd,
                    timeout=timeout or self.config.timeout,
                    merge_output_streams=True,
                    env=self.config.env if self.config.env else None,
                )
            )

        output = self._worker.run_coroutine(_do_execute())
        return {
            "output": output.stdout,
            "returncode": output.exit_code,
        }

    def _patched_stop(self):
        """Patched stop: stops deployment on the background thread, then stops the thread."""
        try:
            self._worker.run_coroutine(
                asyncio.wait_for(self.deployment.stop(), timeout=10),
                timeout=15,
            )
        except Exception:
            pass
        finally:
            self._worker.stop()

    # Apply the patches
    SwerexModalEnvironment.__init__ = _patched_init
    SwerexModalEnvironment.execute = _patched_execute
    SwerexModalEnvironment.stop = _patched_stop

    logger.debug("Patched SwerexModalEnvironment for async-safe operation")


def apply_patches():
    """
    Apply all monkey patches needed for Atropos compatibility.

    Safe to call multiple times -- patches are only applied once.
    Safe for normal CLI use -- patched code works identically when
    there is no running event loop.
    """
    global _patches_applied
    if _patches_applied:
        return

    _patch_swerex_modal()

    _patches_applied = True
