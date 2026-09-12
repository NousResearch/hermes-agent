"""Profile-scoped Work Router lifetime at upstream Gateway phase boundaries.

An enabled router without a native sender is a startup error, not a dry-run
worker: processing with sender=None can strand acknowledged events as prepared.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yaml

from hermes_constants import get_hermes_home
from .config import RouterConfig
from .service import WorkRouter
from .worker import WorkRouterWorker
from .native_sender import NativeRouterSender

logger = logging.getLogger(__name__)


class RouterRuntime:
    def __init__(self, config: RouterConfig, *, sender=None):
        if not callable(sender):
            raise RuntimeError("Work Router native sender is not bound; refusing startup")
        self.router = WorkRouter(config)
        self.sender = sender
        self.router.set_control_sender(sender)
        if isinstance(sender, NativeRouterSender):
            self.router.set_preflight_executor(sender.execute)
        self.adapters = set()
        self._recovery_tasks = set()
        self.stopping = False
        self.closed = False
        self._stop_lock = asyncio.Lock()
        self.worker = WorkRouterWorker(
            self.router, sender=sender,
            ready=sender.ready if isinstance(sender, NativeRouterSender) else None,
        )
        self.task = asyncio.create_task(self.worker.run(), name="work-router")

    def attach(self, adapter):
        if self.stopping:
            raise RuntimeError("Work Router is stopping")
        if adapter in self.adapters:
            return
        adapter.attach_work_router(self.router)
        if isinstance(self.sender, NativeRouterSender):
            self.sender.bind(adapter)
        self.adapters.add(adapter)
        task = asyncio.create_task(self._recover_when_ready(adapter), name="work-router-ingress-recovery")
        self._recovery_tasks.add(task)
        task.add_done_callback(self._recovery_done)

    def _recovery_done(self, task):
        self._recovery_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Work Router ingress recovery failed: %s", type(task.exception()).__name__)

    async def _recover_when_ready(self, adapter):
        # Attach precedes connect/registry publication in native startup and
        # reconnect. Never promote control events until transport ownership is ready.
        while not self.stopping:
            if isinstance(self.sender, NativeRouterSender):
                if self.sender.adapter is not adapter:
                    return  # A replacement receiver owns recovery now.
                ready = self.sender.ready()
            else:
                ready = adapter.is_connected is True
            if ready:
                await adapter.recover_work_router_ingress()
                return
            await asyncio.sleep(0.05)

    async def stop(self):
        # Concurrent stop callers must observe completion, not merely a fence.
        # A timed-out stop may be retried once cancellation-resistant work exits.
        async with self._stop_lock:
            if self.closed:
                return
            if not self.stopping:
                self.stopping = True
                for adapter in self.adapters:
                    adapter.detach_work_router(self.router)
            # Recovery/admission may create meeting/control children while
            # draining. Fence ALL receivers before awaiting any of them, and
            # do not snapshot child tasks or close SQLite until ingress is quiet.
            for adapter in self.adapters:
                if not await adapter.drain_work_router_ingress():
                    logger.error("Work Router ingress drain timed out; retaining fenced store")
                    return
            recovery = set(self._recovery_tasks)
            if recovery:
                _, pending = await asyncio.wait(recovery, timeout=5)
                if pending:
                    logger.error("Work Router recovery drain timed out; retaining fenced store")
                    return
            tasks = {self.task, *self.router._lounge_candidate_tasks}
            for pending in self.router._meeting_send_tasks.values():
                tasks.update(pending)
            for task in tasks:
                task.cancel()
            # Never close SQLite while a processor may still be using it.
            done, pending = await asyncio.wait(tasks, timeout=5)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    logger.error("Work Router task stopped with error: %s", type(task.exception()).__name__)
            if pending:
                logger.error("Work Router stop timed out; retaining fenced store for active tasks")
                return
            self.router.store.close()
            self.closed = True
            self.adapters.clear()


def ensure_runtime(runner, *, profile_home=None):
    """Called before receiver creation, and in the secondary profile scope.

    Each profile reads ONLY its own raw YAML; no default-profile inheritance.
    Reconnect reuses the same owner, store and worker rather than reconstructing.
    """
    home = Path(profile_home or get_hermes_home()).resolve()
    runtimes = getattr(runner, "_work_router_runtimes", None)
    if runtimes is None:
        runtimes = runner._work_router_runtimes = {}
    if home in runtimes:
        return runtimes[home]
    path = home / "config.yaml"
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = RouterConfig.from_mapping(raw.get("work_router"))
    if not config.enabled:
        return None
    config.require_ready()  # Misconfigured opt-in must never silently bypass admission.
    runtime = RouterRuntime(config, sender=NativeRouterSender(runner))
    runtimes[home] = runtime
    return runtime


def attach_adapter(runner, adapter, *, profile_home=None):
    from gateway.config import Platform
    if adapter.platform != Platform.SLACK:
        return
    runtime = ensure_runtime(runner, profile_home=profile_home)
    if runtime is not None:
        runtime.attach(adapter)


async def stop_runtimes(runner):
    for runtime in getattr(runner, "_work_router_runtimes", {}).values():
        try:
            await runtime.stop()
        except Exception:
            # Adapter teardown must still run; admission was fenced first.
            logger.exception("Work Router shutdown failed")
