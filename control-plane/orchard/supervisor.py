"""Supervisor — owns worker lifecycle: wake on demand, keep-alive, idle sleep,
and a hard cap on simultaneously-awake workers (the RAM guard).

Concurrency model:
  * One asyncio.Lock per employee serializes that employee's turns and prevents
    a double-wake.
  * A global state lock guards the active-worker set + capacity/eviction.
  * A background reaper sleeps workers idle longer than idle_ttl_seconds.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .backends.base import WorkerBackend
from .config import Settings
from .models import Employee, WorkerInfo, WorkerStatus

log = logging.getLogger("orchard.supervisor")


class CapacityFull(RuntimeError):
    pass


class Supervisor:
    def __init__(self, settings: Settings, backend: WorkerBackend):
        self.settings = settings
        self.backend = backend
        self._workers: dict[str, WorkerInfo] = {}
        self._emp: dict[str, Employee] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._state_lock = asyncio.Lock()
        self._reaper: asyncio.Task | None = None

    # --- public API ----------------------------------------------------------
    async def handle(self, employee: Employee, session: str, message: str) -> str:
        lock = self._locks.setdefault(employee.id, asyncio.Lock())
        async with lock:
            await self._ensure_capacity(employee.id)
            info = self._workers.get(employee.id)
            if info is None:
                info = WorkerInfo(employee.id, WorkerStatus.WARMING, time.monotonic())
                self._workers[employee.id] = info
                self._emp[employee.id] = employee
                log.info("waking worker %s", employee.id)
                await self.backend.ensure_ready(employee)
                info.status = WorkerStatus.READY
            info.status = WorkerStatus.BUSY
            info.last_used = time.monotonic()
            try:
                return await self.backend.send(employee, session, message)
            finally:
                info.status = WorkerStatus.READY
                info.last_used = time.monotonic()

    async def ensure(self, employee: Employee) -> None:
        """Explicitly wake a worker (no message). Used by the admin UI."""
        lock = self._locks.setdefault(employee.id, asyncio.Lock())
        async with lock:
            if employee.id in self._workers:
                return
            await self._ensure_capacity(employee.id)
            info = WorkerInfo(employee.id, WorkerStatus.WARMING, time.monotonic())
            self._workers[employee.id] = info
            self._emp[employee.id] = employee
            log.info("waking worker %s (explicit)", employee.id)
            await self.backend.ensure_ready(employee)
            info.status = WorkerStatus.READY
            info.last_used = time.monotonic()

    async def put_to_sleep(self, employee: Employee) -> bool:
        """Explicitly sleep a worker. Returns True if it was awake."""
        async with self._state_lock:
            was_awake = self._workers.pop(employee.id, None) is not None
            self._emp.pop(employee.id, None)
        if was_awake:
            log.info("sleeping worker %s (explicit)", employee.id)
            await self.backend.sleep(employee)
        return was_awake

    def status_of(self, employee_id: str) -> WorkerStatus:
        info = self._workers.get(employee_id)
        return info.status if info else WorkerStatus.ASLEEP

    def snapshot(self) -> list[WorkerInfo]:
        return list(self._workers.values())

    async def start(self) -> None:
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap_loop())

    async def stop(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        await self.backend.shutdown_all()

    # --- internals -----------------------------------------------------------
    async def _ensure_capacity(self, new_id: str) -> None:
        cap = self.settings.supervisor.max_active_workers
        async with self._state_lock:
            if new_id in self._workers or len(self._workers) < cap:
                return
            # Evict the least-recently-used READY worker to make room.
            idle = [w for w in self._workers.values() if w.status == WorkerStatus.READY]
            if not idle:
                raise CapacityFull(
                    f"all {cap} worker slots busy; try again shortly"
                )
            victim = min(idle, key=lambda w: w.last_used)
            self._workers.pop(victim.employee_id, None)
            emp = self._emp.pop(victim.employee_id, None)
        log.info("evicting LRU worker %s to free a slot", victim.employee_id)
        if emp:
            await self.backend.sleep(emp)

    async def _reap_loop(self) -> None:
        ttl = self.settings.supervisor.idle_ttl_seconds
        interval = max(5, min(ttl, 60))
        while True:
            await asyncio.sleep(interval)
            await self._reap_once()

    async def _reap_once(self) -> list[str]:
        """Sleep workers idle longer than idle_ttl. Returns reaped ids."""
        ttl = self.settings.supervisor.idle_ttl_seconds
        now = time.monotonic()
        async with self._state_lock:
            stale = [
                w.employee_id for w in self._workers.values()
                if w.status == WorkerStatus.READY and now - w.last_used >= ttl
            ]
            emps = [self._emp.pop(i) for i in stale if i in self._emp]
            for i in stale:
                self._workers.pop(i, None)
        for emp in emps:
            log.info("reaping idle worker %s", emp.id)
            try:
                await self.backend.sleep(emp)
            except Exception:
                log.exception("failed to sleep worker %s", emp.id)
        return stale
