"""Front-owned per-profile worker lifecycle (Tier-2).

The front owns the one shared bot token; each routed profile runs in an isolated
tokenless ``hermes gateway run`` worker exposing only api_server. This module
spawns, readiness-probes, idle-evicts, and crash-respawns those workers, and
enforces the load-bearing **one-agent-loop-per-profile** invariant: it refuses
to spawn a worker for a profile a standalone gateway already owns (design §7).

Process/transport details (Popen, health HTTP, interlock PID check) are injected
so the state machine is exercised without real subprocesses.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    SPAWNING = "spawning"
    PROBING = "probing"
    SERVING = "serving"
    DRAINING = "draining"
    REAPED = "reaped"
    UNHEALTHY = "unhealthy"


class ProfileBusyError(RuntimeError):
    """A profile cannot be served: a standalone owns it, or it circuit-broke."""


@dataclass
class Worker:
    profile: str
    proc: object
    port: int
    key: str
    state: WorkerState = WorkerState.SPAWNING
    last_used: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc.poll() is None


def _free_loopback_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _default_spawn(profile: str) -> tuple[object, int, str]:
    import os
    import subprocess

    from hermes_cli.gateway import _worker_run_args_for_profile

    port = _free_loopback_port()
    key = secrets.token_hex(32)
    args, env = _worker_run_args_for_profile(profile, port, key)
    # Tracked (non-detached) child: the front reaps it via the pool, not s6.
    proc = subprocess.Popen(args, env={**os.environ, **env})
    return proc, port, key


def _default_interlock(profile: str) -> Optional[int]:
    from hermes_cli.profiles import get_profile_dir
    from gateway.status import get_running_pid

    return get_running_pid(get_profile_dir(profile) / "gateway.pid", cleanup_stale=False)


class WorkerPool:
    def __init__(
        self,
        *,
        spawn: Callable[[str], tuple[object, int, str]] = _default_spawn,
        probe: Callable[[Worker], Awaitable[bool]] | None = None,
        interlock: Callable[[str], Optional[int]] = _default_interlock,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_workers: int = 8,
        idle_ttl: float = 300.0,
        probe_interval: float = 0.25,
        probe_timeout: float = 30.0,
        crash_limit: int = 5,
        crash_window: float = 60.0,
        kill_grace: float = 10.0,
        broken_cooldown: float = 60.0,
    ):
        self._spawn = spawn
        self._probe = probe or self._default_probe
        self._interlock = interlock
        self._clock = clock
        self._sleep = sleep
        self.max_workers = max_workers
        self.idle_ttl = idle_ttl
        self.probe_interval = probe_interval
        self.probe_timeout = probe_timeout
        self.crash_limit = crash_limit
        self.crash_window = crash_window
        self.kill_grace = kill_grace
        self.broken_cooldown = broken_cooldown
        self.workers: dict[str, Worker] = {}
        self._crashes: dict[str, deque[float]] = {}
        # profile -> monotonic time the circuit-break cooldown expires.  After
        # it elapses, the next acquire() clears the breaker and retries once.
        self._broken: dict[str, float] = {}
        self._closed = False
        self._lock = asyncio.Lock()

    async def _default_probe(self, worker: Worker) -> bool:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{worker.base_url}/health", timeout=2) as r:
                    return r.status == 200
        except Exception:
            return False

    async def acquire(self, profile: str) -> Worker:
        """Return a SERVING worker for *profile*, spawning + probing if needed."""
        async with self._lock:
            if self._closed:
                raise ProfileBusyError("worker pool is shutting down")
            existing = self.workers.get(profile)
            if existing and existing.state is WorkerState.SERVING and existing.alive():
                existing.last_used = self._clock()
                return existing
            # A previously serving/probing worker that died unexpectedly is a
            # crash.  Record it here, on the lazy-respawn path, so the circuit
            # breaker still trips even when no periodic reap observed the death
            # first (the live gateway respawns on-demand, not via reap_exited).
            if existing and existing.state in (WorkerState.SERVING, WorkerState.PROBING) and not existing.alive():
                self._record_crash(profile)
            self.workers.pop(profile, None)
            self._raise_if_broken(profile)
            if (pid := self._interlock(profile)) is not None:
                raise ProfileBusyError(
                    f"profile {profile!r} is already served by a standalone gateway (pid {pid}); "
                    "refusing to start a second loop on its state.db"
                )
            if len(self.workers) >= self.max_workers:
                raise ProfileBusyError(f"worker pool is full ({self.max_workers})")
            return await self._spawn_and_probe(profile)

    async def _spawn_and_probe(self, profile: str) -> Worker:
        proc, port, key = self._spawn(profile)
        worker = Worker(profile=profile, proc=proc, port=port, key=key, state=WorkerState.PROBING)
        self.workers[profile] = worker
        deadline = self._clock() + self.probe_timeout
        while self._clock() < deadline:
            if not worker.alive():
                break
            if await self._probe(worker):
                worker.state = WorkerState.SERVING
                worker.last_used = self._clock()
                return worker
            await self._sleep(self.probe_interval)
        await self._reap(worker)
        raise ProfileBusyError(f"worker for {profile!r} failed readiness probe")

    async def sweep_idle(self) -> None:
        now = self._clock()
        for worker in list(self.workers.values()):
            if worker.state is WorkerState.SERVING and now - worker.last_used > self.idle_ttl:
                await self.evict(worker.profile)

    async def evict(self, profile: str) -> None:
        worker = self.workers.get(profile)
        if not worker:
            return
        worker.state = WorkerState.DRAINING
        await self._reap(worker)

    async def reap_exited(self) -> None:
        """Record crashes for workers whose process exited unexpectedly."""
        for worker in list(self.workers.values()):
            if worker.state in (WorkerState.SERVING, WorkerState.PROBING) and not worker.alive():
                self._record_crash(worker.profile)
                worker.state = WorkerState.REAPED
                self.workers.pop(worker.profile, None)

    def _record_crash(self, profile: str) -> None:
        now = self._clock()
        hist = self._crashes.setdefault(profile, deque())
        hist.append(now)
        while hist and now - hist[0] > self.crash_window:
            hist.popleft()
        if len(hist) >= self.crash_limit:
            self._broken[profile] = now + self.broken_cooldown
            logger.error(
                "worker for %r circuit-broke after %d crashes in %ss; cooling down %ss",
                profile, len(hist), self.crash_window, self.broken_cooldown,
            )

    def _raise_if_broken(self, profile: str) -> None:
        """Raise while the profile's circuit breaker is open; self-heal after cooldown."""
        until = self._broken.get(profile)
        if until is None:
            return
        if self._clock() < until:
            raise ProfileBusyError(
                f"profile {profile!r} is unhealthy (too many recent crashes); cooling down"
            )
        # Cooldown elapsed — clear the breaker and the stale crash history so
        # the profile gets a fresh chance instead of staying broken forever.
        self._broken.pop(profile, None)
        self._crashes.pop(profile, None)

    async def _reap(self, worker: Worker) -> None:
        if worker.alive():
            worker.proc.terminate()
            await self._sleep(self.kill_grace)
            if worker.alive():
                worker.proc.kill()
        worker.state = WorkerState.REAPED
        self.workers.pop(worker.profile, None)

    async def shutdown(self) -> None:
        # Latch closed under the lock so an acquire() racing shutdown can't spawn
        # a worker after we snapshot the set — otherwise that child would orphan.
        async with self._lock:
            self._closed = True
            profiles = list(self.workers)
        for profile in profiles:
            await self.evict(profile)
