"""Front-owned worker pool: spawn/reuse, one-loop interlock, idle-evict, crash-respawn."""

import pytest

from gateway.worker_pool import WorkerPool, WorkerState, ProfileBusyError


class FakeProc:
    def __init__(self):
        self._rc = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True
        self._rc = 0

    def kill(self):
        self.killed = True
        self._rc = -9

    def crash(self, rc=1):
        self._rc = rc


class Harness:
    """Injectable deps with a controllable clock and spawn/probe stubs."""

    def __init__(self, standalone_pid=None):
        self.t = 1000.0
        self.spawned = []
        self.standalone_pid = standalone_pid

    def clock(self):
        return self.t

    def advance(self, dt):
        self.t += dt

    def spawn(self, profile):
        proc = FakeProc()
        self.spawned.append((profile, proc))
        return proc, 50000 + len(self.spawned), "key-" + profile

    async def probe(self, worker):
        return worker.proc.poll() is None  # healthy while running

    def interlock(self, profile):
        return self.standalone_pid

    async def sleep(self, _):
        return None

    def make_pool(self, **kw):
        return WorkerPool(
            spawn=self.spawn, probe=self.probe, interlock=self.interlock,
            clock=self.clock, sleep=self.sleep, **kw,
        )


@pytest.mark.asyncio
async def test_acquire_spawns_then_reuses():
    h = Harness()
    pool = h.make_pool()
    w1 = await pool.acquire("coder")
    assert w1.state is WorkerState.SERVING
    w2 = await pool.acquire("coder")
    assert w1 is w2
    assert len(h.spawned) == 1


@pytest.mark.asyncio
async def test_standalone_pid_refuses_spawn():
    h = Harness(standalone_pid=4242)
    pool = h.make_pool()
    with pytest.raises(ProfileBusyError):
        await pool.acquire("coder")
    assert h.spawned == []


@pytest.mark.asyncio
async def test_idle_eviction_drains_and_reaps():
    h = Harness()
    pool = h.make_pool(idle_ttl=300)
    w = await pool.acquire("coder")
    h.advance(301)
    await pool.sweep_idle()
    assert w.state is WorkerState.REAPED
    assert "coder" not in pool.workers
    assert w.proc.terminated


@pytest.mark.asyncio
async def test_active_worker_not_evicted():
    h = Harness()
    pool = h.make_pool(idle_ttl=300)
    await pool.acquire("coder")
    h.advance(100)
    await pool.sweep_idle()
    assert pool.workers["coder"].state is WorkerState.SERVING


@pytest.mark.asyncio
async def test_crash_respawns_with_backoff():
    h = Harness()
    pool = h.make_pool()
    w = await pool.acquire("coder")
    w.proc.crash()
    await pool.reap_exited()
    # acquire again gets a fresh process
    w2 = await pool.acquire("coder")
    assert w2.proc is not w.proc
    assert len(h.spawned) == 2


@pytest.mark.asyncio
async def test_circuit_break_after_five_crashes():
    h = Harness()
    pool = h.make_pool(crash_limit=5, crash_window=60)
    for _ in range(5):
        w = await pool.acquire("coder")
        w.proc.crash()
        await pool.reap_exited()
        h.advance(1)
    with pytest.raises(ProfileBusyError):
        await pool.acquire("coder")


@pytest.mark.asyncio
async def test_lazy_respawn_records_crash_without_reap():
    """A worker that dies between messages is counted as a crash on the
    on-demand respawn path — the breaker must trip without reap_exited ever
    running (the live gateway respawns lazily, not via a sweep)."""
    h = Harness()
    pool = h.make_pool(crash_limit=3, crash_window=60)
    for _ in range(3):
        w = await pool.acquire("coder")
        w.proc.crash()  # dies, but no reap_exited() call
        h.advance(1)
    with pytest.raises(ProfileBusyError):
        await pool.acquire("coder")


@pytest.mark.asyncio
async def test_circuit_breaker_recovers_after_cooldown():
    h = Harness()
    pool = h.make_pool(crash_limit=3, crash_window=60, broken_cooldown=120)
    for _ in range(3):
        w = await pool.acquire("coder")
        w.proc.crash()
        h.advance(1)
    with pytest.raises(ProfileBusyError):
        await pool.acquire("coder")
    # After the cooldown elapses the breaker self-heals and a fresh spawn works.
    h.advance(121)
    w = await pool.acquire("coder")
    assert w.state is WorkerState.SERVING


@pytest.mark.asyncio
async def test_shutdown_terminates_all_workers():
    h = Harness()
    pool = h.make_pool()
    a = await pool.acquire("coder")
    b = await pool.acquire("research")
    await pool.shutdown()
    assert a.proc.terminated and b.proc.terminated
    assert pool.workers == {}


@pytest.mark.asyncio
async def test_acquire_after_shutdown_is_refused():
    """Once shut down, the pool spawns nothing — a late acquire can't orphan a
    child process by racing the shutdown snapshot."""
    h = Harness()
    pool = h.make_pool()
    await pool.shutdown()
    with pytest.raises(ProfileBusyError):
        await pool.acquire("coder")
    assert h.spawned == []
