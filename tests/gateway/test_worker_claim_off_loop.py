"""A worker RPC's claim and verification never block the gateway's event loop.

``_claim`` probes the process table (psutil) and reads state.db; ``_verify`` reads state.db. Run on
the loop, one slow claim froze the whole gateway, chat and every other worker's receipt included,
for up to 12 s (py-spy on the live gateway, 2026-11-09). These tests make the claim slow on purpose
and require the loop to keep ticking through it, on every worker lane.
"""
import asyncio
import time

import pytest

import gateway.session_worker as session_worker
from gateway.session_worker import worker_request

_PROFILE = 'engineering-lead'
_SLOW_S = 0.6
_TICK_S = 0.02


class _Actor:
    profile_id = _PROFILE
    capabilities = frozenset({'worker:adopt'})
    subject = 'local'


class _Authority:
    profile_id = _PROFILE
    epoch = 1
    db = object()

    def _require_admission_open(self):
        return None

    def authorize(self, _actor, _ref, _capability):
        return None

    def wake_after_worker(self, _session_id):
        return None


class _Connection:
    authority = _Authority()
    actor = _Actor()


def _params(**extra):
    return {'profile_id': _PROFILE, 'session_id': 'local-owned', 'execution_id': 'w1', 'generation': 1,
            'pid': 4242, 'birth': 1000.0, 'secret': 's' * 32} | extra


def _slow(result=None):
    def blocking(*_args, **_kwargs):
        time.sleep(_SLOW_S)
        return result
    return blocking


async def _longest_loop_gap(call):
    """Run ``call`` while a ticker measures the longest time the loop went without running it."""
    gaps, done = [0.0], asyncio.Event()

    async def ticker():
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(_TICK_S)
            now = time.perf_counter()
            gaps[0] = max(gaps[0], now - last)
            last = now

    tick = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        result = await call
    finally:
        done.set()
        await tick
    return result, gaps[0]


def _stub_lanes(monkeypatch):
    monkeypatch.setattr(session_worker, '_claim', _slow('c' * 64))
    monkeypatch.setattr(session_worker, '_verify', _slow())
    monkeypatch.setattr(session_worker, 'register_worker_execution', lambda *_a, **_k: 'registered')
    monkeypatch.setattr(session_worker, 'adopt_worker_execution', lambda *_a, **_k: 'adopted')
    monkeypatch.setattr(session_worker, 'mutate_worker_execution', lambda *_a, **_k: 'persisted')


@pytest.mark.parametrize('operation, extra, expected', [
    ('register', {'kind': 'compute'}, 'registered'),
    ('adopt', {}, 'adopted'),
    ('persist', {'epoch': 1, 'sequence': 1, 'operation': 'turn.lease', 'payload': {}}, 'persisted'),
])
def test_a_slow_claim_does_not_stall_the_loop(monkeypatch, operation, extra, expected):
    _stub_lanes(monkeypatch)
    result, gap = asyncio.run(_longest_loop_gap(
        worker_request(_Connection(), None, _params(**extra), operation=operation)))
    assert result == expected
    # The claim (and, past register, the verify) each block for _SLOW_S; on the loop the ticker
    # would have starved for at least that long.
    assert gap < _SLOW_S / 2, f'event loop starved for {gap:.2f}s during a {operation}'


def test_the_delegation_lane_claims_off_the_loop(monkeypatch):
    import gateway.session_worker_delegation as delegation
    _stub_lanes(monkeypatch)
    monkeypatch.setattr(delegation, 'persist_worker_delegation', lambda *_a, **_k: 'delegated', raising=False)
    params = _params(epoch=1, sequence=1, operation='delegation.read', payload={})

    async def call():
        try:
            return await delegation.worker_delegation_request(_Connection(), None, params)
        except Exception as exc:  # past the claim the stubbed store may refuse; the loop is the point
            return type(exc).__name__

    _, gap = asyncio.run(_longest_loop_gap(call()))
    assert gap < _SLOW_S / 2, f'event loop starved for {gap:.2f}s during a delegation claim'


def test_the_child_reserve_lane_claims_off_the_loop(monkeypatch):
    import gateway.session_worker_child as child
    _stub_lanes(monkeypatch)
    params = _params(epoch=1, sequence=1, operation='child.reserve', payload={})

    async def call():
        try:
            return await child.worker_child_request(_Connection(), None, params)
        except Exception as exc:  # past the claim the stub authority has no store; the loop is the point
            return type(exc).__name__

    _, gap = asyncio.run(_longest_loop_gap(call()))
    assert gap < _SLOW_S / 2, f'event loop starved for {gap:.2f}s during a child reserve claim'
