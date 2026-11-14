"""A managed worker is reachable through the agent interrupt contract, not only through
``_managed_workers``.

``execute_managed`` drops its worker from ``_managed_workers`` in its ``finally`` the moment the turn
returns, while ``authority.adopted`` keeps naming that worker until settlement. The profile-stop poll
(``join_authority_work`` → ``stop_authority_turns(in_process=True)``, every 50 ms) therefore hands a
``ManagedWorker`` to ``request_hard_interrupt`` with a reason, and ``ManagedWorker.interrupt`` took no
argument: every graceful ``hermes -p <profile> gateway restart --yes`` of a profile running a managed
turn aborted mid-drain with ``TypeError: ManagedWorker.interrupt() takes 1 positional argument but 2
were given`` and left the profile neither stopped nor restarted."""
import queue
from types import SimpleNamespace

import pytest

from agent.interrupt_compat import request_hard_interrupt
from gateway.run_runtime import stop_authority_turns
from gateway.session_managed_worker import ManagedWorker


def _queued_controls(worker):
    frames = []
    while True:
        try:
            frames.append(worker.commands.get_nowait())
        except queue.Empty:
            return frames


def _settlement_window_authority(worker):
    """The window that broke the drain: generation 3 still executes and ``adopted`` still names the
    worker, but ``execute_managed`` already removed it from ``_managed_workers``, so the managed arm
    of ``stop_authority_turns`` never sees it and only the in-process arm does."""
    live = SimpleNamespace(route='r', event_stream=SimpleNamespace(execution={'execution_generation': 3}))
    return SimpleNamespace(sessions={'s': live}, _managed_workers={}, _managed_spawning={},
                           adopted={'s': (3, worker)}, pending_stops={},
                           runner=SimpleNamespace(_running_agents={}))


def test_interrupt_accepts_the_reason_the_agent_contract_passes():
    """``request_hard_interrupt`` finds no ``hard_interrupt`` on a managed worker and falls back to
    the legacy ``interrupt(message)`` ABI, forwarding ``tool_reason`` only because the callable
    advertises it: the worker must take the reason and still do exactly one thing — Stop."""
    worker = ManagedWorker(SimpleNamespace())
    assert request_hard_interrupt(worker, 'Profile stopping', tool_reason='gateway shutdown') is True
    assert _queued_controls(worker) == [{'type': 'stop'}]
    assert worker.stop.is_set()


def test_profile_stop_completes_when_it_reaches_an_adopted_managed_worker():
    """The drain-path call itself: no raise, the Stop control reaches the worker, and the reason the
    drain named is recorded as the turn's stop cause so the settled result says why it was
    interrupted. A managed turn is not an in-process one, so nothing is latched for a later
    generation to consume."""
    worker = ManagedWorker(SimpleNamespace())
    authority = _settlement_window_authority(worker)
    assert stop_authority_turns(authority, in_process=True) == 1
    assert _queued_controls(worker) == [{'type': 'stop'}]
    assert worker.stop_cause == 'Profile stopping'
    assert authority.pending_stops == {}


@pytest.mark.parametrize('arm', ['closing', 'backpressured'])
def test_profile_stop_never_aborts_when_the_control_cannot_be_queued(arm):
    """Both real states of an already-dropped worker must drain quietly, exactly as the managed arm
    tolerates them: ``close`` has begun, so ``control`` refuses with ``managed_worker_lost`` and the
    turn is over; or the control queue is full, and ``control`` latched Stop first, so the owner's
    read loop escalates on that latch alone."""
    worker = ManagedWorker(SimpleNamespace())
    if arm == 'closing':
        worker.closed.set()
    else:
        for _ in range(worker.commands.maxsize):
            worker.commands.put_nowait({'type': 'delta', 'text': '.'})
    authority = _settlement_window_authority(worker)

    assert stop_authority_turns(authority, in_process=True) == 1
    if arm == 'closing':
        assert _queued_controls(worker) == [], 'a closed worker has no turn left to stop'
    else:
        assert worker.stop.is_set() and worker.stopped_at is not None
