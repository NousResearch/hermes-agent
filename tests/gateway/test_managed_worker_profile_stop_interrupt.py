"""A profile stop hard-interrupts every adopted agent through ``request_hard_interrupt``, which falls
back to the legacy ``interrupt(message)`` ABI. A managed worker's turn stays adopted until settlement,
after ``execute_managed`` has already dropped it from ``_managed_workers``: the next stop poll must
reach it without a TypeError (which aborted the profile stop mid-drain) and without a closed-pipe error."""
import queue
from types import SimpleNamespace

from agent.interrupt_compat import request_hard_interrupt
from gateway.run_runtime import stop_authority_turns
from gateway.session_managed_worker import ManagedWorker


def _queued(worker):
    frames = []
    while True:
        try:
            frames.append(worker.commands.get_nowait())
        except queue.Empty:
            return frames


def _authority_with_adopted(worker):
    """The settlement window: the session still executes generation 3 and ``adopted`` names the
    worker, but ``execute_managed``'s finally already removed it from ``_managed_workers``."""
    live = SimpleNamespace(route='r', event_stream=SimpleNamespace(execution={'execution_generation': 3}))
    return SimpleNamespace(sessions={'s': live}, _managed_workers={}, _managed_spawning={},
                           adopted={'s': (3, worker)}, pending_stops={},
                           runner=SimpleNamespace(_running_agents={}))


def test_request_hard_interrupt_with_message_sends_the_stop_control():
    worker = ManagedWorker(SimpleNamespace())
    assert request_hard_interrupt(worker, 'Profile stopping', tool_reason='gateway shutdown') is True
    assert worker.stop.is_set()
    assert _queued(worker) == [{'type': 'stop'}]


def test_profile_stop_reaches_an_adopted_managed_worker_without_raising():
    worker = ManagedWorker(SimpleNamespace())
    authority = _authority_with_adopted(worker)
    assert stop_authority_turns(authority, in_process=True) == 1
    assert _queued(worker) == [{'type': 'stop'}]
    assert authority.pending_stops == {}


def test_profile_stop_treats_a_closed_adopted_managed_worker_as_stopped():
    worker = ManagedWorker(SimpleNamespace())
    worker.closed.set()  # close() began: control() refuses with managed_worker_lost
    authority = _authority_with_adopted(worker)
    assert stop_authority_turns(authority, in_process=True) == 1
    assert _queued(worker) == []
