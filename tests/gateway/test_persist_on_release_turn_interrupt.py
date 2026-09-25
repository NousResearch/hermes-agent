"""persist_on_release jobs must survive the gateway's interrupt-driven reaps too (#41225).

``kill_all`` skips persisted sessions only for ``_LIFECYCLE_KILL_SOURCES``. The gateway's
interrupt paths — ``/new`` while a turn runs (``_busy_new_command``) and eviction while a
turn runs (``_hm_evict_running_agent``) — reap through
``GatewayAgentCacheMixin._interrupt_running_turn`` with ``source="gateway_turn_interrupt"``,
which is not in that set: the persisted overnight batch dies on ``/new``, contradicting the
terminal schema's own contract ("keep the process alive across session end, /new, context
compression, error recovery"). ``/stop`` is the operator's deliberate stop and must keep
killing persisted jobs.
"""

import threading
import time
from types import SimpleNamespace

from gateway.run import (
    _INTERRUPT_REASON_EVICTED,
    _INTERRUPT_REASON_RESET,
    _INTERRUPT_REASON_STOP,
)
from gateway.run_agent_cache import GatewayAgentCacheMixin
from tools.process_registry import ProcessRegistry, ProcessSession


def _make_session(sid: str, owner: str, *, persist: bool = False) -> ProcessSession:
    s = ProcessSession(id=sid, command="sleep 600", task_id=owner)
    s.persist_on_release = persist
    return s


class _RunningAgent:
    _gateway_turn_process_baseline = frozenset()

    def __init__(self, owner: str):
        self._gateway_turn_process_task_id = owner

    def interrupt(self, *_args, **_kwargs):
        return None


class _Runner:
    """Minimal GatewayAgentCacheMixin host — the shape test_reaper_profile_scope.py drives."""

    def __init__(self, owner: str):
        self._state = SimpleNamespace(turn=SimpleNamespace(agent=_RunningAgent(owner)))

    def _peek_session_state(self, _key):
        return self._state

    def _invalidate_session_run_generation(self, _key, *, reason: str = "") -> int:
        return 7

    def _is_session_run_current(self, _key, generation: int) -> bool:
        return generation == 7


def _drive_interrupt(monkeypatch, registry: ProcessRegistry, interrupt_reason: str) -> None:
    """Run the real `_interrupt_running_turn` reap chain against `registry` (kill_process stubbed)."""
    import tools.process_registry as pr_module

    reaped = threading.Event()

    def _fake_kill(session_id, **_kwargs):
        registry._running[session_id].exited = True
        reaped.set()
        return {"status": "killed"}

    # The production seam reads the module singleton at call time (from-import inside
    # _reap_gateway_turn_processes); point it at the fresh registry, keep the real
    # kill_started_since/kill_all targeting (the code under test), stub only OS signalling.
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(registry, "kill_process", _fake_kill)

    assert GatewayAgentCacheMixin._interrupt_running_turn(
        _Runner("session-a"), "session-a",
        interrupt_reason=interrupt_reason,
        invalidation_reason="test",
    ) == 7
    assert reaped.wait(timeout=2.0), "reap thread did not run"


def _seed(registry: ProcessRegistry) -> tuple[ProcessSession, ProcessSession]:
    persisted = _make_session("proc_persisted", "session-a", persist=True)
    volatile = _make_session("proc_volatile", "session-a")
    registry._running[persisted.id] = persisted
    registry._running[volatile.id] = volatile
    return persisted, volatile


def test_new_command_reap_spares_persist_on_release_jobs(monkeypatch):
    """/new while the turn runs is lifecycle cleanup (the schema names it): the persisted job survives."""
    registry = ProcessRegistry()
    persisted, volatile = _seed(registry)

    _drive_interrupt(monkeypatch, registry, _INTERRUPT_REASON_RESET)

    assert volatile.exited is True
    assert persisted.exited is False, (
        "a persist_on_release job died on /new — the terminal schema promises it "
        "survives session end and /new (#41225)"
    )


def test_eviction_reap_spares_persist_on_release_jobs(monkeypatch):
    """Eviction while the turn runs is session end (lifecycle): the persisted job survives."""
    registry = ProcessRegistry()
    persisted, volatile = _seed(registry)

    _drive_interrupt(monkeypatch, registry, _INTERRUPT_REASON_EVICTED)

    assert volatile.exited is True
    assert persisted.exited is False, (
        "a persist_on_release job died on session-end eviction — the terminal schema "
        "promises it survives session end (#41225)"
    )


def test_stop_command_reap_still_kills_persist_on_release_jobs(monkeypatch):
    """/stop is the operator's deliberate stop: it must keep reaching persisted jobs."""
    registry = ProcessRegistry()
    persisted, volatile = _seed(registry)

    _drive_interrupt(monkeypatch, registry, _INTERRUPT_REASON_STOP)

    assert volatile.exited is True
    assert persisted.exited is True, (
        "/stop is an explicit operator stop — it must still kill a persist_on_release job"
    )
