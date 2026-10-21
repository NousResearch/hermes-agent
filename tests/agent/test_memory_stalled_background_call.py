"""A background provider call that never returns pins its mem-sync worker thread forever (Python cannot
stop a thread), and MemoryManagers are rebuilt per agent. The process-wide gate skips calls once
_MAX_STUCK_CALLS calls that started after the provider's last success are stuck, admits one half-open
probe per interval while closed, and caps calls in flight, so a wedged backend pins a bounded number of
threads and a recovered one keeps its memory."""
import threading

import pytest

import agent.memory_manager as mm
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from tests.waiting import wait_until


class _HangingProvider(MemoryProvider):
    def __init__(self, name: str = "hangs"):
        self._name = name
        self.release = threading.Event()
        self.started = threading.Event()
        self.sync_calls = 0
        self.prefetch_calls = 0
        self.hang_next = True

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def get_tool_schemas(self):
        return []

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        self.prefetch_calls += 1

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "") -> None:
        self.sync_calls += 1
        if self.hang_next:
            self.hang_next = False
            self.started.set()
            self.release.wait(10)


_GATE = MemoryManager._BACKGROUND_GATE


def _clear_gate():
    for state in (_GATE.calls, _GATE.last_ok, _GATE.probe_at, _GATE.skips):
        state.clear()


@pytest.fixture(autouse=True)
def _short_stall(monkeypatch):
    monkeypatch.setattr(mm, "_BACKGROUND_CALL_STALL_S", 0.0)
    monkeypatch.setattr(mm, "_PROBE_INTERVAL_S", 1e9)  # no half-open probe unless a test asks for one
    _clear_gate()
    yield
    _clear_gate()


def _manager(provider):
    mgr = MemoryManager()
    mgr.add_provider(provider)
    return mgr


def _mem_sync_threads():
    return [t for t in threading.enumerate() if t.name.startswith("mem-sync")]


def _hang_one_call(name="hangs"):
    """A manager whose provider's first sync_turn never returns (until released)."""
    provider = _HangingProvider(name)
    mgr = _manager(provider)
    mgr.sync_all("hello", "hi")
    assert provider.started.wait(5)
    return mgr, provider


def _skip(mgr, text):
    """sync_all runs the gate on the manager's mem-sync worker; wait for it so the skip is recorded."""
    mgr.sync_all(text, "x")
    assert mgr.flush_pending(timeout=5)
    mgr.shutdown_all()


def test_sustained_wedge_pins_at_most_max_stuck_threads():
    """A backend that stays wedged: every call hangs. Calls keep being admitted until
    _MAX_STUCK_CALLS are stuck, then later managers are skipped, so pinned threads stay bounded."""
    hung = [_hang_one_call() for _ in range(mm._MAX_STUCK_CALLS)]
    before = len(_mem_sync_threads())

    for _ in range(5):
        fresh = _manager(_HangingProvider())
        fresh.sync_all("again", "reply")
        fresh.queue_prefetch_all("query")
        assert fresh.flush_pending(timeout=5)  # returns: nothing blocked behind the hung calls
        fresh.shutdown_all()

    key = hung[0][0]._registry_key(hung[0][1])
    try:
        assert _GATE.skips[key] == 10
        assert len(_GATE.calls[key]) == mm._MAX_STUCK_CALLS
        # Idle workers of shut-down executors exit asynchronously.
        wait_until(lambda: len(_mem_sync_threads()) <= before, timeout=5, what="idle mem-sync workers to exit")
    finally:
        for _, provider in hung:
            provider.release.set()
    for mgr, _ in hung:
        assert mgr.flush_pending(timeout=5)
    assert key not in _GATE.calls
    for mgr, _ in hung:
        mgr.shutdown_all()


def test_healthy_replacement_still_writes_behind_an_orphaned_call():
    """One call that never returns must not silence the provider for the process lifetime: a fresh
    manager with a healthy provider instance (the backend recovered) keeps writing."""
    orphan_mgr, orphan = _hang_one_call()
    healthy = _HangingProvider()
    healthy.hang_next = False
    try:
        for _ in range(5):
            fresh = _manager(healthy)
            fresh.sync_all("again", "reply")
            fresh.queue_prefetch_all("query")
            assert fresh.flush_pending(timeout=5)
            fresh.shutdown_all()
        assert (healthy.sync_calls, healthy.prefetch_calls) == (5, 5)
    finally:
        orphan.release.set()
        orphan_mgr.shutdown_all()


def test_new_outage_warns_on_its_first_skip_even_with_an_orphan_alive(caplog):
    """Orphans keep the key alive after the gate reopens; the next outage must still warn on
    its first skip instead of waiting for the 100th."""
    hung = [_hang_one_call() for _ in range(mm._MAX_STUCK_CALLS)]
    key = hung[0][0]._registry_key(hung[0][1])
    try:
        _skip(_manager(_HangingProvider()), "skipped")
        assert _GATE.skips[key] == 1
        hung[-1][1].release.set()  # one returns: the gate reopens, three orphans keep the key alive
        assert hung[-1][0].flush_pending(timeout=5)
        hung += [_hang_one_call() for _ in range(mm._MAX_STUCK_CALLS)]  # a new outage
        caplog.clear()
        with caplog.at_level("WARNING", logger="agent.memory_manager"):
            _skip(_manager(_HangingProvider()), "skipped again")
        assert _GATE.skips[key] == 1
        assert "background calls stuck" in caplog.text
    finally:
        for _, provider in hung:
            provider.release.set()
    for mgr, _ in hung:
        mgr.shutdown_all()


def _orphan_then_recover(orphans):
    """Each outage leaves one call that never returns, then the backend recovers for one turn."""
    for _ in range(mm._MAX_STUCK_CALLS):
        orphans.append(_hang_one_call())
        recovered = _HangingProvider()
        recovered.hang_next = False
        _skip(_manager(recovered), "recovered")


def test_orphans_from_past_outages_do_not_close_the_gate():
    """Four separate blips over the process lifetime, each followed by recovery, must not turn the
    provider off: a stuck call that started before the last success no longer counts."""
    orphans = []
    healthy = _HangingProvider()
    healthy.hang_next = False
    try:
        _orphan_then_recover(orphans)
        for _ in range(3):
            fresh = _manager(healthy)
            fresh.sync_all("again", "reply")
            fresh.queue_prefetch_all("query")
            assert fresh.flush_pending(timeout=5)
            fresh.shutdown_all()
        assert (healthy.sync_calls, healthy.prefetch_calls) == (3, 3)
    finally:
        for _, provider in orphans:
            provider.release.set()
    for mgr, _ in orphans:
        mgr.shutdown_all()


def test_closed_gate_admits_a_half_open_probe_that_reopens_it(monkeypatch):
    """With the gate closed by calls that never return, one probe per interval goes through; once a
    probe succeeds (the backend recovered) every later call is admitted again."""
    hung = [_hang_one_call() for _ in range(mm._MAX_STUCK_CALLS)]
    healthy = _HangingProvider()
    healthy.hang_next = False
    try:
        _skip(_manager(healthy), "closed")
        assert healthy.sync_calls == 0  # gate closed, probe interval not yet reached
        monkeypatch.setattr(mm, "_PROBE_INTERVAL_S", 0.0)
        _skip(_manager(healthy), "probe")
        assert healthy.sync_calls == 1
        monkeypatch.setattr(mm, "_PROBE_INTERVAL_S", 1e9)  # no further probe: only a reopened gate admits
        _skip(_manager(healthy), "after the probe")
        assert healthy.sync_calls == 2
    finally:
        for _, provider in hung:
            provider.release.set()
    for mgr, _ in hung:
        mgr.shutdown_all()


def test_failed_probe_does_not_restart_the_outage_warning(monkeypatch):
    """A probe that wedges too is not a recovery: the skip count keeps running, so a long outage
    warns on its first skip and every 100th, not after every probe."""
    hung = [_hang_one_call() for _ in range(mm._MAX_STUCK_CALLS)]
    key = hung[0][0]._registry_key(hung[0][1])
    try:
        _skip(_manager(_HangingProvider()), "closed")
        monkeypatch.setattr(mm, "_PROBE_INTERVAL_S", 0.0)
        hung.append(_hang_one_call())  # the probe is admitted and hangs
        monkeypatch.setattr(mm, "_PROBE_INTERVAL_S", 1e9)
        _skip(_manager(_HangingProvider()), "still closed")
        assert _GATE.skips[key] == 2
    finally:
        for _, provider in hung:
            provider.release.set()
    for mgr, _ in hung:
        mgr.shutdown_all()


def test_sustained_wedge_with_probes_never_exceeds_the_inflight_cap(monkeypatch):
    """Every failed probe pins one more thread; at the cap, probing stops, so a backend that stays
    wedged pins a bounded number of threads however long it lasts."""
    monkeypatch.setattr(mm, "_PROBE_INTERVAL_S", 0.0)
    monkeypatch.setattr(mm, "_MAX_INFLIGHT_CALLS", mm._MAX_STUCK_CALLS + 2)
    hung = [_hang_one_call() for _ in range(mm._MAX_STUCK_CALLS)]
    key = hung[0][0]._registry_key(hung[0][1])
    try:
        for _ in range(2):  # two probes admitted, and they wedge too
            hung.append(_hang_one_call())
        for turn in range(5):
            _skip(_manager(_HangingProvider()), f"capped {turn}")
        assert len(_GATE.calls[key]) == mm._MAX_INFLIGHT_CALLS
        assert _GATE.skips[key] == 5
    finally:
        for _, provider in hung:
            provider.release.set()
    for mgr, _ in hung:
        mgr.shutdown_all()


def test_slow_call_below_threshold_is_not_skipped(monkeypatch):
    monkeypatch.setattr(mm, "_BACKGROUND_CALL_STALL_S", 60.0)
    provider = _HangingProvider()
    first = _manager(provider)
    first.sync_all("hello", "hi")
    assert provider.started.wait(5)

    second = _manager(provider)
    try:
        second.sync_all("again", "reply")
        assert second.flush_pending(timeout=5)
        assert provider.sync_calls == 2
    finally:
        provider.release.set()
    for m in (first, second):
        m.shutdown_all()
