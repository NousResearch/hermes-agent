"""#24651: the OpenRouter metadata pre-warm must be claimed atomically.

``_finalize_routing`` read ``_openrouter_prewarm_done.is_set()`` and then called ``.set()`` as
two separate steps.  Concurrent agent inits (gateway sessions, the batch runner, in-process
subagents) could all observe the Event clear and each start an ``openrouter-prewarm`` thread —
the very thread leak the process-level Event was added to prevent.

These run the real ``_finalize_routing`` from several threads; only the metadata fetch itself is
stubbed, so the claim is exercised on the production code path.
"""

import threading
from types import SimpleNamespace

from agent import agent_init
import run_agent


_WORKERS = 8


class _ClaimGate:
    """Holds the claimant's ``set()`` until every worker has reached the claim.

    A worker reaches it by reading the Event or by queueing on the claim lock. A non-atomic
    check-then-set lets every worker read the Event clear before the first ``set()`` lands; an
    atomic claim parks the others on the lock instead. Either way the gate opens on that
    condition, not on the clock, so the race is forced rather than hoped for.
    """

    def __init__(self, workers: int) -> None:
        self._workers = workers
        self._reached: set[int] = set()
        self._cond = threading.Condition()
        self.timed_out = False

    def reach(self) -> None:
        with self._cond:
            self._reached.add(threading.get_ident())
            self._cond.notify_all()

    def hold(self) -> None:
        with self._cond:
            if not self._cond.wait_for(lambda: len(self._reached) >= self._workers, timeout=30):
                self.timed_out = True


class _GatedClaimEvent(threading.Event):
    """An Event whose ``set()`` lands only once the gate opens.

    Every read stays truthful, so this only widens the window between "the claimant checked"
    and "the claim is visible", the gap the guard has to cover. Widening it cannot *create* a
    race: with an atomic claim every later caller reads an Event that is already set.
    """

    def __init__(self, gate: _ClaimGate) -> None:
        super().__init__()
        self._gate = gate

    def is_set(self) -> bool:
        self._gate.reach()
        return super().is_set()

    def set(self) -> None:
        self._gate.hold()
        super().set()


class _ArrivalLock:
    """Stand-in for ``agent_init._OPENROUTER_PREWARM_LOCK`` that reports each caller before it queues."""

    def __init__(self, gate: _ClaimGate) -> None:
        self._gate = gate
        self._lock = threading.Lock()

    def __enter__(self):
        self._gate.reach()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def _openrouter_agent():
    """The minimum an agent needs to reach the pre-warm block of ``_finalize_routing``."""
    return SimpleNamespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4",
        api_mode="chat_completions",
        _transport_cache={},
        _get_transport=lambda: None,
        _is_openrouter_url=lambda: True,
        _is_azure_openai_url=lambda: False,
        _is_direct_openai_url=lambda: False,
        _provider_model_requires_responses_api=lambda *a, **k: False,
    )


def _race_finalize_routing(monkeypatch, event, gate):
    """Drive ``_WORKERS`` concurrent ``_finalize_routing`` calls; return (prewarm calls, errors)."""
    started, errors = [], []
    record_lock = threading.Lock()

    def _record_prewarm():
        with record_lock:
            started.append(threading.current_thread().name)

    monkeypatch.setattr(agent_init, "fetch_model_metadata", _record_prewarm)
    monkeypatch.setattr(run_agent, "_openrouter_prewarm_done", event)
    # raising=False: a tree whose claim takes no lock leaves the stand-in unused, and the race shows.
    monkeypatch.setattr(agent_init, "_OPENROUTER_PREWARM_LOCK", _ArrivalLock(gate), raising=False)

    def _init_one():
        try:
            agent_init._finalize_routing(_openrouter_agent(), "chat_completions", None)
        except Exception as exc:  # surfaced by the caller's assertion
            with record_lock:
                errors.append(exc)

    workers = [threading.Thread(target=_init_one, name=f"init-{i}") for i in range(_WORKERS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    for thread in threading.enumerate():  # let every prewarm thread that DID start be counted
        if thread.name == "openrouter-prewarm":
            thread.join(timeout=30)
    return started, errors


def test_concurrent_init_starts_exactly_one_openrouter_prewarm(monkeypatch):
    """Racing inits claim the pre-warm once; the losers must not spawn their own thread."""
    gate = _ClaimGate(_WORKERS)
    event = _GatedClaimEvent(gate)

    started, errors = _race_finalize_routing(monkeypatch, event, gate)

    assert errors == []
    assert not gate.timed_out, "workers neither read the Event nor queued on the claim lock"
    assert len(started) == 1, f"{len(started)} prewarm threads started, expected exactly 1"
    assert event.is_set() is True


def test_prewarm_is_skipped_once_the_event_is_already_set(monkeypatch):
    """The Event stays the readable "already warmed" flag: a set Event means no spawn at all."""
    event = threading.Event()
    event.set()

    started, errors = _race_finalize_routing(monkeypatch, event, _ClaimGate(_WORKERS))

    assert errors == []
    assert started == []
