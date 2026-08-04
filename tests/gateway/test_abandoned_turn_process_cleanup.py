"""Regression coverage for abandoned gateway-turn subprocess cleanup (#76115)."""

import threading

from gateway.run import (
    _abandon_timed_out_gateway_turn,
    _reap_gateway_turn_processes,
    _watch_gateway_turn_inactivity,
)
from tools.process_registry import process_registry


class _IdleAgent:
    def __init__(self, idle_seconds=60.0):
        self.idle_seconds = idle_seconds
        self.interrupts = []

    def get_activity_summary(self):
        return {"seconds_since_activity": self.idle_seconds}

    def interrupt(self, reason):
        self.interrupts.append(reason)


def _state():
    return threading.Event(), threading.Event(), threading.Lock()


def test_thread_watchdog_reaps_only_processes_created_by_timed_out_turn(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda task_id, baseline, *, source: calls.append(
            (task_id, baseline, source)
        )
        or 1,
    )

    watchdog = threading.Thread(
        target=_watch_gateway_turn_inactivity,
        kwargs={
            "agent_holder": [agent],
            "task_id": "session-a",
            "process_baseline": frozenset({"proc_existing"}),
            "timeout": 30.0,
            "worker_done": worker_done,
            "timeout_fired": timeout_fired,
            "cleanup_lock": cleanup_lock,
            "poll_interval": 0.01,
        },
    )
    watchdog.start()
    watchdog.join(timeout=1)

    assert not watchdog.is_alive()
    assert timeout_fired.is_set()
    assert agent.interrupts == ["Execution timed out (inactivity)"]
    assert calls == [
        (
            "session-a",
            frozenset({"proc_existing"}),
            "gateway_turn_timeout",
        )
    ]


def test_completed_worker_wins_race_and_preserves_background_process(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    worker_done.set()
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed turn must not reap background work")
        ),
    )

    assert not _abandon_timed_out_gateway_turn(
        agent_holder=[agent],
        task_id="session-a",
        process_baseline=frozenset(),
        worker_done=worker_done,
        timeout_fired=timeout_fired,
        cleanup_lock=cleanup_lock,
    )
    assert not timeout_fired.is_set()
    assert agent.interrupts == []


def test_timeout_cleanup_is_idempotent(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_args, **_kwargs: calls.append(True) or 0,
    )
    kwargs = {
        "agent_holder": [agent],
        "task_id": "session-a",
        "process_baseline": frozenset(),
        "worker_done": worker_done,
        "timeout_fired": timeout_fired,
        "cleanup_lock": cleanup_lock,
    }

    assert _abandon_timed_out_gateway_turn(**kwargs)
    assert not _abandon_timed_out_gateway_turn(**kwargs)
    assert len(calls) == 1
    assert len(agent.interrupts) == 1


# ---------------------------------------------------------------------------
# Cross-turn race guard (#76188 review): task_id is session-scoped, not
# turn-scoped, so a replacement turn on the same session could otherwise
# have its freshly-spawned process killed by a stale reaper. Gated on
# run_generation via an injected `is_still_current` check.
# ---------------------------------------------------------------------------


def test_reap_skips_when_a_newer_turn_has_claimed_the_session(monkeypatch):
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    killed = _reap_gateway_turn_processes(
        "session-a",
        frozenset({"proc_old"}),
        source="gateway_turn_timeout",
        is_still_current=lambda: False,
    )

    assert killed == 0
    assert calls == []


def test_reap_proceeds_when_this_turn_is_still_current(monkeypatch):
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda task_id, baseline, *, source: calls.append(
            (task_id, baseline, source)
        )
        or 1,
    )

    killed = _reap_gateway_turn_processes(
        "session-a",
        frozenset({"proc_old"}),
        source="gateway_turn_timeout",
        is_still_current=lambda: True,
    )

    assert killed == 1
    assert calls == [("session-a", frozenset({"proc_old"}), "gateway_turn_timeout")]


def test_reap_fails_open_when_is_still_current_raises(monkeypatch):
    """A bug in the generation-check closure must not silently disable the
    underlying leak fix — it should log and fall through to reaping."""
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    def _boom():
        raise RuntimeError("session state lookup failed")

    killed = _reap_gateway_turn_processes(
        "session-a",
        frozenset(),
        source="gateway_turn_timeout",
        is_still_current=_boom,
    )

    assert killed == 1
    assert calls == [True]


def test_reap_skips_empty_task_id(monkeypatch):
    """ProcessSession.task_id defaults to "" — a blank turn id must never
    fan out into killing unrelated sessionless processes (#76188 review)."""
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    killed = _reap_gateway_turn_processes(
        "",
        frozenset(),
        source="gateway_turn_timeout",
    )

    assert killed == 0
    assert calls == []


def test_timeout_abandon_propagates_is_still_current_to_the_reap(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    assert _abandon_timed_out_gateway_turn(
        agent_holder=[agent],
        task_id="session-a",
        process_baseline=frozenset(),
        worker_done=worker_done,
        timeout_fired=timeout_fired,
        cleanup_lock=cleanup_lock,
        is_still_current=lambda: False,
    )

    # The turn was still marked abandoned (interrupt fired), but the actual
    # reap was skipped because a newer turn already claimed the session.
    assert agent.interrupts == ["Execution timed out (inactivity)"]
    assert calls == []
