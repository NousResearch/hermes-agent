"""An in-turn auto-compaction must carry its count onto the live session (#146153).

The agent's compressor increments on every compaction and emits ``session:compress`` with the new
count, but before this only the manual ``/compress`` receipt wrote ``live.compression_count``.
The gateway evicts and rebuilds a session's agent on ``/stop``, ``/model``, viewer close and
one-turn model switches; the successor is seeded from ``live``, so an unset ``live`` reset the
session's cumulative count to 0 after every eviction (and every restart, via the same seed).
"""
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner
from gateway.session_usage import seed_compression_count


def _authority(live, *, generation=0, executing_generation=0):
    """Just the authority fields ``_carry_live_compression_count`` / ``_adopt_turn_agent`` read."""
    return SimpleNamespace(
        sessions={"s1": live},
        executing={"s1": executing_generation},
        db=SimpleNamespace(get_session=lambda _sid: {"runtime_generation": generation}),
        adopt_agent=lambda sid, gen, agent: None,
    )


def _runner(live, *, authority=True, **authority_fields):
    """A TurnRunner with just the fields ``_event_callback_sync`` reads."""
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        _hooks_ref=SimpleNamespace(loaded_hooks=[], emit=lambda *a, **k: None),
        _loop_for_step=None,
    )
    turn._approval_owner = (
        (_authority(live, **authority_fields), "s1", 0) if authority else None
    )
    return turn


def test_in_turn_compaction_reaches_the_live_session():
    live = SimpleNamespace()  # no compression_count: the gateway never set it on an auto-compaction
    _runner(live)._event_callback_sync("session:compress", {"compression_count": 2})

    assert live.compression_count == 2


def test_the_count_survives_an_agent_eviction():
    """End to end for the reported symptom: two auto-compactions, then evict + seed."""
    live = SimpleNamespace()
    turn = _runner(live)
    turn._event_callback_sync("session:compress", {"compression_count": 2})

    successor = SimpleNamespace(context_compressor=SimpleNamespace(compression_count=0))
    seed_compression_count(live, successor)

    assert successor.context_compressor.compression_count == 2


def test_a_late_lower_count_never_lowers_the_carried_count():
    live = SimpleNamespace(compression_count=5)  # a manual /compress already carried 5 forward
    _runner(live)._event_callback_sync("session:compress", {"compression_count": 2})

    assert live.compression_count == 5


def test_other_events_leave_the_count_alone():
    live = SimpleNamespace()
    _runner(live)._event_callback_sync("agent:step", {"iteration": 2})

    assert not hasattr(live, "compression_count")


def test_a_turn_without_an_owning_session_is_a_noop():
    live = SimpleNamespace()
    _runner(live, authority=False)._event_callback_sync("session:compress", {"compression_count": 2})

    assert not hasattr(live, "compression_count")


def test_a_late_event_after_the_claim_ended_is_dropped():
    """The agent thread outlived its running claim (evicted or timed out): it must not write."""
    live = SimpleNamespace(compression_count=0)  # a fresh conversation, as /new leaves it
    _runner(live, executing_generation=None)._event_callback_sync(
        "session:compress", {"compression_count": 2})

    assert live.compression_count == 0


def test_a_late_event_after_new_is_dropped():
    """/new zeroes the count and bumps the generation; a summary from the old conversation is stale."""
    live = SimpleNamespace(compression_count=0)
    _runner(live, generation=1)._event_callback_sync("session:compress", {"compression_count": 2})

    assert live.compression_count == 0


def test_a_boolean_is_not_a_compression_count():
    live = SimpleNamespace()
    _runner(live)._event_callback_sync("session:compress", {"compression_count": True})

    assert not hasattr(live, "compression_count")


def test_the_adoption_path_seeds_the_successor():
    """Every authority-owned turn seeds its agent, not only local viewers (messaging, muted turns)."""
    live = SimpleNamespace(compression_count=4)
    agent = SimpleNamespace(context_compressor=SimpleNamespace(compression_count=0))

    _runner(live)._adopt_turn_agent(agent)

    assert agent.context_compressor.compression_count == 4
