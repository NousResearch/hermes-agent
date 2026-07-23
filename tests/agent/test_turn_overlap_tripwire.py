"""note_turn_start / note_turn_persisted — the concurrent-turn tripwire.

Two turns interleaving on one session corrupt the durable transcript (flush
order races, identity-dedup row loss, stale history base). The tripwire does
not prevent the overlap; it names the occurrence with both turn ids so the
dispatch route that bypassed the busy guard can be identified from logs.
"""

import logging

import pytest

from agent import agent_runtime_helpers as _helpers
from agent.agent_runtime_helpers import note_turn_start, note_turn_persisted


class _FakeAgent:
    session_id = "s1"


@pytest.fixture(autouse=True)
def _clear_inflight_registry():
    """Isolate the module-level session registry between tests."""
    with _helpers._INFLIGHT_TURNS_LOCK:
        _helpers._INFLIGHT_TURNS_BY_SESSION.clear()
    yield
    with _helpers._INFLIGHT_TURNS_LOCK:
        _helpers._INFLIGHT_TURNS_BY_SESSION.clear()


def test_clean_serial_turns_no_warning(caplog):
    agent = _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        assert note_turn_start(agent, "s1:t1:aaaa") is None
        note_turn_persisted(agent)
        assert note_turn_start(agent, "s1:t2:bbbb") is None
        note_turn_persisted(agent)
    assert not caplog.records


def test_overlap_warns_with_both_turn_ids(caplog):
    agent = _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent, "s1:t1:aaaa")
        # second turn starts before the first persisted
        prev = note_turn_start(agent, "s1:t2:bbbb")
    assert prev == "s1:t1:aaaa"
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "s1:t1:aaaa" in msg and "s1:t2:bbbb" in msg and "s1" in msg


def test_overlap_takes_ownership_no_repeat_warning(caplog):
    """A turn that crashed before its persist warns at most once — the next
    turn takes ownership of the in-flight slot."""
    agent = _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent, "s1:t1:aaaa")   # never persists (crash)
        note_turn_start(agent, "s1:t2:bbbb")   # warns once, takes ownership
        note_turn_persisted(agent)
        note_turn_start(agent, "s1:t3:cccc")   # clean again
    assert len(caplog.records) == 1


def test_same_turn_id_reentry_is_silent(caplog):
    """Re-entering with the same turn_id (retry paths) is not an overlap."""
    agent = _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent, "s1:t1:aaaa")
        note_turn_start(agent, "s1:t1:aaaa")
    assert not caplog.records


def test_cross_agent_same_session_overlap_warns(caplog):
    """#64934 route: two routing keys mapped to one session_id run their
    turns on two different agent objects (the gateway agent cache is keyed
    by routing key), so per-agent state alone can never see the overlap."""
    agent_a, agent_b = _FakeAgent(), _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        assert note_turn_start(agent_a, "s1:t1:aaaa") is None
        prev = note_turn_start(agent_b, "s1:t2:bbbb")
    assert prev == "s1:t1:aaaa"
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "s1:t1:aaaa" in msg and "s1:t2:bbbb" in msg and "s1" in msg
    assert "different agent object" in msg


def test_cross_agent_serial_turns_are_silent(caplog):
    """A persisted turn releases the session slot — a later turn on another
    agent object for the same session is normal (e.g. cache eviction)."""
    agent_a, agent_b = _FakeAgent(), _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent_a, "s1:t1:aaaa")
        note_turn_persisted(agent_a)
        note_turn_start(agent_b, "s1:t2:bbbb")
        note_turn_persisted(agent_b)
    assert not caplog.records


def test_distinct_sessions_never_cross_warn(caplog):
    """Concurrent turns on different session_ids are legitimate parallelism."""
    agent_a, agent_b = _FakeAgent(), _FakeAgent()
    agent_b.session_id = "s2"
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent_a, "s1:t1:aaaa")
        assert note_turn_start(agent_b, "s2:t2:bbbb") is None
    assert not caplog.records


def test_same_agent_overlap_warns_once_not_twice(caplog):
    """A same-agent overlap must not double-report through the session leg."""
    agent = _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent, "s1:t1:aaaa")
        prev = note_turn_start(agent, "s1:t2:bbbb")
    assert prev == "s1:t1:aaaa"
    assert len(caplog.records) == 1


def test_persist_clears_start_session_after_mid_turn_rotation(caplog):
    """Compression rotates agent.session_id mid-turn; the persist must
    release the slot the turn registered under, not the rotated id."""
    agent = _FakeAgent()
    agent.session_id = "s-parent"
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent, "sp:t1:aaaa")
        agent.session_id = "s-child"  # mid-turn compression rotation
        note_turn_persisted(agent)
        # A fresh turn on the parent id must find the slot released.
        other = _FakeAgent()
        other.session_id = "s-parent"
        assert note_turn_start(other, "sp:t2:bbbb") is None
    assert not caplog.records


def test_crashed_cross_agent_turn_warns_once_then_recovers(caplog):
    """A turn that never persists (crash) yields one warning; the next turn
    takes ownership of the session slot and the tripwire goes quiet."""
    agent_a, agent_b, agent_c = _FakeAgent(), _FakeAgent(), _FakeAgent()
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(agent_a, "s1:t1:aaaa")   # never persists (crash)
        note_turn_start(agent_b, "s1:t2:bbbb")   # warns once, takes ownership
        note_turn_persisted(agent_b)
        note_turn_start(agent_c, "s1:t3:cccc")   # clean again
    assert len(caplog.records) == 1


def test_persist_disabled_fork_neither_registers_nor_warns(caplog):
    """Background-review forks share the live parent's session_id for
    prompt-cache warmth but are _persist_disabled — they can never write
    to the transcript, so they must not trip the cross-agent warning
    against the parent's real in-flight turn (in either direction)."""
    parent, fork = _FakeAgent(), _FakeAgent()
    fork._persist_disabled = True
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(parent, "s1:t1:aaaa")     # real turn in flight
        assert note_turn_start(fork, "s1:tr:ffff") is None  # fork: silent
        note_turn_persisted(fork)                 # fork's funnel still runs
        # Reverse order on the next cycle: fork in flight, then real turn.
        note_turn_persisted(parent)
        note_turn_start(fork, "s1:tr:gggg")
        assert note_turn_start(parent, "s1:t2:bbbb") is None
    assert not caplog.records


def test_persist_disabled_fork_persist_does_not_steal_parent_slot(caplog):
    """The fork's persist funnel still runs; it must not pop the parent's
    session slot, or a real cross-agent overlap right after a review fork
    would go unreported."""
    parent, fork, intruder = _FakeAgent(), _FakeAgent(), _FakeAgent()
    fork._persist_disabled = True
    with caplog.at_level(logging.WARNING, logger="agent.agent_runtime_helpers"):
        note_turn_start(parent, "s1:t1:aaaa")     # real turn holds the slot
        note_turn_start(fork, "s1:tr:ffff")
        note_turn_persisted(fork)                 # must NOT release s1
        prev = note_turn_start(intruder, "s1:t2:bbbb")
    assert prev == "s1:t1:aaaa"
    assert len(caplog.records) == 1
