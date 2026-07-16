"""note_turn_start / note_turn_persisted — the concurrent-turn tripwire.

Two turns interleaving on one session corrupt the durable transcript (flush
order races, identity-dedup row loss, stale history base). The tripwire does
not prevent the overlap; it names the occurrence with both turn ids so the
dispatch route that bypassed the busy guard can be identified from logs.
"""

import logging

from agent.agent_runtime_helpers import note_turn_start, note_turn_persisted


class _FakeAgent:
    session_id = "s1"


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
