"""Idle compaction measures the gap from the transcript, not the in-memory clock (#79357).

Every turn entry re-stamps ``agent._last_activity_ts`` before the idle check reads it:

- the durable turn lease's ``start()`` (``agent/turn_facade_lease.py``) calls
  ``_touch_activity("starting new turn")`` right before ``run_conversation``;
- the gateway's ``_init_cached_agent_for_turn`` resets it for cached agents;
- a rebuilt agent (cache eviction, restart) starts with it set to construction time.

So the gap measured from ``_last_activity_ts`` was always ~0 and idle compaction never
fired. The reference is now the newest timestamp among the transcript rows that precede
this turn's user message; ``_last_activity_ts`` is only the fallback for untimestamped
transcripts.
"""

from __future__ import annotations

import time
from pathlib import Path

from hermes_state import SessionDB

from agent.turn_context_compaction import idle_reference_timestamp
from tests.agent.test_idle_compaction_lock_and_guards import _prep_idle_agent, _run_prologue


def _timed_history(last_ts: float, n: int = 20) -> list:
    """Alternating user/assistant rows, one second apart, the newest at ``last_ts``."""
    rows = []
    for i in range(n):
        rows.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"m{i}",
            "timestamp": last_ts - (n - 1 - i),
        })
    return rows


class TestIdleReferenceTimestamp:

    def test_newest_row_before_the_current_turn(self):
        msgs = [{"role": "user", "timestamp": 100.0}, {"role": "assistant", "timestamp": 250.0},
                {"role": "tool", "timestamp": 200.0}, {"role": "user", "timestamp": 9_999.0}]
        assert idle_reference_timestamp(msgs, 3) == 250.0

    def test_current_turn_user_message_is_excluded(self):
        msgs = [{"role": "assistant", "timestamp": 100.0}, {"role": "user", "timestamp": 5_000.0}]
        assert idle_reference_timestamp(msgs, 1) == 100.0

    def test_none_without_usable_timestamps(self):
        msgs = [{"role": "user"}, {"role": "assistant", "timestamp": None},
                {"role": "assistant", "timestamp": True}, {"role": "assistant", "timestamp": "123"},
                {"role": "assistant", "timestamp": 0}, "not-a-dict", {"role": "user", "timestamp": 5.0}]
        assert idle_reference_timestamp(msgs, 6) is None

    def test_missing_index_scans_everything(self):
        msgs = [{"role": "user", "timestamp": 1.0}, {"role": "assistant", "timestamp": 2.0}]
        assert idle_reference_timestamp(msgs, None) == 2.0
        assert idle_reference_timestamp(msgs, -1) == 2.0


def _fresh(tmp_path: Path, sid: str, *, idle_after: int = 300):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(sid, source="cli")
    agent = _prep_idle_agent(db, sid, idle_after=idle_after)
    agent.context_compressor.emit_automatic_compaction_status = False
    return agent


def test_fires_after_turn_lease_restamps_the_activity_clock(tmp_path: Path) -> None:
    """Production order: lease.start() touches activity, then the idle check runs."""
    agent = _fresh(tmp_path, "IDLE_LEASE")
    agent._touch_activity("starting new turn")  # exactly what DurableTurnLease.start() does
    assert time.time() - agent._last_activity_ts < 5

    _run_prologue(agent, _timed_history(time.time() - 3600))

    agent.context_compressor.compress.assert_called_once()


def test_fires_on_a_rebuilt_agent(tmp_path: Path) -> None:
    """Cache eviction / restart: a new agent's clock starts at construction time."""
    agent = _fresh(tmp_path, "IDLE_REBUILT")
    agent._last_activity_ts = time.time()  # construction-time default

    _run_prologue(agent, _timed_history(time.time() - 7200))

    agent.context_compressor.compress.assert_called_once()


def test_recent_transcript_does_not_fire_even_if_the_clock_is_stale(tmp_path: Path) -> None:
    """The transcript wins over the in-memory clock in both directions."""
    agent = _fresh(tmp_path, "IDLE_RECENT")
    agent._last_activity_ts = time.time() - 86_400

    _run_prologue(agent, _timed_history(time.time() - 30))

    agent.context_compressor.compress.assert_not_called()


def test_untimestamped_transcript_falls_back_to_the_activity_clock(tmp_path: Path) -> None:
    agent = _fresh(tmp_path, "IDLE_FALLBACK")
    agent._last_activity_ts = time.time() - 3600

    _run_prologue(agent, [{"role": "user", "content": f"m{i}"} for i in range(20)])

    agent.context_compressor.compress.assert_called_once()
