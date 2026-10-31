"""Turn-end terminal activity stamp tests (#72016 stuck-timestamp fix).

Tests the ``_finalize_activity_after_turn`` method that replaces
``_reset_activity_labels_after_turn`` in the turn ``finally`` block.

Key properties verified:
1. DB ``last_activity_at`` is stamped even when the persist window is fresh.
2. Labels (description/provenance) are cleared atomically in the same write.
3. In-memory ``_last_activity_ts`` is NOT bumped (preserves #15654 watchdog).
4. Kanban bridge is NOT invoked (no ``_touch_activity`` call).
5. Fail-open: DB errors are swallowed, never raise.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import run_agent
from agent.session_activity import ActivityProvenance


def _agent_with_db(session_id: str = "sess-1"):
    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=MagicMock(),
        _last_activity_ts=1_700_000_000.0,
        _last_activity_desc="executing tool: terminal",
        _last_activity_provenance=ActivityProvenance.UNKNOWN,
        _session_activity_last_persist_mono=0.0,
        _current_tool=None,
        _api_call_count=0,
        max_iterations=10,
        iteration_budget=SimpleNamespace(used=0, max_total=10),
    )
    agent._touch_activity = run_agent.AIAgent._touch_activity.__get__(agent, SimpleNamespace)
    agent._persist_session_activity_if_due = (
        run_agent.AIAgent._persist_session_activity_if_due.__get__(agent, SimpleNamespace)
    )
    agent._reset_activity_labels_after_turn = (
        run_agent.AIAgent._reset_activity_labels_after_turn.__get__(agent, SimpleNamespace)
    )
    agent._finalize_activity_after_turn = (
        run_agent.AIAgent._finalize_activity_after_turn.__get__(agent, SimpleNamespace)
    )
    agent.get_activity_summary = run_agent.AIAgent.get_activity_summary.__get__(
        agent, SimpleNamespace
    )
    return agent


def test_finalize_stamps_db_even_when_persist_window_is_fresh(monkeypatch):
    """The terminal stamp must write to the DB even when the heartbeat
    rate-limiter would normally skip (persist window fresh)."""
    agent = _agent_with_db()
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_050.0)
    monkeypatch.setattr(run_agent.time, "monotonic", lambda: 1000.5)  # within 60s window
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent._finalize_activity_after_turn()

    # finalize_session_activity must be called with the in-memory ts
    agent._session_db.finalize_session_activity.assert_called_once_with(
        "sess-1",
        1_700_000_000.0,
    )
    # touch_session_activity must NOT be called (avoids kanban bridge)
    agent._session_db.touch_session_activity.assert_not_called()


def test_finalize_does_not_bump_in_memory_ts(monkeypatch):
    """The terminal stamp must NOT update ``_last_activity_ts`` — the gateway
    stall-watchdog relies on this value being preserved across interrupt-recursive
    turns (#15654)."""
    agent = _agent_with_db()
    original_ts = agent._last_activity_ts
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_999.0)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent._finalize_activity_after_turn()

    assert agent._last_activity_ts == original_ts, (
        "_finalize_activity_after_turn must NOT bump _last_activity_ts "
        "(#15654 watchdog continuity)"
    )


def test_finalize_clears_in_memory_labels(monkeypatch):
    """After finalization, in-memory description and provenance are cleared."""
    agent = _agent_with_db()
    agent._last_activity_desc = "compressing context"
    agent._last_activity_provenance = ActivityProvenance.AGENT_COMPRESSION
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_050.0)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent._finalize_activity_after_turn()

    assert agent._last_activity_desc == ""
    assert agent._last_activity_provenance is ActivityProvenance.UNKNOWN


def test_finalize_swallows_db_errors(monkeypatch):
    """A DB write failure must never raise into turn teardown."""
    agent = _agent_with_db()
    agent._session_db.finalize_session_activity.side_effect = OSError("db locked")
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_050.0)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    # Must not raise
    agent._finalize_activity_after_turn()
    assert agent._session_db.finalize_session_activity.called


def test_finalize_noop_without_session_id(monkeypatch):
    """No session_id → no-op, no crash."""
    agent = _agent_with_db()
    agent.session_id = None
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent._finalize_activity_after_turn()
    agent._session_db.finalize_session_activity.assert_not_called()


def test_finalize_noop_without_session_db(monkeypatch):
    """No session_db → no-op, no crash."""
    agent = _agent_with_db()
    agent._session_db = None
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent._finalize_activity_after_turn()
    # Should not crash, should not touch any DB


def test_finalize_uses_existing_last_activity_ts(monkeypatch):
    """When ``_last_activity_ts`` is set from mid-turn activity, the finalize
    stamps THAT value (not a fresh ``time.time()``) — so the DB reflects the
    last real activity, not the turn-end wall clock."""
    agent = _agent_with_db()
    agent._last_activity_ts = 1_700_000_042.0  # mid-turn activity time
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_099.0)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent._finalize_activity_after_turn()

    agent._session_db.finalize_session_activity.assert_called_once_with(
        "sess-1",
        1_700_000_042.0,  # the mid-turn ts, NOT the turn-end time
    )


def test_existing_reset_activity_labels_still_works(monkeypatch):
    """``_reset_activity_labels_after_turn`` is still callable and unchanged —
    compression and other consumers still use it."""
    agent = _agent_with_db()
    agent._last_activity_ts = 1_700_000_000.0
    agent._last_activity_desc = "compressing context"
    agent._last_activity_provenance = ActivityProvenance.AGENT_COMPRESSION
    agent._session_activity_last_persist_mono = 1_000.0
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent._reset_activity_labels_after_turn()

    assert agent._last_activity_ts == 1_700_000_000.0  # ts preserved
    assert agent._last_activity_desc == ""
    assert agent._last_activity_provenance is ActivityProvenance.UNKNOWN
    agent._session_db.clear_session_activity_labels.assert_called_once_with("sess-1")
    agent._session_db.touch_session_activity.assert_not_called()
