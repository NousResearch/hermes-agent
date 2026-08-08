"""Tests for the date-change note (ported from MoonshotAI/kimi-code#2564).

The system prompt bakes in "Conversation started: <date>" and stays
byte-stable for prompt-cache integrity, so a session that crosses midnight
leaves the model with a stale date. ``maybe_date_change_note`` tracks the
last announced date on the agent and emits a note (delivered via the
api_content sidecar channel) only on genuine rollovers.
"""

from datetime import datetime
from types import SimpleNamespace

from agent.turn_context import maybe_date_change_note


def _agent():
    return SimpleNamespace()


class TestMaybeDateChangeNote:
    def test_first_call_seeds_quietly(self):
        agent = _agent()
        note = maybe_date_change_note(agent, now=datetime(2026, 8, 6, 23, 50))
        assert note == ""
        assert agent._last_known_date == "Thursday, August 06, 2026"

    def test_same_day_no_note(self):
        agent = _agent()
        maybe_date_change_note(agent, now=datetime(2026, 8, 6, 9, 0))
        note = maybe_date_change_note(agent, now=datetime(2026, 8, 6, 23, 59))
        assert note == ""

    def test_midnight_rollover_announces(self):
        agent = _agent()
        maybe_date_change_note(agent, now=datetime(2026, 8, 6, 23, 50))
        note = maybe_date_change_note(agent, now=datetime(2026, 8, 7, 0, 10))
        assert "Friday, August 07, 2026" in note
        assert "Thursday, August 06, 2026" in note
        assert note.startswith("[System note:")

    def test_announces_once_then_quiet(self):
        agent = _agent()
        maybe_date_change_note(agent, now=datetime(2026, 8, 6, 12, 0))
        assert maybe_date_change_note(agent, now=datetime(2026, 8, 7, 12, 0)) != ""
        assert maybe_date_change_note(agent, now=datetime(2026, 8, 7, 18, 0)) == ""

    def test_multi_day_gap_announces_current_date(self):
        agent = _agent()
        maybe_date_change_note(agent, now=datetime(2026, 8, 1, 12, 0))
        note = maybe_date_change_note(agent, now=datetime(2026, 8, 6, 12, 0))
        assert "Thursday, August 06, 2026" in note
        assert "Saturday, August 01, 2026" in note

    def test_updates_tracker_on_rollover(self):
        agent = _agent()
        maybe_date_change_note(agent, now=datetime(2026, 8, 6, 12, 0))
        maybe_date_change_note(agent, now=datetime(2026, 8, 7, 12, 0))
        assert agent._last_known_date == "Friday, August 07, 2026"
