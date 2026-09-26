"""Per-turn model/tool time split (issue #109569) — accumulator tests.

The accumulator lives in ``agent/turn_context.py::note_completed_api_call``,
fed from the single choke point where ``api_duration`` is computed
(``agent/turn_response_check.py``). These tests prove accumulation, reset
coverage, and nonsense-input guards with no network and no agent loop.
"""

from types import SimpleNamespace

import pytest

from agent.turn_context import _PER_TURN_RESET_STATE, note_completed_api_call


def _agent(**overrides):
    base = {"_turn_model_seconds": 0.0, "_turn_first_token_at": None,
            "_last_api_first_chunk_at": None}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestResetCoversTimingKeys:
    def test_timing_keys_reset_each_turn(self):
        names = dict(_PER_TURN_RESET_STATE)
        assert names["_turn_model_seconds"] is None
        assert names["_turn_first_token_at"] is None

    def test_reset_stamps_the_turn_admission_clock(self, monkeypatch):
        """_reset_per_turn_agent_state owns the turn's wall-clock start; the
        verbose stamp reads that one value instead of taking a second one."""
        from agent import turn_context

        monkeypatch.setattr(turn_context, "note_turn_start", lambda *a, **k: None, raising=False)
        agent = SimpleNamespace(
            _memory_store=SimpleNamespace(), _tool_guardrails=SimpleNamespace(reset_for_turn=lambda: None),
            api_mode="anthropic_messages", _compression_warning=None, max_iterations=5,
            run_budget_seconds=None, iteration_budget=None,
        )
        turn_context._reset_per_turn_agent_state(agent)
        assert isinstance(agent._current_turn_timestamp, float)
        assert agent._turn_model_seconds is None
        assert agent._turn_first_token_at is None


class TestNoteCompletedApiCall:
    def test_accumulates_across_calls(self):
        agent = _agent()
        note_completed_api_call(agent, 1.2)
        note_completed_api_call(agent, 2.5)
        assert agent._turn_model_seconds == pytest.approx(3.7)

    def test_missing_attr_starts_from_zero(self):
        agent = SimpleNamespace(_turn_first_token_at=None, _last_api_first_chunk_at=None)
        note_completed_api_call(agent, 2.0)
        assert agent._turn_model_seconds == pytest.approx(2.0)

    def test_none_means_unavailable_until_first_call(self):
        agent = _agent(_turn_model_seconds=None)
        assert agent._turn_model_seconds is None
        note_completed_api_call(agent, 2.0)
        assert agent._turn_model_seconds == pytest.approx(2.0)

    def test_ignores_nonsense_durations(self):
        agent = _agent()
        for bad in (None, "fast", float("nan"), -1.0):
            note_completed_api_call(agent, bad)
        assert agent._turn_model_seconds == pytest.approx(0.0)

    def test_captures_first_token_only_once(self):
        agent = _agent(_last_api_first_chunk_at=100.5)
        note_completed_api_call(agent, 1.0)
        assert agent._turn_first_token_at == pytest.approx(100.5)
        agent._last_api_first_chunk_at = 101.5
        note_completed_api_call(agent, 1.0)
        assert agent._turn_first_token_at == pytest.approx(100.5)

    def test_no_chunk_leaves_first_token_unset(self):
        agent = _agent()
        note_completed_api_call(agent, 1.0)
        assert agent._turn_first_token_at is None


class TestChokePointWiring:
    def test_response_check_feeds_accumulator(self):
        # The one-line call survives refactors only if the import resolves —
        # a cycle here breaks every agent turn, so import for real.
        import agent.turn_response_check as check
        assert check.note_completed_api_call is note_completed_api_call
