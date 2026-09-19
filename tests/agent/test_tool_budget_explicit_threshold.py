"""Precedence of the explicit tool-result persistence threshold.

``agent.tool_executor._budget_for_agent`` resolves the BudgetConfig used
for tool-result persistence. An explicit
``tools.tool_result_persist_threshold_chars`` value (exposed by agent_init
as ``agent._tool_result_persist_threshold_chars``) must override the
context-scaled default; otherwise context scaling and the DEFAULT_BUDGET
fallback behave exactly as before.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.tool_executor import _budget_for_agent
from tools.budget_config import (
    DEFAULT_BUDGET,
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_TURN_BUDGET_CHARS,
    budget_for_context_window,
)


def _agent(persist_threshold=None, context_length=None):
    compressor = None
    if context_length is not None:
        compressor = MagicMock()
        compressor.context_length = context_length
    return SimpleNamespace(
        _tool_result_persist_threshold_chars=persist_threshold,
        context_compressor=compressor,
    )


class TestExplicitThresholdPrecedence:
    def test_explicit_threshold_wins_over_context_scaling(self):
        agent = _agent(persist_threshold=20_000, context_length=1_000_000)
        cfg = _budget_for_agent(agent)
        assert cfg.default_result_size == 20_000
        assert cfg.turn_budget == DEFAULT_TURN_BUDGET_CHARS
        assert cfg != DEFAULT_BUDGET

    def test_explicit_threshold_keeps_small_model_turn_budget(self):
        # Regression (#23767): an explicit per-result cap must NOT reset the
        # turn budget back to the 200K default on a small-window model.
        agent = _agent(persist_threshold=20_000, context_length=65_536)
        cfg = _budget_for_agent(agent)
        assert cfg.default_result_size == 20_000
        scaled = budget_for_context_window(65_536)
        assert cfg.turn_budget == scaled.turn_budget
        assert cfg.preview_size == scaled.preview_size
        assert cfg.turn_budget < DEFAULT_TURN_BUDGET_CHARS

    def test_explicit_threshold_ignores_context_failure(self):
        agent = _agent(persist_threshold=30_000, context_length="garbage")
        cfg = _budget_for_agent(agent)
        assert cfg.default_result_size == 30_000

    def test_noninteger_explicit_falls_back_to_context_scaling(self):
        # A plugin/test setting a non-integer value must not crash resolution.
        agent = _agent(persist_threshold="not-an-int", context_length=65_536)
        cfg = _budget_for_agent(agent)
        assert cfg is not DEFAULT_BUDGET
        baseline = _budget_for_agent(
            _agent(persist_threshold=None, context_length=65_536)
        )
        assert cfg.default_result_size == baseline.default_result_size
        agent2 = _agent(persist_threshold="not-an-int", context_length=None)
        assert _budget_for_agent(agent2) is DEFAULT_BUDGET

    def test_no_explicit_uses_context_scaling(self):
        agent = _agent(persist_threshold=None, context_length=65_536)
        cfg = _budget_for_agent(agent)
        assert cfg is not DEFAULT_BUDGET
        # 65K model: scaled cap below the 100K default.
        assert cfg.default_result_size < DEFAULT_RESULT_SIZE_CHARS

    def test_no_explicit_large_context_keeps_default_cap(self):
        agent = _agent(persist_threshold=None, context_length=1_000_000)
        cfg = _budget_for_agent(agent)
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS

    def test_no_explicit_no_compressor_returns_default_budget(self):
        agent = _agent(persist_threshold=None, context_length=None)
        assert _budget_for_agent(agent) is DEFAULT_BUDGET

    def test_no_explicit_broken_compressor_returns_default_budget(self):
        agent = _agent(persist_threshold=None, context_length="garbage")
        assert _budget_for_agent(agent) is DEFAULT_BUDGET

    def test_zero_explicit_is_treated_as_unset(self):
        # agent_init normalizes <=0 to None before it reaches the executor;
        # belt-and-braces: a literal 0 must not crash the resolution.
        agent = _agent(persist_threshold=0, context_length=65_536)
        cfg = _budget_for_agent(agent)
        assert cfg.default_result_size < DEFAULT_RESULT_SIZE_CHARS

    def test_bool_explicit_falls_back_to_context_scaling(self):
        # Regression: int(True) == 1 used to turn a programmatically-set
        # boolean into default_result_size == 1 (persist almost everything).
        # The executor must normalize through the same single source of truth
        # as agent_init and fall back to context scaling.
        agent = _agent(persist_threshold=True, context_length=65_536)
        cfg = _budget_for_agent(agent)
        baseline = _budget_for_agent(
            _agent(persist_threshold=None, context_length=65_536)
        )
        assert cfg.turn_budget == baseline.turn_budget
        assert cfg.default_result_size == baseline.default_result_size
        assert cfg.default_result_size != 1
        assert cfg.turn_budget < DEFAULT_TURN_BUDGET_CHARS
        agent2 = _agent(persist_threshold=False, context_length=None)
        assert _budget_for_agent(agent2) is DEFAULT_BUDGET

    def test_float_explicit_falls_back_to_context_scaling(self):
        # int(20000.0) == 20000 is harmless, but int(1.5) == 1 is not; floats
        # are rejected wholesale so the rule stays simple and single-sourced.
        agent = _agent(persist_threshold=20_000.0, context_length=65_536)
        cfg = _budget_for_agent(agent)
        baseline = _budget_for_agent(
            _agent(persist_threshold=None, context_length=65_536)
        )
        assert cfg.default_result_size == baseline.default_result_size
        assert cfg.default_result_size != 20_000

    def test_invalid_programmatic_value_warns_once(self, caplog):
        agent = _agent(persist_threshold="not-an-int", context_length=65_536)
        with caplog.at_level(logging.WARNING, logger="agent.tool_executor"):
            _budget_for_agent(agent)
            _budget_for_agent(agent)

        matches = [
            record
            for record in caplog.records
            if "invalid programmatic tools.tool_result_persist_threshold_chars"
            in record.getMessage()
        ]
        assert len(matches) == 1
