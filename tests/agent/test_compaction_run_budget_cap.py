"""Compression must never claim more of a turn's wall clock than the turn has left.

``resolve_context_compression_timeouts`` floors the idle window and the ceiling at the auxiliary
compression request budget (min 300s, #114594) so prep/chunking before the first streamed token is
never judged silent. That floor is inoperative when the host's own run budget is shorter than it —
the supervisor kills the run at the same instant the watchdog would first have spoken — and the
floor cannot be lowered by configuration. RUAAAAA-5212 measured that as 415 compaction-signed
timeouts across 13 ``hermes_local`` agents in 30 days: passes at +635s / +611s / +922s / +1003s,
SIGKILL at the 1800s wall clock, zero work output.

These tests pin that the run-budget cap closes that gap without touching the #114594 floor for
ordinary turns.
"""

from __future__ import annotations

import time

import pytest

from agent.conversation_compression_deadline import (
    cap_compaction_to_run_budget,
    resolve_context_compression_timeouts,
)


class TestCapCompactionToRunBudget:
    @pytest.fixture(autouse=True)
    def _no_aux_floor(self, monkeypatch):
        """Pin the aux floor off so these tests judge the run-budget cap on its own."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_effective_aux_timeout", lambda task, timeout: 0.0)

    @staticmethod
    def _agent(run_budget=None, elapsed=0.0):
        class _Agent:
            pass

        agent = _Agent()
        agent.run_budget_seconds = run_budget
        agent._run_budget_started_at = (time.time() - elapsed) if run_budget else None
        return agent

    def test_short_run_budget_caps_the_watchdog(self):
        """A 300s run cannot afford the 300s/600s windows: the pass is bounded well inside it."""
        idle, ceiling = cap_compaction_to_run_budget(self._agent(run_budget=300.0), 300.0, 600.0)

        assert ceiling < 300.0, "one compaction pass must not be able to consume the whole run"
        assert idle <= ceiling, "idle must never exceed the ceiling"
        # Leaves the majority of the run for actual work.
        assert ceiling <= 150.0

    def test_partially_elapsed_run_caps_harder(self):
        """The cap tracks the REMAINING budget, not the original one."""
        fresh = cap_compaction_to_run_budget(self._agent(run_budget=1800.0), 300.0, 600.0)
        spent = cap_compaction_to_run_budget(
            self._agent(run_budget=1800.0, elapsed=1500.0), 300.0, 600.0
        )

        assert fresh == (300.0, 600.0), "a roomy run keeps the #114594 floor"
        assert spent[1] <= 180.0, "a nearly exhausted run must not admit a 600s pass"

    def test_explicit_disable_is_never_resurrected(self):
        """``context_timeout_seconds <= 0`` means no owned wrapper; the cap must not re-arm it."""
        assert cap_compaction_to_run_budget(self._agent(run_budget=300.0), 0.0, 600.0) == (0.0, 600.0)
        assert cap_compaction_to_run_budget(self._agent(run_budget=300.0), -1.0, 600.0) == (-1.0, 600.0)

    def test_no_active_run_budget_is_a_no_op(self):
        """Interactive/CLI turns have no run budget; the resolved windows govern them."""
        assert cap_compaction_to_run_budget(self._agent(), 300.0, 600.0) == (300.0, 600.0)
        assert cap_compaction_to_run_budget(self._agent(run_budget=0.0), 120.0, 600.0) == (120.0, 600.0)

    def test_roomy_run_budget_keeps_the_floor(self):
        """A run budget larger than the resolved windows changes nothing."""
        assert cap_compaction_to_run_budget(
            self._agent(run_budget=7200.0), 300.0, 600.0
        ) == (300.0, 600.0)


class TestConfiguredFloorStillCannotBeLowered:
    """The RUAAAAA-5212 amplifier: the aux floor overrides even an explicit lower value, so the gap
    above could not be closed by configuration alone. Pinned so a future fix is a deliberate,
    test-visible change to #114594 semantics rather than an accident."""

    def test_explicit_value_below_the_aux_floor_is_overridden(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_effective_aux_timeout", lambda task, timeout: 300.0)

        assert resolve_context_compression_timeouts({"context_timeout_seconds": 30}) == (300.0, 300.0)
        assert resolve_context_compression_timeouts({"context_timeout_seconds": 1}) == (300.0, 300.0)