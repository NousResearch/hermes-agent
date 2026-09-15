"""The compression outer timeouts must never out-tighten the inner deadline.

Context compression runs an auxiliary summarisation call inside a
progress-aware wrapper.  Two outer budgets bound that call:

* ``compression.context_timeout_seconds`` (default 120) -- the no-progress
  watchdog.
* ``compression.context_total_ceiling_seconds`` (default 600) -- the hard stop
  on the whole pass.

Both wrap an inner ``auxiliary.compression.timeout`` that is floored to 300 s
(``_COMPRESSION_TIMEOUT_FLOOR_SECONDS``) because summarising a large context
legitimately takes minutes.

When an outer budget is tighter than the inner deadline it is not a safety net,
it is a guaranteed kill:

* the outer watchdog fires first and abandons the worker;
* ``call_llm`` never raises, because nothing timed out at the HTTP layer;
* so the ``except`` branch that walks ``fallback_providers`` is unreachable.

The ceiling axis has a second, worse consequence.  The summary runs in a
detached executor thread, so when the ceiling fires mid-stream the worker keeps
running and **completes** after the host stopped waiting.  The commit fence then
correctly refuses the late commit -- discarding a fully successful summary while
reporting that the summariser stalled.
"""

from agent.compression_timeout_floor import (
    DERIVED_CEILING_CAP_SECONDS,
    DERIVED_IDLE_CAP_SECONDS,
    reconcile_ceiling,
    reconcile_idle_timeout,
    reconcile_timeouts,
)


class TestIdleGuardNeverUndercutsInnerDeadline:
    def test_default_guard_is_lifted_above_the_inner_deadline(self):
        assert reconcile_idle_timeout(120.0, 300.0) > 300.0

    def test_invariant_holds_across_inner_deadlines(self):
        for inner in (60.0, 120.0, 300.0, 600.0):
            assert reconcile_idle_timeout(120.0, inner) >= inner

    def test_explicit_operator_value_is_honoured_verbatim(self):
        assert reconcile_idle_timeout(120.0, 300.0, explicit=True) == 120.0
        # hermetic tests pin tiny values; they must not be clamped
        assert reconcile_idle_timeout(0.01, 300.0, explicit=True) == 0.01

    def test_zero_or_negative_disables_and_is_never_resurrected(self):
        assert reconcile_idle_timeout(0.0, 300.0) == 0.0
        assert reconcile_idle_timeout(-1.0, 300.0) == -1.0

    def test_an_already_generous_guard_is_never_lowered(self):
        assert reconcile_idle_timeout(900.0, 300.0) == 900.0

    def test_unknown_inner_deadline_is_a_noop(self):
        assert reconcile_idle_timeout(120.0, None) == 120.0
        assert reconcile_idle_timeout(120.0, 0) == 120.0
        assert reconcile_idle_timeout(120.0, "nonsense") == 120.0  # type: ignore[arg-type]

    def test_derived_guard_is_capped(self):
        assert reconcile_idle_timeout(120.0, 10_000.0) == DERIVED_IDLE_CAP_SECONDS


class TestCeilingAdmitsAFallbackAttempt:
    def test_default_ceiling_is_lifted_to_admit_a_fallback(self):
        idle, ceiling = reconcile_timeouts(120.0, 600.0, 300.0)
        assert idle == 360.0
        assert ceiling >= idle + 300.0, (
            f"ceiling cannot contain a stall-fallback: idle={idle} ceiling={ceiling}"
        )
        assert ceiling == 720.0

    def test_explicit_operator_ceiling_is_honoured_verbatim(self):
        # Tested above the idle window: the `ceiling >= idle` invariant is
        # orthogonal and would legitimately raise a value below idle.
        _, ceiling = reconcile_timeouts(120.0, 500.0, 300.0, explicit_ceiling=True)
        assert ceiling == 500.0
        assert reconcile_ceiling(300.0, 360.0, 300.0, explicit=True) == 300.0

    def test_a_generous_ceiling_is_never_lowered(self):
        _, ceiling = reconcile_timeouts(120.0, 3600.0, 300.0)
        assert ceiling == 3600.0

    def test_derived_ceiling_is_capped(self):
        _, ceiling = reconcile_timeouts(120.0, 600.0, 10_000.0)
        assert ceiling == DERIVED_CEILING_CAP_SECONDS

    def test_unknown_inner_deadline_is_a_noop(self):
        assert reconcile_ceiling(600.0, 360.0, None) == 600.0
        assert reconcile_ceiling(600.0, 360.0, 0) == 600.0

    def test_ceiling_is_never_below_the_idle_window(self):
        idle, ceiling = reconcile_timeouts(120.0, 600.0, 300.0)
        assert ceiling >= idle


class TestResolverWiring:
    """The RESOLVER must apply both invariants, not just the pure helpers.

    A helper fix the production resolver never consults ships inert.
    """

    def test_resolver_applies_both_invariants(self, monkeypatch):
        from agent import conversation_compression as cc

        monkeypatch.setattr(
            "agent.auxiliary_client._effective_aux_timeout",
            lambda task, timeout: 300.0,
        )
        idle, ceiling = cc.resolve_context_compression_timeouts({})
        assert idle > 300.0, "idle must be lifted above the inner deadline"
        assert ceiling >= idle + 300.0, "ceiling must admit a fallback attempt"

    def test_a_default_valued_key_is_not_mistaken_for_operator_intent(
        self, monkeypatch
    ):
        """DEFAULT_CONFIG is deep-merged, so presence proves nothing.

        Passing the shipped defaults explicitly must still reconcile; treating
        them as operator-set is what makes this fix inert in production.
        """
        from agent import conversation_compression as cc

        monkeypatch.setattr(
            "agent.auxiliary_client._effective_aux_timeout",
            lambda task, timeout: 300.0,
        )
        idle, ceiling = cc.resolve_context_compression_timeouts(
            {"context_timeout_seconds": 120, "context_total_ceiling_seconds": 600}
        )
        assert idle > 300.0
        assert ceiling >= idle + 300.0

    def test_operator_values_still_win(self, monkeypatch):
        from agent import conversation_compression as cc

        monkeypatch.setattr(
            "agent.auxiliary_client._effective_aux_timeout",
            lambda task, timeout: 300.0,
        )
        idle, ceiling = cc.resolve_context_compression_timeouts(
            {"context_timeout_seconds": 45.0, "context_total_ceiling_seconds": 1200.0}
        )
        assert idle == 45.0
        assert ceiling == 1200.0

    def test_resolver_survives_aux_config_failure(self, monkeypatch):
        from agent import conversation_compression as cc

        def _boom(*a, **k):
            raise RuntimeError("aux config unavailable")

        monkeypatch.setattr("agent.auxiliary_client._effective_aux_timeout", _boom)
        idle, _ = cc.resolve_context_compression_timeouts({})
        assert idle == cc.DEFAULT_CONTEXT_TIMEOUT_SECONDS
