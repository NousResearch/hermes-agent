"""A delivered cron run's success must survive a transient post-delivery fire-claim blip (#105861).

Agent-mode cron jobs hold a durable fire claim that a heartbeat re-validates every 60s. The run
wrapper (``_run_one_job_body``) used to short-circuit to ``_record_fire_ownership_lost`` whenever
``d.side_effect_ownership_lost or _fire_claim_ownership_lost()`` was truthy *after* the
save/compose/deliver phase. ``fence.lost()`` (``_fire_claim_ownership_lost``) is a sampled flag that
flips permanently on a single transient missed heartbeat tick, so a completed, already-delivered run
got its ``last_status`` overwritten with ``Interrupted by shutdown before terminal completion.`` —
poisoning job history and firing false watchdog alerts even though the message was sent.

The post-delivery interruption decision is now ``_delivery_phase_interrupted(d)``, which is True only
when the claim was lost *during* the side effect (delivery did not complete). A completed delivery is
left to ``_finish_completed_run``'s owner-fenced ``mark_job_run``, the authoritative ownership check.
"""

import inspect

from cron import scheduler
from cron.scheduler import _delivery_phase_interrupted


def _delivery(*, side_effect_ownership_lost=False, delivery_attempted=True, delivery_error=None,
              success=True):
    d = scheduler._RunDelivery(job={"id": "j1"}, success=success, error=None)
    d.side_effect_ownership_lost = side_effect_ownership_lost
    d.delivery_attempted = delivery_attempted
    d.delivery_error = delivery_error
    return d


class TestDeliveryPhaseInterrupted:
    def test_completed_delivery_is_not_interrupted(self):
        # The regression: a run whose delivery completed cleanly must NOT be recorded as an
        # ownership-lost interruption, no matter what a later sampled heartbeat reads.
        assert _delivery_phase_interrupted(_delivery()) is False

    def test_loss_during_side_effect_is_interrupted(self):
        # Delivery raised _FireClaimLostDuringSideEffect before completing → genuine interruption.
        assert _delivery_phase_interrupted(
            _delivery(side_effect_ownership_lost=True, delivery_attempted=False)) is True

    def test_completed_delivery_with_a_delivery_error_is_not_an_ownership_interruption(self):
        # A normal (non-fence) delivery failure is a completed run with a delivery error, handled by
        # _finish_completed_run — not an ownership-lost interruption.
        assert _delivery_phase_interrupted(
            _delivery(delivery_error="discord 500", success=True)) is False

    def test_predicate_does_not_consult_a_sampled_heartbeat(self):
        # The decision is a pure function of the run-delivery outcome; it must not re-sample
        # ownership (which is what flipped a delivered success to an error).
        sig = inspect.signature(_delivery_phase_interrupted)
        assert list(sig.parameters) == ["d"]

    def test_run_wrapper_uses_the_predicate(self):
        # Guard the wiring: the post-delivery block must route through _delivery_phase_interrupted
        # rather than re-adding the sampled `or _fire_claim_ownership_lost()` disjunction.
        src = inspect.getsource(scheduler._run_one_job_body)
        assert "_delivery_phase_interrupted(d)" in src
        assert "d.side_effect_ownership_lost or _fire_claim_ownership_lost()" not in src
