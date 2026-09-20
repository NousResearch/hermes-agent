"""Attempt-ownership guards for overlapping compression attempts (#96634).

The stall-fallback path (#78981, PR #96634) deliberately DETACHES a
timed-out primary compression worker — the fence cancel wins and the
future stays on the shared pool — then immediately runs a fallback attempt
against the SAME ``ContextCompressor``. donovan-yohan's post-merge
adversarial review identified two interleavings where the still-unwinding
primary clobbers fallback-owned state:

1. The late primary's unwind calls ``_restore_compressor_attempt_state``
   with the PRIMARY's pre-attempt snapshot. Landing after the fallback's
   commit, it rolls ``_previous_summary`` / cooldown / provenance /
   telemetry back to pre-primary values.
2. ``_compression_cancelled_check`` is one shared attribute: the late
   primary's ``finally`` clears the callback the fallback just installed.

Both are now guarded by a monotonic per-compressor attempt generation
(``_claim_compressor_attempt``): restores and callback set/clear are keyed
to the claiming generation and no-op when a newer attempt owns the
compressor. These tests drive both interleavings deterministically —
no timing, no threads.
"""

from types import SimpleNamespace

from agent.conversation_compression import (
    _claim_compressor_attempt,
    _clear_compression_cancelled_check_if_owner,
    _compressor_attempt_is_current,
    _install_compression_cancelled_check,
    _restore_compressor_attempt_state,
    _snapshot_compressor_attempt_state,
)


def _compressor(**overrides):
    """Bare compressor stand-in carrying only attempt-state fields."""
    base = {
        "_previous_summary": "primary-era summary",
        "_summary_failure_cooldown_until": 0.0,
        "_last_summary_error": None,
        "_cooldown_persist_failed": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestLatePrimaryRestoreAfterFallbackCommit:
    """Claim 1: a stale attempt's snapshot restore must no-op."""

    def test_stale_restore_noops_and_preserves_fallback_state(self):
        compressor = _compressor()

        # Primary attempt starts: snapshot + claim.
        primary_snapshot = _snapshot_compressor_attempt_state(compressor)
        primary_gen = _claim_compressor_attempt(compressor)

        # Primary "runs" and mutates state, then stalls; the host detaches
        # it and starts the fallback attempt, which claims a NEWER generation
        # and commits its own state.
        compressor._previous_summary = "primary partial work"
        fallback_gen = _claim_compressor_attempt(compressor)
        assert fallback_gen > primary_gen
        compressor._previous_summary = "FALLBACK COMMITTED SUMMARY"
        compressor._summary_failure_cooldown_until = 123.0

        # The detached primary finally unwinds and tries to restore its
        # pre-attempt snapshot. Stale generation → must no-op.
        _restore_compressor_attempt_state(
            compressor, primary_snapshot, attempt_generation=primary_gen
        )

        assert compressor._previous_summary == "FALLBACK COMMITTED SUMMARY"
        assert compressor._summary_failure_cooldown_until == 123.0

    def test_current_attempt_restore_still_works(self):
        """The guard must not break the legitimate same-attempt rollback."""
        compressor = _compressor()
        snapshot = _snapshot_compressor_attempt_state(compressor)
        gen = _claim_compressor_attempt(compressor)

        compressor._previous_summary = "attempt scribbles"
        _restore_compressor_attempt_state(
            compressor, snapshot, attempt_generation=gen
        )

        assert compressor._previous_summary == "primary-era summary"

    def test_legacy_callers_without_generation_are_unchanged(self):
        """attempt_generation=None preserves the historical always-restore."""
        compressor = _compressor()
        snapshot = _snapshot_compressor_attempt_state(compressor)
        _claim_compressor_attempt(compressor)  # someone else claims

        compressor._previous_summary = "scribbles"
        _restore_compressor_attempt_state(compressor, snapshot)

        assert compressor._previous_summary == "primary-era summary"

    def test_slotted_compressor_disables_guard_gracefully(self):
        """A compressor that rejects attribute writes yields generation 0
        (guard off) rather than raising into the compression path."""

        class Frozen:
            __slots__ = ()

        gen = _claim_compressor_attempt(Frozen())
        assert gen == 0
        # Generation 0 always reports current — legacy behavior.
        assert _compressor_attempt_is_current(Frozen(), 0) is True


class TestCancelledCheckOwnership:
    """Claim 2: only the installing attempt may clear the shared callback."""

    def test_stale_primary_finally_cannot_clear_fallback_callback(self):
        compressor = _compressor()

        primary_gen = _claim_compressor_attempt(compressor)
        _install_compression_cancelled_check(
            compressor, lambda: "primary", primary_gen
        )

        # Fallback claims and installs ITS callback while the primary is
        # still unwinding.
        fallback_gen = _claim_compressor_attempt(compressor)
        fallback_check = lambda: "fallback"  # noqa: E731
        _install_compression_cancelled_check(
            compressor, fallback_check, fallback_gen
        )

        # Detached primary's ``finally`` fires late — must be refused.
        cleared = _clear_compression_cancelled_check_if_owner(
            compressor, primary_gen
        )

        assert cleared is False
        assert compressor._compression_cancelled_check is fallback_check

        # The owner can still clear its own callback afterwards.
        assert _clear_compression_cancelled_check_if_owner(
            compressor, fallback_gen
        ) is True
        assert compressor._compression_cancelled_check is None

    def test_owner_clear_roundtrip(self):
        compressor = _compressor()
        gen = _claim_compressor_attempt(compressor)
        check = lambda: True  # noqa: E731
        _install_compression_cancelled_check(compressor, check, gen)
        assert compressor._compression_cancelled_check is check

        assert _clear_compression_cancelled_check_if_owner(compressor, gen)
        assert compressor._compression_cancelled_check is None
        assert compressor._compression_cancelled_check_owner is None

    def test_generation_zero_clear_is_unconditional(self):
        """Guard-disabled (legacy/slotted) attempts keep the old clear."""
        compressor = _compressor()
        gen = _claim_compressor_attempt(compressor)
        _install_compression_cancelled_check(compressor, lambda: True, gen)

        # generation 0 = guard disabled → clears like the historical code.
        assert _clear_compression_cancelled_check_if_owner(compressor, 0)
        assert compressor._compression_cancelled_check is None


class TestSummaryRoutePinSingleUse:
    """The pin is single-use: consumed by the one summary call per attempt."""

    def test_pin_is_consumed_once(self):
        import contextvars

        def _probe():
            from agent.context_compressor import (
                pin_summary_route,
                take_pinned_summary_route,
            )

            route = {"provider": "fallback-prov", "model": "fallback-model"}
            with pin_summary_route(route):
                # Summary call consumes the pin (single-use preserved)...
                consumed = take_pinned_summary_route()
                assert consumed == route
                # ...and the main-model retry never re-issues it.
                assert take_pinned_summary_route() is None
            return True

        # Fresh context per test: state must not leak in from any other
        # test that touched the contextvars.
        assert contextvars.copy_context().run(_probe) is True

    def test_consume_is_context_local(self):
        """A consume in one context cannot leak into an unrelated attempt's."""
        import contextvars

        def _consume_in_isolated_context():
            from agent.context_compressor import (
                pin_summary_route,
                take_pinned_summary_route,
            )

            with pin_summary_route({"provider": "p", "model": "m"}):
                take_pinned_summary_route()

        ctx = contextvars.copy_context()
        ctx.run(_consume_in_isolated_context)

        # Outer context never saw the pin.
        from agent.context_compressor import take_pinned_summary_route

        assert take_pinned_summary_route() is None


class TestMidRestoreClaimRace:
    """TOCTOU: a claim landing between entry check and write must void the restore."""

    def test_claim_during_restore_body_voids_the_write(self, monkeypatch):
        import agent.conversation_compression as cc

        compressor = _compressor()
        snapshot = _snapshot_compressor_attempt_state(compressor)
        primary_gen = _claim_compressor_attempt(compressor)
        compressor._previous_summary = "FALLBACK STATE"

        # Interleave deterministically: the fallback claims the compressor
        # while the primary is inside the restore body (during deepcopy of
        # the snapshot, i.e. after the entry check passed).
        real_deepcopy = cc.copy.deepcopy
        state = {"claimed": False}

        def claiming_deepcopy(obj, *a, **kw):
            if not state["claimed"] and obj is snapshot:
                state["claimed"] = True
                _claim_compressor_attempt(compressor)  # fallback arrives NOW
            return real_deepcopy(obj, *a, **kw)

        monkeypatch.setattr(cc.copy, "deepcopy", claiming_deepcopy)
        _restore_compressor_attempt_state(
            compressor, snapshot, attempt_generation=primary_gen
        )

        # The write-time re-check must have refused the stale restore.
        assert compressor._previous_summary == "FALLBACK STATE"
