"""Summary auth/network failure flags are scoped to the failure episode.

``_last_summary_auth_failure`` / ``_last_summary_network_failure`` are set by
``_generate_summary()`` on a terminal failure so ``compress()`` aborts instead
of falling through to the destructive static-fallback (#29559 / #25585).

Two invariants:

1. While the failure cooldown is ARMED, a re-entrant ``compress()`` must keep
   the flags: ``_generate_summary()`` short-circuits in its cooldown
   early-return without re-asserting them, so clearing eagerly would drop the
   abort guard and destroy the middle window — the #29559 data loss.

2. Once the cooldown has EXPIRED, a stale flag from a long-resolved blip must
   NOT persist for the session lifetime: it would force every later generic
   summary failure onto the abort path, overriding
   ``abort_on_summary_failure=False``.  ``compress()`` clears the flags at the
   start of the call; if the error persists, the fresh ``_generate_summary()``
   attempt re-asserts them and the abort guard still fires.
"""

import time
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _make_compressor():
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100000,
    ):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )


def _tiny_transcript():
    # Small enough that compress() returns on the "cannot compress" early
    # return — AFTER the per-call state reset block we are testing, BEFORE
    # any summary generation / LLM machinery.
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


class TestFlagsPreservedWhileCooldownArmed:
    def test_auth_flag_survives_compress_during_active_cooldown(self):
        c = _make_compressor()
        c._last_summary_auth_failure = True
        c._summary_failure_cooldown_until = time.monotonic() + 60

        c.compress(_tiny_transcript())

        # Re-entrant compress during cooldown must stay abort-eligible.
        assert c._last_summary_auth_failure is True

    def test_network_flag_survives_compress_during_active_cooldown(self):
        c = _make_compressor()
        c._last_summary_network_failure = True
        c._summary_failure_cooldown_until = time.monotonic() + 60

        c.compress(_tiny_transcript())

        assert c._last_summary_network_failure is True


class TestFlagsClearedOnceCooldownExpires:
    def test_stale_auth_flag_cleared_after_cooldown_expiry(self):
        c = _make_compressor()
        c._last_summary_auth_failure = True
        c._summary_failure_cooldown_until = time.monotonic() - 1

        c.compress(_tiny_transcript())

        assert c._last_summary_auth_failure is False

    def test_stale_network_flag_cleared_after_cooldown_expiry(self):
        c = _make_compressor()
        c._last_summary_network_failure = True
        c._summary_failure_cooldown_until = time.monotonic() - 1

        c.compress(_tiny_transcript())

        assert c._last_summary_network_failure is False

    def test_never_armed_cooldown_counts_as_expired(self):
        # Fresh compressor: cooldown 0.0 — a stale flag (e.g. restored from
        # persisted state) must not survive a fresh compress() call.
        c = _make_compressor()
        c._last_summary_auth_failure = True
        c._last_summary_network_failure = True
        assert c._summary_failure_cooldown_until == 0.0

        c.compress(_tiny_transcript())

        assert c._last_summary_auth_failure is False
        assert c._last_summary_network_failure is False


class TestForcedCompressClearsFlags:
    def test_manual_force_clears_cooldown_then_flags(self):
        # /compress (force=True) clears the cooldown first, so the flag clear
        # applies even while the auto-compress cooldown would still be armed —
        # the user asked for a fresh attempt; _generate_summary() re-asserts
        # the flags if the credential/network is still broken.
        c = _make_compressor()
        c._last_summary_auth_failure = True
        c._last_summary_network_failure = True
        c._summary_failure_cooldown_until = time.monotonic() + 60

        c.compress(_tiny_transcript(), force=True)

        assert c._summary_failure_cooldown_until == 0.0
        assert c._last_summary_auth_failure is False
        assert c._last_summary_network_failure is False
