"""Reason-surfacing in the fallback announce (2026-07-07).

A same-model cross-provider failover reads like a rate-limit blip; a content-policy
refusal (fable declined → opus answered) is a materially different event. Thread the
FailoverReason into _emit_fallback_announce so the line says WHY:
  🔄 Model fallback (safety refusal): claude-app/claude-fable-5 → claude-api-proxy-f1/claude-opus-4-8
"""

import pytest

from agent.chat_completion_helpers import _emit_fallback_announce, _fallback_reason_label
from agent.error_classifier import FailoverReason


class _StubAgent:
    def __init__(self):
        self.emitted = []
        self._last_fallback_announced = None
        self._last_fallback_event = None
        self.context_compressor = None
        self._pending_stream_error_reason = None

    def _emit_status(self, msg):
        self.emitted.append(msg)


def _announce(reason=None, kind="fallback", old_model="claude-fable-5", new_model="claude-opus-4-8",
              old_provider="claude-app", new_provider="claude-api-proxy-f1"):
    a = _StubAgent()
    _emit_fallback_announce(
        a, old_model, new_model, new_provider,
        old_provider=old_provider,
        announce_enabled=True,
        record_event=False,
        kind=kind,
        reason=reason,
    )
    return a.emitted


class TestReasonLabel:
    def test_content_policy_maps_to_safety_refusal(self):
        assert _fallback_reason_label(FailoverReason.content_policy_blocked) == "safety refusal"

    def test_rate_limit_maps(self):
        assert _fallback_reason_label(FailoverReason.rate_limit) == "rate limit"
        assert _fallback_reason_label(FailoverReason.upstream_rate_limit) == "rate limit"

    def test_enum_value_string_also_maps(self):
        # Callers may pass the .value string rather than the enum member.
        assert _fallback_reason_label("content_policy_blocked") == "safety refusal"

    def test_timeout_maps_to_connection_dropped(self):
        # A peer-closed / mid-stream transport drop classifies to `timeout`; it
        # must read as "connection dropped", not the ambiguous bare "timeout".
        assert _fallback_reason_label(FailoverReason.timeout) == "connection dropped"

    def test_ssl_cert_maps(self):
        assert _fallback_reason_label(FailoverReason.ssl_cert_verification) == "TLS error"

    def test_unmapped_reason_falls_to_generic_floor(self):
        # A classified-but-unmapped reason must STILL surface a rider so the
        # announce is never silent about WHY (the 2026-07-10 peer-closed
        # stream-drop that failed over with a bare line). "connection issue" is
        # the honest floor.
        assert _fallback_reason_label(FailoverReason.unknown) == "connection issue"

    def test_none_reason_is_still_none(self):
        # A genuine no-op call (no reason object at all) renders no suffix.
        assert _fallback_reason_label(None) is None


class TestAnnounceRendersReason:
    def test_safety_refusal_suffix(self):
        out = _announce(reason=FailoverReason.content_policy_blocked)
        assert len(out) == 1
        assert "🔄 Model fallback (safety refusal): claude-app/claude-fable-5 → claude-api-proxy-f1/claude-opus-4-8" == out[0]

    def test_rate_limit_suffix(self):
        out = _announce(reason=FailoverReason.rate_limit)
        assert "(rate limit)" in out[0]
        assert "Model fallback (rate limit):" in out[0]

    def test_no_reason_bare_line_unchanged(self):
        out = _announce(reason=None)
        assert out[0].startswith("🔄 Model fallback: ")
        assert "(" not in out[0].split(":")[0]  # no suffix in the verb segment

    def test_unmapped_reason_shows_generic_floor(self):
        # A classified-but-unmapped reason renders the "(connection issue)" floor
        # rather than a bare line — the announce always says WHY.
        out = _announce(reason=FailoverReason.unknown)
        assert "🔄 Model fallback (connection issue): " in out[0]

    def test_timeout_shows_connection_dropped(self):
        # The exact 2026-07-10 class: a peer-closed stream drop (→ timeout) now
        # names itself instead of a bare "Model fallback: A → B".
        out = _announce(reason=FailoverReason.timeout)
        assert "Model fallback (connection dropped):" in out[0]

    def test_recovery_never_shows_reason(self):
        # A recovery-to-primary must not render a reason suffix even if one is passed.
        out = _announce(reason=FailoverReason.content_policy_blocked, kind="recovery")
        assert out and out[0].startswith("🔄 Model recovery: ")
        assert "safety refusal" not in out[0]


class TestStreamErrorReasonThreading:
    """The reason-resolution invariant behind naming WHY a stream-drop failover
    changed the route (the 2026-07-10 peer-closed drop that fired a bare line).
    Tests the pure resolver directly so explicit-wins / backfill / consume-once
    are asserted on the EFFECTIVE reason, not just a cleared stamp."""

    def _agent(self, pending):
        a = _StubAgent()
        a._pending_stream_error_reason = pending
        return a

    def test_reasonless_call_backfills_from_stamp(self):
        from agent.chat_completion_helpers import _resolve_failover_reason
        a = self._agent(FailoverReason.timeout)
        # reason=None (the loop's stub failover sites) → backfilled from the stamp
        eff = _resolve_failover_reason(a, None)
        assert eff == FailoverReason.timeout          # the announce will name it
        assert a._pending_stream_error_reason is None  # consume-once

    def test_explicit_reason_wins_over_stamp(self):
        from agent.chat_completion_helpers import _resolve_failover_reason
        a = self._agent(FailoverReason.timeout)
        # An explicit reason at the call site must be the EFFECTIVE reason —
        # the stamp must not override it (this is the invariant the name promises).
        eff = _resolve_failover_reason(a, FailoverReason.rate_limit)
        assert eff == FailoverReason.rate_limit
        assert a._pending_stream_error_reason is None  # stamp still consumed

    def test_reasonless_call_no_stamp_returns_none(self):
        from agent.chat_completion_helpers import _resolve_failover_reason
        a = self._agent(None)
        eff = _resolve_failover_reason(a, None)
        assert eff is None                             # → bare line, correctly
        assert a._pending_stream_error_reason is None

    def test_backfilled_reason_renders_rider_end_to_end(self):
        # The stamped reason, once resolved, must produce a real announce rider.
        from agent.chat_completion_helpers import _resolve_failover_reason
        a = self._agent(FailoverReason.timeout)
        eff = _resolve_failover_reason(a, None)
        out = _announce(reason=eff)
        assert "Model fallback (connection dropped):" in out[0]
