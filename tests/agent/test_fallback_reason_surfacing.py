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

    def test_unmapped_reason_none(self):
        assert _fallback_reason_label(FailoverReason.unknown) is None
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

    def test_unmapped_reason_bare_line(self):
        out = _announce(reason=FailoverReason.unknown)
        assert out[0].startswith("🔄 Model fallback: ")

    def test_recovery_never_shows_reason(self):
        # A recovery-to-primary must not render a reason suffix even if one is passed.
        out = _announce(reason=FailoverReason.content_policy_blocked, kind="recovery")
        assert out and out[0].startswith("🔄 Model recovery: ")
        assert "safety refusal" not in out[0]
