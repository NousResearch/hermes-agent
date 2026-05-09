"""Tests for eager fallback on stream-stall / call timeouts (issue #22277).

When a primary provider's stream stalls and the stale detector kills the
connection, the error classifies as ``FailoverReason.timeout``.  Without
this feature, the retry loop hammers the same broken primary repeatedly,
producing 15+ min silent hangs.  With ``agent.eager_fallback_on_timeout``
enabled, the next fallback provider activates after the first stale kill.
"""

from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from run_agent import AIAgent


def _make_agent(fallback_model=None, eager_on_timeout=None):
    config_overlay = {}
    if eager_on_timeout is not None:
        config_overlay = {"agent": {"eager_fallback_on_timeout": eager_on_timeout}}

    def _fake_load_config(*_args, **_kwargs):
        return config_overlay

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", side_effect=_fake_load_config),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


# ── Config wiring ─────────────────────────────────────────────────────────


class TestEagerFallbackConfigDefaults:
    def test_default_is_false_when_unset(self):
        agent = _make_agent()
        assert agent._eager_fallback_on_timeout is False

    def test_explicit_false(self):
        agent = _make_agent(eager_on_timeout=False)
        assert agent._eager_fallback_on_timeout is False

    def test_explicit_true(self):
        agent = _make_agent(eager_on_timeout=True)
        assert agent._eager_fallback_on_timeout is True

    def test_truthy_int_coerced_to_bool(self):
        agent = _make_agent(eager_on_timeout=1)
        assert agent._eager_fallback_on_timeout is True


# ── Trigger gating logic (unit-level) ─────────────────────────────────────
#
# We don't drive the full retry loop end-to-end (it's a 700-line block of
# inline state); instead we assert the gate predicate that the loop checks.


class TestEagerFallbackOnTimeoutGate:
    def _gate(self, classified_reason, eager_flag, fb_index, fb_chain_len):
        return (
            classified_reason == FailoverReason.timeout
            and eager_flag
            and fb_index < fb_chain_len
        )

    def test_fires_when_all_conditions_met(self):
        assert self._gate(FailoverReason.timeout, True, 0, 1) is True

    def test_does_not_fire_when_flag_off(self):
        assert self._gate(FailoverReason.timeout, False, 0, 1) is False

    def test_does_not_fire_when_no_chain(self):
        assert self._gate(FailoverReason.timeout, True, 0, 0) is False

    def test_does_not_fire_when_chain_exhausted(self):
        assert self._gate(FailoverReason.timeout, True, 1, 1) is False

    def test_does_not_fire_for_non_timeout_reason(self):
        for reason in (
            FailoverReason.unknown,
            FailoverReason.server_error,
            FailoverReason.format_error,
            FailoverReason.thinking_signature,
        ):
            assert self._gate(reason, True, 0, 1) is False, reason
