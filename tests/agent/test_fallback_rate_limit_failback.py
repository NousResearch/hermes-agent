"""Runtime failover must tell the chat, and a fallback 429 must try the primary
before sleeping its cooldown (#141025)."""

import time
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_runtime_helpers import (
    primary_restored_notice, try_failback_before_fallback_rate_limit,
)
from agent.error_classifier import FailoverReason
from hermes_cli.fallback_config import pre_agent_fallback_notice, runtime_fallback_notice


def test_runtime_notice_matches_the_pre_agent_sentence():
    notice = runtime_fallback_notice("custom", "local-brain", "nous", "step-5-preview:free")
    assert notice == pre_agent_fallback_notice("custom", "local-brain", "nous", "step-5-preview:free")
    cooled = runtime_fallback_notice(
        "custom", "local-brain", "nous", "step-5-preview:free", cooldown_remaining=60,
    )
    assert cooled.startswith(notice)
    assert "Primary retry eligible in ~60 s" in cooled


def test_rate_limit_failback_prefixes_the_restored_line():
    agent = SimpleNamespace(model="local-brain", provider="custom")
    plain = primary_restored_notice(agent, "step-5-preview:free", "nous", from_fallback_rate_limit=False)
    probed = primary_restored_notice(agent, "step-5-preview:free", "nous", from_fallback_rate_limit=True)
    assert plain.startswith("✅ Primary model restored:")
    assert probed.startswith("⚠️ Fallback rate-limited, retrying primary now.")
    assert "Primary model restored" in probed


def _agent(*, active=True, probe_at=0.0, restore=True):
    agent = SimpleNamespace(
        _provider_fallback_active=active,
        _fallback_activated=active,
        _fallback_primary_probe_at=probe_at,
        _failback_from_fallback_rate_limit=False,
        restore_calls=0,
    )

    def _restore():
        agent.restore_calls += 1
        assert agent._failback_from_fallback_rate_limit is True
        if not restore:
            return False
        agent._provider_fallback_active = False
        return True

    agent._restore_primary_runtime = _restore
    return agent


def test_fallback_rate_limit_probes_the_primary_once():
    agent = _agent()
    assert try_failback_before_fallback_rate_limit(agent) is True
    assert agent.restore_calls == 1
    assert agent._fallback_primary_probe_at > 0

    again = _agent(probe_at=time.monotonic())
    assert try_failback_before_fallback_rate_limit(again) is False
    assert again.restore_calls == 0


def test_primary_still_cooling_down_is_not_stamped():
    agent = _agent(restore=False)
    assert try_failback_before_fallback_rate_limit(agent) is False
    assert agent.restore_calls == 1
    assert agent._fallback_primary_probe_at == 0
    assert agent._failback_from_fallback_rate_limit is False


def test_live_switch_notice_is_not_repeated_when_the_turn_succeeds():
    from agent.chat_completion_helpers import _buffer_fallback_notice
    from agent.status_output import StatusOutputMixin

    class _Emitter(StatusOutputMixin):
        def __init__(self):
            self.sent = []

        def _buffer_diagnostic_status(self, message):
            return None

        def _emit_status(self, message):
            self.sent.append(message)

        def _emit_diagnostic_status(self, message):
            self.sent.append(message)

    agent = _Emitter()
    notice = runtime_fallback_notice("custom", "local-brain", "nous", "step-5-preview:free")
    _buffer_fallback_notice(agent, notice)
    agent._emit_pending_fallback_notice()
    assert agent.sent == [notice]
    assert agent._pending_fallback_notice is None


def test_idle_primary_is_not_probed():
    agent = _agent(active=False)
    assert try_failback_before_fallback_rate_limit(agent) is False
    assert agent.restore_calls == 0


def test_settle_breaks_to_primary_instead_of_sleeping_the_fallback_429():
    from agent.turn_api_error import settle_unrecovered_error

    agent = _agent()
    slept = []
    classified = SimpleNamespace(
        reason=FailoverReason.rate_limit, retryable=True, should_compress=False, should_fallback=True,
    )
    retry = SimpleNamespace(
        reasoning_mandatory_retry_attempted=False, image_shrink_retry_attempted=False,
        copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=False,
        restart_with_redirected_messages=False, restart_with_rebuilt_messages=False,
    )
    err = Exception("rate limited")
    with patch("agent.turn_api_error.free_tier_cooldown_ends_turn", return_value=False), patch(
        "agent.turn_api_error.compute_error_backoff", side_effect=lambda *a, **k: slept.append(True),
    ), patch(
        "agent.turn_api_error.interruptible_backoff_sleep", return_value=None,
    ):
        verdict = settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=429,
            error_msg="rate limited", is_context_length_error=False, is_rate_limited=True,
            _is_zai_coding_overload=False, _provider="nous", _base="https://inference-api.nousresearch.com/v1",
            _model="step-5-preview:free", messages=[], api_messages=[{"role": "user", "content": "hi"}],
            api_kwargs={}, active_system_prompt="prompt", conversation_history=None, approx_tokens=10,
            retry_count=1, max_retries=3, compression_attempts=0, api_call_count=2,
        )
    assert verdict.action == "break"
    assert verdict.retry_count == 0
    assert agent.restore_calls == 1
    assert retry.restart_with_rebuilt_messages is True
    assert slept == []
