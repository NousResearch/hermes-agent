"""Two coding-plan overload invariants across model aliases and quota errors.

These phase-boundary tests control unrelated recovery and side effects.
The routing invariant uses the imported classifier on structured 429 bodies. The detector, routing, settlement and backoff functions run unchanged.
They supplement, and do not replace, the required full AIAgent integration run.
"""
from types import SimpleNamespace

from unittest.mock import Mock

import pytest

import agent.conversation_loop as loop

import agent.retry_utils as retry

import agent.turn_api_error as errors

import agent.turn_recovery as recovery

from agent.error_classifier import ClassifiedError, FailoverReason, classify_api_error

CODING_URL = "https://api.z.ai/api/coding/paas/v4"

MODELS = ["glm-5.2", "glm-5.3", "z-ai/glm-5.3", "opaque-alias", None]

class ProviderError(Exception):
    def __init__(self, message="The service may be temporarily overloaded", *, status=429,
                 code="1305", reason=FailoverReason.overloaded, retry_after=None):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"message": message, "code": code}}
        self.reason = reason
        self.response = SimpleNamespace(headers={} if retry_after is None else {"Retry-After": retry_after})

class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.after_sleep = None

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.after_sleep:
            self.after_sleep()

def classified(error):
    return ClassifiedError(reason=error.reason, status_code=error.status_code,
                           retryable=error.reason not in {FailoverReason.billing, FailoverReason.auth},
                           should_fallback=True)

@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    states = SimpleNamespace(
        auth_failover_attempted=False, primary_recovery_attempted=False,
        has_retried_429=False, restart_with_redirected_messages=False,
        restart_with_rebuilt_messages=False, copilot_stale_cred_retry_attempted=False,
    )
    agent = SimpleNamespace(
        model="glm-5.3", provider="zai", base_url=CODING_URL, log_prefix="",
        thinking_callback=None, _interrupt_requested=False, _credential_pool=None,
        _fallback_index=0, _fallback_chain=["fallback"], _fallback_activated=False,
        _buffer_status=Mock(), _emit_status=Mock(), _emit_wait_notice=Mock(),
        _buffer_vprint=Mock(), _vprint=Mock(), _persist_session=Mock(),
        _touch_activity=Mock(), _invoke_api_request_error_hook=Mock(),
        _extract_api_error_context=lambda error: {}, _client_log_context=lambda: "test-client",
        _try_recover_primary_transport=Mock(return_value=False),
        _clean_error_message=str, _flush_status_buffer=Mock(),
    )
    agent._has_pending_fallback = lambda: agent._fallback_index < len(agent._fallback_chain)

    def activate(**kwargs):
        if not agent._has_pending_fallback():
            return False
        agent._fallback_index += 1
        agent._fallback_activated = True
        agent.provider, agent.model, agent.base_url = "other", "fallback", "https://fallback.invalid/v1"
        return True

    agent._try_activate_fallback = Mock(side_effect=activate)
    agent.clear_interrupt = Mock(return_value=False)

    def arm(agent, messages, prompt, state):
        state.restart_with_rebuilt_messages = True
        return "rebuilt-system"

    arm_mock = Mock(side_effect=arm)
    pool_mock = Mock(return_value=False)
    monkeypatch.setattr(loop, "_arm_fallback_restart", arm_mock)
    monkeypatch.setattr(loop, "_ra", lambda: SimpleNamespace(_pool_may_recover_from_rate_limit=pool_mock))
    monkeypatch.setattr(loop, "_is_copilot_provider", lambda a: False)
    monkeypatch.setattr(loop, "_is_stale_copilot_credential_error", lambda *args: False)
    monkeypatch.setattr(recovery, "time", clock)
    monkeypatch.setattr(errors, "time", clock)
    monkeypatch.setattr(retry, "jittered_backoff", lambda attempt, **kw:
                        min(kw["base_delay"] * 2 ** (attempt - 1), kw["max_delay"]))
    monkeypatch.setattr(recovery, "is_output_cap_error", lambda message: False)
    monkeypatch.setattr(recovery, "parse_available_output_tokens_from_error", lambda message: None)
    monkeypatch.setattr(errors, "classify_api_error", lambda error, **kwargs: classified(error))
    monkeypatch.setattr(errors, "recover_before_classification", lambda a, e, **kw:
                        (False, kw["active_system_prompt"]))
    monkeypatch.setattr(errors, "recover_after_classification", lambda *args, **kw: (False, False))
    monkeypatch.setattr(errors, "log_api_error_attempt", lambda a, e, **kw:
                        (type(e).__name__, str(e).lower(), a.provider, a.base_url, a.model))
    monkeypatch.setattr(errors, "recover_from_overflow", lambda a, e, c, s, **kw: SimpleNamespace(
        messages=kw["messages"], active_system_prompt=kw["active_system_prompt"],
        conversation_history=kw["conversation_history"], approx_tokens=kw["approx_tokens"],
        compression_attempts=kw["compression_attempts"], is_context_length_error=False,
        provider_overflow_recovery_pending=False, action="fallthrough"))
    monkeypatch.setattr(errors, "max_retries_exhausted_result", lambda *args, **kwargs: {"exhausted": True})
    monkeypatch.setattr(errors, "nonretryable_client_error_result", lambda *args, **kwargs: {"nonretryable": True})
    waits = []
    real_backoff = recovery.compute_error_backoff

    def record(*args, **kwargs):
        result = real_backoff(*args, **kwargs)
        waits.append(result)
        return result

    monkeypatch.setattr(errors, "compute_error_backoff", record)
    return SimpleNamespace(agent=agent, state=states, clock=clock, waits=waits, arm=arm_mock, pool=pool_mock)

def handle(rig, error, count=0, maximum=2):
    return errors.handle_api_error(
        rig.agent, api_error=error, _retry=rig.state, thinking_spinner=None, messages=[],
        api_messages=[], api_kwargs={}, system_message={}, active_system_prompt="system",
        conversation_history=[], approx_tokens=10, retry_count=count, max_retries=maximum,
        compression_attempts=0, max_compression_attempts=3, api_call_count=count + 1,
        api_request_id="test-request", api_start_time=rig.clock.time(), effective_task_id="test", turn_id="test",
    )

@pytest.mark.parametrize("model", MODELS)
def test_coding_overload_detection_is_model_independent_and_narrow(model):
    for message, code in [
        ("busy", "1305"),
        ("该模型当前访问量过大，请您稍后再试", "1305"),
        ("temporarily overloaded", "other"),
    ]:
        assert retry.is_zai_coding_overload_error(
            base_url=CODING_URL, model=model, error=ProviderError(message, code=code))
    for url, status, message, code in [
        (CODING_URL, 429, "quota exceeded", "quota"),
        (CODING_URL, 429, "insufficient credits", "billing"),
        ("https://api.z.ai/api/v4", 429, "temporarily overloaded", "1305"),
        ("https://other.invalid/v1", 429, "temporarily overloaded", "1305"),
        (None, 429, "temporarily overloaded", "1305"),
        (CODING_URL, 500, "temporarily overloaded", "1305"),
    ]:
        assert not retry.is_zai_coding_overload_error(
            base_url=url, model=model, error=ProviderError(message, status=status, code=code))


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("message,code,should_wait", [
    ("The service may be temporarily overloaded", "1305", True),
    ("该模型当前访问量过大，请您稍后再试", "1305", True),
    ("", "1305", True),
    ("rate limit exceeded", "rate_limit", False),
    ("insufficient credits", "billing", False),
    ("insufficient credits", "1305", False),
])
def test_overload_budget_precedes_fallback_but_quota_remains_immediate(
    rig, monkeypatch, model, message, code, should_wait,
):
    rig.agent.model = model
    error = ProviderError(message, code=code)
    # Use the production classifier binding, not the fixture's synthetic verdict.
    monkeypatch.setattr(errors, "classify_api_error", classify_api_error)
    reason = classify_api_error(error, provider="zai", model=model or "").reason
    ceiling = retry.zai_coding_overload_retry_ceiling()
    count, maximum = 0, 2
    for failure in range(1, ceiling + 1):
        result = handle(rig, error, count, maximum)
        count, maximum = result.retry_count, result.max_retries
        if result.action == "break":
            break
        assert result.action == "fallthrough"
        rig.agent._try_activate_fallback.assert_not_called()
    assert result.action == "break"
    assert failure == (ceiling if should_wait else 1)
    assert rig.waits == ([2.0, 4.0, 8.0, 30.0, 60.0, 90.0, 120.0] if should_wait else [])
    rig.agent._try_activate_fallback.assert_called_once_with(reason=reason)
    assert rig.state.restart_with_rebuilt_messages
    assert result.retry_count == result.compression_attempts == 0
    assert result.active_system_prompt == "rebuilt-system"
    rig.agent._try_recover_primary_transport.assert_not_called()
