"""Bounded post-exhaustion auto-recovery ladder (#85426, #107307): fallback first, transient
outages only, visible on every surface, interrupt cancels."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_recovery_autorecover import ladder_wait_seconds
from agent.turn_retry_state import TurnRetryState


class _Err(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = SimpleNamespace(headers=headers or {})
        self.body = None


class _Agent:
    """Real seams: fallback state, the status/wait callbacks the surfaces read, the streamed-text
    gate and the interrupt flag; everything else the terminal path touches is a no-op."""
    log_prefix = ""
    verbose = False
    provider = "custom"
    platform = "cli"
    _credential_pool = None
    _current_streamed_assistant_text = ""
    _interrupt_requested = False
    providers_order: list[str] | None = None
    providers_allowed = None
    providers_ignored = None
    provider_sort = None
    provider_require_parameters = None
    provider_data_collection = None
    allow_provider_fallbacks = None
    sleep_outcome = None  # what the (patched) interruptible wait returns

    def __init__(self, *, fallback_left=False, cycles=5, platform="cli"):
        self.statuses, self.waits, self.activated = [], [], []
        self._fallback_left = fallback_left
        self._auto_recovery_cycles = cycles
        self.platform = platform

    def _has_pending_fallback(self):
        return self._fallback_left

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(True)
        return self._fallback_left

    def _try_recover_primary_transport(self, *a, **k):
        return False

    def _recover_with_credential_pool(self, **kw):
        return False, kw.get("has_retried_429", False)

    def _has_content_after_think_block(self, text):
        return bool(text.strip())

    def _emit_diagnostic_status(self, text):
        self.statuses.append(str(text))

    def _emit_diagnostic_wait(self, text):
        self.waits.append(str(text))

    def _summarize_api_error(self, error):
        return str(error)

    def _client_log_context(self):
        return ""

    def clear_interrupt(self, **kwargs):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, retry=None):
    retry = retry or TurnRetryState(primary_recovery_attempted=True)
    classified = classify_api_error(err, provider=agent.provider)
    with patch("agent.conversation_loop._credential_refresh_profile", lambda a: None), \
            patch("agent.conversation_loop._arm_fallback_restart", lambda a, m, s, r: s), \
            patch("agent.turn_recovery.interruptible_backoff_sleep", lambda agent, *a, **k: agent.sleep_outcome), \
            patch("agent.turn_api_error.max_retries_exhausted_result", lambda *a, **k: {"failed": True}), \
            patch("agent.turn_api_error.nonretryable_client_error_result", lambda *a, **k: {"failed": True, "nonretryable": True}):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=err.status_code,
            error_msg=str(err).lower(), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=agent.provider, _base="http://x", _model="m",
            messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None,
            conversation_history=None, approx_tokens=0, retry_count=3, max_retries=3,
            compression_attempts=0, api_call_count=1,
        ), retry


@pytest.mark.parametrize("status,message", [(503, "Service temporarily unavailable"), (529, "Overloaded")])
def test_transient_exhaustion_without_fallback_enters_ladder_and_retries(status, message):
    """No fallback left + 5xx/overloaded + nothing delivered -> one visible cycle, then retry from 0."""
    agent = _Agent()
    verdict, retry = _settle(agent, _Err(status, message))
    assert (verdict.action, verdict.retry_count, retry.auto_recovery_cycles_used) == ("continue", 0, 1)
    assert agent.statuses == agent.waits and len(agent.statuses) == 1
    line = agent.statuses[0]
    assert "(cycle 1/5)" in line


def test_ladder_is_bounded_and_yields_to_fallback_and_shuns_nonretryable():
    agent = _Agent()
    spent = TurnRetryState(primary_recovery_attempted=True, auto_recovery_cycles_used=5)
    verdict, _ = _settle(agent, _Err(503, "down"), retry=spent)
    assert verdict.action == "return" and verdict.result == {"failed": True}
    assert any("gave up after 5 cycles" in s for s in agent.statuses)

    # Fallback first: a remaining fallback engages and the ladder never runs.
    fb = _Agent(fallback_left=True)
    verdict, retry = _settle(fb, _Err(503, "down"))
    assert (verdict.action, fb.activated, retry.auto_recovery_cycles_used) == ("break", [True], 0)

    # Non-retryable classes never enter (401 here) and delivered text is never replayed.
    auth = _Agent()
    verdict, retry = _settle(auth, _Err(401, "Incorrect API key provided"))
    assert verdict.result.get("nonretryable") and retry.auto_recovery_cycles_used == 0
    streamed = _Agent()
    streamed._current_streamed_assistant_text = "Here is the first half of the answer"
    verdict, retry = _settle(streamed, _Err(503, "down"))
    assert (verdict.action, retry.auto_recovery_cycles_used) == ("return", 0)

    # Disabled by config.
    off = _Agent(cycles=0)
    verdict, retry = _settle(off, _Err(503, "down"))
    assert (verdict.action, retry.auto_recovery_cycles_used, off.statuses) == ("return", 0, [])


def test_ladder_schedule_honours_retry_after_and_platform_stop_hint(monkeypatch):
    from agent import retry_utils
    calls = []

    def _recording_backoff(attempt, *, base_delay, max_delay, jitter_ratio):
        calls.append((attempt, base_delay, max_delay))
        return min(base_delay * 2 ** (attempt - 1), max_delay)
    monkeypatch.setattr(retry_utils, "jittered_backoff", _recording_backoff)

    plain = _Err(503, "down")
    assert [ladder_wait_seconds(c, plain) for c in (1, 2, 3, 4, 5)] == [15, 30, 60, 60, 60]
    assert calls[0] == (1, 15.0, 60.0)
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "4"})) == 4
    # Past the 60 s cap the provider's own cooldown is honoured, up to 120 s.
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "90"})) == 90
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "900"})) == 120

    tg = _Agent(platform="telegram")
    _settle(tg, plain)
    assert "send /stop to cancel" in tg.statuses[0]
    cron = _Agent(platform="cron")
    _settle(cron, plain)
    assert cron.statuses[0].endswith("(cycle 1/5)")


def test_interrupt_during_ladder_wait_returns_interrupted_result():
    agent = _Agent()
    agent.sleep_outcome = {"interrupted": True, "completed": False}
    verdict, retry = _settle(agent, _Err(503, "down"))
    assert verdict.action == "return" and verdict.result["interrupted"] is True
    assert retry.auto_recovery_cycles_used == 1


def test_terminal_fallback_429_wraps_back_to_primary_runtime_once():
    """CF-60: when already on the terminal fallback provider (_fallback_activated=True,
    no pending fallback left) and retries exhaust on a 429 (e.g. Vertex TPM quota exhausted),
    settle_unrecovered_error restores the primary runtime once and breaks to restart."""
    agent = _Agent(fallback_left=False)
    agent._fallback_activated = True
    agent._rate_limited_until = 999999999.0
    restored = []
    agent._restore_primary_runtime = lambda: restored.append(True) or True

    verdict, retry = _settle(agent, _Err(429, "TPM quota exhausted. No spare capacity is currently available"))
    assert verdict.action == "break"
    assert verdict.retry_count == 0
    assert retry.primary_fallback_wrap_attempted is True
    assert agent._rate_limited_until == 0
    assert restored == [True]

    # Second exhaustion on the same call must NOT infinite-loop wrap-back
    verdict2, _ = _settle(agent, _Err(429, "TPM quota exhausted"), retry=retry)
    assert verdict2.action == "return" and verdict2.result == {"failed": True}
    assert restored == [True]


def test_openrouter_upstream_429_records_provider_failure_and_reorders_upstreams():
    """CF-60: when OpenRouter returns a 429 from an upstream provider (e.g. Google AI Studio),
    provider_latency_circuit_breaker.record_provider_failure demotes that upstream so
    adjust_provider_order moves google-vertex/global to slot #1."""
    from agent.provider_latency_circuit_breaker import (
        adjust_provider_order,
        record_provider_failure,
        reset_circuit_breaker,
    )

    reset_circuit_breaker()
    try:
        order = ["google-ai-studio/flex", "google-vertex/global", "google-ai-studio"]
        assert adjust_provider_order("google/gemini-4-argon", order) == order

        record_provider_failure("google/gemini-4-argon", "Google AI Studio")
        reordered = adjust_provider_order("google/gemini-4-argon", order)
        assert reordered == ["google-vertex/global", "google-ai-studio/flex", "google-ai-studio"]
    finally:
        reset_circuit_breaker()


def test_openrouter_upstream_429_retries_reordered_upstream_before_cross_provider_fallback():
    """When an unwrapped OpenRouter 429 demotes Google AI Studio and promotes google-vertex/global
    in extra_body.provider.order, attempt 1/3 retries on OpenRouter with the reordered order before
    activating cross-provider fallback; if the reordered order also 429s, fallback activates."""
    from agent.provider_latency_circuit_breaker import is_provider_demoted, reset_circuit_breaker
    from agent.turn_api_error import handle_api_error

    reset_circuit_breaker()
    try:
        agent = _Agent(fallback_left=True)
        agent.provider = "openrouter"
        agent.model = "google/gemini-4-argon"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._fallback_index = 0
        agent._fallback_chain = [{"provider": "vertex", "model": "google/gemini-4-argon"}]
        agent.providers_order = ["google-ai-studio/flex", "google-vertex/global", "google-ai-studio"]
        agent._extract_api_error_context = lambda err: {"status_code": 429}

        err1 = _Err(429, "Error code: 429 - Provider returned error")
        err1.body = {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "provider_name": "Google AI Studio",
                "limit_source": "upstream_provider_shared_pool",
            },
        }
        api_kwargs = {
            "extra_body": {
                "provider": {
                    "order": ["google-ai-studio/flex", "google-vertex/global", "google-ai-studio"]
                }
            }
        }
        retry = TurnRetryState()
        with patch("agent.turn_api_error.recover_before_classification", lambda *a, **k: (False, None)), \
                patch("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **k: None):
            v1 = handle_api_error(
                agent,
                api_error=err1,
                _retry=retry,
                messages=[],
                api_messages=[],
                api_kwargs=api_kwargs,
                active_system_prompt=None,
                conversation_history=None,
                approx_tokens=50000,
                retry_count=0,
                max_retries=3,
                compression_attempts=0,
                max_compression_attempts=2,
                api_call_count=1,
                thinking_spinner=None,
                system_message="",
                api_request_id=None,
                api_start_time=0.0,
                effective_task_id=None,
                turn_id="t1",
            )

        assert is_provider_demoted("google/gemini-4-argon", "google-ai-studio/flex")
        assert api_kwargs["extra_body"]["provider"]["order"] == [
            "google-vertex/global",
            "google-ai-studio/flex",
            "google-ai-studio",
        ]
        assert v1.action == "fallthrough"
        assert v1.retry_count == 1
        assert agent.activated == []

        # If attempt 2/3 also 429s (either on Google or with no further reorder available),
        # cross-provider fallback activates immediately.
        err2 = _Err(429, "Error code: 429 - Provider returned error")
        err2.body = {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "provider_name": "Google",
                "limit_source": "upstream_provider_shared_pool",
            },
        }
        with patch("agent.turn_api_error.recover_before_classification", lambda *a, **k: (False, None)), \
                patch("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **k: None):
            v2 = handle_api_error(
                agent,
                api_error=err2,
                _retry=retry,
                messages=[],
                api_messages=[],
                api_kwargs=api_kwargs,
                active_system_prompt=None,
                conversation_history=None,
                approx_tokens=50000,
                retry_count=v1.retry_count,
                max_retries=3,
                compression_attempts=0,
                max_compression_attempts=2,
                api_call_count=1,
                thinking_spinner=None,
                system_message="",
                api_request_id=None,
                api_start_time=0.0,
                effective_task_id=None,
                turn_id="t1",
            )

        assert is_provider_demoted("google/gemini-4-argon", "google-vertex/global")
        assert v2.action == "break"
        assert agent.activated == [True]
    finally:
        reset_circuit_breaker()


def test_route_classified_error_does_not_eagerly_wrap_back_during_inflight_rebuild_or_sticky_content_filter():
    """CF-74 (RC-2 & RC-4): when on terminal fallback, route_classified_error must NOT
    eagerly wrap back to primary on attempt 1/3 when _rebuilt_restarts_since_response > 0
    (primary just failed in this same unresponded step) or when _content_filter_fallback_sticky
    is True (primary failed on 403 content_filter); when both are False/0 and cooldown expired,
    it wraps back once."""
    from agent.error_classifier import FailoverReason, classify_api_error
    from agent.turn_recovery import route_classified_error

    def _run_route(rebuilt_restarts: int, sticky_cf: bool):
        agent = _Agent(fallback_left=False)
        agent.provider = "vertex"
        agent.model = "google/gemini-4-argon"
        agent.base_url = "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._fallback_chain = [{"provider": "vertex", "model": "google/gemini-4-argon"}]
        agent._rate_limited_until = 0
        agent._rebuilt_restarts_since_response = rebuilt_restarts
        agent._content_filter_fallback_sticky = sticky_cf
        restored = []
        agent._restore_primary_runtime = lambda: restored.append(True) or True
        err = _Err(429, "Resource exhausted (DSQ 429)")
        classified = classify_api_error(err, provider="vertex", model="google/gemini-4-argon")
        assert classified.reason == FailoverReason.rate_limit
        retry = TurnRetryState()
        verdict = route_classified_error(
            agent,
            err,
            classified,
            retry,
            error_msg=str(err),
            error_context={"status_code": 429},
            recovered_with_pool=False,
            base_url=agent.base_url,
            model=agent.model,
            messages=[],
            api_messages=[],
            system_message=None,
            active_system_prompt=None,
            conversation_history=[],
            retry_count=0,
            max_retries=3,
            compression_attempts=0,
            max_compression_attempts=2,
            api_call_count=1,
            effective_task_id=None,
        )
        return verdict, retry, restored

    # Negative control A: in-flight rebuild chain (_rebuilt_restarts_since_response=1) must fallthrough to retry Vertex
    v_inflight, retry_inflight, restored_inflight = _run_route(rebuilt_restarts=1, sticky_cf=False)
    assert v_inflight.action == "fallthrough"
    assert restored_inflight == []
    assert retry_inflight.primary_fallback_wrap_attempted is False

    # Negative control B: sticky content filter (_content_filter_fallback_sticky=True) must fallthrough to retry Vertex
    v_sticky, retry_sticky, restored_sticky = _run_route(rebuilt_restarts=0, sticky_cf=True)
    assert v_sticky.action == "fallthrough"
    assert restored_sticky == []
    assert retry_sticky.primary_fallback_wrap_attempted is False

    # Positive control: steady-state fallback with expired cooldown wraps back once
    v_ok, retry_ok, restored_ok = _run_route(rebuilt_restarts=0, sticky_cf=False)
    assert v_ok.action == "break"
    assert restored_ok == [True]
    assert retry_ok.primary_fallback_wrap_attempted is True
    assert retry_ok.restart_with_rebuilt_messages is True


def test_openrouter_gemini_corrupted_thought_signature_and_preserved_reasoning_classify_as_thinking_signature():
    """CF-74 (RC-3): OpenRouter/Gemini cross-upstream thoughtSignature rejections (both HTTP 400
    'Corrupted thought signature.' and status-less SSE stream error 'Gemini models require OpenRouter
    reasoning details to be preserved in each request.') classify as FailoverReason.thinking_signature
    and trigger one-shot reasoning_details strip in _recover_format_errors."""
    import copy
    from agent.error_classifier import FailoverReason, classify_api_error
    from agent.turn_recovery import _recover_format_errors

    err_400 = _Err(
        400,
        "Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, "
        "'metadata': {'raw': 'Corrupted thought signature.'}}}",
    )
    c_400 = classify_api_error(err_400, provider="openrouter", model="google/gemini-4-argon")
    assert c_400.reason == FailoverReason.thinking_signature

    err_sse = Exception(
        "Gemini models require OpenRouter reasoning details to be preserved in each request. "
        "Please refer to our docs: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens"
    )
    c_sse = classify_api_error(err_sse, provider="openrouter", model="google/gemini-4-argon")
    assert c_sse.reason == FailoverReason.thinking_signature

    # Negative control: unrelated 400 stays format_error
    err_other = _Err(400, "Invalid JSON payload received")
    c_other = classify_api_error(err_other, provider="openrouter", model="google/gemini-4-argon")
    assert c_other.reason == FailoverReason.format_error

    # Verify _recover_format_errors strips reasoning_details from api_messages without mutating canonical messages
    canonical = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "corrupted", "id": "c1"}],
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    before = copy.deepcopy(canonical)
    api_msgs = copy.deepcopy(canonical)
    retry = TurnRetryState()
    agent = _Agent()
    agent.api_mode = "chat_completions"
    assert _recover_format_errors(agent, err_sse, c_sse, retry, canonical, api_msgs) is True
    assert retry.thinking_sig_retry_attempted is True
    assert "reasoning_details" not in api_msgs[1]
    assert canonical == before


def test_merge_assistant_into_carries_reasoning_details_for_appended_tool_calls():
    """CF-74 (RC-3b): when _merge_assistant_into folds a tool-call assistant turn into a preceding
    text-only assistant turn, it preserves the tool-call turn's reasoning_details."""
    from agent.agent_runtime_helpers import _merge_assistant_into

    prev = {
        "role": "assistant",
        "content": "Text preface",
        "reasoning_details": [{"type": "reasoning.text", "text": "plan", "index": 0}],
        "_db_persisted": True,
    }
    later = {
        "role": "assistant",
        "content": "",
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "sig_c1", "id": "call_1", "format": "google-gemini-v1", "index": 0}
        ],
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "skill_view", "arguments": "{}"}}],
    }
    _merge_assistant_into(prev, later)
    assert len(prev["reasoning_details"]) == 2
    assert any(d.get("id") == "call_1" for d in prev["reasoning_details"])
    assert "_db_persisted" not in prev




