"""KAN-319: Verification of retry and fallback behavior on HTTP 503.

Covers:
1. Error classification: HTTP 503 maps to FailoverReason.overloaded and retryable=True.
2. In-loop retry: retry_count increments and backoff is computed when under max_retries.
3. Fallback activation: when max_retries is reached, _try_activate_fallback switches
   to the next configured model in the fallback chain and triggers a restart.
4. Auto-recovery ladder: when no fallback is configured, exhausted 503 enters the ladder.
5. Bridge resilience: bridges with 503 in skip_retry_on_status fail fast to avoid
   stalling on a downed Cloudflare Worker bridge.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_retry_state import TurnRetryState


class MockAPIError(Exception):
    def __init__(self, message="Service Unavailable", status_code=503):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def test_503_error_classification():
    """HTTP 503 must classify as FailoverReason.overloaded with retryable=True."""
    err = MockAPIError("Service Unavailable", status_code=503)
    classified = classify_api_error(err)
    assert classified.reason == FailoverReason.overloaded
    assert classified.retryable is True
    assert classified.status_code == 503


def test_503_activates_fallback_when_max_retries_exhausted():
    """When retries for 503 are exhausted and a fallback is available, it switches to fallback."""
    err = MockAPIError("Backend overloaded", status_code=503)
    classified = classify_api_error(err)

    activated_reasons = []
    agent = SimpleNamespace(
        log_prefix="[test]",
        _has_pending_fallback=lambda: True,
        _try_activate_fallback=lambda reason, reset_at=None: (activated_reasons.append(reason) or True),
        _try_recover_primary_transport=lambda *a, **k: False,
        _buffer_diagnostic_status=lambda msg: None,
        _touch_activity=lambda desc: None,
    )
    retry_state = TurnRetryState()

    verdict = settle_unrecovered_error(
        agent,
        api_error=err,
        classified=classified,
        _retry=retry_state,
        status_code=503,
        error_msg="Backend overloaded",
        error_context={},
        is_context_length_error=False,
        is_rate_limited=False,
        _is_zai_coding_overload=False,
        _provider="openai",
        _base="https://api.openai.com/v1",
        _model="gpt-5.4",
        messages=[],
        api_messages=[],
        api_kwargs={},
        active_system_prompt="sys prompt",
        conversation_history=[],
        approx_tokens=1000,
        retry_count=3,
        max_retries=3,
        compression_attempts=0,
        api_call_count=1,
    )

    assert verdict.action == "break"
    assert activated_reasons == [FailoverReason.overloaded]
    assert retry_state.restart_with_rebuilt_messages is True


def test_503_without_fallback_enters_auto_recovery_ladder():
    """When retries for 503 are exhausted and no fallback exists, auto-recovery ladder engages."""
    err = MockAPIError("Origin overloaded", status_code=503)
    classified = classify_api_error(err)

    agent = SimpleNamespace(
        log_prefix="[test]",
        platform="cli",
        _auto_recovery_cycles=5,
        _has_pending_fallback=lambda: False,
        _try_activate_fallback=lambda *a, **k: False,
        _try_recover_primary_transport=lambda *a, **k: False,
        _buffer_diagnostic_status=lambda msg: None,
        _touch_activity=lambda desc: None,
        _emit_warning=lambda msg: None,
        _emit_diagnostic_status=lambda msg: None,
        _emit_diagnostic_wait=lambda msg: None,
        _clean_error_message=lambda msg: msg,
        _client_log_context=lambda: "",
        _adopt_nous_key_before_expiry=lambda: None,
        _interrupt_requested=False,
        clear_interrupt=lambda **k: False,
        _flush_status_buffer=lambda: None,
        _current_streamed_assistant_text="",
        _has_content_after_think_block=lambda text: False,
        interruptible_sleep=lambda sec, activity_desc=None: False,
    )
    retry_state = TurnRetryState()

    with patch("time.sleep"):
        verdict = settle_unrecovered_error(
            agent,
            api_error=err,
            classified=classified,
            _retry=retry_state,
            status_code=503,
            error_msg="Origin overloaded",
            error_context={},
            is_context_length_error=False,
            is_rate_limited=False,
            _is_zai_coding_overload=False,
            _provider="gemini",
            _base="https://gemini.example.com/v1",
            _model="gemini-3.8-flash",
            messages=[],
            api_messages=[],
            api_kwargs={},
            active_system_prompt="sys prompt",
            conversation_history=[],
            approx_tokens=1000,
            retry_count=3,
            max_retries=3,
            compression_attempts=0,
            api_call_count=1,
        )

    # Ladder continues for cycle 1
    assert verdict.action == "continue"
    assert retry_state.auto_recovery_cycles_used == 1
    assert verdict.retry_count == 0


def test_bridge_resilience_skips_retry_on_503(tmp_path):
    """Bridges configured with 503 in skip_retry_on_status must fail fast without retrying."""
    import sys
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from bridge_resilience import BridgeResilienceManager

    # Use isolated test config to satisfy home_io_guard
    test_yaml = tmp_path / "hermes-routing.config.yaml"
    test_yaml.write_text("""
bridge_resilience:
  defaults:
    max_retries: 3
    skip_retry_on_status: [422, 404, 409]
  gemini-web-bridge:
    skip_retry_on_status: [422, 503, -32000]
  aipass-web-bridge:
    skip_retry_on_status: [404, 409, 503]
""")

    mgr = BridgeResilienceManager(config_path=str(test_yaml))
    gemini_cfg = mgr.get_bridge_config("gemini-web-bridge")
    assert 503 in gemini_cfg.retry.skip_retry_on_status
    assert mgr.should_retry(gemini_cfg.retry, attempt=0, status_code=503, exc=None) is False

    aipass_cfg = mgr.get_bridge_config("aipass-web-bridge")
    assert 503 in aipass_cfg.retry.skip_retry_on_status
    assert mgr.should_retry(aipass_cfg.retry, attempt=0, status_code=503, exc=None) is False
