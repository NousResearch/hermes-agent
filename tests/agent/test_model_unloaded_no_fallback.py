"""LM Studio unloaded-model errors must not activate the provider fallback chain."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error


class _LMStudioError(Exception):
    status_code = 404
    response = None

    def __init__(self):
        super().__init__("No models loaded")
        self.body = {"error": {"message": "No models loaded"}}


class _Agent:
    """Minimal fallback seam for the real terminal-error settlement path."""

    log_prefix = ""
    verbose = False
    provider = "lmstudio"
    _fallback_chain = [object()]
    _fallback_index = 0
    _credential_pool = None

    def __init__(self):
        self.activated = []

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(True)
        return True

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_model_unloaded_settlement_skips_fallback():
    agent = _Agent()
    error = _LMStudioError()
    classified = classify_api_error(error, provider="lmstudio")
    retry = SimpleNamespace(
        copilot_stale_cred_retry_attempted=False,
        primary_recovery_attempted=False,
    )

    assert classified.reason is FailoverReason.model_not_found
    assert classified.should_fallback is False
    assert agent._has_pending_fallback() is True
    fallback_index = agent._fallback_index

    with patch("agent.conversation_loop._is_copilot_provider", lambda candidate: False):
        verdict = settle_unrecovered_error(
            agent,
            api_error=error,
            classified=classified,
            _retry=retry,
            status_code=404,
            error_msg=str(error),
            is_context_length_error=False,
            is_rate_limited=False,
            _is_zai_coding_overload=False,
            _provider="lmstudio",
            _base="http://127.0.0.1:1234/v1",
            _model="local-model",
            messages=[],
            api_messages=[],
            api_kwargs={},
            active_system_prompt="",
            conversation_history=None,
            approx_tokens=10,
            retry_count=0,
            max_retries=3,
            compression_attempts=0,
            api_call_count=1,
        )

    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == FailoverReason.model_not_found.value
    assert agent.activated == []
    assert agent._fallback_index == fallback_index
    assert agent._has_pending_fallback() is True
