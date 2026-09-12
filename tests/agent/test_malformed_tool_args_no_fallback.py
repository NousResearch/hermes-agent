"""A 400 for the model's own malformed tool-call JSON must not walk the fallback chain (#12770)."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error


class _Err(Exception):
    status_code = 400
    response = None

    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body or {"error": {"message": message, "type": "invalid_request_error"}}


@pytest.mark.parametrize("wording", ["invalid tool call arguments", "Invalid function_call arguments"])
def test_malformed_tool_args_400_is_terminal_without_fallback(wording):
    """Large session included: the bare-ish message must not read as context overflow either."""
    verdict = classify_api_error(_Err(f"Error code: 400 - {wording}"), provider="ollama",
                                 approx_tokens=90_000, context_length=128_000, num_messages=120)
    assert verdict.reason is FailoverReason.format_error
    assert (verdict.retryable, verdict.should_compress, verdict.should_fallback) == (False, False, False)

    # Unrelated request-shape 400s keep their fallback (another provider may accept the request).
    assert classify_api_error(_Err("Unsupported parameter: 'max_tokens'")).should_fallback is True


class _Agent:
    """Only the fallback seam is real; every other helper the terminal path touches is a no-op."""
    log_prefix = ""
    verbose = False
    provider = "ollama"
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


def test_client_error_settlement_skips_fallback_when_classifier_says_so():
    agent = _Agent()
    retry = SimpleNamespace(copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=False)
    err = _Err("invalid tool call arguments")
    classified = classify_api_error(err, provider="ollama")

    with patch("agent.conversation_loop._is_copilot_provider", lambda a: False):
        verdict = settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=400, error_msg=str(err),
            is_context_length_error=False, is_rate_limited=False, _is_zai_coding_overload=False,
            _provider="ollama", _base="http://127.0.0.1:11434/v1", _model="glm", messages=[], api_messages=[],
            api_kwargs={}, active_system_prompt="", conversation_history=None, approx_tokens=10,
            retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
        )

    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == FailoverReason.format_error.value
    assert agent.activated == []
