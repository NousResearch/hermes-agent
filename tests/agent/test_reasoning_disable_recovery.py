"""Main-loop recovery for a route that rejects a reasoning disable by field name (#114460).

The classifier maps the rejection to ``reasoning_mandatory``; ``recover_after_classification``
drops the disable once; a SECOND rejection in the same turn means the configured reasoning
controls themselves are refused, so the settle path takes the fallback chain instead of
replaying the identical request until ``max_retries``.
"""
from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_retry_state import TurnRetryState

_REVERSED_400 = "reasoning_effort 'none' unsupported; use minimal|low|medium|high|xhigh"


class _FakeApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(f"Error code: {status_code} - {{'error': {{'message': {message!r}}}}}")
        self.status_code = status_code
        self.message = message
        self.body = {"error": {"message": message, "type": "invalid_request_error"}}


class _Agent:
    log_prefix = ""
    verbose = False
    provider = "custom"
    model = "chat-only"
    _fallback_chain = [object()]
    _fallback_index = 0
    _credential_pool = None

    def __init__(self):
        self.activated = []
        self._reasoning_disable_rejected = False

    def _recover_with_credential_pool(self, **kwargs):
        return False, False

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(True)
        return True

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(disable_drop_attempted):
    from agent.turn_api_error import settle_unrecovered_error

    agent = _Agent()
    err = _FakeApiError(400, _REVERSED_400)
    classified = classify_api_error(err, provider="custom", model=agent.model)
    assert classified.reason == FailoverReason.reasoning_mandatory
    retry = SimpleNamespace(
        reasoning_mandatory_retry_attempted=disable_drop_attempted, image_shrink_retry_attempted=False,
        copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=False,
        restart_with_redirected_messages=False,
    )
    with patch("agent.conversation_loop._is_copilot_provider", lambda a: False), patch(
        "agent.conversation_loop._arm_fallback_restart", lambda agent, msgs, prompt, retry: prompt
    ), patch("agent.turn_api_error.compute_error_backoff", lambda *a, **k: 0), patch(
        "agent.turn_api_error.interruptible_backoff_sleep", lambda *a, **k: None
    ):
        verdict = settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=400, error_msg=str(err),
            is_context_length_error=False, is_rate_limited=False, _is_zai_coding_overload=False,
            _provider="custom", _base="http://relay.example/v1", _model=agent.model, messages=[],
            api_messages=[], api_kwargs={}, active_system_prompt="", conversation_history=None,
            approx_tokens=10, retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
        )
    return agent, verdict


def test_spent_disable_drop_falls_back_instead_of_replaying_the_request():
    agent, verdict = _settle(disable_drop_attempted=True)
    assert verdict.action == "break"
    assert agent.activated == [True]

    # Control: before the drop has run the verdict stays retryable (the rung gets its shot;
    # the one-shot drop itself is pre-existing ``turn_recovery`` behaviour, not asserted here).
    agent, verdict = _settle(disable_drop_attempted=False)
    assert verdict.action == "fallthrough"
    assert agent.activated == []


# #100536: a Responses relay rejects an unsupported reasoning LEVEL (``reasoning.effort: max`` on
# an ENABLED config) with a message-less structured 400. The classifier routes it to
# reasoning_mandatory; the rung must drop the effort for the retry instead of resending it.
_STRUCTURED_LEVEL_400 = (
    "Error code: 400 - {'error': {'param': 'reasoning.effort', "
    "'error_code': 'invalid_reasoning_effort', 'retryable': False}}"
)


class _WireAgent(_Agent):
    api_mode = "chat_completions"
    base_url = "http://relay.example/v1"
    request_overrides = None

    def __init__(self, reasoning_config):
        super().__init__()
        self.reasoning_config = reasoning_config
        self.notices = []
        # ``_Agent.__getattr__`` fabricates callables for unknown names; wire flags must read False.
        self._ephemeral_reasoning_off = False
        self._reasoning_effort_rejected = False
        self._fast_until = 0.0
        self.service_tier = None

    def _vprint(self, line, **kwargs):
        self.notices.append(line)


def _wire_reasoning_config(agent):
    """What the main-loop request builder hands the transport as ``reasoning_config``."""
    from agent.chat_completion_helpers import _build_api_kwargs_for_mode

    def _capture(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id):
        return {"reasoning_config": reasoning_config}

    with patch("agent.chat_completion_helpers._build_chat_completions_kwargs", _capture):
        return _build_api_kwargs_for_mode(agent, [], [])["reasoning_config"]


def _recover(agent, message):
    from agent.turn_recovery import recover_after_classification

    err = _FakeApiError(400, message)
    classified = classify_api_error(err, provider="custom", model=agent.model)
    assert classified.reason == FailoverReason.reasoning_mandatory
    retry = TurnRetryState()
    with patch("hermes_cli.models_reasoning_caps.refresh_reasoning_caps_async", lambda provider: None):
        retry_now, _ = recover_after_classification(
            agent, err, classified, retry, status_code=400, error_context={}, messages=[], api_messages=[],
        )
    assert retry_now is True
    return agent


def test_rejected_reasoning_level_retries_without_the_effort():
    agent = _WireAgent({"enabled": True, "effort": "max"})
    assert _wire_reasoning_config(agent) == {"enabled": True, "effort": "max"}  # the request that 400ed
    _recover(agent, _STRUCTURED_LEVEL_400)
    assert _wire_reasoning_config(agent) is None, "retry must not resend reasoning.effort=max"
    assert any("rejects reasoning effort max" in n for n in agent.notices)
    assert not any("rejects disabling reasoning" in n for n in agent.notices)

    # Control: a rejected DISABLE keeps the existing contract — the user's own enabled config
    # goes out verbatim on the retry and the notice names the disable.
    agent = _WireAgent({"enabled": True, "effort": "high"})
    agent._ephemeral_reasoning_off = True
    assert _wire_reasoning_config(agent) == {"enabled": False, "effort": "none"}  # the request that 400ed
    _recover(agent, _REVERSED_400)
    assert _wire_reasoning_config(agent) == {"enabled": True, "effort": "high"}
    assert any("rejects disabling reasoning" in n for n in agent.notices)
