"""Terminal failed-turn results carry a specific ``failure_reason`` and plain-language chat
copy on every surface (the CLI-only 💡 hints used to be the only place with a next step).

Behaviour contracts, not snapshots: each test asserts the relationship between a result
and what ``agent/error_surface.py`` derives from it, plus the presence of the exact command
the user is told to run.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.error_classifier import classify_api_error
from agent.error_surface import LAYER_GATEWAY, LAYER_PROVIDER, build_error_surface_from_result
from agent.turn_loop_errors import handle_outer_loop_error
from agent.turn_recovery import max_retries_exhausted_result, nonretryable_client_error_result
from agent.turn_failure_copy import SITE_FAILURE_CODES, provider_label_for
from agent.turn_response_check import retry_invalid_response


class _Agent:
    log_prefix = ""
    verbose = False
    verbose_logging = False
    provider = "openrouter"
    model = "gpt-5-turbo"
    base_url = "https://openrouter.ai/api/v1"
    max_iterations = 30
    suppress_status_output = True
    _fallback_chain = ()
    _fallback_index = 0

    def _summarize_api_error(self, error):
        return str(error)

    def _clean_error_message(self, msg):
        return msg

    def _has_pending_fallback(self):
        return False

    def _try_activate_fallback(self):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Http(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def _nonretryable(status, message, provider="openrouter", model="gpt-5-turbo", agent=None):
    error = _Http(status, message)
    classified = classify_api_error(error, provider=provider, model=model)
    return nonretryable_client_error_result(
        agent or _Agent(), error, classified, status_code=status, api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=1, approx_tokens=10, provider=provider,
        base_url="https://openrouter.ai/api/v1", model=model,
    )


def test_model_not_found_chat_text_points_at_model_picker_not_http():
    result = _nonretryable(404, "HTTP 404: The model `gpt-5-turbo` does not exist")
    text = result["final_response"]
    assert "/model" in text and "gpt-5-turbo" in text
    assert not text.startswith("HTTP")
    assert result["failure_reason"] == "model_not_found"
    assert build_error_surface_from_result(result, provider="openrouter")["retryable"] is False


def test_api_key_rejection_chat_text_names_the_fix_and_the_provider_label():
    result = _nonretryable(401, "HTTP 401: Invalid API key provided")
    text = result["final_response"]
    assert "hermes setup" in text and "OpenRouter" in text
    assert "Provider said:" in text  # raw detail demoted to a trailing line
    assert text.index("hermes setup") < text.index("Provider said:")


def test_oauth_rejection_chat_text_names_the_provider_slug_and_the_failing_profile(tmp_path, monkeypatch):
    """A revoked Codex grant must send the user to THAT profile's own sign-in (profiles are
    islands, 93889b770da) and put the provider slug in the text the goal judge reads (#114012)."""
    profile_home = tmp_path / ".hermes" / "profiles" / "codex"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    hints = []

    class _Recorder(_Agent):
        def _vprint(self, msg, **_kw):
            hints.append(msg)

    result = _nonretryable(
        401, "HTTP 401: Encountered invalidated oauth token for user, failing request (code: token_revoked)",
        provider="openai-codex", model="gpt-5.6-sol", agent=_Recorder(),
    )
    text = result["final_response"]
    assert "`hermes -p codex auth add openai-codex --type oauth`" in text
    assert "<provider>" not in text
    assert "token_revoked" in text  # the raw error survives for the judge to quote
    # The CLI 💡 hint names the same command; it no longer sends the user to a bare `hermes auth`.
    cli_hint = "\n".join(hints)
    assert "`hermes -p codex auth add openai-codex --type oauth`" in cli_hint, cli_hint
    assert "`hermes auth`" not in cli_hint, cli_hint


def test_max_retries_exhausted_chat_text_has_next_step_and_no_mechanism_lead():
    error = _Http(503, "HTTP 503: upstream unavailable")
    classified = classify_api_error(error, provider="openrouter", model="m")
    result = max_retries_exhausted_result(
        _Agent(), error, classified, max_retries=3, is_rate_limited=False, error_msg=str(error).lower(),
        api_kwargs=None, api_messages=[], messages=[], conversation_history=None, api_call_count=1,
        approx_tokens=10, provider="openrouter", base_url="https://openrouter.ai/api/v1", model="m",
    )
    text = result["final_response"]
    assert "/retry" in text and "/model" in text and "hermes fallback add" in text
    assert not text.startswith("API call failed")
    assert result["failure_reason"] == classified.reason.value
    assert result["failure_retryable"] is True


def test_exhausted_retry_collapses_stream_continuation_into_durable_partial():
    error = _Http(503, "HTTP 503: upstream unavailable")
    classified = classify_api_error(error, provider="openrouter", model="m")
    messages = [
        {"role": "user", "content": "Write a long answer"},
        {
            "role": "assistant",
            "content": "First half",
            "_length_continuation_fragment": True,
        },
        {
            "role": "user",
            "content": "Continue exactly where you left off",
            "_length_continuation_nudge": True,
        },
        {
            "role": "assistant",
            "content": "second half",
            "_length_continuation_fragment": True,
        },
        {
            "role": "user",
            "content": "Continue exactly where you left off",
            "_length_continuation_nudge": True,
        },
    ]

    result = max_retries_exhausted_result(
        _Agent(), error, classified, max_retries=3, is_rate_limited=False,
        error_msg=str(error).lower(), api_kwargs=None, api_messages=[], messages=messages,
        conversation_history=None, api_call_count=3, approx_tokens=10, provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="m", current_turn_user_idx=0,
    )

    assert result["partial"] is True
    assert result["final_response"] == "First half\nsecond half"
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "First half\nsecond half"
    assert result["messages"][-1]["finish_reason"] == "error"
    assert not any(
        message.get("_length_continuation_fragment") or message.get("_length_continuation_nudge")
        for message in result["messages"]
    )


def test_exhausted_retry_preserves_prior_turn_continuation_marker():
    error = _Http(503, "HTTP 503: upstream unavailable")
    classified = classify_api_error(error, provider="openrouter", model="m")
    prior_fragment = {
        "role": "assistant",
        "content": "Earlier crash fragment",
        "_length_continuation_fragment": True,
    }
    messages = [
        {"role": "user", "content": "Earlier turn"},
        prior_fragment,
        {"role": "user", "content": "Current turn"},
        {
            "role": "assistant",
            "content": "Visible partial",
            "_length_continuation_fragment": True,
        },
        {
            "role": "user",
            "content": "Continue exactly where you left off",
            "_length_continuation_nudge": True,
        },
    ]

    result = max_retries_exhausted_result(
        _Agent(), error, classified, max_retries=3, is_rate_limited=False,
        error_msg=str(error).lower(), api_kwargs=None, api_messages=[], messages=messages,
        conversation_history=None, api_call_count=3, approx_tokens=10, provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="m", current_turn_user_idx=2,
    )

    assert result["final_response"] == "Visible partial"
    assert result["messages"].count(prior_fragment) == 1
    assert result["messages"][-1]["content"] == "Visible partial"
    assert not any(
        message.get("_length_continuation_fragment") or message.get("_length_continuation_nudge")
        for message in result["messages"][3:]
    )


def test_exhausted_plan_quota_429_names_the_reset_window_not_wait_a_minute():
    """The real usage-limit envelope: ``_summarize_api_error`` reduces the body to ``HTTP 429: The
    usage limit has been reached``, so the reset must travel through the classifier, not the text (#89401)."""
    import httpx
    import openai
    from agent.api_error_summary import ApiErrorSummaryMixin

    body = {"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached",
                      "resets_in_seconds": 30995, "plan_type": "pro"}}
    response = httpx.Response(429, json=body, request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"))
    error = openai.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)
    classified = classify_api_error(error, provider="openai-codex", model="gpt-5.3-codex")
    agent = _Agent()
    agent._summarize_api_error = ApiErrorSummaryMixin._summarize_api_error
    result = max_retries_exhausted_result(
        agent, error, classified, max_retries=3, is_rate_limited=True, error_msg=str(error).lower(),
        api_kwargs=None, api_messages=[], messages=[], conversation_history=None, api_call_count=3,
        approx_tokens=10, provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex", model="gpt-5.3-codex",
    )
    text = result["final_response"]
    assert result["error"] == "HTTP 429: The usage limit has been reached"
    assert "resets in ~9h" in text and "/retry" in text and "/model" in text
    assert "Wait a minute" not in text
    # A throttle with no reset window keeps the short-wait copy.
    short = _Http(429, "HTTP 429: Rate limit exceeded")
    plain = max_retries_exhausted_result(
        _Agent(), short, classify_api_error(short, provider="openrouter", model="m"), max_retries=3,
        is_rate_limited=True, error_msg=str(short).lower(), api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=3, approx_tokens=10, provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="m",
    )
    assert "Wait a minute" in plain["final_response"] and "resets in" not in plain["final_response"]


def test_invalid_response_stamps_reason_from_embedded_provider_code():
    """An HTTP-200 body carrying a 429 is rate limiting for the UI, not 'unknown'."""
    agent = _Agent()
    response = SimpleNamespace(error=SimpleNamespace(code=429, metadata={"provider_name": "Acme"}), choices=[])
    verdict = retry_invalid_response(
        agent, response=response, error_details=["no choices"],
        _retry=SimpleNamespace(restart_with_redirected_messages=False), thinking_spinner=None,
        messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None, conversation_history=None,
        retry_count=2, max_retries=3, compression_attempts=0, api_call_count=1, api_request_id="r",
        api_start_time=0.0, api_duration=0.4, effective_task_id="t", turn_id="turn",
    )
    assert verdict.action == "return"
    result = verdict.result
    assert result["failure_reason"] == "rate_limit"
    assert "/retry" in result["final_response"] and "Acme" in result["final_response"]
    assert "Invalid API response" not in result["final_response"]


def test_outer_loop_error_copy_has_no_apology_and_routes_to_gateway_layer():
    """Deterministic local bugs are failed, non-retryable turns with a plain what-now."""
    agent = _Agent()
    try:
        raise TypeError("expected str, got list")
    except TypeError as exc:
        verdict = handle_outer_loop_error(
            agent, e=exc, _outer_error_count=7, api_call_count=2, messages=[], conversation_history=None,
            _turn_exit_reason="unknown", failed=False, final_response=None,
        )
    assert verdict.action == "break" and verdict.failed is True
    text = verdict.final_response
    assert "apologize" not in text.lower() and "OpenAI-compatible" not in text
    assert "hermes doctor" in text and "/new" in text
    assert text.rstrip().endswith("expected str, got list")  # raw detail last, not first
    from agent.turn_failure_copy import exit_reason_failure

    exit_failure = exit_reason_failure(verdict._turn_exit_reason)
    assert exit_failure.fails_turn is True  # the outer-error cap IS a failed turn
    surface = build_error_surface_from_result(
        {"failed": True, "error": text, "failure_reason": exit_failure.reason,
         "failure_retryable": exit_failure.retryable}
    )
    assert surface["layer"] == LAYER_GATEWAY and surface["code"] == "loop_error"


def test_invalid_response_copy_never_names_a_model_id_as_the_provider():
    """describe_invalid_response falls back to 'model=<id>' for OpenRouter bodies; that is not a
    provider name and must not be spliced into the sentence."""
    agent = _Agent()
    response = SimpleNamespace(error=None, choices=[], model="anthropic/claude-opus")
    verdict = retry_invalid_response(
        agent, response=response, error_details=["no choices"],
        _retry=SimpleNamespace(restart_with_redirected_messages=False), thinking_spinner=None,
        messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None, conversation_history=None,
        retry_count=2, max_retries=3, compression_attempts=0, api_call_count=1, api_request_id="r",
        api_start_time=0.0, api_duration=0.4, effective_task_id="t", turn_id="turn",
    )
    text = verdict.result["final_response"]
    assert "model=" not in text
    assert text.startswith(provider_label_for(agent.provider))


def test_interpreter_shutdown_copy_substitutes_the_real_session_id():
    agent = _Agent()
    agent.session_id = "20260914_abc"
    verdict = handle_outer_loop_error(
        agent, e=RuntimeError("cannot schedule new futures after interpreter shutdown"),
        _outer_error_count=0, api_call_count=1, messages=[], conversation_history=None,
        _turn_exit_reason="unknown", failed=False, final_response=None,
    )
    assert "hermes --resume 20260914_abc" in verdict.final_response
    assert "<session-id>" not in verdict.final_response


@pytest.mark.parametrize("code", sorted(SITE_FAILURE_CODES))
def test_site_failure_codes_never_collapse_to_unknown(code):
    """Every site code is listed in error_surface's layer table (no fall-through guesswork)."""
    from agent.error_surface import _REASON_TO_LAYER

    surface = build_error_surface_from_result({"failed": True, "error": "x", "failure_reason": code})
    assert surface["code"] == code
    assert code in _REASON_TO_LAYER
    assert surface["layer"] in (LAYER_GATEWAY, LAYER_PROVIDER)


def test_model_caused_codes_stay_on_the_provider_layer_and_runtime_codes_on_gateway():
    """Cut-off / empty / broken replies come from the model (provider layer, so the client's
    per-code copy applies); a busy session or loop bug is Hermes-side (gateway layer, so the
    client never offers Switch provider for it)."""
    layers = {c: build_error_surface_from_result({"failed": True, "error": "x", "failure_reason": c})["layer"]
              for c in ("truncated", "empty_response", "invalid_response", "session_busy", "loop_error")}
    assert layers["truncated"] == layers["empty_response"] == layers["invalid_response"] == LAYER_PROVIDER
    assert layers["session_busy"] == layers["loop_error"] == LAYER_GATEWAY
