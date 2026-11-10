"""Behaviour contract for the non-retryable API-error echo burn loop (#143940).

When a messaging surface re-admits a terminal 401/402/403 notice as a user turn,
the gateway must stop the loop: suppress the diagnostic status, discard echo
follow-ups after a terminal failure, and hold the session against matching
inbound until a dissimilar human message arrives.
"""

from types import SimpleNamespace

import pytest

from gateway.api_error_echo import (
    HOLD_META_KEY,
    api_error_hold_active,
    arm_api_error_hold,
    clear_api_error_hold,
    discard_pending_api_error_echo,
    is_terminal_nonretryable_api_failure,
    looks_like_api_error_echo,
    looks_like_terminal_nonretryable_diagnostic,
    maybe_arm_hold_from_result,
    should_drop_text_under_api_error_hold,
)
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message, _sanitize_gateway_final_response


DIAGNOSTIC_402 = (
    "❌ custom rejected the request and retrying won't help: HTTP 402 "
    "spend limit exceeded"
)
BILLING_FINAL = "Billing or credits exhausted: HTTP 402 spend limit exceeded"
SANITIZED_LOOP = (
    "⚠️ The AI model service kept failing. Use /retry to try again, or /model "
    "to switch models. Details are in the gateway log (`hermes logs`)."
)


@pytest.mark.parametrize(
    "text",
    [
        DIAGNOSTIC_402,
        BILLING_FINAL,
        "❌ Billing or credits exhausted — HTTP 402: out of credits",
        "HTTP 402: payment required",
        "Error: HTTP 401 unauthorized",
    ],
)
def test_terminal_diagnostic_shapes_are_recognized(text):
    assert looks_like_terminal_nonretryable_diagnostic(text)
    assert looks_like_api_error_echo(text)


def test_ordinary_prose_citing_http_is_not_a_diagnostic():
    prose = (
        "When the docs say the server returns HTTP 402 for spend limits, "
        "they mean you should top up — not that Hermes should loop."
    )
    assert not looks_like_terminal_nonretryable_diagnostic(prose)
    assert not looks_like_api_error_echo(prose)


def test_sanitized_gateway_reply_is_echo_under_hold_only():
    assert looks_like_api_error_echo(SANITIZED_LOOP)
    # Status suppression uses the stricter diagnostic predicate so ordinary
    # chat is never silenced by the sanitized copy alone.
    assert not looks_like_terminal_nonretryable_diagnostic(SANITIZED_LOOP)


@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SLACK, Platform.FEISHU, "discord"])
def test_messaging_status_suppresses_terminal_nonretryable_diagnostics(platform):
    assert _prepare_gateway_status_message(platform, "lifecycle", DIAGNOSTIC_402) is None
    assert _prepare_gateway_status_message(platform, "lifecycle", BILLING_FINAL) is None


def test_local_surface_keeps_raw_diagnostic_status():
    # CLI/local keep the diagnostic stream for operators.
    assert _prepare_gateway_status_message(Platform.LOCAL, "lifecycle", DIAGNOSTIC_402) == DIAGNOSTIC_402


def test_final_response_sanitizes_billing_envelope_on_messaging():
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, BILLING_FINAL)
    assert sanitized
    assert "HTTP 402" not in sanitized
    assert looks_like_api_error_echo(sanitized)


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"failed": True, "failure_reason": "billing", "failure_retryable": False}, True),
        ({"failed": True, "failure_reason": "auth_permanent", "failure_retryable": False}, True),
        ({"failed": True, "failure_reason": "auth", "failure_retryable": False}, True),
        ({"failed": True, "billing_block": {"url": "https://example"}, "failure_retryable": False}, True),
        ({"failed": True, "failure_reason": "rate_limit", "failure_retryable": True}, False),
        ({"failed": True, "failure_reason": "server_error", "failure_retryable": True}, False),
        ({"failed": False, "final_response": "ok"}, False),
        (None, False),
    ],
)
def test_terminal_failure_detection(result, expected):
    assert is_terminal_nonretryable_api_failure(result) is expected


def test_discard_pending_echo_after_terminal_failure():
    result = {"failed": True, "failure_reason": "billing", "failure_retryable": False}
    assert discard_pending_api_error_echo(result, DIAGNOSTIC_402) is True
    assert discard_pending_api_error_echo(result, "please continue") is False
    assert discard_pending_api_error_echo(
        {"failed": True, "failure_reason": "rate_limit", "failure_retryable": True},
        DIAGNOSTIC_402,
    ) is False


def test_hold_arms_drops_echo_and_clears_on_human_message():
    entry = SimpleNamespace(metadata={})
    assert maybe_arm_hold_from_result(
        entry, {"failed": True, "failure_reason": "billing", "failure_retryable": False}
    )
    assert api_error_hold_active(entry)
    assert HOLD_META_KEY in entry.metadata

    drop, cleared = should_drop_text_under_api_error_hold(entry, DIAGNOSTIC_402)
    assert drop is True and cleared is False
    assert api_error_hold_active(entry)

    drop, cleared = should_drop_text_under_api_error_hold(entry, "I topped up, try again")
    assert drop is False and cleared is True
    assert not api_error_hold_active(entry)


def test_hold_allows_slash_commands_without_clearing():
    entry = SimpleNamespace(metadata={})
    arm_api_error_hold(entry, reason="billing")
    drop, cleared = should_drop_text_under_api_error_hold(
        entry, "/status", is_command=True,
    )
    assert drop is False and cleared is False
    assert api_error_hold_active(entry)
    clear_api_error_hold(entry)
    assert not api_error_hold_active(entry)


def test_ordinary_final_response_quoting_billing_copy_is_not_replaced():
    reply = "The message 'Billing or credits exhausted' means the account has no credit."
    assert not looks_like_terminal_nonretryable_diagnostic(reply)
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, reply) == reply


def test_human_question_quoting_retry_diagnostic_releases_hold():
    entry = SimpleNamespace(metadata={})
    arm_api_error_hold(entry, reason="billing")
    question = "What does 'retrying won't help' mean for my API key?"
    assert not looks_like_api_error_echo(question)
    assert should_drop_text_under_api_error_hold(entry, question) == (False, True)
    assert not api_error_hold_active(entry)


# Real terminal envelopes from the emitters, beyond the fixtures at the top of
# this file. These must stay recognized after the predicate is anchored.
_UNVERIFIED_BILLING_FINAL = (
    "Provider reported usage/credit exhaustion (unverified — the same "
    "error can be a content-filter rejection, not billing): HTTP 402 spend limit exceeded"
)
_UNVERIFIED_BILLING_STATUS = (
    "❌ Provider reported usage/credit exhaustion "
    "(unverified — may be a content-filter rejection) — HTTP 402 spend limit exceeded"
)
_NONRETRYABLE_COPY_FINAL = (
    "OpenRouter rejected the request and retrying won't help. Pick another model with /model, "
    "or check the details in `/home/user/.hermes/logs/agent.log`."
    "\n\nProvider said: HTTP 403 forbidden"
)
_POLICY_COPY_FINAL = (
    "OpenRouter refused this request because of a policy on your account (its data/privacy "
    "settings, or a block the model's upstream provider placed on the account), so the model "
    "didn't answer and retrying won't help. Check the account with the provider, or switch "
    "models with /model."
)


@pytest.mark.parametrize(
    "text",
    [
        _UNVERIFIED_BILLING_FINAL,
        _UNVERIFIED_BILLING_STATUS,
        _NONRETRYABLE_COPY_FINAL,
        _POLICY_COPY_FINAL,
    ],
)
def test_real_emitter_envelopes_stay_detected(text):
    assert looks_like_terminal_nonretryable_diagnostic(text)
    assert looks_like_api_error_echo(text)


@pytest.mark.parametrize(
    "text",
    [
        "The message 'Billing or credits exhausted' means the account has no credit.",
        "What does 'retrying won't help' mean for my API key?",
        "Notes: 'Provider reported usage/credit exhaustion' is the unverified wording.",
        "⚠️ Billing or credits exhausted — switching to fallback provider...",
        "⚠️ Provider reported usage/credit exhaustion "
        "(unverified — may be a content-filter rejection) "
        "— switching to fallback provider...",
    ],
)
def test_quoted_phrases_and_fallback_status_are_not_terminal_diagnostics(text):
    assert not looks_like_terminal_nonretryable_diagnostic(text)


def test_messaging_still_sanitizes_real_nonretryable_copy():
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, _NONRETRYABLE_COPY_FINAL)
    assert sanitized
    assert sanitized != _NONRETRYABLE_COPY_FINAL
    assert "retrying won't help" not in sanitized.lower()
    assert looks_like_api_error_echo(sanitized)


def test_messaging_still_sanitizes_unverified_billing_final():
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, _UNVERIFIED_BILLING_FINAL)
    assert sanitized
    assert sanitized != _UNVERIFIED_BILLING_FINAL
    assert "usage/credit exhaustion" not in sanitized.lower()
    assert looks_like_api_error_echo(sanitized)
