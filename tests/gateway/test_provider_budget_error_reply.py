"""Regression tests for gateway provider account/workspace budget-cap classification."""

import pytest

from gateway.config import Platform
from gateway.run import (
    _gateway_provider_error_reply,
    _looks_like_gateway_provider_error,
    _sanitize_gateway_final_response,
)

_OPENROUTER_BUDGET_RAW = (
    "Error code: 403 - {'error': {'message': "
    "'Workspace monthly budget of $30.00 exceeded. Contact your org admin.', "
    "'code': 403}}"
)


def test_budget_cap_error_gets_specific_safe_reply():
    reply = _gateway_provider_error_reply(_OPENROUTER_BUDGET_RAW)
    assert "budget" in reply.lower()
    assert "provider failed after retries" not in reply.lower()
    assert "$30.00" not in reply
    assert "Contact your org admin" not in reply


def test_budget_error_is_recognized_as_a_provider_error_envelope():
    assert _looks_like_gateway_provider_error(_OPENROUTER_BUDGET_RAW)


@pytest.mark.parametrize("phrase", [
    "insufficient credits",
    "spending limit exceeded",
    "credit limit reached",
    "billing quota exceeded",
])
def test_budget_phrase_variants_map_to_budget_reply(phrase):
    reply = _gateway_provider_error_reply(f"Error code: 402 - {phrase}. Please add funds.")
    assert "budget" in reply.lower()
    assert phrase not in reply.lower()


def test_ordinary_rate_limit_is_not_reclassified_as_budget():
    reply = _gateway_provider_error_reply(
        "Error code: 429 - Rate limit exceeded, please try again later."
    )
    assert "rate-limiting" in reply.lower()
    assert "budget" not in reply.lower()


def test_invalid_api_key_is_not_reclassified_as_budget():
    reply = _gateway_provider_error_reply(
        "Error code: 403 - Incorrect API key provided."
    )
    assert "authentication failed" in reply.lower()
    assert "budget" not in reply.lower()


def test_bare_quota_alone_stays_a_rate_limit():
    """A bare 'quota' with no budget/credit/spending/billing context stays rate-limit."""
    reply = _gateway_provider_error_reply("Error code: 429 - quota exceeded for this minute.")
    assert "rate-limiting" in reply.lower()
    assert "budget" not in reply.lower()


def test_telegram_final_response_sanitizes_budget_error():
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, _OPENROUTER_BUDGET_RAW)
    assert "budget" in sanitized.lower()
    assert "$30.00" not in sanitized
    assert "Contact your org admin" not in sanitized
    assert "Workspace" not in sanitized


def test_local_surface_keeps_raw_budget_error():
    assert _sanitize_gateway_final_response("local", _OPENROUTER_BUDGET_RAW) == _OPENROUTER_BUDGET_RAW
